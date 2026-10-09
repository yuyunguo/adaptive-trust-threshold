"""
eICU Feature Extractor
Extracts clinical features from the eICU Collaborative Research Database,
aligned to the same feature schema as the MIMIC-III/IV pipeline.

Key schema differences handled:
  - eICU uses `patientunitstayid` as primary key (not subject_id/hadm_id)
  - Time offsets are in minutes from ICU admission (not timestamps)
  - Labs use text labels (labname) instead of numeric itemids
  - Diagnoses use free-text diagnosisstring (not ICD codes)
  - Age stored as varchar with '> 89' for nonagenarians
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional


class EicuFeatureExtractor:
    """Feature extraction from eICU database, aligned to MIMIC schema."""

    def __init__(self, connection, schema: str = 'eicu_crd'):
        self.connection = connection
        self.schema = schema
        self.logger = logging.getLogger(__name__)
        self.extraction_config = {
            'min_age': 18,
            'max_age': 100,
            'min_icu_los_minutes': 60,
            'vital_window_minutes': 1440,   # 24 hours
            'lab_window_minutes': 1440,     # 24 hours
            'prolonged_stay_minutes': 4320, # 72 hours
            'max_missing_rate': 0.8,
        }

    def extract_cohort(self, limit: int = 5000, seed: int = 42) -> pd.DataFrame:
        self.logger.info(f"Extracting eICU cohort (n={limit}, seed={seed})")

        cursor = self.connection.cursor()
        try:
            # Step 1: Patient base with outcomes
            self.logger.debug("Building patient base temp table")
            cursor.execute(self._patient_base_query(limit, seed))
            n_base = cursor.rowcount
            self.logger.info(f"Patient base: {n_base} patients")

            # Step 2: Vital signs from vitalperiodic (first 24h)
            self.logger.debug("Extracting vital signs")
            df_vs = self._fetch_df(cursor, self._vital_signs_query())
            self.logger.info(f"Vital signs: {len(df_vs)} rows")

            # Step 3: Labs (first 24h)
            self.logger.debug("Extracting lab features")
            df_lab = self._fetch_df(cursor, self._lab_query())
            self.logger.info(f"Lab features: {len(df_lab)} rows")

            # Step 4: Diagnoses
            self.logger.debug("Extracting diagnosis features")
            df_diag = self._fetch_df(cursor, self._diagnosis_query())
            self.logger.info(f"Diagnosis features: {len(df_diag)} rows")

            # Step 5: Medications
            self.logger.debug("Extracting medication features")
            df_med = self._fetch_df(cursor, self._medication_query())
            self.logger.info(f"Medication features: {len(df_med)} rows")

            # Step 6: Patient base features
            cursor.execute(self._pb_features_query())
            df_pb = self._cursor_to_df(cursor)
            self.logger.info(f"Patient base features: {len(df_pb)} rows")

            # Cleanup
            cursor.execute("DROP TABLE IF EXISTS _eicu_pb")

            # Merge
            df = df_pb
            for sub_df in [df_vs, df_lab, df_diag, df_med]:
                if not sub_df.empty:
                    df = df.merge(sub_df, on='patientunitstayid', how='left')

            self.logger.info(f"Merged: {len(df)} rows, {len(df.columns)} cols")

            # Rename for MIMIC alignment
            df = self._align_columns(df)

            # Fill NaN counts with 0 (patients without those records)
            count_cols = [
                'unique_lab_tests', 'abnormal_labs_count', 'unique_diagnoses',
                'insulin_orders', 'heparin_orders', 'vasopressor_orders',
                'unique_medications',
            ]
            for c in count_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(0)

            # Data quality
            df = self._validate_data_quality(df)

            self.logger.info(
                f"Final eICU cohort: {len(df)} patients, "
                f"mortality={df['mortality'].mean():.2%}, "
                f"prolonged_stay={df['prolonged_stay'].mean():.2%}"
            )
            return df

        finally:
            try:
                cursor.execute("DROP TABLE IF EXISTS _eicu_pb")
            except Exception:
                pass
            cursor.close()

    # ── SQL queries ──────────────────────────────────────────────

    def _patient_base_query(self, limit: int, seed: int) -> str:
        return f"""
        SELECT
            p.patientunitstayid,
            CASE WHEN p.age = '> 89' THEN 90
                 WHEN p.age ~ '^[0-9]+$' THEN p.age::numeric
                 ELSE NULL END as age,
            CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END as gender,
            p.unitdischargeoffset / 60.0 as icu_los_hours,
            CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END as mortality,
            CASE WHEN p.unitdischargestatus = 'Expired' THEN 1 ELSE 0 END as unit_mortality,
            CASE WHEN p.hospitaladmitsource ILIKE '%emergency%'
                   OR p.hospitaladmitsource ILIKE '%ed%'
                   OR p.hospitaladmitsource ILIKE '%direct admit%er%'
                 THEN 1 ELSE 0 END as emergency_admit,
            CASE WHEN p.unitdischargeoffset > {self.extraction_config['prolonged_stay_minutes']}
                 THEN 1 ELSE 0 END as prolonged_stay,
            p.unittype
        INTO TEMP TABLE _eicu_pb
        FROM {self.schema}.patient p
        WHERE (p.age ~ '^[0-9]+$' OR p.age = '> 89')
          AND p.unitdischargeoffset > {self.extraction_config['min_icu_los_minutes']}
          AND CASE WHEN p.age = '> 89' THEN 90
                   WHEN p.age ~ '^[0-9]+$' THEN p.age::numeric
                   ELSE NULL END
              BETWEEN {self.extraction_config['min_age']} AND {self.extraction_config['max_age']}
        ORDER BY RANDOM()
        LIMIT {limit}
        """

    def _vital_signs_query(self) -> str:
        w = self.extraction_config['vital_window_minutes']
        return f"""
        WITH vp_agg AS (
            SELECT
                pb.patientunitstayid,
                AVG(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_mean,
                MIN(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_min,
                MAX(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_max,
                STDDEV(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_std,
                AVG(CASE WHEN vp.systemicsystolic > 0 THEN vp.systemicsystolic END) as sbp_mean,
                MIN(CASE WHEN vp.systemicsystolic > 0 THEN vp.systemicsystolic END) as sbp_min,
                MAX(CASE WHEN vp.systemicsystolic > 0 THEN vp.systemicsystolic END) as sbp_max,
                AVG(CASE WHEN vp.systemicdiastolic > 0 THEN vp.systemicdiastolic END) as dbp_mean,
                AVG(CASE WHEN vp.respiration > 0 THEN vp.respiration END) as resp_mean,
                AVG(CASE WHEN vp.temperature > 20 AND vp.temperature < 45 THEN vp.temperature END) as temp_mean,
                AVG(CASE WHEN vp.sao2 > 0 THEN vp.sao2 END) as spo2_mean
            FROM _eicu_pb pb
            JOIN {self.schema}.vitalperiodic vp
                ON pb.patientunitstayid = vp.patientunitstayid
            WHERE vp.observationoffset <= {w}
            GROUP BY pb.patientunitstayid
        ),
        va_agg AS (
            SELECT
                pb.patientunitstayid,
                AVG(CASE WHEN va.noninvasivesystolic > 0 THEN va.noninvasivesystolic END) as nibp_sbp,
                MIN(CASE WHEN va.noninvasivesystolic > 0 THEN va.noninvasivesystolic END) as nibp_sbp_min,
                MAX(CASE WHEN va.noninvasivesystolic > 0 THEN va.noninvasivesystolic END) as nibp_sbp_max,
                AVG(CASE WHEN va.noninvasivediastolic > 0 THEN va.noninvasivediastolic END) as nibp_dbp
            FROM _eicu_pb pb
            JOIN {self.schema}.vitalaperiodic va
                ON pb.patientunitstayid = va.patientunitstayid
            WHERE va.observationoffset <= {w}
            GROUP BY pb.patientunitstayid
        )
        SELECT
            COALESCE(v.patientunitstayid, n.patientunitstayid) as patientunitstayid,
            v.hr_mean, v.hr_min, v.hr_max, v.hr_std,
            COALESCE(v.sbp_mean, n.nibp_sbp) as sbp_mean,
            COALESCE(v.sbp_min, n.nibp_sbp_min) as sbp_min,
            COALESCE(v.sbp_max, n.nibp_sbp_max) as sbp_max,
            v.resp_mean, v.temp_mean, v.spo2_mean,
            COALESCE(v.dbp_mean, n.nibp_dbp) as dbp_mean
        FROM vp_agg v
        FULL OUTER JOIN va_agg n ON v.patientunitstayid = n.patientunitstayid
        """

    def _lab_query(self) -> str:
        w = self.extraction_config['lab_window_minutes']
        return f"""
        SELECT
            pb.patientunitstayid,
            AVG(CASE WHEN l.labname = 'creatinine' THEN l.labresult END) as creatinine_mean,
            AVG(CASE WHEN l.labname = 'potassium' THEN l.labresult END) as potassium_mean,
            AVG(CASE WHEN l.labname = 'sodium' THEN l.labresult END) as sodium_mean,
            AVG(CASE WHEN l.labname = 'BUN' THEN l.labresult END) as bun_mean,
            AVG(CASE WHEN l.labname ILIKE '%bilirubin%' THEN l.labresult END) as bilirubin_mean,
            AVG(CASE WHEN l.labname = 'pH' THEN l.labresult END) as ph_mean,
            AVG(CASE WHEN l.labname = 'chloride' THEN l.labresult END) as chloride_mean,
            AVG(CASE WHEN l.labname = 'glucose' THEN l.labresult END) as glucose_mean,
            AVG(CASE WHEN l.labname = 'Hgb' THEN l.labresult END) as hemoglobin_mean,
            AVG(CASE WHEN l.labname ILIKE '%WBC%' THEN l.labresult END) as wbc_mean,
            COUNT(DISTINCT l.labname) as unique_lab_tests,
            COUNT(CASE WHEN l.labresulttext ILIKE '%abnormal%' OR l.labresulttext ILIKE '%high%' OR l.labresulttext ILIKE '%low%' THEN 1 END) as abnormal_labs_count
        FROM _eicu_pb pb
        JOIN {self.schema}.lab l
            ON pb.patientunitstayid = l.patientunitstayid
        WHERE l.labresultoffset <= {w}
          AND l.labresult IS NOT NULL
          AND l.labresult > 0
        GROUP BY pb.patientunitstayid
        """

    def _diagnosis_query(self) -> str:
        return f"""
        SELECT
            pb.patientunitstayid,
            COUNT(DISTINCT d.diagnosisstring) as unique_diagnoses,
            MAX(CASE WHEN d.diagnosisstring ILIKE '%diabet%' THEN 1 ELSE 0 END) as diabetes,
            MAX(CASE WHEN d.diagnosisstring ILIKE '%hypertension%' THEN 1 ELSE 0 END) as hypertension,
            MAX(CASE WHEN d.diagnosisstring ILIKE '%myocardial infarction%'
                       OR d.diagnosisstring ILIKE '%acute coronary%' THEN 1 ELSE 0 END) as mi,
            MAX(CASE WHEN d.diagnosisstring ILIKE '%heart failure%'
                       OR d.diagnosisstring ILIKE '%congestive heart failure%' THEN 1 ELSE 0 END) as heart_failure,
            MAX(CASE WHEN d.diagnosisstring ILIKE '%respiratory failure%'
                       OR d.diagnosisstring ILIKE '%pneumonia%'
                       OR d.diagnosisstring ILIKE '%COPD%'
                       OR d.diagnosisstring ILIKE '%pulmonary%'
                       OR d.diagnosisstring ILIKE '%respiratory distress%' THEN 1 ELSE 0 END) as respiratory_disease
        FROM _eicu_pb pb
        LEFT JOIN {self.schema}.diagnosis d
            ON pb.patientunitstayid = d.patientunitstayid
        GROUP BY pb.patientunitstayid
        """

    def _medication_query(self) -> str:
        return f"""
        SELECT
            pb.patientunitstayid,
            COUNT(DISTINCT m.drugname) as unique_medications,
            COUNT(CASE WHEN m.drugname ILIKE '%insulin%' THEN 1 END) as insulin_orders,
            COUNT(CASE WHEN m.drugname ILIKE '%heparin%' THEN 1 END) as heparin_orders,
            COUNT(CASE WHEN m.drugname ILIKE '%norepinephrine%'
                        OR m.drugname ILIKE '%dopamine%'
                        OR m.drugname ILIKE '%vasopressin%'
                        OR m.drugname ILIKE '%epinephrine%'
                        OR m.drugname ILIKE '%phenylephrine%' THEN 1 END) as vasopressor_orders
        FROM _eicu_pb pb
        LEFT JOIN {self.schema}.medication m
            ON pb.patientunitstayid = m.patientunitstayid
        WHERE m.drugordercancelled IS NULL OR m.drugordercancelled = 'No'
        GROUP BY pb.patientunitstayid
        """

    def _pb_features_query(self) -> str:
        return """
        SELECT patientunitstayid, age, gender, icu_los_hours,
               mortality, emergency_admit, prolonged_stay
        FROM _eicu_pb ORDER BY patientunitstayid
        """

    # ── Helpers ──────────────────────────────────────────────────

    def _fetch_df(self, cursor, query: str) -> pd.DataFrame:
        cursor.execute(query)
        return self._cursor_to_df(cursor)

    @staticmethod
    def _cursor_to_df(cursor) -> pd.DataFrame:
        cols = [desc[0] for desc in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=cols)

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            'subject_id_col': 'subject_id',
        })
        df = df.rename(columns={'patientunitstayid': 'subject_id'})
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['unittype']:
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().sum() > 0:
                        df[col] = converted
                except (ValueError, TypeError):
                    pass
        return df

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.dropna(subset=['age', 'mortality'])
        vital_bounds = {
            'hr_mean': (30, 200), 'sbp_mean': (50, 250),
            'dbp_mean': (30, 150), 'temp_mean': (32, 42),
            'resp_mean': (5, 50), 'spo2_mean': (70, 100),
        }
        for col, (lo, hi) in vital_bounds.items():
            if col in df.columns:
                mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
                if mask.sum() > 0:
                    df.loc[mask, col] = np.nan
        self.logger.info(f"Data quality: {initial} → {len(df)} patients")
        return df
