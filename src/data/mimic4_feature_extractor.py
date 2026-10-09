"""
MIMIC-IV Feature Extractor
Clinical feature extraction from MIMIC-IV multi-schema database
Adapted from MIMIC-III EnhancedRealMimicFeatureExtractor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging


class Mimic4FeatureExtractor:
    """Feature extraction from MIMIC-IV database"""

    def __init__(self, connection, schema_core: str = 'mimic_core',
                 schema_hosp: str = 'mimic_hosp', schema_icu: str = 'mimic_icu'):
        self.connection = connection
        self.sc = schema_core
        self.sh = schema_hosp
        self.si = schema_icu
        self.scaler = StandardScaler()
        self.encoders = {}
        self.logger = logging.getLogger(__name__)

        self.extraction_config = {
            'min_age': 18,
            'max_age': 100,
            'min_icu_los': 0.5,
            'max_missing_rate': 0.8,
            'vital_signs_window': 24,
            'lab_aggregation_methods': ['mean', 'min', 'max', 'std']
        }

    def extract_comprehensive_cohort(self, limit: int = 1000) -> pd.DataFrame:
        self.logger.info(f"Extracting MIMIC-IV patient cohort (n={limit})")

        cursor = self.connection.cursor()

        try:
            # Step 1: Create temp table with patient base for faster joins
            self.logger.debug("Creating patient base temp table")
            cursor.execute(self._build_patient_base_query(limit))
            patient_count = cursor.rowcount
            self.logger.info(f"Patient base: {patient_count} rows")

            # Step 2: Extract vital signs features
            self.logger.debug("Extracting vital signs features")
            cursor.execute(self._build_vital_signs_query())
            vs_cols = [desc[0] for desc in cursor.description]
            vs_data = cursor.fetchall()
            df_vs = pd.DataFrame(vs_data, columns=vs_cols)
            self.logger.info(f"Vital signs: {len(df_vs)} rows")

            # Step 3: Extract lab features
            self.logger.debug("Extracting lab features")
            cursor.execute(self._build_lab_features_query())
            lf_cols = [desc[0] for desc in cursor.description]
            lf_data = cursor.fetchall()
            df_lf = pd.DataFrame(lf_data, columns=lf_cols)
            self.logger.info(f"Lab features: {len(df_lf)} rows")

            # Step 4: Extract procedure features
            self.logger.debug("Extracting procedure features")
            cursor.execute(self._build_procedure_features_query())
            pf_cols = [desc[0] for desc in cursor.description]
            pf_data = cursor.fetchall()
            df_pf = pd.DataFrame(pf_data, columns=pf_cols)
            self.logger.info(f"Procedure features: {len(df_pf)} rows")

            # Step 5: Extract diagnosis features
            self.logger.debug("Extracting diagnosis features")
            cursor.execute(self._build_diagnosis_features_query())
            df_cols = [desc[0] for desc in cursor.description]
            df_data = cursor.fetchall()
            df_diag = pd.DataFrame(df_data, columns=df_cols)
            self.logger.info(f"Diagnosis features: {len(df_diag)} rows")

            # Step 6: Extract medication features
            self.logger.debug("Extracting medication features")
            cursor.execute(self._build_medication_features_query())
            mf_cols = [desc[0] for desc in cursor.description]
            mf_data = cursor.fetchall()
            df_mf = pd.DataFrame(mf_data, columns=mf_cols)
            self.logger.info(f"Medication features: {len(df_mf)} rows")

            # Step 7: Extract patient base features
            self.logger.debug("Extracting patient base features")
            cursor.execute(self._build_pb_features_query())
            pb_cols = [desc[0] for desc in cursor.description]
            pb_data = cursor.fetchall()
            df_pb = pd.DataFrame(pb_data, columns=pb_cols)
            self.logger.info(f"Patient base features: {len(df_pb)} rows")

            # Step 8: Drop temp table
            cursor.execute("DROP TABLE IF EXISTS _m4_pb")

            # Step 9: Merge all features
            self.logger.debug("Merging features")
            df = df_pb
            for suffix, sub_df in [('', df_vs), ('', df_lf), ('', df_pf),
                                    ('', df_diag), ('', df_mf)]:
                if not sub_df.empty:
                    df = df.merge(
                        sub_df, on=['subject_id', 'hadm_id'], how='left',
                        suffixes=('', suffix)
                    )

            self.logger.info(f"Merged dataframe: {len(df)} rows, {len(df.columns)} cols")

            # Step 10: Convert object columns to numeric where possible
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().sum() > 0:
                        df[col] = converted
                except (ValueError, TypeError):
                    pass

            # Step 11: Engineering + validation
            df = self._engineer_clinical_features(df)
            df = self._validate_data_quality(df)

            self.logger.info(f"Final cohort: {len(df)} patients, {len(df.columns)} features")
            return df

        except Exception as e:
            self.logger.error(f"Error extracting MIMIC-IV cohort: {e}")
            raise
        finally:
            try:
                cursor.execute("DROP TABLE IF EXISTS _m4_pb")
            except Exception:
                pass
            cursor.close()

    def _build_patient_base_query(self, limit: int) -> str:
        return f"""
        SELECT DISTINCT
            p.subject_id,
            a.hadm_id,
            i.stay_id,
            (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year) as age,
            CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END as gender_male,
            EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 as icu_los_hours,
            a.hospital_expire_flag as mortality,
            CASE WHEN a.admission_type = 'EMERGENCY' THEN 1 ELSE 0 END as emergency_admit,
            CASE WHEN i.first_careunit IN ('MICU', 'SICU', 'TSICU') THEN 1 ELSE 0 END as intensive_unit,
            CASE WHEN EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 > 72 THEN 1 ELSE 0 END as prolonged_stay,
            a.admittime,
            i.intime,
            i.outtime
        INTO TEMP TABLE _m4_pb
        FROM {self.sc}.patients p
        JOIN {self.sc}.admissions a ON p.subject_id = a.subject_id
        JOIN {self.si}.icustays i ON a.hadm_id = i.hadm_id
        WHERE (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year)
              BETWEEN {self.extraction_config['min_age']} AND {self.extraction_config['max_age']}
        AND EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 > {self.extraction_config['min_icu_los']}
        ORDER BY p.subject_id, a.hadm_id
        LIMIT {limit}
        """

    def _build_vital_signs_query(self) -> str:
        return f"""
        SELECT
            pb.subject_id,
            pb.hadm_id,
            AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_mean,
            MIN(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_min,
            MAX(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) as hr_max,
            AVG(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) as sbp_mean,
            MIN(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) as sbp_min,
            MAX(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) as sbp_max,
            AVG(CASE WHEN ce.itemid IN (220051, 220180) THEN ce.valuenum END) as dbp_mean,
            AVG(CASE WHEN ce.itemid = 223762 THEN ce.valuenum END) as temp_mean,
            AVG(CASE WHEN ce.itemid IN (220210, 224690) THEN ce.valuenum END) as resp_mean,
            AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) as spo2_mean,
            COUNT(DISTINCT ce.itemid) as vital_measurements
        FROM _m4_pb pb
        LEFT JOIN {self.si}.chartevents ce ON pb.subject_id = ce.subject_id
            AND pb.hadm_id = ce.hadm_id
            AND ce.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '{self.extraction_config['vital_signs_window']} hours'
            AND ce.valuenum IS NOT NULL
            AND ce.valuenum > 0
        GROUP BY pb.subject_id, pb.hadm_id
        """

    def _build_lab_features_query(self) -> str:
        return f"""
        SELECT
            pb.subject_id,
            pb.hadm_id,
            COUNT(DISTINCT l.itemid) as unique_lab_tests,
            COUNT(l.itemid) as total_lab_measurements,
            COUNT(CASE WHEN l.flag = 'abnormal' THEN 1 END) as abnormal_labs,
            AVG(CASE WHEN l.itemid = 50912 THEN l.valuenum END) as creatinine_mean,
            AVG(CASE WHEN l.itemid = 50971 THEN l.valuenum END) as potassium_mean,
            AVG(CASE WHEN l.itemid = 50983 THEN l.valuenum END) as sodium_mean,
            AVG(CASE WHEN l.itemid = 51006 THEN l.valuenum END) as bun_mean,
            AVG(CASE WHEN l.itemid = 50885 THEN l.valuenum END) as bilirubin_mean,
            AVG(CASE WHEN l.itemid = 50820 THEN l.valuenum END) as ph_mean,
            AVG(CASE WHEN l.itemid = 50902 THEN l.valuenum END) as chloride_mean
        FROM _m4_pb pb
        LEFT JOIN {self.sh}.labevents l ON pb.subject_id = l.subject_id
            AND pb.hadm_id = l.hadm_id
            AND l.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '{self.extraction_config['vital_signs_window']} hours'
            AND l.valuenum IS NOT NULL
        GROUP BY pb.subject_id, pb.hadm_id
        """

    def _build_procedure_features_query(self) -> str:
        return f"""
        SELECT
            pb.subject_id,
            pb.hadm_id,
            COUNT(DISTINCT pr.icd_code) as unique_procedures,
            COUNT(pr.icd_code) as total_procedures,
            COUNT(CASE WHEN (pr.icd_version = 9 AND pr.icd_code LIKE '36%')
                       OR (pr.icd_version = 10 AND pr.icd_code LIKE '02%') THEN 1 END) as cardiac_procedures,
            COUNT(CASE WHEN (pr.icd_version = 9 AND pr.icd_code LIKE '39%')
                       OR (pr.icd_version = 10 AND pr.icd_code LIKE '03%') THEN 1 END) as vascular_procedures
        FROM _m4_pb pb
        LEFT JOIN {self.sh}.procedures_icd pr ON pb.subject_id = pr.subject_id
            AND pb.hadm_id = pr.hadm_id
        GROUP BY pb.subject_id, pb.hadm_id
        """

    def _build_diagnosis_features_query(self) -> str:
        return f"""
        SELECT
            pb.subject_id,
            pb.hadm_id,
            COUNT(DISTINCT d.icd_code) as unique_diagnoses,
            COUNT(d.icd_code) as total_diagnoses,
            COUNT(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '250%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'E08%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'E09%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'E10%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'E11%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'E13%') THEN 1 END) as diabetes_diagnosis,
            COUNT(CASE WHEN (d.icd_version = 9 AND (d.icd_code LIKE '401%' OR d.icd_code LIKE '402%'))
                       OR (d.icd_version = 10 AND (d.icd_code LIKE 'I10%' OR d.icd_code LIKE 'I11%'))
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'I12%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'I13%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'I15%') THEN 1 END) as hypertension_diagnosis,
            COUNT(CASE WHEN (d.icd_version = 9 AND (d.icd_code LIKE '410%' OR d.icd_code LIKE '411%'))
                       OR (d.icd_version = 10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%'))
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'I24%') THEN 1 END) as mi_diagnosis,
            COUNT(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '428%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'I50%') THEN 1 END) as heart_failure_diagnosis,
            COUNT(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '518%')
                       OR (d.icd_version = 10 AND (d.icd_code LIKE 'J40%' OR d.icd_code LIKE 'J41%'))
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J42%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J43%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J44%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J45%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J46%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J47%')
                       OR (d.icd_version = 10 AND d.icd_code LIKE 'J96%') THEN 1 END) as respiratory_diagnosis
        FROM _m4_pb pb
        LEFT JOIN {self.sh}.diagnoses_icd d ON pb.subject_id = d.subject_id
            AND pb.hadm_id = d.hadm_id
        GROUP BY pb.subject_id, pb.hadm_id
        """

    def _build_medication_features_query(self) -> str:
        return f"""
        SELECT
            pb.subject_id,
            pb.hadm_id,
            COUNT(DISTINCT pr.drug) as unique_medications,
            COUNT(pr.drug) as total_medication_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%insulin%' THEN 1 END) as insulin_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%heparin%' THEN 1 END) as heparin_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%dopamine%' OR LOWER(pr.drug) LIKE '%norepinephrine%' THEN 1 END) as vasopressor_orders
        FROM _m4_pb pb
        LEFT JOIN {self.sh}.prescriptions pr ON pb.subject_id = pr.subject_id
            AND pb.hadm_id = pr.hadm_id
        GROUP BY pb.subject_id, pb.hadm_id
        """

    def _build_pb_features_query(self) -> str:
        return f"""
        SELECT
            pb.subject_id,
            pb.hadm_id,
            pb.stay_id,
            pb.age,
            pb.gender_male,
            pb.icu_los_hours,
            pb.mortality,
            pb.emergency_admit,
            pb.intensive_unit,
            pb.prolonged_stay
        FROM _m4_pb pb
        ORDER BY pb.subject_id, pb.hadm_id
        """

    def _engineer_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Engineering clinical features")

        df['clinical_complexity'] = (
            df['unique_diagnoses'] * 0.3 +
            df['unique_procedures'] * 0.4 +
            df['unique_medications'] * 0.2 +
            df['abnormal_labs'] * 0.1
        )

        df['hemodynamic_instability'] = (
            (df['hr_max'] - df['hr_min']) / df['hr_mean'] +
            (df['sbp_max'] - df['sbp_min']) / df['sbp_mean']
        ).fillna(0)

        df['lab_abnormality_rate'] = (
            df['abnormal_labs'] / (df['total_lab_measurements'] + 1)
        )

        df['organ_system_score'] = (
            df['diabetes_diagnosis'].astype(bool).astype(int) +
            df['hypertension_diagnosis'].astype(bool).astype(int) +
            df['mi_diagnosis'].astype(bool).astype(int) +
            df['heart_failure_diagnosis'].astype(bool).astype(int) +
            df['respiratory_diagnosis'].astype(bool).astype(int)
        )

        df['critical_care_intensity'] = (
            df['vasopressor_orders'] * 3 +
            df['cardiac_procedures'] * 2 +
            df['intensive_unit'] * 1
        )

        df['age_adjusted_risk'] = df['age'] * (1 + df['clinical_complexity'] / 10)

        df['medication_burden'] = np.log1p(df['total_medication_orders'])

        vital_cols = ['hr_mean', 'sbp_mean', 'dbp_mean', 'temp_mean', 'resp_mean', 'spo2_mean']
        normal_ranges = {
            'hr_mean': (60, 100),
            'sbp_mean': (90, 140),
            'dbp_mean': (60, 90),
            'temp_mean': (36.0, 37.5),
            'resp_mean': (12, 20),
            'spo2_mean': (95, 100)
        }

        stability_scores = []
        for _, row in df.iterrows():
            score = 0
            for col in vital_cols:
                if col in normal_ranges:
                    min_val, max_val = normal_ranges[col]
                    value = row[col]
                    if min_val <= value <= max_val:
                        score += 1
            stability_scores.append(score / len(vital_cols))

        df['vital_stability_score'] = stability_scores

        self.logger.info(f"Engineered {8} additional features")
        return df

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Validating data quality")

        initial_rows = len(df)

        df = df.drop_duplicates(subset=['subject_id', 'hadm_id', 'stay_id'])

        missing_rates = df.isnull().sum() / len(df)
        high_missing_cols = missing_rates[
            missing_rates > self.extraction_config['max_missing_rate']
        ].index.tolist()

        if high_missing_cols:
            self.logger.warning(f"Dropping {len(high_missing_cols)} features with >80% missing")
            df = df.drop(columns=high_missing_cols)

        vital_outlier_bounds = {
            'hr_mean': (30, 200),
            'sbp_mean': (50, 250),
            'dbp_mean': (30, 150),
            'temp_mean': (32, 42),
            'resp_mean': (5, 50),
            'spo2_mean': (70, 100)
        }

        for col, (min_val, max_val) in vital_outlier_bounds.items():
            if col in df.columns:
                outlier_mask = (df[col] < min_val) | (df[col] > max_val)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    self.logger.warning(f"Removing {outlier_count} outliers from {col}")
                    df = df[~outlier_mask]

        if len(df) < 50:
            self.logger.warning(f"Very small sample size after cleaning: {len(df)} patients")

        self.logger.info(f"Data quality: {initial_rows} → {len(df)} patients")
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.logger.info("Preparing features and target variable")

        target = df['mortality']

        exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        features = df[feature_cols].copy()

        features = features.fillna(features.median())

        numerical_cols = features.select_dtypes(include=[np.number]).columns
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])

        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            features[col] = encoder.fit_transform(features[col].astype(str))
            self.encoders[col] = encoder

        self.logger.info(f"Feature preparation: {len(feature_cols)} features ready")
        return features, target

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            'age': 'Patient age at admission (years)',
            'gender_male': 'Male gender indicator (1=male, 0=female)',
            'icu_los_hours': 'ICU length of stay (hours)',
            'emergency_admit': 'Emergency admission indicator',
            'intensive_unit': 'Intensive care unit type indicator',
            'prolonged_stay': 'ICU stay >72 hours indicator',
            'hr_mean': 'Mean heart rate (bpm)',
            'hr_min': 'Minimum heart rate (bpm)',
            'hr_max': 'Maximum heart rate (bpm)',
            'sbp_mean': 'Mean systolic blood pressure (mmHg)',
            'sbp_min': 'Minimum systolic blood pressure (mmHg)',
            'sbp_max': 'Maximum systolic blood pressure (mmHg)',
            'dbp_mean': 'Mean diastolic blood pressure (mmHg)',
            'temp_mean': 'Mean body temperature (C)',
            'resp_mean': 'Mean respiratory rate (breaths/min)',
            'spo2_mean': 'Mean oxygen saturation (%)',
            'vital_measurements': 'Total vital sign measurements',
            'unique_lab_tests': 'Number of unique laboratory tests',
            'total_lab_measurements': 'Total laboratory measurements',
            'abnormal_labs': 'Number of abnormal lab results',
            'creatinine_mean': 'Mean serum creatinine (mg/dL)',
            'potassium_mean': 'Mean serum potassium (mEq/L)',
            'sodium_mean': 'Mean serum sodium (mEq/L)',
            'bun_mean': 'Mean blood urea nitrogen (mg/dL)',
            'bilirubin_mean': 'Mean total bilirubin (mg/dL)',
            'ph_mean': 'Mean arterial pH',
            'chloride_mean': 'Mean serum chloride (mEq/L)',
            'unique_procedures': 'Number of unique procedures',
            'total_procedures': 'Total number of procedures',
            'cardiac_procedures': 'Number of cardiac procedures',
            'vascular_procedures': 'Number of vascular procedures',
            'unique_diagnoses': 'Number of unique diagnoses',
            'total_diagnoses': 'Total number of diagnoses',
            'diabetes_diagnosis': 'Diabetes diagnosis indicator',
            'hypertension_diagnosis': 'Hypertension diagnosis indicator',
            'mi_diagnosis': 'Myocardial infarction diagnosis indicator',
            'heart_failure_diagnosis': 'Heart failure diagnosis indicator',
            'respiratory_diagnosis': 'Respiratory disease diagnosis indicator',
            'unique_medications': 'Number of unique medications',
            'total_medication_orders': 'Total medication orders',
            'insulin_orders': 'Number of insulin orders',
            'heparin_orders': 'Number of heparin orders',
            'vasopressor_orders': 'Number of vasopressor orders',
            'clinical_complexity': 'Composite clinical complexity score',
            'hemodynamic_instability': 'Hemodynamic instability indicator',
            'lab_abnormality_rate': 'Rate of abnormal laboratory results',
            'organ_system_score': 'Number of affected organ systems',
            'critical_care_intensity': 'Critical care intensity score',
            'age_adjusted_risk': 'Age-adjusted risk score',
            'medication_burden': 'Log-transformed medication burden',
            'vital_stability_score': 'Vital sign stability score (0-1)'
        }

    def export_cohort_summary(self, df: pd.DataFrame, save_path: str) -> Dict[str, Any]:
        summary = {
            'cohort_characteristics': {
                'total_patients': len(df),
                'mortality_rate': df['mortality'].mean() if 'mortality' in df.columns else None,
                'age_distribution': {
                    'mean': float(df['age'].mean()),
                    'std': float(df['age'].std()),
                    'median': float(df['age'].median()),
                    'range': [float(df['age'].min()), float(df['age'].max())]
                } if 'age' in df.columns else None,
                'gender_distribution': {
                    'male_percentage': float(df['gender_male'].mean() * 100)
                } if 'gender_male' in df.columns else None,
                'icu_los_distribution': {
                    'mean_hours': float(df['icu_los_hours'].mean()),
                    'median_hours': float(df['icu_los_hours'].median())
                } if 'icu_los_hours' in df.columns else None
            },
            'feature_summary': {
                'total_features': len(df.columns),
                'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'missing_data_summary': df.isnull().sum().to_dict()
            },
            'clinical_indicators': {
                'emergency_admission_rate': float(df['emergency_admit'].mean()) if 'emergency_admit' in df.columns else None,
                'intensive_unit_rate': float(df['intensive_unit'].mean()) if 'intensive_unit' in df.columns else None,
                'prolonged_stay_rate': float(df['prolonged_stay'].mean()) if 'prolonged_stay' in df.columns else None
            },
            'extraction_metadata': {
                'extraction_timestamp': pd.Timestamp.now().isoformat(),
                'database': 'MIMIC-IV',
                'schemas': {'core': self.sc, 'hosp': self.sh, 'icu': self.si},
                'extraction_config': self.extraction_config,
                'feature_descriptions': self.get_feature_descriptions()
            }
        }

        import json
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Cohort summary exported to {save_path}")
        return summary
