"""
Enhanced MIMIC-III Feature Extractor
Advanced clinical feature extraction with comprehensive patient cohort analysis
"""

import numpy as np
import pandas as pd
import psycopg2
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging


class EnhancedRealMimicFeatureExtractor:
    """Enhanced feature extraction from real MIMIC-III database"""

    def __init__(self, connection, schema: str = 'mimiciii'):
        """
        Initialize Enhanced MIMIC Feature Extractor

        Args:
            connection: Database connection object
            schema: MIMIC-III database schema name
        """
        self.connection = connection
        self.schema = schema
        self.scaler = StandardScaler()
        self.encoders = {}
        self.logger = logging.getLogger(__name__)

        # Feature extraction parameters
        self.extraction_config = {
            'min_age': 18,
            'max_age': 100,
            'min_icu_los': 0.5,  # At least 12 hours
            'max_missing_rate': 0.8,  # Features with >80% missing excluded
            'vital_signs_window': 24,  # First 24 hours
            'lab_aggregation_methods': ['mean', 'min', 'max', 'std']
        }

    def extract_comprehensive_cohort(self, limit: int = 1000) -> pd.DataFrame:
        """Extract comprehensive patient cohort with rich clinical features"""
        self.logger.info(f"Extracting comprehensive patient cohort (n={limit})")

        cursor = self.connection.cursor()

        try:
            # Build comprehensive patient cohort query
            query = self._build_cohort_query(limit)

            self.logger.debug("Executing cohort extraction query")
            cursor.execute(query)

            # Fetch results with column names
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()

            self.logger.info(f"Retrieved {len(data)} patients from database")

            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)

            # Enhanced feature engineering
            df = self._engineer_clinical_features(df)

            # Quality validation
            df = self._validate_data_quality(df)

            self.logger.info(f"Final cohort size: {len(df)} patients with {len(df.columns)} features")

            return df

        except Exception as e:
            self.logger.error(f"Error extracting cohort: {e}")
            raise
        finally:
            cursor.close()

    def _build_cohort_query(self, limit: int) -> str:
        """Build comprehensive cohort extraction query"""

        query = f"""
        WITH patient_base AS (
            SELECT DISTINCT
                p.subject_id,
                a.hadm_id,
                i.icustay_id,
                EXTRACT(YEAR FROM a.admittime) - EXTRACT(YEAR FROM p.dob) as age,
                CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END as gender_male,
                EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 as icu_los_hours,
                a.hospital_expire_flag as mortality,
                CASE
                    WHEN a.admission_type = 'EMERGENCY' THEN 1 ELSE 0
                END as emergency_admit,
                CASE
                    WHEN i.first_careunit IN ('MICU', 'SICU', 'TSICU') THEN 1 ELSE 0
                END as intensive_unit,
                CASE
                    WHEN EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 > 72 THEN 1
                    ELSE 0
                END as prolonged_stay,
                a.admittime,
                i.intime,
                i.outtime
            FROM {self.schema}.patients p
            JOIN {self.schema}.admissions a ON p.subject_id = a.subject_id
            JOIN {self.schema}.icustays i ON a.hadm_id = i.hadm_id
            WHERE EXTRACT(YEAR FROM a.admittime) - EXTRACT(YEAR FROM p.dob)
                  BETWEEN {self.extraction_config['min_age']} AND {self.extraction_config['max_age']}
            AND EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 > {self.extraction_config['min_icu_los']}
        ),

        vital_signs AS (
            SELECT
                pb.subject_id,
                pb.hadm_id,
                AVG(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) as hr_mean,
                MIN(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) as hr_min,
                MAX(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) as hr_max,
                AVG(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) as sbp_mean,
                MIN(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) as sbp_min,
                MAX(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) as sbp_max,
                AVG(CASE WHEN ce.itemid IN (8368, 8440, 8441, 8555, 220180, 220051) THEN ce.valuenum END) as dbp_mean,
                AVG(CASE WHEN ce.itemid IN (223761, 678) THEN ce.valuenum END) as temp_mean,
                AVG(CASE WHEN ce.itemid IN (618, 615, 220210, 224690) THEN ce.valuenum END) as resp_mean,
                AVG(CASE WHEN ce.itemid IN (646, 220277) THEN ce.valuenum END) as spo2_mean,
                COUNT(DISTINCT ce.itemid) as vital_measurements
            FROM patient_base pb
            LEFT JOIN {self.schema}.chartevents ce ON pb.subject_id = ce.subject_id
                AND pb.hadm_id = ce.hadm_id
                AND ce.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '{self.extraction_config['vital_signs_window']} hours'
                AND ce.valuenum IS NOT NULL
                AND ce.valuenum > 0
            GROUP BY pb.subject_id, pb.hadm_id
        ),

        lab_features AS (
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
            FROM patient_base pb
            LEFT JOIN {self.schema}.labevents l ON pb.subject_id = l.subject_id
                AND pb.hadm_id = l.hadm_id
                AND l.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '{self.extraction_config['vital_signs_window']} hours'
                AND l.valuenum IS NOT NULL
            GROUP BY pb.subject_id, pb.hadm_id
        ),

        procedure_features AS (
            SELECT
                pb.subject_id,
                pb.hadm_id,
                COUNT(DISTINCT pr.icd9_code) as unique_procedures,
                COUNT(pr.icd9_code) as total_procedures,
                COUNT(CASE WHEN pr.icd9_code LIKE '36%' THEN 1 END) as cardiac_procedures,
                COUNT(CASE WHEN pr.icd9_code LIKE '39%' THEN 1 END) as vascular_procedures
            FROM patient_base pb
            LEFT JOIN {self.schema}.procedures_icd pr ON pb.subject_id = pr.subject_id
                AND pb.hadm_id = pr.hadm_id
            GROUP BY pb.subject_id, pb.hadm_id
        ),

        diagnosis_features AS (
            SELECT
                pb.subject_id,
                pb.hadm_id,
                COUNT(DISTINCT d.icd9_code) as unique_diagnoses,
                COUNT(d.icd9_code) as total_diagnoses,
                COUNT(CASE WHEN d.icd9_code LIKE '250%' THEN 1 END) as diabetes_diagnosis,
                COUNT(CASE WHEN d.icd9_code LIKE '401%' OR d.icd9_code LIKE '402%' THEN 1 END) as hypertension_diagnosis,
                COUNT(CASE WHEN d.icd9_code LIKE '410%' OR d.icd9_code LIKE '411%' THEN 1 END) as mi_diagnosis,
                COUNT(CASE WHEN d.icd9_code LIKE '428%' THEN 1 END) as heart_failure_diagnosis,
                COUNT(CASE WHEN d.icd9_code LIKE '518%' THEN 1 END) as respiratory_diagnosis
            FROM patient_base pb
            LEFT JOIN {self.schema}.diagnoses_icd d ON pb.subject_id = d.subject_id
                AND pb.hadm_id = d.hadm_id
            GROUP BY pb.subject_id, pb.hadm_id
        ),

        medication_features AS (
            SELECT
                pb.subject_id,
                pb.hadm_id,
                COUNT(DISTINCT pr.drug) as unique_medications,
                COUNT(pr.drug) as total_medication_orders,
                COUNT(CASE WHEN LOWER(pr.drug) LIKE '%insulin%' THEN 1 END) as insulin_orders,
                COUNT(CASE WHEN LOWER(pr.drug) LIKE '%heparin%' THEN 1 END) as heparin_orders,
                COUNT(CASE WHEN LOWER(pr.drug) LIKE '%dopamine%' OR LOWER(pr.drug) LIKE '%norepinephrine%' THEN 1 END) as vasopressor_orders
            FROM patient_base pb
            LEFT JOIN {self.schema}.prescriptions pr ON pb.subject_id = pr.subject_id
                AND pb.hadm_id = pr.hadm_id
            GROUP BY pb.subject_id, pb.hadm_id
        )

        SELECT
            pb.subject_id,
            pb.hadm_id,
            pb.icustay_id,
            pb.age,
            pb.gender_male,
            pb.icu_los_hours,
            pb.mortality,
            pb.emergency_admit,
            pb.intensive_unit,
            pb.prolonged_stay,

            -- Vital Signs Features
            COALESCE(vs.hr_mean, 80) as hr_mean,
            COALESCE(vs.hr_min, 60) as hr_min,
            COALESCE(vs.hr_max, 100) as hr_max,
            COALESCE(vs.sbp_mean, 120) as sbp_mean,
            COALESCE(vs.sbp_min, 90) as sbp_min,
            COALESCE(vs.sbp_max, 140) as sbp_max,
            COALESCE(vs.dbp_mean, 80) as dbp_mean,
            COALESCE(vs.temp_mean, 36.5) as temp_mean,
            COALESCE(vs.resp_mean, 18) as resp_mean,
            COALESCE(vs.spo2_mean, 98) as spo2_mean,
            COALESCE(vs.vital_measurements, 0) as vital_measurements,

            -- Laboratory Features
            COALESCE(lf.unique_lab_tests, 0) as unique_lab_tests,
            COALESCE(lf.total_lab_measurements, 0) as total_lab_measurements,
            COALESCE(lf.abnormal_labs, 0) as abnormal_labs,
            COALESCE(lf.creatinine_mean, 1.0) as creatinine_mean,
            COALESCE(lf.potassium_mean, 4.0) as potassium_mean,
            COALESCE(lf.sodium_mean, 140) as sodium_mean,
            COALESCE(lf.bun_mean, 20) as bun_mean,
            COALESCE(lf.bilirubin_mean, 1.0) as bilirubin_mean,
            COALESCE(lf.ph_mean, 7.4) as ph_mean,
            COALESCE(lf.chloride_mean, 102) as chloride_mean,

            -- Procedure Features
            COALESCE(pf.unique_procedures, 0) as unique_procedures,
            COALESCE(pf.total_procedures, 0) as total_procedures,
            COALESCE(pf.cardiac_procedures, 0) as cardiac_procedures,
            COALESCE(pf.vascular_procedures, 0) as vascular_procedures,

            -- Diagnosis Features
            COALESCE(df.unique_diagnoses, 0) as unique_diagnoses,
            COALESCE(df.total_diagnoses, 0) as total_diagnoses,
            COALESCE(df.diabetes_diagnosis, 0) as diabetes_diagnosis,
            COALESCE(df.hypertension_diagnosis, 0) as hypertension_diagnosis,
            COALESCE(df.mi_diagnosis, 0) as mi_diagnosis,
            COALESCE(df.heart_failure_diagnosis, 0) as heart_failure_diagnosis,
            COALESCE(df.respiratory_diagnosis, 0) as respiratory_diagnosis,

            -- Medication Features
            COALESCE(mf.unique_medications, 0) as unique_medications,
            COALESCE(mf.total_medication_orders, 0) as total_medication_orders,
            COALESCE(mf.insulin_orders, 0) as insulin_orders,
            COALESCE(mf.heparin_orders, 0) as heparin_orders,
            COALESCE(mf.vasopressor_orders, 0) as vasopressor_orders

        FROM patient_base pb
        LEFT JOIN vital_signs vs ON pb.subject_id = vs.subject_id AND pb.hadm_id = vs.hadm_id
        LEFT JOIN lab_features lf ON pb.subject_id = lf.subject_id AND pb.hadm_id = lf.hadm_id
        LEFT JOIN procedure_features pf ON pb.subject_id = pf.subject_id AND pb.hadm_id = pf.hadm_id
        LEFT JOIN diagnosis_features df ON pb.subject_id = df.subject_id AND pb.hadm_id = df.hadm_id
        LEFT JOIN medication_features mf ON pb.subject_id = mf.subject_id AND pb.hadm_id = mf.hadm_id

        ORDER BY pb.subject_id, pb.hadm_id
        LIMIT {limit}
        """

        return query

    def _engineer_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional clinical features from extracted data"""
        self.logger.info("Engineering clinical features")

        # Clinical complexity scores
        df['clinical_complexity'] = (
            df['unique_diagnoses'] * 0.3 +
            df['unique_procedures'] * 0.4 +
            df['unique_medications'] * 0.2 +
            df['abnormal_labs'] * 0.1
        )

        # Hemodynamic stability indicators
        df['hemodynamic_instability'] = (
            (df['hr_max'] - df['hr_min']) / df['hr_mean'] +
            (df['sbp_max'] - df['sbp_min']) / df['sbp_mean']
        ).fillna(0)

        # Laboratory abnormality rate
        df['lab_abnormality_rate'] = (
            df['abnormal_labs'] / (df['total_lab_measurements'] + 1)
        )

        # Organ system involvement score
        df['organ_system_score'] = (
            df['diabetes_diagnosis'].astype(bool).astype(int) +
            df['hypertension_diagnosis'].astype(bool).astype(int) +
            df['mi_diagnosis'].astype(bool).astype(int) +
            df['heart_failure_diagnosis'].astype(bool).astype(int) +
            df['respiratory_diagnosis'].astype(bool).astype(int)
        )

        # Critical care intensity
        df['critical_care_intensity'] = (
            df['vasopressor_orders'] * 3 +
            df['cardiac_procedures'] * 2 +
            df['intensive_unit'] * 1
        )

        # Age-adjusted risk score
        df['age_adjusted_risk'] = df['age'] * (1 + df['clinical_complexity'] / 10)

        # Medication burden
        df['medication_burden'] = np.log1p(df['total_medication_orders'])

        # Vital sign stability score
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

        self.logger.info(f"Added {len(df.columns) - len(df.columns) + 8} engineered features")
        return df

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data quality"""
        self.logger.info("Validating data quality")

        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates(subset=['subject_id', 'hadm_id', 'icustay_id'])

        # Handle missing values
        missing_rates = df.isnull().sum() / len(df)
        high_missing_cols = missing_rates[
            missing_rates > self.extraction_config['max_missing_rate']
        ].index.tolist()

        if high_missing_cols:
            self.logger.warning(f"Dropping {len(high_missing_cols)} features with >80% missing: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)

        # Remove outliers for vital signs
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

        # Ensure minimum sample size
        if len(df) < 50:
            self.logger.warning(f"Very small sample size after cleaning: {len(df)} patients")

        self.logger.info(f"Data quality validation complete: {initial_rows} → {len(df)} patients")
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable for ML pipeline"""
        self.logger.info("Preparing features and target variable")

        # Define target variable
        target = df['mortality']

        # Define feature columns (exclude IDs and target)
        exclude_cols = ['subject_id', 'hadm_id', 'icustay_id', 'mortality']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        features = df[feature_cols].copy()

        # Handle remaining missing values
        features = features.fillna(features.median())

        # Scale numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])

        # Encode categorical features (if any)
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            features[col] = encoder.fit_transform(features[col].astype(str))
            self.encoders[col] = encoder

        self.logger.info(f"Feature preparation complete: {len(feature_cols)} features prepared")

        return features, target

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all extracted features"""

        descriptions = {
            # Demographics
            'age': 'Patient age at admission (years)',
            'gender_male': 'Male gender indicator (1=male, 0=female)',

            # ICU characteristics
            'icu_los_hours': 'ICU length of stay (hours)',
            'emergency_admit': 'Emergency admission indicator',
            'intensive_unit': 'Intensive care unit type indicator',
            'prolonged_stay': 'ICU stay >72 hours indicator',

            # Vital signs (first 24h)
            'hr_mean': 'Mean heart rate (bpm)',
            'hr_min': 'Minimum heart rate (bpm)',
            'hr_max': 'Maximum heart rate (bpm)',
            'sbp_mean': 'Mean systolic blood pressure (mmHg)',
            'sbp_min': 'Minimum systolic blood pressure (mmHg)',
            'sbp_max': 'Maximum systolic blood pressure (mmHg)',
            'dbp_mean': 'Mean diastolic blood pressure (mmHg)',
            'temp_mean': 'Mean body temperature (°C)',
            'resp_mean': 'Mean respiratory rate (breaths/min)',
            'spo2_mean': 'Mean oxygen saturation (%)',
            'vital_measurements': 'Total vital sign measurements',

            # Laboratory values
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

            # Procedures
            'unique_procedures': 'Number of unique procedures',
            'total_procedures': 'Total number of procedures',
            'cardiac_procedures': 'Number of cardiac procedures',
            'vascular_procedures': 'Number of vascular procedures',

            # Diagnoses
            'unique_diagnoses': 'Number of unique diagnoses',
            'total_diagnoses': 'Total number of diagnoses',
            'diabetes_diagnosis': 'Diabetes diagnosis indicator',
            'hypertension_diagnosis': 'Hypertension diagnosis indicator',
            'mi_diagnosis': 'Myocardial infarction diagnosis indicator',
            'heart_failure_diagnosis': 'Heart failure diagnosis indicator',
            'respiratory_diagnosis': 'Respiratory disease diagnosis indicator',

            # Medications
            'unique_medications': 'Number of unique medications',
            'total_medication_orders': 'Total medication orders',
            'insulin_orders': 'Number of insulin orders',
            'heparin_orders': 'Number of heparin orders',
            'vasopressor_orders': 'Number of vasopressor orders',

            # Engineered features
            'clinical_complexity': 'Composite clinical complexity score',
            'hemodynamic_instability': 'Hemodynamic instability indicator',
            'lab_abnormality_rate': 'Rate of abnormal laboratory results',
            'organ_system_score': 'Number of affected organ systems',
            'critical_care_intensity': 'Critical care intensity score',
            'age_adjusted_risk': 'Age-adjusted risk score',
            'medication_burden': 'Log-transformed medication burden',
            'vital_stability_score': 'Vital sign stability score (0-1)'
        }

        return descriptions

    def export_cohort_summary(self, df: pd.DataFrame, save_path: str) -> Dict[str, Any]:
        """Export comprehensive cohort summary statistics"""

        summary = {
            'cohort_characteristics': {
                'total_patients': len(df),
                'mortality_rate': df['mortality'].mean() if 'mortality' in df.columns else None,
                'age_distribution': {
                    'mean': df['age'].mean(),
                    'std': df['age'].std(),
                    'median': df['age'].median(),
                    'range': [df['age'].min(), df['age'].max()]
                } if 'age' in df.columns else None,
                'gender_distribution': {
                    'male_percentage': df['gender_male'].mean() * 100
                } if 'gender_male' in df.columns else None,
                'icu_los_distribution': {
                    'mean_hours': df['icu_los_hours'].mean(),
                    'median_hours': df['icu_los_hours'].median()
                } if 'icu_los_hours' in df.columns else None
            },
            'feature_summary': {
                'total_features': len(df.columns),
                'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'missing_data_summary': df.isnull().sum().to_dict()
            },
            'clinical_indicators': {
                'emergency_admission_rate': df['emergency_admit'].mean() if 'emergency_admit' in df.columns else None,
                'intensive_unit_rate': df['intensive_unit'].mean() if 'intensive_unit' in df.columns else None,
                'prolonged_stay_rate': df['prolonged_stay'].mean() if 'prolonged_stay' in df.columns else None
            },
            'extraction_metadata': {
                'extraction_timestamp': pd.Timestamp.now().isoformat(),
                'database_schema': self.schema,
                'extraction_config': self.extraction_config,
                'feature_descriptions': self.get_feature_descriptions()
            }
        }

        # Save summary
        import json
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Cohort summary exported to {save_path}")
        return summary