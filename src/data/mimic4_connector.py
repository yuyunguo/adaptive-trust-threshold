"""
MIMIC-IV Database Connector
Handles database connections for MIMIC-IV multi-schema PostgreSQL database
"""

import psycopg2
import pandas as pd
import logging
import os
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()


class Mimic4Connector:
    """
    Database connector for MIMIC-IV PostgreSQL database
    MIMIC-IV uses three schemas: mimic_core, mimic_hosp, mimic_icu
    """

    def __init__(self, config: Dict[str, str] = None):
        if config is None:
            self.config = {
                'host': os.getenv('DB_HOST', '192.168.1.249'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'mimiciv'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres'),
                'schema_core': os.getenv('DB_SCHEMA_CORE', 'mimic_core'),
                'schema_hosp': os.getenv('DB_SCHEMA_HOSP', 'mimic_hosp'),
                'schema_icu': os.getenv('DB_SCHEMA_ICU', 'mimic_icu'),
                'schema': os.getenv('DB_SCHEMA', 'mimic_core')
            }
        else:
            self.config = config

        self.connection = None
        self.logger = logging.getLogger(__name__)

        required_keys = ['host', 'port', 'database', 'user', 'password']
        for key in required_keys:
            if key not in self.config or self.config[key] is None:
                raise ValueError(f"Missing required config key: {key}")

    @property
    def sc(self):
        """Shortcut for mimic_core schema"""
        return self.config.get('schema_core', 'mimic_core')

    @property
    def sh(self):
        """Shortcut for mimic_hosp schema"""
        return self.config.get('schema_hosp', 'mimic_hosp')

    @property
    def si(self):
        """Shortcut for mimic_icu schema"""
        return self.config.get('schema_icu', 'mimic_icu')

    def connect(self) -> None:
        try:
            connection_params = {
                'host': self.config['host'],
                'port': self.config['port'],
                'database': self.config['database'],
                'user': self.config['user'],
                'password': self.config['password']
            }
            self.connection = psycopg2.connect(**connection_params)
            self.logger.info("Successfully connected to MIMIC-IV database")
        except Exception as e:
            self.logger.error(f"Failed to connect to MIMIC-IV: {e}")
            raise

    def disconnect(self) -> None:
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")

    @contextmanager
    def get_connection(self):
        if not self.connection:
            self.connect()
        try:
            yield self.connection
        except Exception as e:
            self.logger.error(f"Database operation failed: {e}")
            if self.connection:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            raise

    def execute_query(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        with self.get_connection() as conn:
            try:
                df = pd.read_sql(query, conn, params=params)
                self.logger.debug(f"Query executed, returned {len(df)} rows")
                return df
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise

    def get_patient_demographics(self, subject_id: int) -> pd.DataFrame:
        query = f"""
        SELECT p.subject_id, p.gender, p.anchor_age as age, p.dod,
               a.hadm_id, a.admittime, a.dischtime, a.deathtime,
               a.admission_type, a.race, a.insurance,
               a.hospital_expire_flag
        FROM {self.sc}.patients p
        LEFT JOIN {self.sc}.admissions a ON p.subject_id = a.subject_id
        WHERE p.subject_id = %s
        ORDER BY a.admittime DESC
        """
        return self.execute_query(query, [subject_id])

    def get_icustays(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        where = "WHERE i.subject_id = %s"
        params = [subject_id]
        if hadm_id:
            where += " AND i.hadm_id = %s"
            params.append(hadm_id)
        query = f"""
        SELECT i.subject_id, i.hadm_id, i.stay_id,
               i.first_careunit, i.last_careunit,
               i.intime, i.outtime, i.los
        FROM {self.si}.icustays i
        {where}
        ORDER BY i.intime
        """
        return self.execute_query(query, params)

    def get_vital_signs(self, subject_id: int, hadm_id: Optional[int] = None,
                       stay_id: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        schema_icu = self.si
        where_conditions = ["c.subject_id = %s"]
        params = [subject_id]

        if hadm_id:
            where_conditions.append("c.hadm_id = %s")
            params.append(hadm_id)
        if stay_id:
            where_conditions.append("c.stay_id = %s")
            params.append(stay_id)

        where_clause = " AND ".join(where_conditions)

        query = f"""
        SELECT c.subject_id, c.hadm_id, c.stay_id, c.charttime,
               c.itemid, d.label, c.value, c.valuenum, c.valueuom
        FROM {schema_icu}.chartevents c
        JOIN {schema_icu}.d_items d ON c.itemid = d.itemid
        {where_clause}
        AND c.itemid IN (
            220045, -- Heart Rate
            220179, -- Non Invasive Blood Pressure systolic
            220180, -- Non Invasive Blood Pressure diastolic
            220210, -- Respiratory Rate
            220277, -- O2 saturation pulseoxymetry
            223762, -- Temperature Celsius
            220050, -- Arterial BP systolic
            220051  -- Arterial BP diastolic
        )
        AND c.charttime IS NOT NULL
        AND c.valuenum IS NOT NULL
        AND c.valuenum > 0
        ORDER BY c.charttime DESC
        LIMIT %s
        """
        params.append(limit)
        return self.execute_query(query, params)

    def get_lab_results(self, subject_id: int, hadm_id: Optional[int] = None,
                       limit: int = 1000) -> pd.DataFrame:
        schema_hosp = self.sh
        where_conditions = ["l.subject_id = %s"]
        params = [subject_id]

        if hadm_id:
            where_conditions.append("l.hadm_id = %s")
            params.append(hadm_id)

        where_clause = " AND ".join(where_conditions)

        query = f"""
        SELECT l.subject_id, l.hadm_id, l.charttime, l.labevent_id,
               l.itemid, d.label, l.value, l.valuenum, l.valueuom, l.flag
        FROM {schema_hosp}.labevents l
        JOIN {schema_hosp}.d_labitems d ON l.itemid = d.itemid
        {where_clause}
        AND l.itemid IN (
            50912,  -- Creatinine
            50902,  -- Chloride
            50931,  -- Glucose
            51221,  -- Hematocrit
            51222,  -- Hemoglobin
            50960,  -- Magnesium
            50970,  -- Phosphate
            50971,  -- Potassium
            50983,  -- Sodium
            51006,  -- Urea Nitrogen
            50820,  -- pH
            50885,  -- Bilirubin, Total
            51300,  -- WBC Count
            51301   -- White Blood Cells
        )
        AND l.charttime IS NOT NULL
        AND l.valuenum IS NOT NULL
        ORDER BY l.charttime DESC
        LIMIT %s
        """
        params.append(limit)
        return self.execute_query(query, params)

    def get_diagnoses(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        where = "WHERE d.subject_id = %s"
        params = [subject_id]
        if hadm_id:
            where += " AND d.hadm_id = %s"
            params.append(hadm_id)
        query = f"""
        SELECT d.subject_id, d.hadm_id, d.seq_num,
               d.icd_code, d.icd_version
        FROM {self.sh}.diagnoses_icd d
        {where}
        ORDER BY d.seq_num
        """
        return self.execute_query(query, params)

    def get_procedures(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        where = "WHERE p.subject_id = %s"
        params = [subject_id]
        if hadm_id:
            where += " AND p.hadm_id = %s"
            params.append(hadm_id)
        query = f"""
        SELECT p.subject_id, p.hadm_id, p.seq_num,
               p.icd_code, p.icd_version, p.chartdate
        FROM {self.sh}.procedures_icd p
        {where}
        ORDER BY p.seq_num
        """
        return self.execute_query(query, params)

    def get_prescriptions(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        where = "WHERE p.subject_id = %s"
        params = [subject_id]
        if hadm_id:
            where += " AND p.hadm_id = %s"
            params.append(hadm_id)
        query = f"""
        SELECT p.subject_id, p.hadm_id, p.pharmacy_id,
               p.drug, p.drug_type, p.route,
               p.starttime, p.stoptime,
               p.dose_val_rx, p.dose_unit_rx
        FROM {self.sh}.prescriptions p
        {where}
        ORDER BY p.starttime
        """
        return self.execute_query(query, params)

    def test_connection(self) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                return result[0] == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]

                info = {}
                for schema_name in [self.sc, self.sh, self.si]:
                    cursor.execute("""
                        SELECT COUNT(*) FROM information_schema.tables
                        WHERE table_schema = %s
                    """, (schema_name,))
                    info[f'{schema_name}_tables'] = cursor.fetchone()[0]

                cursor.execute(f"SELECT COUNT(*) FROM {self.sc}.patients")
                patient_count = cursor.fetchone()[0]

                cursor.execute(f"SELECT COUNT(*) FROM {self.sc}.admissions")
                admission_count = cursor.fetchone()[0]

                cursor.execute(f"SELECT COUNT(*) FROM {self.si}.icustays")
                icustay_count = cursor.fetchone()[0]

                cursor.close()

                return {
                    'database_version': version,
                    'total_patients': patient_count,
                    'total_admissions': admission_count,
                    'total_icustays': icustay_count,
                    'schemas': info,
                    'connection_status': 'connected'
                }

        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {
                'connection_status': 'error',
                'error': str(e)
            }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
