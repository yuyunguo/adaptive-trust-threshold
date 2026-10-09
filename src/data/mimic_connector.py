"""
MIMIC-III Database Connector
Handles database connections and basic query operations
"""

import psycopg2
import pandas as pd
import logging
import os
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MimicConnector:
    """
    Database connector for MIMIC-III PostgreSQL database
    Provides connection management and basic query functionality
    """

    def __init__(self, config: Dict[str, str] = None):
        """
        Initialize MIMIC database connector

        Args:
            config: Database configuration dictionary (optional, uses .env if not provided)
        """
        if config is None:
            # Load from environment variables
            self.config = {
                'host': os.getenv('DB_HOST', '192.168.1.207'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv(DB_PASSWORD, 'postgres'),
                'schema': os.getenv('DB_SCHEMA', 'mimiciii')
            }
        else:
            self.config = config

        self.connection = None
        self.logger = logging.getLogger(__name__)

        # Validate configuration
        required_keys = ['host', 'port', 'database', 'user', 'password']
        for key in required_keys:
            if key not in self.config or self.config[key] is None:
                raise ValueError(f"Missing required config key: {key}. Please set DB_{key.upper()} in .env file")

    def connect(self) -> None:
        """Establish database connection"""
        try:
            connection_params = {
                'host': self.config['host'],
                'port': self.config['port'],
                'database': self.config['database'],
                'user': self.config['user'],
                'password': self.config['password']
            }

            self.connection = psycopg2.connect(**connection_params)
            self.logger.info("Successfully connected to MIMIC-III database")

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if not self.connection:
            self.connect()

        try:
            yield self.connection
        except Exception as e:
            self.logger.error(f"Database operation failed: {e}")
            # Attempt to rollback on error
            if self.connection:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            raise

    def execute_query(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            DataFrame with query results
        """
        with self.get_connection() as conn:
            try:
                df = pd.read_sql(query, conn, params=params)
                self.logger.debug(f"Query executed successfully, returned {len(df)} rows")
                return df
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise

    def get_patient_demographics(self, subject_id: int) -> pd.DataFrame:
        """Get patient demographics"""
        schema = self.config.get('schema', 'mimiciii')
        query = f"""
        SELECT p.subject_id, p.gender, p.dob, p.dod,
               a.hadm_id, a.admittime, a.dischtime,
               a.admission_type, a.ethnicity, a.insurance
        FROM {schema}.patients p
        LEFT JOIN {schema}.admissions a ON p.subject_id = a.subject_id
        WHERE p.subject_id = %s
        ORDER BY a.admittime DESC
        """
        return self.execute_query(query, [subject_id])

    def get_vital_signs(self, subject_id: int, hadm_id: Optional[int] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Get vital signs for a patient"""
        schema = self.config.get('schema', 'mimiciii')
        where_clause = "WHERE c.subject_id = %s"
        params = [subject_id]

        if hadm_id:
            where_clause += " AND c.hadm_id = %s"
            params.append(hadm_id)

        query = f"""
        SELECT c.subject_id, c.hadm_id, c.icustay_id, c.charttime,
               c.itemid, d.label, c.value, c.valuenum, c.valueuom
        FROM {schema}.chartevents c
        JOIN {schema}.d_items d ON c.itemid = d.itemid
        {where_clause}
        AND c.itemid IN (
            211,    -- Heart Rate
            220045, -- Heart Rate
            51,     -- Arterial BP [Systolic]
            220179, -- Non Invasive Blood Pressure systolic
            8368,   -- Respiratory Rate
            220210, -- Respiratory Rate
            646,    -- SpO2
            220277  -- SpO2
        )
        AND c.charttime IS NOT NULL
        ORDER BY c.charttime DESC
        LIMIT %s
        """
        params.append(limit)

        return self.execute_query(query, params)

    def get_lab_results(self, subject_id: int, hadm_id: Optional[int] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Get laboratory results for a patient"""
        schema = self.config.get('schema', 'mimiciii')
        where_clause = "WHERE l.subject_id = %s"
        params = [subject_id]

        if hadm_id:
            where_clause += " AND l.hadm_id = %s"
            params.append(hadm_id)

        query = f"""
        SELECT l.subject_id, l.hadm_id, l.charttime,
               l.itemid, d.label, l.value, l.valuenum, l.valueuom, l.flag
        FROM {schema}.labevents l
        JOIN {schema}.d_labitems d ON l.itemid = d.itemid
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
            51300,  -- WBC Count
            51301   -- White Blood Cells
        )
        AND l.charttime IS NOT NULL
        ORDER BY l.charttime DESC
        LIMIT %s
        """
        params.append(limit)

        return self.execute_query(query, params)

    def test_connection(self) -> bool:
        """Test database connection"""
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
        """Get basic database information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get version
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]

                # Get table count
                schema = self.config.get('schema', 'mimiciii')
                cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s
                """, (schema,))
                table_count = cursor.fetchone()[0]

                # Get patient count
                cursor.execute(f"SELECT COUNT(*) FROM {schema}.patients")
                patient_count = cursor.fetchone()[0]

                cursor.close()

                return {
                    'database_version': version,
                    'total_tables': table_count,
                    'total_patients': patient_count,
                    'connection_status': 'connected'
                }

        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {
                'connection_status': 'error',
                'error': str(e)
            }

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def create_mock_connector() -> MimicConnector:
    """
    Create a mock connector for testing (uses SQLite in-memory database)
    """
    import sqlite3

    class MockMimicConnector:
        """Mock connector for testing without actual MIMIC database"""

        def __init__(self):
            self.connection = sqlite3.connect(':memory:')
            self.logger = logging.getLogger(__name__)
            self._setup_mock_data()

        def _setup_mock_data(self):
            """Setup mock tables and data"""
            cursor = self.connection.cursor()

            # Create patients table
            cursor.execute("""
            CREATE TABLE patients (
                subject_id INTEGER PRIMARY KEY,
                gender VARCHAR(1),
                dob DATE,
                dod DATE
            )
            """)

            # Create admissions table
            cursor.execute("""
            CREATE TABLE admissions (
                subject_id INTEGER,
                hadm_id INTEGER PRIMARY KEY,
                admittime TIMESTAMP,
                dischtime TIMESTAMP,
                admission_type VARCHAR(50),
                ethnicity VARCHAR(50),
                insurance VARCHAR(50)
            )
            """)

            # Insert sample data
            cursor.execute("""
            INSERT INTO patients VALUES
            (12345, 'M', '1960-05-15', NULL),
            (12346, 'F', '1970-08-22', NULL),
            (12347, 'M', '1955-12-10', NULL)
            """)

            cursor.execute("""
            INSERT INTO admissions VALUES
            (12345, 100001, '2020-03-15 14:30:00', '2020-03-18 10:15:00', 'EMERGENCY', 'WHITE', 'Medicare'),
            (12346, 100002, '2020-04-10 09:45:00', '2020-04-12 16:20:00', 'ELECTIVE', 'BLACK', 'Private'),
            (12347, 100003, '2020-05-01 08:00:00', '2020-05-05 18:30:00', 'EMERGENCY', 'HISPANIC', 'Medicaid')
            """)

            self.connection.commit()
            cursor.close()

        def execute_query(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
            """Execute query on mock database"""
            try:
                # Convert PostgreSQL syntax to SQLite if needed
                query = query.replace('%s', '?')
                if params:
                    df = pd.read_sql_query(query, self.connection, params=params)
                else:
                    df = pd.read_sql_query(query, self.connection)
                return df
            except Exception as e:
                self.logger.error(f"Mock query execution failed: {e}")
                return pd.DataFrame()

        @property
        def cursor(self):
            """Provide cursor for compatibility"""
            return self.connection.cursor()

        def get_patient_demographics(self, subject_id: int) -> pd.DataFrame:
            query = """
            SELECT p.subject_id, p.gender, p.dob, p.dod,
                   a.hadm_id, a.admittime, a.dischtime,
                   a.admission_type, a.ethnicity, a.insurance
            FROM patients p
            LEFT JOIN admissions a ON p.subject_id = a.subject_id
            WHERE p.subject_id = ?
            ORDER BY a.admittime DESC
            """
            return self.execute_query(query, [subject_id])

        def get_vital_signs(self, subject_id: int, hadm_id: Optional[int] = None,
                          limit: int = 1000) -> pd.DataFrame:
            # Return empty DataFrame for mock
            return pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime',
                                       'itemid', 'label', 'valuenum'])

        def get_lab_results(self, subject_id: int, hadm_id: Optional[int] = None,
                          limit: int = 1000) -> pd.DataFrame:
            # Return empty DataFrame for mock
            return pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime',
                                       'itemid', 'label', 'valuenum'])

        def test_connection(self) -> bool:
            return True

        def get_database_info(self) -> Dict[str, Any]:
            return {
                'database_version': 'SQLite Mock',
                'total_tables': 2,
                'total_patients': 3,
                'connection_status': 'mock_connected'
            }

        def disconnect(self):
            if self.connection:
                self.connection.close()

    return MockMimicConnector()