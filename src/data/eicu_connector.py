"""
eICU Collaborative Research Database Connector
Handles database connections for eICU PostgreSQL database (schema: eicu_crd).
"""

import psycopg2
import pandas as pd
import logging
import os
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()


class EicuConnector:
    """Database connector for eICU Collaborative Research Database."""

    def __init__(self, config: Dict[str, str] = None):
        if config is None:
            self.config = {
                'host': os.getenv('EICU_DB_HOST', '192.168.1.249'),
                'port': int(os.getenv('EICU_DB_PORT', 5432)),
                'database': os.getenv('EICU_DB_NAME', 'eicu'),
                'user': os.getenv('EICU_DB_USER', 'postgres'),
                'password': os.getenv('EICU_DB_PASSWORD', 'postgres'),
                'schema': os.getenv('EICU_DB_SCHEMA', 'eicu_crd'),
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
    def schema(self):
        return self.config.get('schema', 'eicu_crd')

    def connect(self) -> None:
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
            )
            self.logger.info("Connected to eICU database")
        except Exception as e:
            self.logger.error(f"Failed to connect to eICU: {e}")
            raise

    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    @contextmanager
    def get_connection(self):
        if not self.connection:
            self.connect()
        try:
            yield self.connection
        except Exception:
            if self.connection:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            raise

    def execute_query(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params)

    def test_connection(self) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM eicu_crd.patient")
                count = cursor.fetchone()[0]
                cursor.close()
                self.logger.info(f"eICU patient count: {count}")
                return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
