"""
Database connection helper
Handles both real MIMIC-III connections and mock data for development
"""

import os
import logging
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_mimic_connection(use_mock: bool = None) -> Any:
    """
    Get MIMIC database connection (real or mock)

    Args:
        use_mock: If True, use mock connector. If None, auto-detect based on environment

    Returns:
        Database connection object
    """

    # Auto-detect if not specified
    if use_mock is None:
        # Use mock if no database credentials are set
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        use_mock = not (db_user and db_password)

    logger = logging.getLogger(__name__)

    if use_mock:
        logger.info("Using mock MIMIC connector for development")
        from src.data.mimic_connector import create_mock_connector
        return create_mock_connector()
    else:
        logger.info("Connecting to real MIMIC-III PostgreSQL database")
        from src.data.mimic_connector import MimicConnector

        # Validate environment variables
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}. "
                           f"Please set them in your .env file.")

        try:
            connector = MimicConnector()
            connector.connect()
            return connector
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.info("Falling back to mock connector")
            from src.data.mimic_connector import create_mock_connector
            return create_mock_connector()


def test_database_connection() -> bool:
    """
    Test database connection

    Returns:
        True if connection successful, False otherwise
    """

    try:
        conn = get_mimic_connection()
        result = conn.test_connection()

        if hasattr(conn, 'disconnect'):
            conn.disconnect()

        return result
    except Exception as e:
        logging.error(f"Database connection test failed: {e}")
        return False