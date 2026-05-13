import os
from qdrant_client import QdrantClient
from app.config.env_config import config
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class StartupService:
    """
    Service to handle startup validation and initialization.
    """
    @staticmethod
    def validate_environment():
        """
        Validates the environment configuration.
        """
        logger.info("Validating environment configuration...")
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set in .env")
        logger.info("Environment validation successful.")

    @staticmethod
    def validate_qdrant_connectivity():
        """
        Validates that Qdrant is accessible.
        """
        logger.info(f"Validating Qdrant connectivity at {config.QDRANT_PATH}...")
        try:
            client = QdrantClient(path=config.QDRANT_PATH)
            client.get_collections()
            logger.info("Qdrant connectivity validation successful.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise RuntimeError(f"Qdrant connection failed: {str(e)}")

    @classmethod
    def run_all_checks(cls):
        """
        Runs all startup checks.
        """
        logger.info("Starting production-style backend initialization...")
        cls.validate_environment()
        cls.validate_qdrant_connectivity()
        logger.info("All startup checks passed.")
