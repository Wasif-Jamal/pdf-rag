from qdrant_client import QdrantClient
from app.config.env_config import config
from app.config.log_config import LogConfig

logger = LogConfig.get_logger(__name__)

class QdrantConfig:
    """
    Manager for Qdrant client initialization and lifecycle.
    """
    def __init__(self):
        self._client = None

    def get_client(self) -> QdrantClient:
        """
        Returns the singleton Qdrant client instance.
        """
        if self._client is None:
            logger.info(f"Initializing Qdrant client at {config.QDRANT_PATH}")
            try:
                self._client = QdrantClient(path=config.QDRANT_PATH)
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {str(e)}")
                raise RuntimeError(f"Qdrant connection failed: {str(e)}")
        return self._client

# Singleton instance
qdrant_config = QdrantConfig()
