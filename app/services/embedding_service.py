from langchain_huggingface import HuggingFaceEmbeddings
from app.config.env_config import config
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class EmbeddingService:
    """
    Service for managing the embedding model.
    """
    def __init__(self):
        self._model = None

    def get_model(self) -> HuggingFaceEmbeddings:
        """
        Returns the initialized HuggingFace embeddings model.
        Initializes it on the first call.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
            self._model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        return self._model

# Create a singleton instance
embedding_service = EmbeddingService()
