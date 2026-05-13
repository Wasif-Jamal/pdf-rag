from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance

from app.config.env_config import config
from app.config.log_config import LogConfig
from app.config.qdrant_config import qdrant_config
from app.services.embedding_service import EmbeddingService, embedding_service

logger = LogConfig.get_logger(__name__)

class VectorStoreService:
    """
    Service for managing Qdrant vector store operations.
    """
    def __init__(self, embedding_svc: EmbeddingService = embedding_service):
        self.embedding_svc = embedding_svc
        self._vectorstore = None

    @property
    def client(self):
        """Returns the Qdrant client from central config."""
        return qdrant_config.get_client()

    def _ensure_collection_exists(self):
        """Ensures the required collection exists in Qdrant."""
        if not self.client.collection_exists(collection_name=config.COLLECTION_NAME):
            logger.info(f"Creating collection: {config.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def get_vectorstore(self) -> QdrantVectorStore:
        """
        Returns the initialized Qdrant vector store.
        """
        if self._vectorstore is None:
            self._ensure_collection_exists()
            embeddings = self.embedding_svc.get_model()
            self._vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=config.COLLECTION_NAME,
                embedding=embeddings,
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds chunks of documents to the Qdrant vector store.
        """
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store.")
        vectorstore = self.get_vectorstore()
        vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search in the vector store.
        """
        logger.info(f"Similarity search: '{query}'")
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search(query=query, k=k)

# Create a singleton instance
vectorstore_service = VectorStoreService()
