from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from app.config.env_config import config
from app.services.embedding_service import EmbeddingService, embedding_service
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class VectorStoreService:
    """
    Service for managing Qdrant vector store operations.
    """
    def __init__(self, embedding_svc: EmbeddingService = embedding_service):
        self.embedding_svc = embedding_svc
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self._vectorstore = None

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
            logger.warning("No documents to add to vector store.")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store.")
        vectorstore = self.get_vectorstore()
        vectorstore.add_documents(documents)
        logger.info("Documents added successfully.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search in the vector store.
        """
        logger.info(f"Performing similarity search for: '{query}' (k={k})")
        vectorstore = self.get_vectorstore()
        results = vectorstore.similarity_search(query=query, k=k)
        logger.info(f"Retrieved {len(results)} relevant chunks.")
        return results

# Create a singleton instance
vectorstore_service = VectorStoreService()
