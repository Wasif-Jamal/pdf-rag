from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.rag.embeddings import get_embedding_model

QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "pdf_documents"

client = QdrantClient(path=QDRANT_PATH)

def _ensure_collection_exists():
    """Ensures the required collection exists in Qdrant."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def get_vectorstore() -> QdrantVectorStore:
    """
    Returns the initialized Qdrant vector store.
    """
    _ensure_collection_exists()
    embeddings = get_embedding_model()
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vectorstore

def add_documents_to_vectorstore(documents: List[Document]) -> None:
    """
    Adds chunks of documents to the Qdrant vector store.
    """
    if not documents:
        return
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
