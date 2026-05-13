from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.services.vectorstore_service import VectorStoreService

@patch("app.services.vectorstore_service.QdrantClient")
@patch("app.services.vectorstore_service.QdrantVectorStore")
def test_vectorstore_initialization(mock_qdrant_store, mock_client):
    """Test vectorstore initialization logic within the service."""
    mock_client_instance = mock_client.return_value
    mock_client_instance.collection_exists.return_value = True
    
    mock_embedding_svc = MagicMock()
    service = VectorStoreService(embedding_svc=mock_embedding_svc)
    vs = service.get_vectorstore()
    
    mock_qdrant_store.assert_called_once()
    assert vs == mock_qdrant_store.return_value

@patch("app.services.vectorstore_service.QdrantClient")
def test_add_documents_to_vectorstore(mock_client):
    """Test adding documents via VectorStoreService with mocked client."""
    mock_vs = MagicMock()
    service = VectorStoreService()
    service._vectorstore = mock_vs
    
    docs = [Document(page_content="test", metadata={})]
    service.add_documents(docs)
    
    mock_vs.add_documents.assert_called_once_with(docs)
