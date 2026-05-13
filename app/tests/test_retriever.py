from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.services.vectorstore_service import VectorStoreService

@patch("app.services.vectorstore_service.qdrant_config.get_client")
@patch("app.services.vectorstore_service.QdrantVectorStore")
def test_vectorstore_similarity_search(mock_qdrant_vs, mock_get_client):
    """Test document retrieval through VectorStoreService."""
    mock_embedding_svc = MagicMock()
    service = VectorStoreService(embedding_svc=mock_embedding_svc)
    
    mock_vs_instance = mock_qdrant_vs.return_value
    mock_docs = [Document(page_content="result", metadata={})]
    mock_vs_instance.similarity_search.return_value = mock_docs
    
    results = service.similarity_search("query", k=2)
    
    mock_vs_instance.similarity_search.assert_called_once_with(query="query", k=2)
    assert results == mock_docs
