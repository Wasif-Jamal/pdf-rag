from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.rag.vectorstore import get_vectorstore, add_documents_to_vectorstore

@patch("app.rag.vectorstore.client")
@patch("app.rag.vectorstore.QdrantVectorStore")
@patch("app.rag.vectorstore.get_embedding_model")
def test_get_vectorstore(mock_get_embedding, mock_qdrant_store, mock_client):
    """Test vectorstore initialization."""
    mock_client.collection_exists.return_value = True
    
    vs = get_vectorstore()
    
    mock_qdrant_store.assert_called_once()
    assert vs == mock_qdrant_store.return_value

@patch("app.rag.vectorstore.get_vectorstore")
def test_add_documents_to_vectorstore(mock_get_vectorstore):
    """Test adding documents to vectorstore."""
    mock_vs = MagicMock()
    mock_get_vectorstore.return_value = mock_vs
    
    docs = [Document(page_content="test", metadata={})]
    add_documents_to_vectorstore(docs)
    
    mock_vs.add_documents.assert_called_once_with(docs)
