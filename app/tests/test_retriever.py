from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.rag.retriever import retrieve_documents

@patch("app.rag.retriever.get_vectorstore")
def test_retrieve_documents(mock_get_vectorstore):
    """Test document retrieval."""
    mock_vs = MagicMock()
    mock_get_vectorstore.return_value = mock_vs
    
    mock_docs = [Document(page_content="result", metadata={})]
    mock_vs.similarity_search.return_value = mock_docs
    
    results = retrieve_documents("query", k=2)
    
    mock_vs.similarity_search.assert_called_once_with(query="query", k=2)
    assert results == mock_docs
