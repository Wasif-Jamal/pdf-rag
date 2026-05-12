from fastapi.testclient import TestClient
from unittest.mock import patch
from langchain_core.documents import Document
from app.main import app

client = TestClient(app)

@patch("app.controller.retrieval_controller.retrieve_documents")
def test_successful_retrieval(mock_retrieve_documents):
    """Test successful retrieval endpoint."""
    mock_retrieve_documents.return_value = [
        Document(page_content="Test content", metadata={"page": 1})
    ]
    
    response = client.post(
        "/retrieve",
        json={"query": "test query", "k": 1}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "Test content"
    assert data["results"][0]["metadata"]["page"] == 1
