import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

@patch("app.controller.chat_controller.generate_rag_response")
def test_chat_endpoint(mock_rag_pipeline):
    """Test the POST /chat endpoint."""
    # Mock pipeline response
    mock_rag_pipeline.return_value = {
        "answer": "This is a mocked answer.",
        "retrieved_sources": [
            {"content": "Source context", "metadata": {"page": 1}}
        ],
        "total_sources": 1
    }
    
    response = client.post("/chat", json={"query": "Hello?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert len(data["retrieved_sources"]) == 1
    assert data["total_sources"] == 1

def test_chat_endpoint_error():
    """Test error handling in /chat endpoint."""
    with patch("app.controller.chat_controller.generate_rag_response", side_effect=Exception("Pipeline failure")):
        response = client.post("/chat", json={"query": "Hello?"})
        assert response.status_code == 500
        assert "Pipeline failure" in response.json()["detail"]
