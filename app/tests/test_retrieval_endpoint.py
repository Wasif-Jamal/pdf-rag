import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.schema.chat_schema import RetrievalResponse, RetrievedDocument

client = TestClient(app)

@pytest.mark.anyio
@patch("app.routes.retrieval.retrieval_service.retrieve")
async def test_successful_retrieval(mock_retrieve):
    """Test successful retrieval endpoint."""
    mock_retrieve.return_value = RetrievalResponse(
        query="test query",
        results=[
            RetrievedDocument(content="Test content", metadata={"page": 1})
        ]
    )
    
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
