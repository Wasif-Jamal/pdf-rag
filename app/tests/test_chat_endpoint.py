import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.schema.chat_schema import ChatResponse, ChatSource

client = TestClient(app, raise_server_exceptions=False)

@pytest.mark.anyio
@patch("app.routes.chat.rag_service.generate_response")
async def test_chat_endpoint(mock_generate_response):
    """Test the POST /chat endpoint."""
    # Mock service response
    mock_generate_response.return_value = ChatResponse(
        answer="This is a mocked answer.",
        retrieved_sources=[
            ChatSource(content="Source context", metadata={"page": 1})
        ],
        total_sources=1
    )
    
    response = client.post("/chat", json={"session_id": "550e8400-e29b-41d4-a716-446655440000", "query": "Hello?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert len(data["retrieved_sources"]) == 1
    assert data["total_sources"] == 1

from fastapi import HTTPException

def test_chat_endpoint_error():
    """Test error handling in /chat endpoint."""
    with patch("app.routes.chat.rag_service.generate_response", side_effect=HTTPException(status_code=500, detail="Service failure")):
        response = client.post("/chat", json={"session_id": "550e8400-e29b-41d4-a716-446655440000", "query": "Hello?"})
        assert response.status_code == 500
        assert "Service failure" in response.json()["detail"]
