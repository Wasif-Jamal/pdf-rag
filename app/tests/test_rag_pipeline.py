import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_community.chat_models import FakeListChatModel
from app.services.rag_service import RAGService
from app.schema.chat_schema import ChatRequest

@pytest.mark.anyio
@patch("app.services.vectorstore_service.VectorStoreService.similarity_search")
@patch("app.services.rag_service.get_llm")
async def test_rag_service_generate_response(mock_get_llm, mock_similarity_search):
    """Test the complete RAG service logic."""
    # 1. Mock retrieval
    mock_similarity_search.return_value = [
        Document(page_content="This is a test context about AI.", metadata={"source": "test.pdf"})
    ]
    
    # 2. Mock LLM
    mock_get_llm.return_value = FakeListChatModel(responses=["The answer is AI."])
    
    # 3. Call service
    service = RAGService()
    request = ChatRequest(query="What is AI?")
    response = await service.generate_response(request)
    
    # 4. Assertions
    assert response.answer == "The answer is AI."
    assert len(response.retrieved_sources) == 1
    assert response.total_sources == 1
    assert response.retrieved_sources[0].content == "This is a test context about AI."
