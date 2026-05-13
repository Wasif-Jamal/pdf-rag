from unittest.mock import patch
from app.rag.pipeline import generate_rag_response
from langchain_core.documents import Document
from langchain_community.chat_models import FakeListChatModel

@patch("app.rag.pipeline.retrieve_documents")
@patch("app.rag.pipeline.get_llm")
def test_generate_rag_response(mock_get_llm, mock_retrieve_docs):
    """Test the complete RAG pipeline with mocked components."""
    # 1. Mock retrieval
    mock_retrieve_docs.return_value = [
        Document(page_content="This is a test context about AI.", metadata={"source": "test.pdf"})
    ]
    
    # 2. Mock LLM using FakeListChatModel
    mock_get_llm.return_value = FakeListChatModel(responses=["The answer is AI."])
    
    # 3. Call pipeline
    response = generate_rag_response("What is AI?")
    
    # 4. Assertions
    assert response["answer"] == "The answer is AI."
    assert "retrieved_sources" in response
    assert response["total_sources"] == 1
    assert response["retrieved_sources"][0]["content"] == "This is a test context about AI."
    # Since we use StrOutputParser, the answer should be a string
    # In our mock, if parser is working, it extracts .content or returns string
    # We'll see if the mock setup works for StrOutputParser
