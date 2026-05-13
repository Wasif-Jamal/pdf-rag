import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from app.services.pdf_service import pdf_service

@patch("app.services.pdf_service.PyPDFLoader")
def test_load_pdf_documents_success(mock_pypdf_loader):
    """Test successful LangChain document loading."""
    mock_instance = mock_pypdf_loader.return_value
    mock_instance.load.return_value = [
        Document(page_content="Page 1 text", metadata={"page": 1}),
        Document(page_content="Page 2 text", metadata={"page": 2})
    ]
    
    documents = pdf_service.load_documents("dummy.pdf")
    assert len(documents) == 2
    assert documents[0].page_content == "Page 1 text"
    assert documents[1].metadata["page"] == 2

def test_load_pdf_documents_error():
    """Test error handling for invalid PDF paths."""
    with pytest.raises(ValueError):
        pdf_service.load_documents("non_existent.pdf")
