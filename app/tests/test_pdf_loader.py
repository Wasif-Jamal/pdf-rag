import pytest
from unittest.mock import patch, MagicMock
from app.ingestion.pdf_loader import load_pdf_text

@patch("app.ingestion.pdf_loader.PdfReader")
def test_load_pdf_text_success(mock_pdf_reader):
    """Test successful text extraction from PDF pages."""
    mock_instance = mock_pdf_reader.return_value
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "Page 1 text."
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "Page 2 text."
    
    mock_instance.pages = [mock_page_1, mock_page_2]
    
    result = load_pdf_text("dummy.pdf")
    assert result == "Page 1 text.\nPage 2 text."

@patch("app.ingestion.pdf_loader.PdfReader")
def test_load_pdf_text_empty_pages(mock_pdf_reader):
    """Test text extraction skips empty pages."""
    mock_instance = mock_pdf_reader.return_value
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = ""
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "Page 2 text."
    
    mock_instance.pages = [mock_page_1, mock_page_2]
    
    result = load_pdf_text("dummy.pdf")
    assert result == "Page 2 text."

def test_load_pdf_text_error():
    """Test error handling for non-existent PDF."""
    with pytest.raises(ValueError):
        load_pdf_text("non_existent.pdf")
