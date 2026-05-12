import io
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

@patch("app.controller.upload_controller.load_pdf_text")
@patch("app.controller.upload_controller.chunk_text")
def test_successful_pdf_upload(mock_chunk_text, mock_load_pdf_text):
    """Test successful PDF upload and processing flow."""
    mock_load_pdf_text.return_value = "Mocked PDF content"
    mock_chunk_text.return_value = ["Mocked PDF content"]

    file_content = b"%PDF-1.4 mock pdf content"
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["status"] == "success"
    assert "data/raw/test.pdf" in data["saved_path"]
    assert data["total_characters"] == len("Mocked PDF content")
    assert data["total_chunks"] == 1

def test_invalid_file_upload():
    """Test rejecting non-PDF files."""
    file_content = b"Not a PDF"
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    )
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are allowed"
