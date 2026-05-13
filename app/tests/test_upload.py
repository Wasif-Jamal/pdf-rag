import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.schema.document_schema import UploadResponse

client = TestClient(app)

@pytest.mark.anyio
@patch("app.routes.upload.upload_service.process_upload")
async def test_successful_pdf_upload(mock_process_upload):
    """Test successful PDF upload and processing flow."""
    mock_process_upload.return_value = UploadResponse(
        filename="test.pdf",
        saved_path="data/raw/test.pdf",
        status="success",
        total_pages=1,
        total_chunks=1,
        vectorstore_status="stored"
    )

    file_content = b"%PDF-1.4 mock pdf content"
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["status"] == "success"
    assert data["total_pages"] == 1

def test_invalid_file_upload():
    """Test rejecting non-PDF files."""
    file_content = b"Not a PDF"
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    )
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are allowed"
