from fastapi import APIRouter, File, UploadFile

from app.schema.document_schema import UploadResponse
from app.services.upload_service import upload_service

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Endpoint to upload and process a PDF file.
    """
    return await upload_service.process_upload(file)
