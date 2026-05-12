from fastapi import APIRouter, File, UploadFile

from app.schema.document_schema import UploadResponse
from app.controller.upload_controller import process_pdf_upload

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Endpoint to upload and process a PDF file.
    """
    return await process_pdf_upload(file)
