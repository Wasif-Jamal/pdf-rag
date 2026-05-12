import os
import shutil
from fastapi import UploadFile, HTTPException

from app.schema.document_schema import UploadResponse
from app.ingestion.pdf_loader import load_pdf_documents
from app.rag.chunking import chunk_documents

RAW_DATA_DIR = "data/raw"

async def process_pdf_upload(file: UploadFile) -> UploadResponse:
    """
    Handles the uploaded PDF file:
    1. Validates the file type.
    2. Saves the file locally to data/raw/.
    3. Loads the PDF into LangChain Documents.
    4. Chunks the documents.
    5. Returns the processing metrics.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Create directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Ensure safe filename
    filename = file.filename if file.filename else "uploaded.pdf"
    file_path = os.path.join(RAW_DATA_DIR, filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Extract and chunk using LangChain
    try:
        documents = load_pdf_documents(file_path)
        chunks = chunk_documents(documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
        
    return UploadResponse(
        filename=filename,
        saved_path=file_path,
        status="success",
        total_pages=len(documents),
        total_chunks=len(chunks)
    )
