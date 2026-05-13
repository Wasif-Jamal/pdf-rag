import os
import shutil
from fastapi import UploadFile, HTTPException

from app.schema.document_schema import UploadResponse
from app.services.pdf_service import PDFService, pdf_service
from app.services.chunking_service import ChunkingService, chunking_service
from app.services.vectorstore_service import VectorStoreService, vectorstore_service

class UploadService:
    """
    Orchestrator service for handling PDF uploads and ingestion.
    """
    def __init__(
        self,
        pdf_svc: PDFService = pdf_service,
        chunk_svc: ChunkingService = chunking_service,
        vectorstore_svc: VectorStoreService = vectorstore_service,
        raw_data_dir: str = "data/raw"
    ):
        self.pdf_svc = pdf_svc
        self.chunk_svc = chunk_svc
        self.vectorstore_svc = vectorstore_svc
        self.raw_data_dir = raw_data_dir

    async def process_upload(self, file: UploadFile) -> UploadResponse:
        """
        Processes an uploaded PDF file through the ingestion pipeline.
        """
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create directory if it doesn't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Ensure safe filename
        filename = file.filename if file.filename else "uploaded.pdf"
        file_path = os.path.join(self.raw_data_dir, filename)
        
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
            
        # Ingestion pipeline
        try:
            documents = self.pdf_svc.load_documents(file_path)
            chunks = self.chunk_svc.chunk_documents(documents)
            self.vectorstore_svc.add_documents(chunks)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
        return UploadResponse(
            filename=filename,
            saved_path=file_path,
            status="success",
            total_pages=len(documents),
            total_chunks=len(chunks),
            vectorstore_status="stored"
        )

# Create a singleton instance
upload_service = UploadService()
