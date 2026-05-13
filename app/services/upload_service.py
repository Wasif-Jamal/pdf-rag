import shutil
import os
from fastapi import UploadFile, HTTPException

from app.config.env_config import config
from app.config.log_config import LogConfig
from app.schema.document_schema import UploadResponse
from app.services.pdf_service import PDFService, pdf_service
from app.services.chunking_service import ChunkingService, chunking_service
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.utils.file_utils import FileUtils
from app.utils.validators import UploadValidator

logger = LogConfig.get_logger(__name__)

class UploadService:
    """
    Orchestrator service for handling PDF uploads and ingestion.
    """
    def __init__(
        self,
        pdf_svc: PDFService = pdf_service,
        chunk_svc: ChunkingService = chunking_service,
        vectorstore_svc: VectorStoreService = vectorstore_service,
    ):
        self.pdf_svc = pdf_svc
        self.chunk_svc = chunk_svc
        self.vectorstore_svc = vectorstore_svc

    async def process_upload(self, file: UploadFile) -> UploadResponse:
        """
        Processes an uploaded PDF file through the ingestion pipeline.
        """
        logger.info(f"Upload request: {file.filename}")
        
        UploadValidator.validate_pdf(file)
        
        FileUtils.ensure_dir(config.RAW_DATA_DIR)
        safe_filename = FileUtils.sanitize_filename(file.filename)
        file_path = os.path.join(config.RAW_DATA_DIR, safe_filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise HTTPException(status_code=500, detail="File save failure.")
            
        try:
            documents = self.pdf_svc.load_documents(file_path)
            chunks = self.chunk_svc.chunk_documents(documents)
            self.vectorstore_svc.add_documents(chunks)
            
            return UploadResponse(
                filename=safe_filename,
                saved_path=file_path,
                status="success",
                total_pages=len(documents),
                total_chunks=len(chunks),
                vectorstore_status="stored"
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Create a singleton instance
upload_service = UploadService()
