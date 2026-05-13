from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from app.config.log_config import LogConfig

logger = LogConfig.get_logger(__name__)

class PDFService:
    """
    Service for loading PDF documents.
    """
    def load_documents(self, file_path: str) -> List[Document]:
        logger.info(f"Loading PDF: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"PDF Load Error: {str(e)}")
            raise ValueError(f"Failed to load PDF: {str(e)}")

# Create a singleton instance
pdf_service = PDFService()
