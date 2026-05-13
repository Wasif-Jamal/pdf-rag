from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class PDFService:
    """
    Service for loading PDF documents.
    """
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and returns a list of LangChain Document objects.
        """
        logger.info(f"Loading PDF from: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} pages.")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise ValueError(f"Failed to load PDF documents: {str(e)}")

# Create a singleton instance
pdf_service = PDFService()
