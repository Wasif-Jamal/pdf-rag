from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFService:
    """
    Service for loading PDF documents.
    """
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and returns a list of LangChain Document objects.
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            raise ValueError(f"Failed to load PDF documents: {str(e)}")

# Create a singleton instance
pdf_service = PDFService()
