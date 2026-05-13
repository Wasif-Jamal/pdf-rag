from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class ChunkingService:
    """
    Service for chunking documents.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits LangChain documents into smaller chunks.
        """
        logger.info(f"Chunking {len(documents)} pages...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

# Create a singleton instance
chunking_service = ChunkingService()
