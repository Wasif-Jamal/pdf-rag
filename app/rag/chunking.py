from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits LangChain documents into smaller chunks.
    Chunk size: 1000 characters
    Overlap: 200 characters
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
