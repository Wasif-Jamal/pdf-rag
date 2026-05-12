from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf_documents(file_path: str) -> List[Document]:
    """
    Loads a PDF file and returns a list of LangChain Document objects.
    Each document typically represents a page.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise ValueError(f"Failed to load PDF documents: {str(e)}")
