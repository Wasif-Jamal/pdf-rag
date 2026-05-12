from langchain_core.documents import Document
from app.rag.chunking import chunk_documents

def test_chunk_documents_empty():
    """Test chunking with empty documents list."""
    assert chunk_documents([]) == []

def test_chunk_documents_short():
    """Test chunking with text shorter than chunk size."""
    docs = [Document(page_content="Short text", metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text"

def test_chunk_documents_long():
    """Test chunking with text longer than chunk size to verify overlap."""
    text = "A" * 1500
    docs = [Document(page_content=text, metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs)
    
    assert len(chunks) == 2
    assert len(chunks[0].page_content) == 1000
    assert len(chunks[1].page_content) == 700
