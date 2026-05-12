from app.rag.chunking import chunk_text

def test_chunk_text_empty():
    """Test chunking with empty text."""
    assert chunk_text("") == []

def test_chunk_text_short():
    """Test chunking with text shorter than chunk size."""
    text = "Short text"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_long():
    """Test chunking with text longer than chunk size to verify overlap."""
    text = "A" * 1500
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 700
