from typing import List

def chunk_text(text: str) -> List[str]:
    """
    Splits text into chunks of specified size with overlap.
    Chunk size: 1000 characters
    Overlap: 200 characters
    """
    chunk_size = 1000
    overlap = 200
    chunks = []
    
    if not text:
        return chunks
        
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks
