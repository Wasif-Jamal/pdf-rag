from typing import List
from pydantic import BaseModel

class RetrievalRequest(BaseModel):
    """Schema for a retrieval query request."""
    query: str
    k: int = 4

class RetrievedDocument(BaseModel):
    """Schema for a single retrieved document chunk."""
    content: str
    metadata: dict

class RetrievalResponse(BaseModel):
    """Schema for the response of a retrieval query."""
    query: str
    results: List[RetrievedDocument]
