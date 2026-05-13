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

class ChatRequest(BaseModel):
    """Schema for a chat request."""
    query: str

class ChatSource(BaseModel):
    """Schema for a source in chat response."""
    content: str
    metadata: dict

class ChatResponse(BaseModel):
    """Schema for the chat response."""
    answer: str
    retrieved_sources: List[ChatSource]
    total_sources: int
