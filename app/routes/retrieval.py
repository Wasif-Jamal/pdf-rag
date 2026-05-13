from fastapi import APIRouter
from app.schema.chat_schema import RetrievalRequest, RetrievalResponse
from app.services.retrieval_service import retrieval_service

router = APIRouter(tags=["retrieval"])

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest) -> RetrievalResponse:
    """
    Endpoint to retrieve relevant document chunks.
    """
    return await retrieval_service.retrieve(request)
