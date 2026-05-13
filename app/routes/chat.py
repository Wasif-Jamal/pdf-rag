from fastapi import APIRouter
from app.services.rag_service import rag_service
from app.schema.chat_schema import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Handles the chat request by running the RAG pipeline.
    """
    return await rag_service.generate_response(request)
