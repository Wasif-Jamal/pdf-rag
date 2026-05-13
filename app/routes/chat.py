from fastapi import APIRouter
from app.schema.chat_schema import ChatRequest, ChatResponse
from app.services.rag_service import rag_service

router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Handles RAG chat requests.
    """
    return await rag_service.generate_response(request)
