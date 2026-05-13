from fastapi import HTTPException
from app.rag.pipeline import generate_rag_response
from app.schema.chat_schema import ChatRequest, ChatResponse

async def chat_handler(request: ChatRequest) -> ChatResponse:
    """
    Handles the chat request by running the RAG pipeline.
    """
    try:
        response_data = generate_rag_response(request.query)
        return ChatResponse(**response_data)
    except Exception as e:
        # In a real app, you'd log the error here
        raise HTTPException(status_code=500, detail=str(e))
