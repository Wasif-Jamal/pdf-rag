from fastapi import APIRouter
from app.schema.chat_schema import RetrievalRequest, RetrievalResponse
from app.controller.retrieval_controller import process_retrieval

router = APIRouter()

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents_endpoint(request: RetrievalRequest) -> RetrievalResponse:
    """
    Endpoint to retrieve relevant document chunks based on a query.
    """
    return await process_retrieval(request)
