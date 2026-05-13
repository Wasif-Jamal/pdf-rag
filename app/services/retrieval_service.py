from fastapi import HTTPException
from app.schema.chat_schema import RetrievalRequest, RetrievalResponse, RetrievedDocument
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class RetrievalService:
    """
    Service for orchestrating document retrieval.
    """
    def __init__(self, vectorstore_svc: VectorStoreService = vectorstore_service):
        self.vectorstore_svc = vectorstore_svc

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Handles semantic retrieval of documents.
        """
        logger.info(f"Retrieval request: query='{request.query}', k={request.k}")
        try:
            docs = self.vectorstore_svc.similarity_search(query=request.query, k=request.k)
            
            results = [
                RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in docs
            ]
            
            return RetrievalResponse(
                query=request.query,
                results=results
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Retrieval engine error.")

# Create a singleton instance
retrieval_service = RetrievalService()
