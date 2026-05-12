from fastapi import HTTPException
from app.schema.chat_schema import RetrievalRequest, RetrievalResponse, RetrievedDocument
from app.rag.retriever import retrieve_documents

async def process_retrieval(request: RetrievalRequest) -> RetrievalResponse:
    """
    Handles semantic retrieval of documents from Qdrant.
    """
    try:
        docs = retrieve_documents(query=request.query, k=request.k)
        
        results = []
        for doc in docs:
            results.append(
                RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata
                )
            )
            
        return RetrievalResponse(
            query=request.query,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retrieval: {str(e)}")
