from fastapi import APIRouter

router = APIRouter(tags=["system"])

@router.get("/health")
def health_check():
    """
    Service health check endpoint.
    """
    return {"status": "healthy", "service": "pdf-rag-backend"}
