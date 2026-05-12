from typing import Dict

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the service status.
    """
    return {
        "status": "ok",
        "service": "pdf-rag"
    }
