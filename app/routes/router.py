from fastapi import APIRouter

from app.routes import health, upload, retrieval

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(upload.router)
api_router.include_router(retrieval.router)
