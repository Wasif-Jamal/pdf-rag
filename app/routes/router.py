from fastapi import APIRouter

from app.routes import health, upload

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(upload.router)
