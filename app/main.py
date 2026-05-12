from fastapi import FastAPI

from app.routes.router import api_router

app = FastAPI(title="pdf-rag")

app.include_router(api_router)
