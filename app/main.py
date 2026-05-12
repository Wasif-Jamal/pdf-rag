from typing import Dict

from fastapi import FastAPI

from app.routes import health

app = FastAPI(title="pdf-rag")

app.include_router(health.router)
