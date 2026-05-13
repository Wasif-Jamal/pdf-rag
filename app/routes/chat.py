from fastapi import APIRouter
from app.controller.chat_controller import chat_handler
from app.schema.chat_schema import ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

router.post("", response_model=ChatResponse)(chat_handler)
