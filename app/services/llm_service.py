import os
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.env_config import config
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class LLMService:
    """
    Service for managing the LLM (Gemini) instance.
    """
    def __init__(self):
        self._llm = None

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Initializes and returns the Gemini LLM instance.
        """
        if self._llm is None:
            api_key = config.GOOGLE_API_KEY
            if not api_key:
                logger.error("GOOGLE_API_KEY is missing from configuration")
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            logger.info(f"Initializing LLM: {config.GEMINI_MODEL_NAME}")
            self._llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL_NAME,
                google_api_key=api_key,
                temperature=0,
            )
        return self._llm

# Singleton instance
llm_service = LLMService()
