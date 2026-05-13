import pytest
from unittest.mock import patch
from app.services.llm_service import LLMService
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.env_config import config

def test_llm_service_initialization():
    """Test that LLMService returns a ChatGoogleGenerativeAI instance."""
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        # We need to refresh config if it was already initialized without key
        # But for test, we can just mock the config object
        with patch("app.services.llm_service.config") as mock_config:
            mock_config.GOOGLE_API_KEY = "test-key"
            mock_config.GEMINI_MODEL_NAME = "gemini-2.0-flash"
            
            service = LLMService()
            llm = service.get_llm()
            assert isinstance(llm, ChatGoogleGenerativeAI)
            assert llm.model == "gemini-2.0-flash"

def test_llm_service_error_without_key():
    """Test that LLMService raises error if key is missing."""
    with patch("app.services.llm_service.config") as mock_config:
        mock_config.GOOGLE_API_KEY = None
        service = LLMService()
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
            service.get_llm()
