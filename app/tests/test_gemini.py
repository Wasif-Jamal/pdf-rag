import pytest
from unittest.mock import patch
from app.llms.gemini import get_llm
from langchain_google_genai import ChatGoogleGenerativeAI

def test_get_llm_without_key():
    """Test that get_llm raises ValueError if GOOGLE_API_KEY is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
            get_llm()

def test_get_llm_with_key():
    """Test that get_llm returns a ChatGoogleGenerativeAI instance."""
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        llm = get_llm()
        assert isinstance(llm, ChatGoogleGenerativeAI)
        assert llm.model == "gemini-2.0-flash"
