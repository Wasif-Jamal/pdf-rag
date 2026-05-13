from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class EnvConfig(BaseSettings):
    """
    Environment configuration using pydantic-settings.
    """
    GOOGLE_API_KEY: str
    QDRANT_PATH: str = "data/qdrant"
    RAW_DATA_DIR: str = "data/raw"
    COLLECTION_NAME: str = "pdf_documents"
    
    # Model configs
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_config() -> EnvConfig:
    """Returns a cached instance of the environment configuration."""
    return EnvConfig()

# Singleton instance
config = get_config()
