from unittest.mock import patch
from app.services.embedding_service import EmbeddingService
from app.config.env_config import config

@patch("app.services.embedding_service.HuggingFaceEmbeddings")
def test_get_embedding_model(mock_hf_embeddings):
    """Test embedding model initialization with config."""
    service = EmbeddingService()
    model = service.get_model()
    mock_hf_embeddings.assert_called_once_with(
        model_name=config.EMBEDDING_MODEL_NAME
    )
    assert model == mock_hf_embeddings.return_value
