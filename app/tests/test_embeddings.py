from unittest.mock import patch
from app.services.embedding_service import EmbeddingService

@patch("app.services.embedding_service.HuggingFaceEmbeddings")
def test_get_embedding_model(mock_hf_embeddings):
    """Test embedding model initialization."""
    service = EmbeddingService()
    model = service.get_model()
    mock_hf_embeddings.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert model == mock_hf_embeddings.return_value
