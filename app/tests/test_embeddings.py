from unittest.mock import patch
from app.rag.embeddings import get_embedding_model

@patch("app.rag.embeddings.HuggingFaceEmbeddings")
def test_get_embedding_model(mock_hf_embeddings):
    """Test embedding model initialization."""
    model = get_embedding_model()
    mock_hf_embeddings.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert model == mock_hf_embeddings.return_value
