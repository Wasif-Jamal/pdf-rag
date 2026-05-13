from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    """
    Service for managing the embedding model.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def get_model(self) -> HuggingFaceEmbeddings:
        """
        Returns the initialized HuggingFace embeddings model.
        Initializes it on the first call.
        """
        if self._model is None:
            self._model = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._model

# Create a singleton instance for reuse across the app
embedding_service = EmbeddingService()
