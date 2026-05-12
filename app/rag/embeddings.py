from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the configured HuggingFace embeddings model.
    Model: sentence-transformers/all-MiniLM-L6-v2
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
