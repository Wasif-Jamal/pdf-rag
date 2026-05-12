from typing import List
from langchain_core.documents import Document
from app.rag.vectorstore import get_vectorstore

def retrieve_documents(query: str, k: int = 4) -> List[Document]:
    """
    Retrieves the most relevant documents for a given query using similarity search.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query=query, k=k)
    return results
