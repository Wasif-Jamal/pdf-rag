from typing import Dict, Any, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.rag.retriever import retrieve_documents
from app.llms.gemini import get_llm
from app.config.prompts import get_rag_prompt

def format_docs(docs: List) -> str:
    """Formats retrieved documents into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

def generate_rag_response(query: str) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline: retrieval -> formatting -> generation.
    
    Args:
        query (str): The user's query.
        
    Returns:
        Dict[str, Any]: Answer and source information.
    """
    # 1. Retrieve documents
    docs = retrieve_documents(query)
    
    # 2. Setup LLM and Prompt
    llm = get_llm()
    prompt = get_rag_prompt()
    
    # 3. Build the chain (LangChain Expression Language - LCEL)
    rag_chain = (
        {"context": lambda x: format_docs(docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 4. Invoke the chain
    answer = rag_chain.invoke(query)
    
    # 5. Format sources
    sources = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    
    return {
        "answer": answer,
        "retrieved_sources": sources,
        "total_sources": len(sources)
    }
