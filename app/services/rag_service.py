from typing import Dict, Any, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import HTTPException

from app.schema.chat_schema import ChatRequest, ChatResponse, ChatSource
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.llms.gemini import get_llm
from app.config.prompts import get_rag_prompt

class RAGService:
    """
    Service for the complete RAG pipeline.
    """
    def __init__(self, vectorstore_svc: VectorStoreService = vectorstore_service):
        self.vectorstore_svc = vectorstore_svc

    def _format_docs(self, docs: List) -> str:
        """Formats retrieved documents into a single string context."""
        return "\n\n".join(doc.page_content for doc in docs)

    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        """
        Executes the full RAG pipeline: retrieval -> formatting -> generation.
        """
        try:
            # 1. Retrieve documents
            docs = self.vectorstore_svc.similarity_search(request.query)
            
            # 2. Setup LLM and Prompt
            llm = get_llm()
            prompt = get_rag_prompt()
            
            # 3. Build the chain
            rag_chain = (
                {"context": lambda x: self._format_docs(docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 4. Invoke the chain
            answer = rag_chain.invoke(request.query)
            
            # 5. Format sources
            sources = [
                ChatSource(content=doc.page_content, metadata=doc.metadata)
                for doc in docs
            ]
            
            return ChatResponse(
                answer=answer,
                retrieved_sources=sources,
                total_sources=len(sources)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Create a singleton instance
rag_service = RAGService()
