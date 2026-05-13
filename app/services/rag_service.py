from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import HTTPException

from app.config.log_config import LogConfig
from app.schema.chat_schema import ChatRequest, ChatResponse, ChatSource
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.services.llm_service import LLMService, llm_service
from app.prompts.rag_prompt import RAGPromptManager, prompt_manager

logger = LogConfig.get_logger(__name__)

class RAGService:
    """
    Service for the complete RAG pipeline.
    """
    def __init__(
        self, 
        vectorstore_svc: VectorStoreService = vectorstore_service,
        llm_svc: LLMService = llm_service,
        prompt_mgr: RAGPromptManager = prompt_manager
    ):
        self.vectorstore_svc = vectorstore_svc
        self.llm_svc = llm_svc
        self.prompt_mgr = prompt_mgr

    def _format_docs(self, docs: List) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        """
        Executes the full RAG pipeline.
        """
        logger.info(f"RAG Request: '{request.query}'")
        try:
            docs = self.vectorstore_svc.similarity_search(request.query)
            
            llm = self.llm_svc.get_llm()
            prompt = self.prompt_mgr.get_rag_prompt()
            
            rag_chain = (
                {"context": lambda x: self._format_docs(docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(request.query)
            
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
            logger.error(f"RAG Error: {str(e)}")
            raise HTTPException(status_code=500, detail="Generation failed.")

# Create a singleton instance
rag_service = RAGService()
