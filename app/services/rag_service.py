from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import HTTPException

from app.config.log_config import LogConfig
from app.schema.chat_schema import ChatRequest, ChatResponse, ChatSource
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.services.llm_service import LLMService, llm_service
from app.prompts.rag_prompt import RAGPromptManager, prompt_manager
from app.services.short_memory_service import (
    ShortMemoryService,
    short_memory_service,
)

logger = LogConfig.get_logger(__name__)

class RAGService:
    """
    Service for the complete RAG pipeline.
    """
    def __init__(
        self, 
        vectorstore_svc: VectorStoreService = vectorstore_service,
        llm_svc: LLMService = llm_service,
        prompt_mgr: RAGPromptManager = prompt_manager,
        short_memory_svc: ShortMemoryService = short_memory_service
    ):
        self.vectorstore_svc = vectorstore_svc
        self.llm_svc = llm_svc
        self.prompt_mgr = prompt_mgr
        self.short_memory_svc = short_memory_svc

    def _format_docs(self, docs: List) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_chat_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous conversation."

        return "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in history
        )

    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        """
        Executes the full RAG pipeline.
        """
        logger.info(f"RAG Request: '{request.query}'")
        try:
            docs = self.vectorstore_svc.similarity_search(request.query)
            
            llm = self.llm_svc.get_llm()
            prompt = self.prompt_mgr.get_rag_prompt()
            chat_history = await self.short_memory_svc.get_history(
                request.session_id
            )
            formatted_history = self._format_chat_history(
                chat_history
            )
            
            rag_chain = (
                {
                    "context": lambda x: self._format_docs(docs),
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: formatted_history
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(request.query)

            await self.short_memory_svc.add_message(
                session_id=request.session_id,
                role="user",
                content=request.query,
            )

            await self.short_memory_svc.add_message(
                session_id=request.session_id,
                role="assistant",
                content=answer,
            )
            
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
