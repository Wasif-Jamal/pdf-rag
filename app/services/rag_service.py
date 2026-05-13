from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import HTTPException

from app.schema.chat_schema import ChatRequest, ChatResponse, ChatSource
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.services.llm_service import LLMService, llm_service
from app.prompts.rag_prompt import RAGPromptManager, prompt_manager
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

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
        """Formats retrieved documents into a single string context."""
        return "\n\n".join(doc.page_content for doc in docs)

    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        """
        Executes the full RAG pipeline: retrieval -> formatting -> generation.
        """
        logger.info(f"RAG chat request: '{request.query}'")
        try:
            # 1. Retrieve documents
            docs = self.vectorstore_svc.similarity_search(request.query)
            
            if not docs:
                logger.warning("No relevant documents found for the query.")
            
            # 2. Setup LLM and Prompt
            llm = self.llm_svc.get_llm()
            prompt = self.prompt_mgr.get_rag_prompt()
            
            # 3. Build the chain
            rag_chain = (
                {"context": lambda x: self._format_docs(docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 4. Invoke the chain
            logger.info("Generating response via LLM...")
            answer = rag_chain.invoke(request.query)
            logger.info("Response generated successfully.")
            
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
            logger.error(f"RAG pipeline error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="AI generation engine failure.")

# Create a singleton instance
rag_service = RAGService()
