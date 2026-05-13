from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routes.router import api_router
from app.config.env_config import config
from app.config.log_config import LogConfig
from app.config.qdrant_config import qdrant_config
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.utils.exception_handlers import global_exception_handler, http_exception_handler

logger = LogConfig.get_logger(__name__)

class AppStarter:
    """
    Factory class to bootstrap the FastAPI application using lifespan.
    """
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Handles application startup and shutdown events.
        """
        logger.info("Starting PDF RAG Backend initialization...")
        
        # 1. Validate Environment
        if not config.GOOGLE_API_KEY:
            logger.critical("GOOGLE_API_KEY is missing!")
            raise ValueError("GOOGLE_API_KEY must be set in .env")
        
        # 2. Warm up services (Initializes singletons)
        try:
            logger.info("Checking Qdrant connectivity...")
            client = qdrant_config.get_client()
            client.get_collections()
            
            logger.info("Warming up Embedding model...")
            embedding_service.get_model()
            
            logger.info("Warming up LLM service...")
            llm_service.get_llm()
            
            logger.info("Lifespan startup complete. Service is ready.")
        except Exception as e:
            logger.critical(f"Startup failed: {str(e)}")
            raise e
            
        yield
        
        logger.info("Shutting down PDF RAG Backend...")

    def __init__(self):
        self.app = FastAPI(
            title="PDF RAG Backend",
            description="Production-style RAG system with Lifespan support.",
            version="1.1.0",
            lifespan=self.lifespan
        )

    def _add_middlewares(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self):
        self.app.include_router(api_router)

    def _attach_exception_handlers(self):
        self.app.add_exception_handler(Exception, global_exception_handler)
        self.app.add_exception_handler(HTTPException, http_exception_handler)

    def create_app(self) -> FastAPI:
        self._add_middlewares()
        self._attach_exception_handlers()
        self._register_routes()
        return self.app
