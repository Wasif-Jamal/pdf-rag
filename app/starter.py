from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routes.router import api_router
from app.services.startup_service import StartupService
from app.utils.exception_handlers import global_exception_handler, http_exception_handler
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)

class AppStarter:
    """
    Factory class to bootstrap the FastAPI application.
    """
    def __init__(self):
        self.app = FastAPI(
            title="PDF RAG Production Backend",
            description="A clean, service-oriented RAG system using Gemini and Qdrant.",
            version="1.0.0",
        )

    def _add_middlewares(self):
        """Adds necessary middlewares."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self):
        """Registers the main API router."""
        self.app.include_router(api_router)

    def _attach_exception_handlers(self):
        """Attaches global exception handlers."""
        self.app.add_exception_handler(Exception, global_exception_handler)
        self.app.add_exception_handler(HTTPException, http_exception_handler)

    def _run_startup_logic(self):
        """Runs startup validation and initialization."""
        @self.app.on_event("startup")
        async def startup_event():
            try:
                StartupService.run_all_checks()
                logger.info("Application startup sequence completed.")
            except Exception as e:
                logger.critical(f"Application failed to start: {str(e)}")
                # In production, you might want to exit here
                # import sys; sys.exit(1)

    def create_app(self) -> FastAPI:
        """
        Complete app initialization sequence.
        """
        self._add_middlewares()
        self._attach_exception_handlers()
        self._register_routes()
        self._run_startup_logic()
        return self.app
