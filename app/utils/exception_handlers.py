from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from app.config.log_config import LogConfig

logger = LogConfig.get_logger(__name__)

async def global_exception_handler(request: Request, exc: Exception):
    """
    Handles all unhandled exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles FastAPI HTTPExceptions.
    """
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
