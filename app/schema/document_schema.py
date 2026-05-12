from pydantic import BaseModel

class UploadResponse(BaseModel):
    """
    Schema for the response of a successful PDF upload and processing.
    """
    filename: str
    saved_path: str
    status: str
    total_pages: int
    total_chunks: int
    vectorstore_status: str
