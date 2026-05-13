from fastapi import HTTPException, UploadFile

class UploadValidator:
    """
    Validator for file uploads.
    """
    @staticmethod
    def validate_pdf(file: UploadFile) -> None:
        """
        Validates that the uploaded file is a PDF.
        """
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Only PDF is allowed."
            )
        
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail="File extension must be .pdf"
            )
