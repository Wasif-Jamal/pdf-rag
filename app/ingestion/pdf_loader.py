from pypdf import PdfReader

def load_pdf_text(file_path: str) -> str:
    """
    Extracts text from all pages of a PDF file using pypdf.
    Handles empty pages safely by only appending non-empty text.
    """
    try:
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text_parts.append(extracted_text)
        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {str(e)}")
