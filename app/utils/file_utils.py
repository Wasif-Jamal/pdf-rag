import os
import re

class FileUtils:
    """
    Utility class for file-related operations.
    """
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitizes a filename by removing special characters.
        """
        # Remove anything that isn't alphanumeric, a dot, or a dash
        return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)

    @staticmethod
    def ensure_dir(path: str) -> None:
        """
        Ensures a directory exists.
        """
        os.makedirs(path, exist_ok=True)
