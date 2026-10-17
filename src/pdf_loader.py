import os
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFLoaderError(Exception):
    """Base exception for PDF loading and parsing errors."""
    pass


class NoTextExtractedError(PDFLoaderError):
    """Raised when a PDF contains no extractable text (e.g. scanned or image-only PDF)."""
    pass


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF file page-by-page, preserving page numbers for citations.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        A list of dictionaries, where each dictionary contains:
            - 'page_number' (int): 1-indexed page number.
            - 'text' (str): The extracted text of that page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a regular file.
        PdfReadError: If the file is corrupted, invalid, or password-protected.
        NoTextExtractedError: If the PDF contains no extractable text.
    """
    # 1. Validate file existence
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: '{pdf_path}'")

    if not os.path.isfile(pdf_path):
        raise ValueError(f"Path is not a regular file: '{pdf_path}'")

    # 2. Parse the PDF document
    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as e:
        raise PdfReadError(f"Invalid or corrupted PDF file '{pdf_path}': {e}") from e

    # 3. Check for encryption / password protection
    if reader.is_encrypted:
        try:
            # Attempt to decrypt with empty password (common for read-only protected PDFs)
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise PdfReadError(f"The PDF file '{pdf_path}' is password-protected and cannot be read without a password.")
        except PdfReadError:
            raise
        except Exception as e:
            raise PdfReadError(f"Failed to decrypt password-protected PDF '{pdf_path}': {e}") from e

    # 4. Extract text page-by-page
    pages = []
    has_text = False

    try:
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                has_text = True
            pages.append({
                "page_number": page_number,
                "text": page_text,
            })
    except PdfReadError as e:
        raise PdfReadError(f"Error reading pages from '{pdf_path}': {e}") from e

    # 5. Check if document has no pages or no extractable text
    if not reader.pages or not has_text:
        raise NoTextExtractedError(
            f"No extractable text found in '{pdf_path}'. "
            "The document may be empty or contain only scanned images/photos (which require OCR)."
        )

    return pages
