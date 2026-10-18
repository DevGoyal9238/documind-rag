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
        TypeError: If pdf_path is not a string.
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a regular file.
        PdfReadError: If the file is corrupted, invalid, or password-protected.
        PDFLoaderError: If an unexpected error occurs while reading or decoding the PDF.
        NoTextExtractedError: If the PDF contains no extractable text.
    """
    # 0. Validate argument type
    if not isinstance(pdf_path, str):
        raise TypeError(f"pdf_path must be a string, got {type(pdf_path).__name__}")

    # 1. Validate file existence
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: '{pdf_path}'")

    if not os.path.isfile(pdf_path):
        raise ValueError(f"Path is not a regular file: '{pdf_path}'")

    reader: PdfReader | None = None
    try:
        # 2. Parse the PDF document
        try:
            reader = PdfReader(pdf_path)
        except PdfReadError as e:
            raise PdfReadError(f"Invalid or corrupted PDF file '{pdf_path}': {e}") from e
        except Exception as e:
            raise PDFLoaderError(f"Failed to load or parse PDF file '{pdf_path}': {e}") from e

        # 3. Check for encryption / password protection
        if reader.is_encrypted:
            try:
                # Attempt to decrypt with empty password (common for read-only protected PDFs)
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise PdfReadError(
                        f"The PDF file '{pdf_path}' is password-protected and cannot be read without a password."
                    )
            except PdfReadError:
                raise
            except Exception as e:
                raise PDFLoaderError(f"Failed to decrypt password-protected PDF '{pdf_path}': {e}") from e

        # 4. Extract text page-by-page
        pages: list[dict] = []
        has_text = False

        try:
            raw_pages = list(reader.pages)
        except PdfReadError as e:
            raise PdfReadError(f"Error reading pages from '{pdf_path}': {e}") from e
        except Exception as e:
            raise PDFLoaderError(f"Error reading pages from '{pdf_path}': {e}") from e

        if not raw_pages:
            raise NoTextExtractedError(
                f"No extractable text found in '{pdf_path}'. "
                "The document may be empty or contain only scanned images/photos (which require OCR)."
            )

        for page_number, page in enumerate(raw_pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except PdfReadError as e:
                raise PdfReadError(f"Error reading page {page_number} from '{pdf_path}': {e}") from e
            except Exception as e:
                raise PDFLoaderError(
                    f"Unexpected decoder failure on page {page_number} of '{pdf_path}': {e}"
                ) from e

            if page_text.strip():
                has_text = True
            pages.append({
                "page_number": page_number,
                "text": page_text,
            })

        # 5. Check if document has no extractable text
        if not has_text:
            raise NoTextExtractedError(
                f"No extractable text found in '{pdf_path}'. "
                "The document may be empty or contain only scanned images/photos (which require OCR)."
            )

        return pages

    finally:
        # Guarantee reader stream handle cleanup (essential on Windows to avoid file locks)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
