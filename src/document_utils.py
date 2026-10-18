"""
DocuMind - Document Utilities Module (Milestone 8, Phase 2, Step 1)
Provides shared content-addressable document identification helpers using SHA-256
cryptographic hashing, ensuring consistency across CLI and Streamlit workflows.
"""

import hashlib
from pathlib import Path


def compute_document_id(pdf_bytes: bytes) -> str:
    """
    Computes a deterministic, content-addressable document ID using the full
    64-character lowercase SHA-256 cryptographic digest of the PDF binary bytes.

    Format: 'doc_<64-character-lowercase-hex>' (total length 68 characters).

    Args:
        pdf_bytes: Raw binary content of the PDF file (bytes or bytearray).

    Returns:
        Document ID string in the format 'doc_<64-hex-sha256>'.

    Raises:
        TypeError: If pdf_bytes is not an instance of bytes or bytearray.
        ValueError: If pdf_bytes is empty.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise TypeError(f"pdf_bytes must be bytes or bytearray, got {type(pdf_bytes).__name__}")

    if not pdf_bytes:
        raise ValueError("Cannot compute document ID for empty byte payload.")

    digest = hashlib.sha256(pdf_bytes).hexdigest().lower()
    return f"doc_{digest}"


def compute_document_id_from_file(file_path: str | Path) -> str:
    """
    Reads a file from disk in binary mode and computes its deterministic document ID.

    Args:
        file_path: Path to the target file.

    Returns:
        Document ID string in the format 'doc_<64-hex-sha256>'.

    Raises:
        TypeError: If file_path is not a string or Path.
        FileNotFoundError: If the file does not exist.
        ValueError: If file_path is not a regular file.
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be a string or Path, got {type(file_path).__name__}")

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    if not path_obj.is_file():
        raise ValueError(f"Path is not a regular file: '{file_path}'")

    with open(path_obj, "rb") as f:
        content = f.read()

    return compute_document_id(content)
