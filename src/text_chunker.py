"""
DocuMind - Text Chunking Module
Provides chunking functionality for extracted PDF text, preserving page-level
metadata and assigning unique sequential chunk IDs.
"""


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Splits page-level text into fixed-size character chunks with overlap.

    Args:
        pages: List of dictionaries returned by extract_text_from_pdf.
               Each dictionary must contain 'page_number' (int) and 'text' (str).
        chunk_size: Maximum number of characters in each chunk. Must be > 0.
                    Default is 500.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
                       Must be >= 0 and strictly less than chunk_size.
                       Default is 100.

    Returns:
        A list of chunk dictionaries, each containing:
            - 'chunk_id' (int): 1-indexed sequential identifier for the chunk.
            - 'page_number' (int): The original PDF page number the chunk came from.
            - 'text' (str): The chunk's text content.

    Raises:
        TypeError: If input parameters or page entries have invalid types.
        ValueError: If chunk_size <= 0, chunk_overlap < 0,
                    chunk_overlap >= chunk_size, page_number <= 0,
                    or page dictionaries lack required keys.
    """
    # 1. Validate chunk_size and chunk_overlap types and values
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer greater than 0, got {chunk_size}")

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer (>= 0), got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    # 2. Validate pages structure
    if not isinstance(pages, list):
        raise TypeError(f"pages must be a list of dictionaries, got {type(pages).__name__}")

    chunks: list[dict] = []
    chunk_id = 1
    step = chunk_size - chunk_overlap

    # 3. Process each page independently to preserve page boundaries
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise TypeError(f"Page at index {index} must be a dictionary, got {type(page).__name__}")

        if "page_number" not in page or "text" not in page:
            raise ValueError(
                f"Page at index {index} missing required keys ('page_number', 'text'). "
                f"Found keys: {list(page.keys())}"
            )

        if not isinstance(page["page_number"], int) or isinstance(page["page_number"], bool):
            raise TypeError(
                f"'page_number' at index {index} must be an integer, got {type(page['page_number']).__name__}"
            )

        if page["page_number"] <= 0:
            raise ValueError(
                f"'page_number' at index {index} must be a positive integer greater than 0, got {page['page_number']}"
            )

        if not isinstance(page["text"], str):
            raise TypeError(
                f"'text' at index {index} must be a string, got {type(page['text']).__name__}"
            )

        raw_text = page["text"]

        # Skip empty or whitespace-only pages
        if not raw_text.strip():
            continue

        # Chunk the non-empty text preserving original text content
        start = 0
        text_length = len(raw_text)

        while start < text_length:
            chunk_text = raw_text[start : start + chunk_size]
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "text": chunk_text,
            })
            chunk_id += 1

            # If current chunk already reaches or covers the end of text, no more new text remains
            if start + chunk_size >= text_length:
                break

            start += step

    return chunks
