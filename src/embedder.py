"""
DocuMind - Text Embedding Module
Generates dense numerical vector representations for text chunks using
the pretrained SentenceTransformer model 'sentence-transformers/all-MiniLM-L6-v2'.

Why are embeddings vectors?
A text embedding maps human language into a continuous multi-dimensional geometric
space (for all-MiniLM-L6-v2, a 384-dimensional vector). Semantically similar sentences
(e.g., "The weather is lovely" and "It is a sunny day") are mapped to nearby coordinates
in this vector space, allowing mathematical comparisons (such as cosine similarity)
rather than relying on brittle exact keyword matching.
"""

from typing import Any
from sentence_transformers import SentenceTransformer

# Default pretrained model for DocuMind
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Module-level cache to avoid reloading neural network weights on every call
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Retrieves a cached SentenceTransformer model instance or loads and caches it.

    Reusing the model avoids re-reading weights from disk and re-initializing
    PyTorch layers on repeated embedding requests.

    Args:
        model_name: Name or path of the pretrained Sentence Transformer model.

    Returns:
        The loaded SentenceTransformer instance.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """
    Converts a list of text chunk dictionaries into dense vector embeddings using batch encoding.

    Args:
        chunks: List of chunk dictionaries. Each dictionary must contain:
            - 'chunk_id' (int): Unique sequential identifier.
            - 'page_number' (int): Original PDF page number.
            - 'text' (str): Text content to embed.
        model: Optional pre-loaded SentenceTransformer instance (or mock) for dependency injection.
        model_name: Pretrained model name to load if model is not provided.
                    Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        A list of new chunk dictionaries preserving all original metadata:
            - 'chunk_id' (int)
            - 'page_number' (int)
            - 'text' (str)
            - 'embedding' (list[float]): 384-dimensional dense vector of Python floats.

    Raises:
        TypeError: If chunks is not a list, if any chunk is not a dictionary,
                   or if required fields have incorrect data types.
        ValueError: If any chunk dictionary is missing required keys,
                    if chunk_id <= 0 or page_number <= 0,
                    or if the model output count/dimension is malformed.
    """
    # 1. Validate top-level input type
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list of dictionaries, got {type(chunks).__name__}")

    # 2. Early return for empty input: Avoid loading PyTorch model if there is nothing to embed
    if not chunks:
        return []

    # 3. Validate structure and types of each chunk dictionary
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk at index {index} must be a dictionary, got {type(chunk).__name__}")

        if "chunk_id" not in chunk or "page_number" not in chunk or "text" not in chunk:
            raise ValueError(
                f"Chunk at index {index} missing required keys ('chunk_id', 'page_number', 'text'). "
                f"Found keys: {list(chunk.keys())}"
            )

        if not isinstance(chunk["chunk_id"], int) or isinstance(chunk["chunk_id"], bool):
            raise TypeError(
                f"'chunk_id' at index {index} must be an integer, got {type(chunk['chunk_id']).__name__}"
            )
        if chunk["chunk_id"] <= 0:
            raise ValueError(
                f"'chunk_id' at index {index} must be a positive integer greater than 0, got {chunk['chunk_id']}"
            )

        if not isinstance(chunk["page_number"], int) or isinstance(chunk["page_number"], bool):
            raise TypeError(
                f"'page_number' at index {index} must be an integer, got {type(chunk['page_number']).__name__}"
            )
        if chunk["page_number"] <= 0:
            raise ValueError(
                f"'page_number' at index {index} must be a positive integer greater than 0, got {chunk['page_number']}"
            )

        if not isinstance(chunk["text"], str):
            raise TypeError(
                f"'text' at index {index} must be a string, got {type(chunk['text']).__name__}"
            )

    # 4. Resolve embedding model (use provided model or retrieve from cache)
    if model is None:
        model = get_embedding_model(model_name)

    # 5. Extract texts for batch encoding (encoding all texts together is significantly faster
    #    than calling the neural network in a loop for each individual chunk)
    texts = [chunk["text"] for chunk in chunks]

    # model.encode returns a NumPy ndarray of shape (len(texts), 384)
    raw_embeddings = model.encode(texts, show_progress_bar=False)

    # Validate that model returned exactly one embedding per input chunk
    if len(raw_embeddings) != len(chunks):
        raise ValueError(
            f"Model output count mismatch: expected {len(chunks)} embeddings for "
            f"{len(chunks)} input chunks, but model returned {len(raw_embeddings)}"
        )

    # 6. Assemble output dictionaries with native Python float lists
    embedded_chunks: list[dict[str, Any]] = []

    for chunk, raw_emb in zip(chunks, raw_embeddings):
        # Convert NumPy array or PyTorch tensor to standard Python list of floats
        if hasattr(raw_emb, "tolist"):
            emb_list = raw_emb.tolist()
        else:
            emb_list = list(raw_emb)

        # Validate vector dimension before constructing output
        if len(emb_list) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch for chunk_id {chunk['chunk_id']}: "
                f"expected {EMBEDDING_DIMENSION} dimensions, but got {len(emb_list)}"
            )

        # Ensure all elements are standard Python floats
        float_embedding = [float(val) for val in emb_list]

        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "embedding": float_embedding,
        })

    return embedded_chunks
