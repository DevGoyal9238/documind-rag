"""
DocuMind - Semantic Search Module
Performs semantic similarity retrieval over embedded text chunks using
cosine similarity and the pretrained all-MiniLM-L6-v2 embedding model.
"""

import math
from typing import Any
import numpy as np

from src.embedder import (
    get_embedding_model,
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
)


def compute_cosine_similarity(
    vec_a: list[float] | tuple[float, ...] | np.ndarray,
    vec_b: list[float] | tuple[float, ...] | np.ndarray,
) -> float:
    """
    Computes the cosine similarity between two 1D numerical vectors of dimension 384.

    Formula:
        cosine_similarity(a, b) = (a . b) / (||a||_2 * ||b||_2)

    Cosine similarity measures the angular alignment between two vectors in multi-dimensional
    space, producing a score between -1.0 (exact opposite direction) and +1.0 (identical direction).
    Unlike Euclidean distance, it is invariant to vector magnitude, making it ideal for text
    representations where directional orientation captures semantic topic.

    Args:
        vec_a: First vector (list, tuple, or 1D NumPy array with exactly 384 numeric values).
        vec_b: Second vector (list, tuple, or 1D NumPy array with exactly 384 numeric values).

    Returns:
        Cosine similarity score as a float in the range [-1.0, 1.0].

    Raises:
        TypeError: If either vector is not a list, tuple, or ndarray, or contains non-numeric
                   types or booleans.
        ValueError: If vector dimensions differ, if vectors are not 1-dimensional,
                    if vector length is not 384, if vectors contain non-finite numbers (NaN/Inf),
                    or if either vector has zero magnitude (zero norm).
    """
    if not isinstance(vec_a, (list, tuple, np.ndarray)) or not isinstance(vec_b, (list, tuple, np.ndarray)):
        raise TypeError(
            f"Vectors must be a list, tuple, or np.ndarray, got {type(vec_a).__name__} and {type(vec_b).__name__}"
        )

    # Validate elements for vec_a
    if isinstance(vec_a, np.ndarray):
        if vec_a.dtype == bool or not np.issubdtype(vec_a.dtype, np.number):
            raise TypeError("vec_a must contain numeric values and not booleans or strings")
    else:
        for idx, val in enumerate(vec_a):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"vec_a element at index {idx} must be a number and not boolean, got {type(val).__name__}")

    # Validate elements for vec_b
    if isinstance(vec_b, np.ndarray):
        if vec_b.dtype == bool or not np.issubdtype(vec_b.dtype, np.number):
            raise TypeError("vec_b must contain numeric values and not booleans or strings")
    else:
        for idx, val in enumerate(vec_b):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"vec_b element at index {idx} must be a number and not boolean, got {type(val).__name__}")

    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Vectors must be 1-dimensional, got ndim={a.ndim} and {b.ndim}")

    if a.shape != (EMBEDDING_DIMENSION,) or b.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"Both vectors must have exactly {EMBEDDING_DIMENSION} dimensions, "
            f"got {a.shape} and {b.shape}"
        )

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Vectors must contain only finite numbers (no NaN, Inf, or -Inf)")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot calculate cosine similarity for a zero-magnitude vector")

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp float precision within [-1.0, 1.0] to guard against IEEE 754 rounding errors
    return max(-1.0, min(1.0, similarity))


def search_chunks(
    query: str,
    embedded_chunks: list[dict[str, Any]],
    top_k: int = 3,
    model: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """
    Searches a collection of embedded chunks for the most semantically relevant matches to a query.

    Args:
        query: User search query string. Must be a non-empty string after stripping whitespace.
        embedded_chunks: List of chunk dictionaries containing 'chunk_id', 'page_number',
                         'text', and 'embedding'.
        top_k: Maximum number of ranked results to return. Must be a positive integer (> 0).
               Default is 3.
        model: Optional SentenceTransformer model instance (or mock) for dependency injection.
        model_name: Pretrained model identifier to load if model is not provided.
                    Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        A new list of top_k chunk dictionaries sorted by similarity_score in descending order:
        Each dictionary preserves all existing chunk metadata fields, copies the embedding list,
        and adds 'similarity_score'.

    Raises:
        TypeError: If query is not a string, if top_k is not an integer or is a boolean,
                   if embedded_chunks is not a list, or if any field has an invalid type.
        ValueError: If query is empty/whitespace, if top_k <= 0, if chunks are malformed,
                    or if embeddings have invalid dimensions, non-finite values, or zero magnitude.
    """
    # 1. Validate query
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query cannot be empty or whitespace-only")

    # 2. Validate top_k (reject booleans because bool is a subclass of int in Python)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")

    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    # 3. Validate embedded_chunks collection
    if not isinstance(embedded_chunks, list):
        raise TypeError(f"embedded_chunks must be a list of dictionaries, got {type(embedded_chunks).__name__}")

    # Early exit: return empty list immediately without loading the embedding model
    if not embedded_chunks:
        return []

    # 4. Validate structure, metadata, and embeddings of each chunk
    required_keys = {"chunk_id", "page_number", "text", "embedding"}

    for index, chunk in enumerate(embedded_chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk at index {index} must be a dictionary, got {type(chunk).__name__}")

        missing = required_keys - set(chunk.keys())
        if missing:
            raise ValueError(f"Chunk at index {index} missing required keys: {sorted(list(missing))}")

        if not isinstance(chunk["chunk_id"], int) or isinstance(chunk["chunk_id"], bool):
            raise TypeError(f"'chunk_id' at index {index} must be an integer, got {type(chunk['chunk_id']).__name__}")
        if chunk["chunk_id"] <= 0:
            raise ValueError(f"'chunk_id' at index {index} must be a positive integer (> 0), got {chunk['chunk_id']}")

        if not isinstance(chunk["page_number"], int) or isinstance(chunk["page_number"], bool):
            raise TypeError(f"'page_number' at index {index} must be an integer, got {type(chunk['page_number']).__name__}")
        if chunk["page_number"] <= 0:
            raise ValueError(f"'page_number' at index {index} must be a positive integer (> 0), got {chunk['page_number']}")

        if not isinstance(chunk["text"], str):
            raise TypeError(f"'text' at index {index} must be a string, got {type(chunk['text']).__name__}")

        raw_emb = chunk["embedding"]
        if not isinstance(raw_emb, list):
            raise TypeError(f"'embedding' at index {index} must be a list of floats, got {type(raw_emb).__name__}")

        if len(raw_emb) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Chunk at index {index} has invalid embedding dimension: "
                f"expected {EMBEDDING_DIMENSION}, got {len(raw_emb)}"
            )

        # Ensure all elements in chunk embedding are finite numbers and not booleans
        for elem_idx, val in enumerate(raw_emb):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"Chunk at index {index} embedding element at index {elem_idx} "
                    f"must be a numeric float, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise ValueError(
                    f"Chunk at index {index} embedding contains non-finite value at index {elem_idx}: {val}"
                )

    # 5. Resolve embedding model (reuses get_embedding_model cache from Milestone 4)
    if model is None:
        model = get_embedding_model(model_name)

    # 6. Encode the query into a dense vector (allow genuine model errors to propagate)
    raw_query_emb = model.encode([clean_query], show_progress_bar=False)

    # Require model output to have exactly the shape (1, 384)
    if hasattr(raw_query_emb, "shape"):
        query_shape = tuple(raw_query_emb.shape)
    elif isinstance(raw_query_emb, list):
        if len(raw_query_emb) > 0 and isinstance(raw_query_emb[0], list):
            query_shape = (len(raw_query_emb), len(raw_query_emb[0]))
        else:
            query_shape = (len(raw_query_emb),)
    else:
        query_shape = np.asarray(raw_query_emb).shape

    expected_shape = (1, EMBEDDING_DIMENSION)
    if query_shape != expected_shape:
        raise ValueError(
            f"Query embedding output shape mismatch: expected {expected_shape}, got {query_shape}"
        )

    # Extract 1D query vector and validate
    if isinstance(raw_query_emb, np.ndarray):
        if raw_query_emb.dtype == bool or not np.issubdtype(raw_query_emb.dtype, np.number):
            raise TypeError("Query embedding must contain numeric values and not booleans or strings")
        query_vec_1d = raw_query_emb[0]
    elif isinstance(raw_query_emb, list):
        row = raw_query_emb[0]
        for elem in row:
            if isinstance(elem, bool) or not isinstance(elem, (int, float)):
                raise TypeError("Query embedding elements must be numeric, not booleans or strings")
        query_vec_1d = np.asarray(row, dtype=np.float64)
    else:
        query_vec_1d = np.asarray(raw_query_emb[0], dtype=np.float64)

    if not np.all(np.isfinite(query_vec_1d)):
        raise ValueError("Query embedding contains non-finite numbers (NaN, Inf, or -Inf)")

    query_norm = float(np.linalg.norm(query_vec_1d))
    if query_norm == 0.0:
        raise ValueError("Cannot calculate semantic search for a zero-magnitude query embedding")

    query_embedding = query_vec_1d.tolist() if hasattr(query_vec_1d, "tolist") else list(query_vec_1d)

    # 7. Compute similarity for each chunk and rank
    scored_chunks: list[dict[str, Any]] = []

    for chunk in embedded_chunks:
        similarity = compute_cosine_similarity(query_embedding, chunk["embedding"])

        # Preserve every original metadata field using {**chunk, ...}, copy embedding, and add similarity_score
        scored_chunk = {
            **chunk,
            "embedding": list(chunk["embedding"]),
            "similarity_score": similarity,
        }
        scored_chunks.append(scored_chunk)

    # 8. Sort by similarity score descending (highest semantic relevance first)
    scored_chunks.sort(key=lambda item: item["similarity_score"], reverse=True)

    # 9. Return at most top_k results
    return scored_chunks[:top_k]
