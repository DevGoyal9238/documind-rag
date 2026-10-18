"""
DocuMind - Vector Store Module (ChromaDB Integration)
Provides modular, testable functions for persistent vector storage and retrieval
using ChromaDB with cosine distance and precomputed 384-dimensional embeddings.
"""

import math
from typing import Any
import numpy as np
import chromadb

from src.embedder import EMBEDDING_DIMENSION

DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "documind_chunks"


def cosine_distance_to_similarity(
    distance: float | int,
    round_digits: int | None = 4,
) -> float:
    """
    Converts ChromaDB cosine distance to cosine similarity.

    In ChromaDB, when configured with `metadata={"hnsw:space": "cosine"}`,
    the returned distance is defined as:
        cosine_distance = 1.0 - cosine_similarity

    Therefore:
        cosine_similarity = 1.0 - cosine_distance

    Because of floating-point inaccuracies, identical vectors might produce
    slight negative distances (e.g., -1.5e-6), yielding similarities slightly
    above 1.0. This function clamps the result to the valid range [-1.0, 1.0].

    Args:
        distance: Numeric distance value returned by ChromaDB.
        round_digits: Optional number of decimal places to round to (default 4).

    Returns:
        Cosine similarity score clamped to [-1.0, 1.0].

    Raises:
        TypeError: If distance is not a float or int, or is a boolean.
        ValueError: If distance is non-finite (NaN, Inf, -Inf).
    """
    if isinstance(distance, (bool, np.bool_)) or not isinstance(distance, (int, float, np.number)):
        raise TypeError(f"distance must be a numeric float or int, got {type(distance).__name__}")

    if not math.isfinite(distance):
        raise ValueError(f"distance must be a finite number, got {distance}")

    similarity = 1.0 - float(distance)
    # Clamp float precision within [-1.0, 1.0] to guard against floating-point rounding
    clamped = max(-1.0, min(1.0, similarity))

    if round_digits is not None:
        if isinstance(round_digits, bool) or not isinstance(round_digits, int):
            raise TypeError(f"round_digits must be an integer or None, got {type(round_digits).__name__}")
        if round_digits < 0:
            raise ValueError(f"round_digits must be non-negative, got {round_digits}")
        return round(clamped, round_digits)

    return clamped


def get_chroma_client(persist_directory: str = DEFAULT_CHROMA_DIR) -> Any:
    """
    Creates or connects to a persistent ChromaDB client storing data on disk.

    Args:
        persist_directory: Directory path where ChromaDB stores its SQLite and vector index files.

    Returns:
        A chromadb.PersistentClient instance.

    Raises:
        TypeError: If persist_directory is not a string.
        ValueError: If persist_directory is empty or whitespace-only.
    """
    if not isinstance(persist_directory, str):
        raise TypeError(f"persist_directory must be a string, got {type(persist_directory).__name__}")

    clean_dir = persist_directory.strip()
    if not clean_dir:
        raise ValueError("persist_directory cannot be empty or whitespace-only")

    return chromadb.PersistentClient(path=clean_dir)


def get_or_create_collection(
    client: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """
    Retrieves an existing ChromaDB collection or creates a new one configured with
    the cosine distance metric (hnsw:space = cosine).

    Note: We explicitly do NOT pass an embedding_function to ChromaDB because we generate
    and supply our own 384-dimensional dense vectors from Sentence Transformers.

    Args:
        client: ChromaDB client instance (e.g., PersistentClient).
        collection_name: Name of the collection to get or create.

    Returns:
        ChromaDB Collection instance.

    Raises:
        TypeError: If client is None or collection_name is not a string.
        ValueError: If collection_name is empty or whitespace-only.
    """
    if client is None:
        raise TypeError("client cannot be None")

    if not isinstance(collection_name, str):
        raise TypeError(f"collection_name must be a string, got {type(collection_name).__name__}")

    clean_name = collection_name.strip()
    if not clean_name:
        raise ValueError("collection_name cannot be empty or whitespace-only")

    return client.get_or_create_collection(
        name=clean_name,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    embedded_chunks: list[dict[str, Any]],
    collection: Any,
    document_id: str = "doc",
) -> int:
    """
    Validates and upserts precomputed chunk embeddings, text, and metadata into ChromaDB.

    To prevent chunk ID collisions across multiple documents in a shared collection,
    each chunk is assigned a stable, document-scoped ID:
        f"{document_id}_chunk_{chunk['chunk_id']}"

    Document identity is also recorded in the chunk's metadata alongside 'chunk_id'
    and 'page_number'.

    Limitation note:
        Calling upsert prevents duplicate chunk records at the database level (idempotent write),
        but our pipeline still extracts and re-embeds the PDF chunks before calling store_chunks.
        Upsert does not eliminate embedding computation time.

    Args:
        embedded_chunks: List of chunk dictionaries containing 'chunk_id', 'page_number',
                         'text', and 'embedding'.
        collection: Target ChromaDB collection.
        document_id: Identifier for the source document (e.g. PDF filename or stem).

    Returns:
        The number of chunks upserted into the collection.

    Raises:
        TypeError: If arguments are of invalid types or chunk fields contain invalid types.
        ValueError: If collection is None, document_id is empty, or chunk fields are malformed.
    """
    if collection is None:
        raise TypeError("collection cannot be None")

    if not isinstance(document_id, str):
        raise TypeError(f"document_id must be a string, got {type(document_id).__name__}")

    clean_doc_id = document_id.strip()
    if not clean_doc_id:
        raise ValueError("document_id cannot be empty or whitespace-only")

    if not isinstance(embedded_chunks, list):
        raise TypeError(f"embedded_chunks must be a list, got {type(embedded_chunks).__name__}")

    if not embedded_chunks:
        return 0

    required_keys = {"chunk_id", "page_number", "text", "embedding"}

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, chunk in enumerate(embedded_chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk at index {idx} must be a dictionary, got {type(chunk).__name__}")

        missing = required_keys - set(chunk.keys())
        if missing:
            raise ValueError(f"Chunk at index {idx} missing required keys: {sorted(list(missing))}")

        chunk_id = chunk["chunk_id"]
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise TypeError(f"'chunk_id' at index {idx} must be an integer, got {type(chunk_id).__name__}")
        if chunk_id <= 0:
            raise ValueError(f"'chunk_id' at index {idx} must be a positive integer (> 0), got {chunk_id}")

        page_number = chunk["page_number"]
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise TypeError(f"'page_number' at index {idx} must be an integer, got {type(page_number).__name__}")
        if page_number <= 0:
            raise ValueError(f"'page_number' at index {idx} must be a positive integer (> 0), got {page_number}")

        text = chunk["text"]
        if not isinstance(text, str):
            raise TypeError(f"'text' at index {idx} must be a string, got {type(text).__name__}")

        raw_emb = chunk["embedding"]
        if not isinstance(raw_emb, (list, tuple, np.ndarray)):
            raise TypeError(f"'embedding' at index {idx} must be a list, tuple, or ndarray, got {type(raw_emb).__name__}")

        if len(raw_emb) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Chunk at index {idx} embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, got {len(raw_emb)}"
            )

        clean_emb: list[float] = []
        for elem_idx, val in enumerate(raw_emb):
            if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.number)):
                raise TypeError(
                    f"Chunk at index {idx} embedding element at index {elem_idx} "
                    f"must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise ValueError(
                    f"Chunk at index {idx} embedding contains non-finite value at index {elem_idx}: {val}"
                )
            clean_emb.append(float(val))

        scoped_id = f"{clean_doc_id}_chunk_{chunk_id}"
        if scoped_id in seen_ids:
            raise ValueError(
                f"Duplicate document-scoped chunk ID detected in input batch: '{scoped_id}'"
            )
        seen_ids.add(scoped_id)

        ids.append(scoped_id)
        documents.append(text)
        embeddings.append(clean_emb)
        metadatas.append({
            "chunk_id": chunk_id,
            "page_number": page_number,
            "document_id": clean_doc_id,
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


def query_chunks(
    query_embedding: list[float] | tuple[float, ...] | np.ndarray,
    collection: Any,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Queries ChromaDB for the most relevant chunks using a precomputed 384-dimensional query embedding.

    Converts returned cosine distances into cosine similarity scores and formats results
    consistently with the rest of the DocuMind pipeline.

    Args:
        query_embedding: Precomputed 384-dimensional dense vector for the query.
        collection: ChromaDB collection to search against.
        top_k: Maximum number of ranked results to return (positive integer > 0). Default is 3.
        document_id: Optional document identifier to filter results to a specific document.

    Returns:
        List of result dictionaries sorted by similarity_score in descending order:
        Each dictionary contains:
            - 'id': Document-scoped chunk ID (e.g. 'sample_chunk_1')
            - 'chunk_id': Original 1-indexed chunk number within the document
            - 'page_number': Source PDF page number
            - 'document_id': Identifier of the source document
            - 'text': Chunk text content
            - 'similarity_score': Cosine similarity score in [-1.0, 1.0]
            - 'cosine_distance': Raw distance returned by ChromaDB

    Raises:
        TypeError: If query_embedding, collection, top_k, or document_id have invalid types.
        ValueError: If query_embedding has wrong dimensions, non-finite values, zero magnitude,
                    or if top_k <= 0.
    """
    if collection is None:
        raise TypeError("collection cannot be None")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    if not isinstance(query_embedding, (list, tuple, np.ndarray)):
        raise TypeError(
            f"query_embedding must be a list, tuple, or ndarray, got {type(query_embedding).__name__}"
        )

    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"query_embedding must have exactly {EMBEDDING_DIMENSION} dimensions, "
            f"got {len(query_embedding)}"
        )

    clean_emb: list[float] = []
    for elem_idx, val in enumerate(query_embedding):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.number)):
            raise TypeError(
                f"query_embedding element at index {elem_idx} must be numeric, got {type(val).__name__}"
            )
        if not math.isfinite(val):
            raise ValueError(
                f"query_embedding contains non-finite value at index {elem_idx}: {val}"
            )
        clean_emb.append(float(val))

    # Validate non-zero norm
    emb_array = np.asarray(clean_emb, dtype=np.float64)
    norm = float(np.linalg.norm(emb_array))
    if norm == 0.0:
        raise ValueError("Cannot query with a zero-magnitude query embedding")

    clean_doc_id: str | None = None
    if document_id is not None:
        if not isinstance(document_id, str):
            raise TypeError(f"document_id must be a string, got {type(document_id).__name__}")
        clean_doc_id = document_id.strip()
        if not clean_doc_id:
            raise ValueError("document_id cannot be empty or whitespace-only")

    # Early exit if collection is empty
    if collection.count() == 0:
        return []

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [clean_emb],
        "n_results": top_k,
    }

    if clean_doc_id is not None:
        query_kwargs["where"] = {"document_id": clean_doc_id}

    raw_results = collection.query(**query_kwargs)

    if not isinstance(raw_results, dict):
        return []

    raw_ids = raw_results.get("ids")
    if not raw_ids or not isinstance(raw_ids, (list, tuple)) or not raw_ids[0]:
        return []

    raw_ids_sub = raw_ids[0]
    if not isinstance(raw_ids_sub, (list, tuple)):
        return []

    # Filter out None or empty string IDs while preserving valid indices
    valid_indices: list[int] = [
        idx for idx, item in enumerate(raw_ids_sub)
        if item is not None and str(item).strip()
    ]
    if not valid_indices:
        return []

    # Safely extract documents sub-array
    raw_documents = raw_results.get("documents")
    doc_list: list[Any] = []
    if raw_documents and isinstance(raw_documents, (list, tuple)) and raw_documents[0] is not None:
        if isinstance(raw_documents[0], (list, tuple)):
            doc_list = list(raw_documents[0])

    # Safely extract metadatas sub-array
    raw_metadatas = raw_results.get("metadatas")
    meta_list: list[Any] = []
    if raw_metadatas and isinstance(raw_metadatas, (list, tuple)) and raw_metadatas[0] is not None:
        if isinstance(raw_metadatas[0], (list, tuple)):
            meta_list = list(raw_metadatas[0])

    # Safely extract distances sub-array
    raw_distances = raw_results.get("distances")
    dist_list: list[Any] = []
    if raw_distances and isinstance(raw_distances, (list, tuple)) and raw_distances[0] is not None:
        if isinstance(raw_distances[0], (list, tuple)):
            dist_list = list(raw_distances[0])

    formatted_results: list[dict[str, Any]] = []
    for idx in valid_indices:
        # 1. Inspect text content: skip if missing, None, empty, or whitespace-only
        if idx >= len(doc_list):
            continue
        raw_doc = doc_list[idx]
        if raw_doc is None:
            continue
        text_str = str(raw_doc)
        if not text_str.strip():
            continue

        chunk_id_str = str(raw_ids_sub[idx])

        # 2. Distance value with safe fallback (default 1.0 -> similarity 0.0)
        dist = 1.0
        if idx < len(dist_list):
            val = dist_list[idx]
            if val is not None and not isinstance(val, (bool, np.bool_)):
                if isinstance(val, (int, float, np.number)) and math.isfinite(val):
                    dist = float(val)

        sim = cosine_distance_to_similarity(dist, round_digits=4)

        # 3. Metadata dictionary with safe fallback (preserve record even if metadata is missing)
        meta: dict[str, Any] = {}
        if idx < len(meta_list):
            m = meta_list[idx]
            if isinstance(m, dict):
                meta = m

        formatted_results.append({
            "id": chunk_id_str,
            "chunk_id": meta.get("chunk_id"),
            "page_number": meta.get("page_number"),
            "document_id": meta.get("document_id"),
            "text": text_str,
            "similarity_score": sim,
            "cosine_distance": dist,
        })

    # Sort descending by similarity score
    formatted_results.sort(key=lambda item: item["similarity_score"], reverse=True)
    return formatted_results[:top_k]


def verify_existing_index(
    collection: Any,
    document_id: str,
    expected_chunk_count: int,
) -> bool:
    """
    Verifies whether ChromaDB already holds a complete, contiguous, and uncorrupted
    index for the given document ID.

    Verification Invariants:
        1. Querying ChromaDB by document_id returns stored records.
        2. Total stored chunks matches expected_chunk_count exactly.
        3. Stored chunk IDs strictly match the expected set {f"{document_id}_chunk_{i}"} for i in 1..N.
        4. Chunk IDs in metadata form an unbroken sequential series from 1 to expected_chunk_count.
        5. Stored document texts are non-empty and non-whitespace strings.
        6. All records belong strictly to the requested document ID.

    Limitation note:
        The ChromaDB chunk metadata schema records document_id, chunk_id, and page_number,
        but does not record chunking configuration (chunk_size, chunk_overlap) or embedding
        model name. Verification establishes structural and count completeness for the given
        chunk count, but assumes consistent project-wide chunking parameters.

    Args:
        collection: Active ChromaDB Collection instance.
        document_id: Scoped document identifier (e.g. doc_<sha256>).
        expected_chunk_count: Expected number of chunks for this document (must be > 0).

    Returns:
        True if the stored index is 100% complete and verified; False if missing,
        incomplete, or structurally invalid.

    Raises:
        RuntimeError: If a database read failure occurs while querying ChromaDB. Callers
            must distinguish this infrastructure failure from a missing/invalid index and
            must not automatically trigger deletion or re-indexing during an outage.
    """
    if collection is None or not isinstance(document_id, str):
        return False

    clean_doc_id = document_id.strip()
    if not clean_doc_id:
        return False

    if isinstance(expected_chunk_count, bool) or not isinstance(expected_chunk_count, int) or expected_chunk_count <= 0:
        return False

    try:
        existing = collection.get(where={"document_id": clean_doc_id})
    except Exception as err:
        raise RuntimeError(
            f"Database read failure while verifying index for document '{clean_doc_id}': {err}"
        ) from err

    if not isinstance(existing, dict):
        return False

    stored_ids = existing.get("ids")
    if not isinstance(stored_ids, list) or len(stored_ids) != expected_chunk_count:
        return False

    metadatas = existing.get("metadatas")
    if not isinstance(metadatas, list) or len(metadatas) != expected_chunk_count:
        return False

    documents = existing.get("documents")
    if not isinstance(documents, list) or len(documents) != expected_chunk_count:
        return False

    expected_ids = {f"{clean_doc_id}_chunk_{i}" for i in range(1, expected_chunk_count + 1)}
    if set(stored_ids) != expected_ids:
        return False

    chunk_ids: list[int] = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            return False
        if meta.get("document_id") != clean_doc_id:
            return False
        cid = meta.get("chunk_id")
        if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
            return False
        page_num = meta.get("page_number")
        if page_num is not None:
            if not isinstance(page_num, int) or isinstance(page_num, bool) or page_num <= 0:
                return False
        chunk_ids.append(cid)

    if sorted(chunk_ids) != list(range(1, expected_chunk_count + 1)):
        return False

    for doc in documents:
        if not isinstance(doc, str) or not doc.strip():
            return False

    return True


def delete_document_chunks(
    collection: Any,
    document_id: str,
) -> None:
    """
    Deletes all vector store records belonging strictly to a specific document ID.

    Scoping deletion strictly by document_id prevents accidental data loss across
    other indexed documents in the shared collection.

    Args:
        collection: Active ChromaDB Collection instance.
        document_id: Scoped document identifier (e.g. doc_<sha256>).

    Raises:
        TypeError: If collection is None or document_id is not a string.
        ValueError: If document_id is empty or whitespace-only.
        RuntimeError: If ChromaDB fails to execute the deletion.
    """
    if collection is None:
        raise TypeError("collection cannot be None")

    if not isinstance(document_id, str):
        raise TypeError(f"document_id must be a string, got {type(document_id).__name__}")

    clean_doc_id = document_id.strip()
    if not clean_doc_id:
        raise ValueError("document_id cannot be empty or whitespace-only")

    try:
        collection.delete(where={"document_id": clean_doc_id})
    except Exception as err:
        raise RuntimeError(
            f"Failed to delete records for document '{clean_doc_id}': {err}"
        ) from err

