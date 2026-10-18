"""
DocuMind - AI PDF Study Assistant (Streamlit Web Interface)
Milestone 7, Step 4: Scoped Question-Answering, Grounded Gemini Synthesis,
and Ground-Truth Source Citations in the Streamlit Web Interface.

Architecture & Execution Model:
    Streamlit re-executes the entire script on every user interaction (clicks, inputs, uploads).
    To maintain stability, state correctness, and performance:
    1. Heavy resources (SentenceTransformer model, ChromaDB persistent client) are cached
       using `@st.cache_resource` so weights and SQLite connections load only once.
    2. Uploaded PDFs are stored in memory (`st.file_uploader`), written to a secure temporary
       file for `pypdf` extraction, and reliably deleted in a `finally` block.
    3. Document IDs are derived from the full 64-character SHA-256 hash of the uploaded PDF bytes
       (`doc_{hash}`), guaranteeing cryptographic collision resistance.
    4. Two-Phase Ingestion Verification:
       - Extraction and chunking run first to establish the expected ground-truth chunk count.
       - ChromaDB is queried to verify whether a complete, sequential index already exists.
       - If verified complete: Reuses existing records instantly, skipping expensive embeddings.
       - If incomplete/corrupted: Safely deletes partial records for this document ID only
         and re-indexes all chunks from scratch.
    5. Document-Scoped Semantic Retrieval & Question-Answering:
       - User questions are encoded into 384-dimensional query embeddings.
       - Retrieval queries ChromaDB strictly filtered by the active document ID (`where={"document_id": doc_id}`).
         This mathematically isolates context and prevents cross-document information leakage.
       - Retrieved context chunks are passed to `generate_answer()`, utilizing grounded prompting,
         greedy argmax decoding ($T=0.0$), pre-LLM empty context guards, and strict abstention.
       - Answers and ground-truth retrieval citations (page numbers, chunk IDs, similarity scores)
         are preserved in `st.session_state` and rendered in the UI.
    6. State synchronization clears Q&A results whenever a file is replaced, removed, or newly processed,
       preventing stale cross-document contamination.
"""

import hashlib
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
import streamlit as st
from pypdf.errors import PdfReadError

from src.pdf_loader import extract_text_from_pdf, NoTextExtractedError, PDFLoaderError
from src.text_chunker import chunk_pages
from src.embedder import get_embedding_model, embed_chunks, EMBEDDING_DIMENSION
from src.vector_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    get_chroma_client,
    get_or_create_collection,
    store_chunks,
    query_chunks,
)
from src.qa_generator import generate_answer, DEFAULT_MODEL_NAME, get_gemini_client

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocuMind — AI PDF Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Pure Helper Functions (Modular & Testable)
# ---------------------------------------------------------
def compute_document_id(pdf_bytes: bytes) -> str:
    """
    Computes a deterministic, content-addressable document ID using the full
    64-character SHA-256 cryptographic digest of the PDF binary bytes.

    Args:
        pdf_bytes: Raw binary content of the PDF file.

    Returns:
        Document ID string in the format 'doc_<64-character-sha256>'.

    Raises:
        ValueError: If pdf_bytes is empty.
    """
    if not pdf_bytes:
        raise ValueError("Cannot compute document ID for empty byte payload.")
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    return f"doc_{digest}"


def sanitize_error_message(err: Exception) -> str:
    """
    Sanitizes raw exception strings to prevent accidental leakage of API keys,
    bearer tokens, or sensitive internal filesystem paths.
    """
    raw = str(err).strip()
    if not raw:
        return type(err).__name__

    # Redact standard Google AI Studio / Gemini API key patterns (AIzaSy...)
    sanitized = re.sub(r"AIza[0-9A-Za-z_-]{30,45}", "[REDACTED_KEY]", raw)
    # Redact query parameters or config keys (key=..., token=..., secret=...)
    sanitized = re.sub(r"(?i)(key|token|auth|secret|password)=[^&\s'\"`]+", r"\1=[REDACTED]", sanitized)
    # Redact Bearer tokens
    sanitized = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+", "Bearer [REDACTED]", sanitized)
    return sanitized.strip()


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
        3. Chunk IDs in metadata form a continuous sequence from 1 to expected_chunk_count.
        4. Stored document texts are non-empty.

    Args:
        collection: Active ChromaDB Collection instance.
        document_id: Scoped document identifier (doc_<sha256>).
        expected_chunk_count: Number of chunks generated from the source PDF.

    Returns:
        True if the stored index is 100% complete and verified; False otherwise.
    """
    if collection is None or not document_id or expected_chunk_count <= 0:
        return False

    try:
        existing = collection.get(where={"document_id": document_id})
    except Exception:
        return False

    if not existing or not existing.get("ids"):
        return False

    stored_ids = existing["ids"]
    if len(stored_ids) != expected_chunk_count:
        return False

    metadatas = existing.get("metadatas", [])
    if len(metadatas) != expected_chunk_count:
        return False

    chunk_ids: list[int] = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            return False
        cid = meta.get("chunk_id")
        if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
            return False
        chunk_ids.append(cid)

    # Verify that chunk IDs form an unbroken sequential series: 1, 2, ..., N
    if sorted(chunk_ids) != list(range(1, expected_chunk_count + 1)):
        return False

    documents = existing.get("documents", [])
    if len(documents) != expected_chunk_count:
        return False

    if any(not isinstance(doc, str) or not doc.strip() for doc in documents):
        return False

    return True


def process_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
    collection: Any,
    embedding_model: Any,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> dict[str, Any]:
    """
    Orchestrates end-to-end PDF ingestion with defensive validation, failure recovery,
    and index verification.

    Workflow:
        1. Validates byte input and computes the full SHA-256 document ID.
        2. Writes bytes to a secure temporary file with guaranteed cleanup in `finally`.
        3. Extracts text and validates that usable, non-whitespace text exists.
        4. Slices text into overlapping chunks and validates chunk output.
        5. Verifies existing index in ChromaDB. If verified, reuses stored records immediately.
        6. If existing records are missing or partial, deletes any stale records for this
           document and computes dense embeddings.
        7. Upserts vector records into ChromaDB and returns ingestion metrics.

    Args:
        pdf_bytes: Raw binary content of the PDF.
        filename: Original filename of the uploaded PDF.
        collection: ChromaDB Collection instance.
        embedding_model: Pre-loaded SentenceTransformer model instance.
        chunk_size: Chunk size in characters (default: 500).
        chunk_overlap: Chunk overlap in characters (default: 100).

    Returns:
        Dictionary containing ingestion metadata:
            - 'document_id' (str)
            - 'filename' (str)
            - 'page_count' (int)
            - 'chunk_count' (int)
            - 'reused_existing' (bool)

    Raises:
        ValueError: If file is empty, or chunking/embedding produces invalid output.
        NoTextExtractedError: If PDF contains no extractable text (e.g. scanned image).
        PdfReadError: If PDF is encrypted, corrupted, or unreadable.
        RuntimeError: If ChromaDB storage or embedding generation fails.
    """
    if not pdf_bytes:
        raise ValueError("Uploaded file is empty (0 bytes). Please upload a valid PDF.")

    if collection is None:
        raise RuntimeError("Vector store collection is not available.")

    doc_id = compute_document_id(pdf_bytes)
    tmp_path: str | None = None

    try:
        # Step 1: Secure temporary file creation
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        # Step 2: Page-by-page text extraction
        pages = extract_text_from_pdf(tmp_path)
        if not pages:
            raise NoTextExtractedError(f"No pages could be read from '{filename}'.")

        # Defensive check: ensure at least one page contains non-whitespace text
        has_text = any(page.get("text", "").strip() for page in pages if isinstance(page, dict))
        if not has_text:
            raise NoTextExtractedError(
                f"No extractable text found in '{filename}'. The document may be empty or contain only scanned images."
            )

        page_count = len(pages)

        # Step 3: Text chunking
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError(f"Document '{filename}' produced 0 chunks after text chunking.")

        chunk_count = len(chunks)

        # Step 4: Index Verification & Reuse
        if verify_existing_index(collection, doc_id, expected_chunk_count=chunk_count):
            return {
                "document_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "reused_existing": True,
            }

        # Step 5: Failure Recovery & Reprocessing
        # If existing index is missing or partial (e.g., from an interrupted previous run),
        # safely delete any orphaned records for THIS document ID before re-indexing.
        try:
            collection.delete(where={"document_id": doc_id})
        except Exception:
            # Collection may not contain any existing records yet; safe to proceed
            pass

        # Step 6: Dense Embedding Generation
        if embedding_model is None:
            raise RuntimeError("Embedding model is not available.")

        embedded_chunks = embed_chunks(chunks, model=embedding_model)
        if not embedded_chunks or len(embedded_chunks) != chunk_count:
            raise ValueError(
                f"Embedding count mismatch: expected {chunk_count} vectors, got {len(embedded_chunks) if embedded_chunks else 0}."
            )

        # Validate that each chunk has a valid 384-dimensional vector
        for chunk in embedded_chunks:
            emb = chunk.get("embedding")
            if not isinstance(emb, list) or len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(f"Malformed embedding vector detected for chunk {chunk.get('chunk_id')}.")

        # Step 7: Persistent ChromaDB Upsert
        upserted = store_chunks(embedded_chunks, collection, document_id=doc_id)
        if upserted != chunk_count:
            raise RuntimeError(
                f"Vector store upsert mismatch: expected {chunk_count} records upserted, got {upserted}."
            )

        return {
            "document_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "reused_existing": False,
        }

    finally:
        # Step 8: Guaranteed temporary file cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def answer_question_for_document(
    query: str,
    document_id: str,
    collection: Any,
    embedding_model: Any,
    top_k: int = 3,
    gemini_client: Any | None = None,
) -> dict[str, Any]:
    """
    Orchestrates scoped semantic retrieval and grounded answer generation for a specific document.

    Workflow:
        1. Validates query and document_id parameters.
        2. Generates a dense 384-dimensional embedding vector for the query.
        3. Queries ChromaDB vector store strictly filtered to the active document ID.
        4. Invokes generate_answer() with retrieved context passages and citation metadata.
        5. Returns a structured dictionary containing query, answer, source metadata,
           retrieved chunk passages, abstention flag, and model metadata.

    Args:
        query: User search query string. Must be a non-empty string.
        document_id: Scoped document identifier (e.g. 'doc_<64-char-sha256>').
        collection: Active ChromaDB Collection instance.
        embedding_model: Pre-loaded SentenceTransformer model instance.
        top_k: Number of most relevant passages to retrieve (default: 3).
        gemini_client: Optional pre-configured genai.Client (for testing/mocking).

    Returns:
        Dictionary containing:
            - 'query' (str)
            - 'document_id' (str)
            - 'answer' (str)
            - 'sources' (list[dict])
            - 'retrieved_chunks' (list[dict])
            - 'abstention' (bool)
            - 'model' (str)

    Raises:
        TypeError: If query, document_id, or collection have invalid types.
        ValueError: If query or document_id are empty or whitespace-only.
        RuntimeError: If collection or embedding_model are missing, or generation fails.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Question cannot be empty or whitespace-only.")

    if not isinstance(document_id, str):
        raise TypeError(f"document_id must be a string, got {type(document_id).__name__}")
    clean_doc_id = document_id.strip()
    if not clean_doc_id:
        raise ValueError("document_id cannot be empty or whitespace-only.")

    if collection is None:
        raise RuntimeError("Vector store collection is not available.")

    if embedding_model is None:
        raise RuntimeError("Embedding model is not available.")

    # Start end-to-end question answering pipeline timer
    t_total_start = time.perf_counter()

    # Step 1: Query Embedding Generation (384-dimensional dense vector)
    t_embed_start = time.perf_counter()
    raw_query_emb = embedding_model.encode([clean_query], show_progress_bar=False)
    query_vector = raw_query_emb[0]
    t_embed_duration = time.perf_counter() - t_embed_start

    # Step 2: Document-Filtered Semantic Search in ChromaDB
    t_retrieve_start = time.perf_counter()
    retrieved_chunks = query_chunks(
        query_embedding=query_vector,
        collection=collection,
        top_k=top_k,
        document_id=clean_doc_id,
    )
    t_retrieve_duration = time.perf_counter() - t_retrieve_start

    # Step 3: Grounded Answer Synthesis with Gemini
    t_gen_start = time.perf_counter()
    answer_data = generate_answer(
        question=clean_query,
        context_chunks=retrieved_chunks,
        client=gemini_client,
    )
    t_gen_duration = time.perf_counter() - t_gen_start
    t_total_duration = time.perf_counter() - t_total_start

    timing = {
        "embedding_ms": round(t_embed_duration * 1000, 1),
        "retrieval_ms": round(t_retrieve_duration * 1000, 1),
        "generation_ms": round(t_gen_duration * 1000, 1),
        "total_seconds": round(t_total_duration, 2),
    }

    return {
        "query": clean_query,
        "document_id": clean_doc_id,
        "answer": answer_data.get("answer", ""),
        "sources": answer_data.get("sources", []),
        "retrieved_chunks": retrieved_chunks,
        "abstention": answer_data.get("abstention", False),
        "model": answer_data.get("model", DEFAULT_MODEL_NAME),
        "timing": timing,
    }


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
def get_cached_embedding_model():
    """Loads and caches the SentenceTransformer model singleton in memory."""
    return get_embedding_model()


@st.cache_resource(show_spinner="Connecting to persistent ChromaDB vector store...")
def get_cached_chroma_collection():
    """Initializes and caches the persistent ChromaDB collection handle."""
    client = get_chroma_client(DEFAULT_CHROMA_DIR)
    return get_or_create_collection(client, DEFAULT_COLLECTION_NAME)


@st.cache_resource(show_spinner="Connecting to Google Gemini API...")
def get_cached_gemini_client():
    """Initializes and caches the Google GenAI client instance to reuse HTTP connection pools."""
    return get_gemini_client()


# ---------------------------------------------------------
# Session State Initialization & Reset Helpers
# ---------------------------------------------------------
def init_session_state():
    """Initializes session state variables required for ingestion and Q&A tracking."""
    defaults = {
        # Ingestion state
        "active_document_id": None,
        "uploaded_filename": None,
        "page_count": 0,
        "chunk_count": 0,
        "document_processed": False,
        "ingestion_success": False,
        "reused_index": False,
        "last_error": None,
        # Q&A state (Step 4)
        "question_input": "",
        "last_question": None,
        "last_answer": None,
        "last_sources": None,
        "last_retrieved_chunks": None,
        "last_abstention": False,
        "last_timing": None,
        "qa_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def reset_qa_session_state(clear_input: bool = True):
    """
    Safely resets Q&A session state and clears widget input before rendering.
    Compatible with Streamlit widget lifecycle rules: only called before the widget is instantiated.
    """
    if clear_input:
        st.session_state["question_input"] = ""
    st.session_state["last_question"] = None
    st.session_state["last_answer"] = None
    st.session_state["last_sources"] = None
    st.session_state["last_retrieved_chunks"] = None
    st.session_state["last_abstention"] = False
    st.session_state["last_timing"] = None
    st.session_state["qa_error"] = None


# ---------------------------------------------------------
# Sidebar: Ingestion & Search Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Ingestion")
    st.caption("Upload a study PDF to extract text, generate embeddings, and store them in ChromaDB.")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Select a PDF document from your local machine to study.",
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True,
        type="primary",
        help="Click to extract text, chunk content, generate dense embeddings, and index in ChromaDB.",
    )

    st.divider()
    st.header("⚙️ Search Settings")
    top_k = st.slider(
        "Context Passages (Top-K)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of most relevant passages retrieved from ChromaDB for Gemini answer synthesis.",
    )


# ---------------------------------------------------------
# State Sync: Detect File Replacement or Removal
# ---------------------------------------------------------
current_doc_id: str | None = None
pdf_bytes: bytes | None = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    if pdf_bytes:
        current_doc_id = compute_document_id(pdf_bytes)

    # Detect if user replaced the uploaded file with a different, unprocessed PDF
    if (
        st.session_state["active_document_id"] is not None
        and st.session_state["active_document_id"] != current_doc_id
    ):
        st.session_state["document_processed"] = False
        st.session_state["ingestion_success"] = False
        st.session_state["active_document_id"] = None
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["page_count"] = 0
        st.session_state["chunk_count"] = 0
        st.session_state["reused_index"] = False
        st.session_state["last_error"] = None
        # Clear Q&A state and input before widget instantiation to prevent stale contamination
        reset_qa_session_state(clear_input=True)
else:
    # User removed the file from the uploader
    if st.session_state["active_document_id"] is not None or st.session_state["document_processed"]:
        st.session_state["active_document_id"] = None
        st.session_state["uploaded_filename"] = None
        st.session_state["page_count"] = 0
        st.session_state["chunk_count"] = 0
        st.session_state["document_processed"] = False
        st.session_state["ingestion_success"] = False
        st.session_state["reused_index"] = False
        st.session_state["last_error"] = None
        # Clear Q&A state and input before widget instantiation on file removal
        reset_qa_session_state(clear_input=True)


# ---------------------------------------------------------
# Ingestion Execution Pipeline (Triggered by Button Click)
# ---------------------------------------------------------
if process_button:
    if uploaded_file is None or not pdf_bytes:
        st.session_state["document_processed"] = False
        st.session_state["ingestion_success"] = False
        st.sidebar.warning("⚠️ Please select and upload a valid PDF file before clicking 'Process PDF'.")
    else:
        # Wrap the entire ingestion sequence in robust error handling
        try:
            with st.spinner("Connecting to persistent vector database..."):
                collection = get_cached_chroma_collection()

            with st.spinner("Loading embedding model..."):
                embedding_model = get_cached_embedding_model()

            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                result = process_uploaded_pdf(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    collection=collection,
                    embedding_model=embedding_model,
                )

            # Atomic State Update: Set success flags ONLY after full success
            st.session_state["active_document_id"] = result["document_id"]
            st.session_state["uploaded_filename"] = result["filename"]
            st.session_state["page_count"] = result["page_count"]
            st.session_state["chunk_count"] = result["chunk_count"]
            st.session_state["reused_index"] = result["reused_existing"]
            st.session_state["document_processed"] = True
            st.session_state["ingestion_success"] = True
            st.session_state["last_error"] = None
            # Reset Q&A state and clear previous question for newly processed document
            reset_qa_session_state(clear_input=True)

            if result["reused_existing"]:
                st.sidebar.success(
                    f"⚡ Loaded existing verified index for **{uploaded_file.name}** "
                    f"({result['chunk_count']} chunks in ChromaDB)."
                )
            else:
                st.sidebar.success(
                    f"✅ Successfully processed **{uploaded_file.name}**!\n\n"
                    f"- Pages: {result['page_count']}\n"
                    f"- Chunks: {result['chunk_count']}\n"
                    f"- ID: `{result['document_id'][:20]}...`"
                )

        except (FileNotFoundError, ValueError, PdfReadError, NoTextExtractedError) as err:
            st.session_state["document_processed"] = False
            st.session_state["ingestion_success"] = False
            clean_err = sanitize_error_message(err)
            st.session_state["last_error"] = clean_err
            st.sidebar.error(f"❌ Ingestion Error: {clean_err}")
        except Exception as err:
            st.session_state["document_processed"] = False
            st.session_state["ingestion_success"] = False
            clean_err = sanitize_error_message(err)
            st.session_state["last_error"] = clean_err
            st.sidebar.error(f"❌ Unexpected Ingestion Failure: {clean_err}")


# ---------------------------------------------------------
# Main Page: Title & Instructions
# ---------------------------------------------------------
st.title("DocuMind — AI PDF Study Assistant")
st.markdown(
    """
    Welcome to **DocuMind**! This interactive study assistant extracts knowledge from your PDFs,
    indexes information using dense semantic vector search, and delivers accurate,
    grounded answers powered by Google's Gemini models with verified source citations.
    """
)

# Getting Started instructions card
with st.container():
    st.subheader("💡 How It Works")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1. Upload**")
        st.write("Select a PDF document in the sidebar.")

    with col2:
        st.markdown("**2. Process**")
        st.write("Click 'Process PDF' to chunk and embed pages.")

    with col3:
        st.markdown("**3. Ask Questions**")
        st.write("Type a question about the document's content.")

    with col4:
        st.markdown("**4. Study Grounded Answers**")
        st.write("Receive answers backed by page-level citations.")

st.divider()


# ---------------------------------------------------------
# Document Status Area
# ---------------------------------------------------------
st.subheader("📊 Document Status")

status_container = st.container()

with status_container:
    if st.session_state["document_processed"] and st.session_state["active_document_id"]:
        reuse_badge = " (⚡ Reused Verified Index)" if st.session_state["reused_index"] else " (✨ Newly Indexed)"
        st.success(
            f"✅ **Active Document:** `{st.session_state['uploaded_filename']}`{reuse_badge}\n\n"
            f"**Document ID:** `{st.session_state['active_document_id']}`"
        )
        # Display key ingestion metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Pages Extracted", st.session_state["page_count"])
        with m_col2:
            st.metric("Chunks Indexed", st.session_state["chunk_count"])
        with m_col3:
            st.metric("Vector Database", "ChromaDB (HNSW)")
    elif uploaded_file is not None and not st.session_state["document_processed"]:
        st.warning(
            f"📄 File **'{uploaded_file.name}'** selected but not yet processed. "
            "Click **'Process PDF'** in the sidebar to extract text, chunk content, and generate embeddings."
        )
    elif st.session_state["last_error"]:
        st.error(f"⚠️ Last ingestion failed: {st.session_state['last_error']}")
    else:
        st.info("No document processed yet. Use the sidebar to upload a PDF and click **'Process PDF'** to begin.")

st.divider()


# ---------------------------------------------------------
# Question & Answering Area (Milestone 7 Step 4)
# ---------------------------------------------------------
st.subheader("💬 Ask a Question")

with st.form("qa_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter your question about the document:",
        placeholder="e.g., What are the main findings in Chapter 3?",
        key="question_input",
        disabled=not st.session_state["document_processed"],
        help="Type a question and submit to query ChromaDB and generate a grounded Gemini answer.",
    )

    ask_button = st.form_submit_button(
        "Ask Assistant",
        type="primary",
        disabled=not st.session_state["document_processed"],
        help="Submit your question for scoped retrieval and Gemini answer synthesis.",
    )

if ask_button:
    if not st.session_state["document_processed"] or not st.session_state["active_document_id"]:
        st.warning("⚠️ Please upload and process a PDF document before asking questions.")
    elif not user_query or not user_query.strip():
        st.warning("⚠️ Please enter a question before clicking 'Ask Assistant'.")
    else:
        clean_query = user_query.strip()
        st.session_state["last_question"] = clean_query
        st.session_state["qa_error"] = None

        try:
            with st.spinner("Connecting to vector database..."):
                collection = get_cached_chroma_collection()

            with st.spinner("Loading embedding model..."):
                embedding_model = get_cached_embedding_model()

            with st.spinner("Connecting to Gemini client..."):
                gemini_client = get_cached_gemini_client()

            with st.spinner("Retrieving relevant passages and synthesizing answer with Gemini (Google AI API inference in progress)..."):
                qa_result = answer_question_for_document(
                    query=clean_query,
                    document_id=st.session_state["active_document_id"],
                    collection=collection,
                    embedding_model=embedding_model,
                    top_k=top_k,
                    gemini_client=gemini_client,
                )

            st.session_state["last_answer"] = qa_result["answer"]
            st.session_state["last_sources"] = qa_result["sources"]
            st.session_state["last_retrieved_chunks"] = qa_result["retrieved_chunks"]
            st.session_state["last_abstention"] = qa_result["abstention"]
            st.session_state["last_timing"] = qa_result.get("timing")
            st.session_state["qa_error"] = None

        except ValueError as err:
            clean_err = sanitize_error_message(err)
            st.session_state["qa_error"] = f"Configuration required: {clean_err}"
            st.session_state["last_answer"] = None
            st.session_state["last_sources"] = None
            st.session_state["last_retrieved_chunks"] = None
            st.session_state["last_timing"] = None
        except RuntimeError as err:
            clean_err = sanitize_error_message(err)
            st.session_state["qa_error"] = f"Generation failure: {clean_err}"
            st.session_state["last_answer"] = None
            st.session_state["last_sources"] = None
            st.session_state["last_retrieved_chunks"] = None
            st.session_state["last_timing"] = None
        except Exception as err:
            clean_err = sanitize_error_message(err)
            st.session_state["qa_error"] = f"Unexpected retrieval/generation error: {clean_err}"
            st.session_state["last_answer"] = None
            st.session_state["last_sources"] = None
            st.session_state["last_retrieved_chunks"] = None
            st.session_state["last_timing"] = None


# ---------------------------------------------------------
# Output Area: Answer & Source Citations
# ---------------------------------------------------------
st.markdown("### 📝 Generated Answer")

if st.session_state["qa_error"]:
    st.error(f"❌ {st.session_state['qa_error']}")

if st.session_state["last_answer"]:
    if st.session_state["last_abstention"]:
        st.info(f"ℹ️ **Notice:** {st.session_state['last_answer']}")
    else:
        st.markdown(st.session_state["last_answer"])

    # Performance Breakdown Display
    if st.session_state.get("last_timing"):
        timing = st.session_state["last_timing"]
        st.caption(
            f"⏱️ **Performance Breakdown:** "
            f"Query Embedding: `{timing['embedding_ms']} ms` | "
            f"ChromaDB Retrieval: `{timing['retrieval_ms']} ms` | "
            f"Gemini API: `{timing['generation_ms'] / 1000:.2f} s` | "
            f"**Total Request: `{timing['total_seconds']} s`**"
        )
elif not st.session_state["document_processed"]:
    st.info("💡 Process a PDF document using the sidebar to unlock question answering.")
elif not st.session_state["qa_error"]:
    st.caption("Ask a question above to generate an answer grounded in the active PDF.")

# ---------------------------------------------------------
# Source Citations Display
# ---------------------------------------------------------
retrieved_chunks = st.session_state["last_retrieved_chunks"]

with st.expander(
    f"🔍 Retrieved Source Citations ({len(retrieved_chunks) if retrieved_chunks else 0} passages)",
    expanded=bool(retrieved_chunks and not st.session_state["last_abstention"]),
):
    if retrieved_chunks:
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            sim_score = chunk.get("similarity_score")
            sim_pct = f"{sim_score * 100:.1f}%" if sim_score is not None else "N/A"
            score_str = f"{sim_score:.4f}" if sim_score is not None else "N/A"
            page_num = chunk.get("page_number", "Unknown")
            chunk_id = chunk.get("chunk_id", "Unknown")
            doc_id_val = chunk.get("document_id") or ""
            doc_id_short = f"{doc_id_val[:16]}..." if len(doc_id_val) > 16 else (doc_id_val or "Unknown")

            st.markdown(
                f"**Passage {rank}** | 📄 Page {page_num} | 🧩 Chunk {chunk_id} | "
                f"🎯 Similarity: `{score_str}` ({sim_pct}) | 🆔 Doc: `{doc_id_short}`"
            )
            st.text_area(
                label=f"Passage {rank} Content",
                value=chunk.get("text", ""),
                height=110,
                disabled=True,
                key=f"chunk_passage_{rank}_{chunk_id}",
            )
            if rank < len(retrieved_chunks):
                st.divider()
    else:
        st.write(
            "No source citations to display. Source citations with exact page numbers, "
            "chunk IDs, and similarity scores will appear here after asking a question."
        )
