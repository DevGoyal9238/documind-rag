import sys
from pathlib import Path
from typing import Any
from pypdf.errors import PdfReadError
import sqlite3

try:
    from chromadb.errors import ChromaError
except ImportError:
    class ChromaError(Exception):  # type: ignore
        pass

from src.pdf_loader import extract_text_from_pdf, NoTextExtractedError, PDFLoaderError
from src.text_chunker import chunk_pages
from src.embedder import embed_chunks, get_embedding_model
from src.semantic_search import search_chunks
from src.vector_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    get_chroma_client,
    get_or_create_collection,
    store_chunks,
    query_chunks,
    verify_existing_index,
    delete_document_chunks,
)
from src.qa_generator import generate_answer, sanitize_error_message
from src.document_utils import compute_document_id


def get_cli_document_id(pdf_path: str) -> str:
    """Reads PDF binary content from disk and computes its SHA-256 document ID."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return compute_document_id(pdf_bytes)


def process_cli_document(
    pdf_path: str,
    collection: Any,
    embedding_model: Any = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> dict[str, Any]:
    """
    Ingests or verifies a PDF for the CLI application with safe index reuse and purge recovery.

    Workflow:
        1. Reads PDF bytes and computes the content-addressable document ID.
        2. Extracts text pages and slices them into chunks to determine expected chunk count.
        3. Checks existing index in ChromaDB using verify_existing_index().
        4. If verified, reuses stored index and skips embedding and storage.
        5. If missing or partial, purges only records for this document ID. If purge fails,
           aborts and raises RuntimeError without proceeding to embedding.
        6. Generates embeddings and stores records into ChromaDB.

    Args:
        pdf_path: Filesystem path to the target PDF.
        collection: Active ChromaDB Collection instance.
        embedding_model: Optional pre-loaded SentenceTransformer model instance.
        chunk_size: Character limit per chunk (default: 500).
        chunk_overlap: Overlapping character count (default: 100).

    Returns:
        Dictionary containing:
            - 'document_id' (str)
            - 'page_count' (int)
            - 'chunk_count' (int)
            - 'chunks' (list[dict])
            - 'embedded_chunks' (list[dict] | None)
            - 'reused_existing' (bool)

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        NoTextExtractedError: If PDF contains no extractable text.
        PdfReadError: If PDF is malformed or unreadable.
        ValueError: If document produces 0 chunks or invalid arguments.
        TypeError: If collection is None.
        RuntimeError: If ChromaDB deletion or upsert fails.
    """
    if collection is None:
        raise TypeError("collection cannot be None")

    doc_id = get_cli_document_id(pdf_path)
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise NoTextExtractedError(f"No pages could be read from '{pdf_path}'.")

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"Document '{pdf_path}' produced 0 chunks after text chunking.")

    chunk_count = len(chunks)
    page_count = len(pages)

    # Step 3: Index Verification & Reuse
    if verify_existing_index(collection, doc_id, expected_chunk_count=chunk_count):
        return {
            "document_id": doc_id,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "chunks": chunks,
            "embedded_chunks": None,
            "reused_existing": True,
        }

    # Step 4: Failure Recovery & Document Purge
    # Purge only records matching THIS document_id. If purge fails, do NOT continue to embedding.
    delete_document_chunks(collection, doc_id)

    # Step 5: Dense Embedding Generation & ChromaDB Storage
    embedded_chunks = embed_chunks(chunks, model=embedding_model)
    if not embedded_chunks or len(embedded_chunks) != chunk_count:
        raise ValueError(
            f"Embedding count mismatch: expected {chunk_count} vectors, got {len(embedded_chunks) if embedded_chunks else 0}."
        )

    upserted = store_chunks(embedded_chunks, collection, document_id=doc_id)
    if upserted != chunk_count:
        raise RuntimeError(
            f"Vector store upsert mismatch: expected {chunk_count} records upserted, got {upserted}."
        )

    return {
        "document_id": doc_id,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "chunks": chunks,
        "embedded_chunks": embedded_chunks,
        "reused_existing": False,
    }


def main():
    print("=" * 45)
    print("      DocuMind - RAG Document Processor")
    print("=" * 45)

    # 1. Accept file path via command line argument or interactive input
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1].strip()
    else:
        try:
            pdf_path = input("Enter path to PDF file: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return

    if not pdf_path:
        print("Error: No file path provided.")
        return

    # 2. Extract text, chunk, verify index, embed/store as needed
    try:
        client = get_chroma_client(DEFAULT_CHROMA_DIR)
        collection = get_or_create_collection(client, DEFAULT_COLLECTION_NAME)

        res = process_cli_document(
            pdf_path=pdf_path,
            collection=collection,
            chunk_size=500,
            chunk_overlap=100,
        )

        doc_id = res["document_id"]
        pages_count = res["page_count"]
        chunks = res["chunks"]
        chunks_count = res["chunk_count"]
        embedded_chunks = res["embedded_chunks"]
        reused_existing = res["reused_existing"]

        print(f"\n[Success] Extracted {pages_count} page(s) from '{pdf_path}'.")
        print(f"[Success] Generated {chunks_count} chunk(s) (chunk_size=500, overlap=100):\n")

        for chunk in chunks:
            print(f"--- Chunk {chunk['chunk_id']} [Page {chunk['page_number']} | {len(chunk['text'])} chars] ---")
            print(chunk["text"])
            print()

        print("=" * 45)
        print("       ChromaDB Vector Database")
        print("=" * 45)

        if reused_existing:
            print(f"[Index Verified] Reusing existing complete index for document '{doc_id}' ({chunks_count} chunk(s)).")
            print("Skipping dense vector embedding generation and ChromaDB storage.")
            print(f"Total chunks in collection '{DEFAULT_COLLECTION_NAME}': {collection.count()}\n")
        else:
            if embedded_chunks:
                dim = len(embedded_chunks[0]["embedding"])
                preview = [round(val, 4) for val in embedded_chunks[0]["embedding"][:5]]
                print(f"Embedding Vector Dimension: {dim}")
                print(f"Sample preview (Chunk 1, first 5 dimensions): {preview} ...\n")
            print(f"[Success] Upserted {chunks_count} chunk(s) into ChromaDB (document_id='{doc_id}').")
            print(f"Total chunks in collection '{DEFAULT_COLLECTION_NAME}': {collection.count()}\n")

        # 5. Interactive semantic search
        print("=" * 45)
        print("           Semantic Search")
        print("=" * 45)

        if len(sys.argv) > 2:
            query = sys.argv[2].strip()
            print(f"Search Query: '{query}'")
        else:
            try:
                query = input("Ask a question about this document (or press Enter to exit): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSearch cancelled by user.")
                return

        if query:
            print(f"\nSearching for chunks relevant to: '{query}'...")
            model = get_embedding_model()

            # NumPy search (Milestone 5 - In-memory exact search)
            if embedded_chunks:
                numpy_results = search_chunks(query, embedded_chunks, top_k=3, model=model)
                print(f"\n--- [NumPy Search Results] (In-memory Exact Dot Product) ---")
                for rank, res_item in enumerate(numpy_results, start=1):
                    score = res_item["similarity_score"]
                    pct = score * 100
                    print(
                        f"Rank {rank} | Chunk {res_item['chunk_id']} [Page {res_item['page_number']}] "
                        f"| Similarity: {score:.4f} ({pct:.1f}%)"
                    )
                    print(res_item["text"])
                    print()
            else:
                print("\n[Note] In-memory NumPy search skipped (reusing verified ChromaDB index).")

            # ChromaDB search (Milestone 5.5 - Persistent vector database)
            raw_query_emb = model.encode([query], show_progress_bar=False)
            query_vector = raw_query_emb[0]
            chroma_results = query_chunks(query_vector, collection, top_k=3, document_id=doc_id)

            print(f"--- [ChromaDB Search Results] (Persistent Cosine Index) ---")
            for rank, res_item in enumerate(chroma_results, start=1):
                score = res_item["similarity_score"]
                dist = res_item["cosine_distance"]
                print(
                    f"Rank {rank} | Chunk {res_item['chunk_id']} [Page {res_item['page_number']}] "
                    f"| ID: {res_item['id']} | Similarity: {score:.4f} | Distance: {dist:.4f}"
                )
                print(res_item["text"])
                print()

            # 6. RAG Answer Generation (Milestone 6 - Gemini LLM)
            print("=" * 45)
            print("       AI Study Assistant (Gemini RAG)")
            print("=" * 45)

            try:
                answer_data = generate_answer(query, chroma_results)
                print(f"\n{answer_data['answer']}\n")

                formatted_sources: list[str] = []
                for src in answer_data.get("sources", []):
                    page = src.get("page_number")
                    chunk = src.get("chunk_id")
                    if page is not None and chunk is not None:
                        formatted_sources.append(f"Page {page} (Chunk {chunk})")
                    elif page is not None:
                        formatted_sources.append(f"Page {page}")

                if formatted_sources:
                    unique_sources: list[str] = []
                    seen_sources: set[str] = set()
                    for s in formatted_sources:
                        if s not in seen_sources:
                            seen_sources.add(s)
                            unique_sources.append(s)
                    citations_str = ", ".join(unique_sources)
                    print(f"Retrieved Sources: {citations_str}\n")

            except ValueError as err:
                # Catches missing or placeholder GEMINI_API_KEY without leaking secrets
                print(f"\n[Note] LLM answer generation skipped: {sanitize_error_message(err)}")
                print("Semantic search results above remain fully available.\n")
            except Exception as err:
                # Catches API connection/quota errors without crashing the application
                print(f"\n[Generation Notice] Could not complete LLM generation: {sanitize_error_message(err)}\n")
        else:
            print("No query provided. Skipping search.\n")

        print("=" * 45)
        print("Processing complete.")
        print("=" * 45)

    except FileNotFoundError as err:
        print(f"\n[File Error] {sanitize_error_message(err)}")
    except ValueError as err:
        print(f"\n[Input Error] {sanitize_error_message(err)}")
    except PdfReadError as err:
        print(f"\n[PDF Error] {sanitize_error_message(err)}")
    except NoTextExtractedError as err:
        print(f"\n[Content Warning] {sanitize_error_message(err)}")
    except PDFLoaderError as err:
        print(f"\n[PDF Error] {sanitize_error_message(err)}")
    except (RuntimeError, ChromaError, sqlite3.Error) as err:
        print(f"\n[Database Error] {sanitize_error_message(err)}")


if __name__ == "__main__":
    main()