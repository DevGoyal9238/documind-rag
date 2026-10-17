import sys
from pypdf.errors import PdfReadError
from src.pdf_loader import extract_text_from_pdf, NoTextExtractedError
from src.text_chunker import chunk_pages
from src.embedder import embed_chunks
from src.semantic_search import search_chunks


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

    # 2. Extract text, chunk, embed, and search with error handling
    try:
        pages = extract_text_from_pdf(pdf_path)
        print(f"\n[Success] Extracted {len(pages)} page(s) from '{pdf_path}'.")

        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)
        print(f"[Success] Generated {len(chunks)} chunk(s) (chunk_size=500, overlap=100):\n")

        for chunk in chunks:
            print(f"--- Chunk {chunk['chunk_id']} [Page {chunk['page_number']} | {len(chunk['text'])} chars] ---")
            print(chunk["text"])
            print()

        # 3. Generate dense vector embeddings
        print("Generating dense vector embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        embedded_chunks = embed_chunks(chunks)
        print(f"[Success] Embedded {len(embedded_chunks)} chunk(s).")

        if embedded_chunks:
            dim = len(embedded_chunks[0]["embedding"])
            preview = [round(val, 4) for val in embedded_chunks[0]["embedding"][:5]]
            print(f"Embedding Vector Dimension: {dim}")
            print(f"Sample preview (Chunk 1, first 5 dimensions): {preview} ...\n")

            # 4. Interactive semantic search
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
                results = search_chunks(query, embedded_chunks, top_k=3)
                print(f"[Success] Retrieved {len(results)} top matching chunk(s):\n")

                for rank, res in enumerate(results, start=1):
                    score = res["similarity_score"]
                    pct = score * 100
                    print(
                        f"--- Rank {rank} | Chunk {res['chunk_id']} [Page {res['page_number']}] "
                        f"| Similarity: {score:.4f} ({pct:.1f}%) ---"
                    )
                    print(res["text"])
                    print()
            else:
                print("No query provided. Skipping search.\n")

        print("=" * 45)
        print("Processing complete.")
        print("=" * 45)

    except FileNotFoundError as err:
        print(f"\n[File Error] {err}")
    except ValueError as err:
        print(f"\n[Input Error] {err}")
    except PdfReadError as err:
        print(f"\n[PDF Error] {err}")
    except NoTextExtractedError as err:
        print(f"\n[Content Warning] {err}")


if __name__ == "__main__":
    main()