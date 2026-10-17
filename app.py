import sys
from pypdf.errors import PdfReadError
from src.pdf_loader import extract_text_from_pdf, NoTextExtractedError
from src.text_chunker import chunk_pages
from src.embedder import embed_chunks


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

    # 2. Extract text, chunk, and embed with error handling
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