import sys
from pypdf.errors import PdfReadError
from src.pdf_loader import extract_text_from_pdf, NoTextExtractedError


def main():
    print("=" * 45)
    print("      DocuMind - PDF Text Extractor")
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

    # 2. Extract text with specific error handling
    try:
        pages = extract_text_from_pdf(pdf_path)
        print(f"\n[Success] Extracted {len(pages)} page(s) from '{pdf_path}':\n")

        for page in pages:
            print(f"--- Page {page['page_number']} ---")
            text = page["text"].strip()
            if text:
                print(text)
            else:
                print("[No text content on this page]")
            print()

        print("=" * 45)
        print("Extraction complete.")
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