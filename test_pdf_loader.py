"""
DocuMind - Unit and Integration Tests for PDF Loader Module (Milestone 8, Phase 1)
Tests page-by-page text extraction, valid PDFs, empty documents, missing files,
non-PDF and malformed files, encrypted PDFs, and defensive exception translation
into PDFLoaderError and NoTextExtractedError using small synthetic PDFs.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pypdf import PdfWriter, PdfReader
from pypdf.errors import PdfReadError

from src.pdf_loader import (
    extract_text_from_pdf,
    PDFLoaderError,
    NoTextExtractedError,
)


class TestPdfLoader(unittest.TestCase):
    """Offline unit and integration tests for extract_text_from_pdf."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    # -------------------------------------------------------------------------
    # 1. Valid PDF Extraction Tests
    # -------------------------------------------------------------------------
    def test_extract_text_from_real_sample_pdf(self):
        """Verifies extracting text from the existing sample.pdf file."""
        sample_path = os.path.join("data", "sample.pdf")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample PDF not found at {sample_path}")

        pages = extract_text_from_pdf(sample_path)
        self.assertIsInstance(pages, list)
        self.assertEqual(len(pages), 2)

        # Page 1 checks
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Hello from Page 1!", pages[0]["text"])

        # Page 2 checks
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertIn("DocuMind RAG Architecture - Page 2", pages[1]["text"])

    def test_extract_text_from_synthetic_pdf(self):
        """Creates a small synthetic PDF by cloning sample.pdf page and verifies extraction."""
        sample_path = os.path.join("data", "sample.pdf")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample PDF not found at {sample_path}")

        reader = PdfReader(sample_path)
        writer = PdfWriter()
        writer.add_page(reader.pages[0])

        synthetic_path = os.path.join(self.temp_dir.name, "synthetic_valid.pdf")
        with open(synthetic_path, "wb") as f:
            writer.write(f)

        pages = extract_text_from_pdf(synthetic_path)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Hello from Page 1!", pages[0]["text"])

    # -------------------------------------------------------------------------
    # 2. Empty Document / No Text Extracted Tests
    # -------------------------------------------------------------------------
    def test_empty_blank_pdf_raises_no_text_extracted_error(self):
        """Existing blank.pdf contains blank pages and should raise NoTextExtractedError."""
        blank_path = os.path.join("data", "blank.pdf")
        if not os.path.exists(blank_path):
            self.skipTest(f"Blank PDF not found at {blank_path}")

        with self.assertRaises(NoTextExtractedError) as ctx:
            extract_text_from_pdf(blank_path)
        self.assertIn("No extractable text found", str(ctx.exception))
        # Ensure it is also an instance of base PDFLoaderError
        self.assertIsInstance(ctx.exception, PDFLoaderError)

    def test_synthetic_blank_page_pdf_raises_no_text_extracted_error(self):
        """Creates a synthetic PDF with blank pages and verifies NoTextExtractedError."""
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        blank_path = os.path.join(self.temp_dir.name, "synthetic_blank.pdf")
        with open(blank_path, "wb") as f:
            writer.write(f)

        with self.assertRaises(NoTextExtractedError) as ctx:
            extract_text_from_pdf(blank_path)
        self.assertIn("No extractable text found", str(ctx.exception))

    def test_zero_pages_pdf_raises_no_text_extracted_error(self):
        """PDF with zero pages raises NoTextExtractedError cleanly."""
        writer = PdfWriter()
        zero_pages_path = os.path.join(self.temp_dir.name, "zero_pages.pdf")
        with open(zero_pages_path, "wb") as f:
            writer.write(f)

        with self.assertRaises(NoTextExtractedError):
            extract_text_from_pdf(zero_pages_path)

    # -------------------------------------------------------------------------
    # 3. Path & File Validation Tests
    # -------------------------------------------------------------------------
    def test_missing_file_raises_file_not_found_error(self):
        """Non-existent file path raises FileNotFoundError."""
        missing_path = os.path.join(self.temp_dir.name, "does_not_exist.pdf")
        with self.assertRaises(FileNotFoundError) as ctx:
            extract_text_from_pdf(missing_path)
        self.assertIn("File not found", str(ctx.exception))

    def test_directory_path_raises_value_error(self):
        """Passing a directory path raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_text_from_pdf(self.temp_dir.name)
        self.assertIn("not a regular file", str(ctx.exception))

    def test_invalid_path_types_raise_type_error(self):
        """Passing non-string arguments raises TypeError."""
        for invalid_path in [123, None, ["path.pdf"], True]:
            with self.subTest(path=invalid_path):
                with self.assertRaises(TypeError):
                    extract_text_from_pdf(invalid_path)  # type: ignore

    # -------------------------------------------------------------------------
    # 4. Non-PDF and Malformed PDF Tests
    # -------------------------------------------------------------------------
    def test_non_pdf_file_raises_pdf_read_error(self):
        """File containing plain text or random binary data raises PdfReadError."""
        fake_pdf = os.path.join(self.temp_dir.name, "plain_text.pdf")
        with open(fake_pdf, "wb") as f:
            f.write(b"This is just a plain text file, not a valid PDF document at all.")

        with self.assertRaises(PdfReadError) as ctx:
            extract_text_from_pdf(fake_pdf)
        self.assertIn("Invalid or corrupted PDF file", str(ctx.exception))

    def test_corrupted_pdf_stream_raises_pdf_read_error(self):
        """File with valid header but truncated/corrupted content raises PdfReadError."""
        corrupt_pdf = os.path.join(self.temp_dir.name, "corrupt.pdf")
        with open(corrupt_pdf, "wb") as f:
            f.write(b"%PDF-1.4\n%corrupt stream content with missing xref and trailer\n%%EOF")

        with self.assertRaises(PdfReadError):
            extract_text_from_pdf(corrupt_pdf)

    # -------------------------------------------------------------------------
    # 5. Encrypted / Password-Protected PDF Tests
    # -------------------------------------------------------------------------
    def test_encrypted_pdf_raises_pdf_read_error(self):
        """Password-protected PDF raises PdfReadError indicating password protection."""
        sample_path = os.path.join("data", "sample.pdf")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample PDF not found at {sample_path}")

        reader = PdfReader(sample_path)
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        writer.encrypt("super_secret_password_123")

        encrypted_path = os.path.join(self.temp_dir.name, "encrypted.pdf")
        with open(encrypted_path, "wb") as f:
            writer.write(f)

        with self.assertRaises(PdfReadError) as ctx:
            extract_text_from_pdf(encrypted_path)
        self.assertIn("password-protected", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 6. Unexpected Parser / Decoder Failure Translation Tests
    # -------------------------------------------------------------------------
    @patch("src.pdf_loader.PdfReader")
    def test_unexpected_reader_parser_failure_translated_to_pdf_loader_error(self, mock_reader_cls):
        """Unexpected decoder error (e.g. KeyError or IndexError) during PdfReader is translated to PDFLoaderError."""
        mock_reader_cls.side_effect = KeyError("Malformed internal font dictionary")

        sample_path = os.path.join("data", "sample.pdf")
        with self.assertRaises(PDFLoaderError) as ctx:
            extract_text_from_pdf(sample_path)

        self.assertIn("Failed to load or parse PDF file", str(ctx.exception))
        self.assertIn("Malformed internal font dictionary", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, KeyError)

    @patch("src.pdf_loader.PdfReader")
    def test_unexpected_page_extraction_failure_translated_to_pdf_loader_error(self, mock_reader_cls):
        """Unexpected decoder failure during extract_text is translated to PDFLoaderError with page number."""
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        mock_page = MagicMock()
        mock_page.extract_text.side_effect = ValueError("Decompression zlib stream error")
        mock_instance.pages = [mock_page]
        mock_reader_cls.return_value = mock_instance

        sample_path = os.path.join("data", "sample.pdf")
        with self.assertRaises(PDFLoaderError) as ctx:
            extract_text_from_pdf(sample_path)

        self.assertIn("Unexpected decoder failure on page 1", str(ctx.exception))
        self.assertIn("Decompression zlib stream error", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    @patch("src.pdf_loader.PdfReader")
    def test_unexpected_decryption_failure_translated_to_pdf_loader_error(self, mock_reader_cls):
        """Unexpected exception during decrypt call is translated to PDFLoaderError."""
        mock_instance = MagicMock()
        mock_instance.is_encrypted = True
        mock_instance.decrypt.side_effect = RuntimeError("Crypto provider initialization failed")
        mock_reader_cls.return_value = mock_instance

        sample_path = os.path.join("data", "sample.pdf")
        with self.assertRaises(PDFLoaderError) as ctx:
            extract_text_from_pdf(sample_path)

        self.assertIn("Failed to decrypt password-protected PDF", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()
