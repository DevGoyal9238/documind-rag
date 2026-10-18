"""
DocuMind - Unit Tests for Streamlit PDF Ingestion Logic
Tests document ID hashing, index verification, tempfile cleanup, defensive validations,
duplicate reuse, crash recovery, and state transitions using mocks.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from pypdf.errors import PdfReadError
from src.pdf_loader import NoTextExtractedError
from streamlit_app import (
    compute_document_id,
    sanitize_error_message,
    verify_existing_index,
    process_uploaded_pdf,
)


class TestComputeDocumentId(unittest.TestCase):
    """Tests cryptographic SHA-256 document ID derivation."""

    def test_compute_document_id_full_64_hex_length(self):
        sample_bytes = b"%PDF-1.4 sample content for documind test"
        doc_id = compute_document_id(sample_bytes)
        # Prefix 'doc_' (4 chars) + 64 hex characters = 68 characters total
        self.assertTrue(doc_id.startswith("doc_"))
        self.assertEqual(len(doc_id), 68)
        # Verify hex characters only after prefix
        hex_part = doc_id[4:]
        int(hex_part, 16)  # Should not raise ValueError

    def test_compute_document_id_deterministic(self):
        bytes_a = b"%PDF-1.4 fixed content"
        bytes_b = b"%PDF-1.4 fixed content"
        self.assertEqual(compute_document_id(bytes_a), compute_document_id(bytes_b))

    def test_compute_document_id_collision_resistance(self):
        bytes_a = b"%PDF-1.4 Document A"
        bytes_b = b"%PDF-1.4 Document B"
        self.assertNotEqual(compute_document_id(bytes_a), compute_document_id(bytes_b))

    def test_compute_document_id_empty_payload_raises(self):
        with self.assertRaises(ValueError):
            compute_document_id(b"")


class TestSanitizeErrorMessage(unittest.TestCase):
    """Tests redaction of secrets, keys, and tokens in error strings."""

    def test_redacts_google_api_key(self):
        fake_diagnostic_key = "AIza" + "SyMockSecretScannerSafePatternTest00"
        raw = f"Error connecting with {fake_diagnostic_key}"
        sanitized = sanitize_error_message(Exception(raw))
        self.assertNotIn(fake_diagnostic_key, sanitized)
        self.assertNotIn("AIza", sanitized)
        self.assertIn("[REDACTED_KEY]", sanitized)

    def test_redacts_query_parameter_secrets(self):
        raw = "Failed request to https://api.example.com?key=secret_12345&auth=token_6789"
        sanitized = sanitize_error_message(Exception(raw))
        self.assertNotIn("secret_12345", sanitized)
        self.assertNotIn("token_6789", sanitized)
        self.assertIn("key=[REDACTED]", sanitized)
        self.assertIn("auth=[REDACTED]", sanitized)

    def test_empty_exception_returns_class_name(self):
        self.assertEqual(sanitize_error_message(ValueError("")), "ValueError")


class TestVerifyExistingIndex(unittest.TestCase):
    """Tests ChromaDB index verification for completeness and integrity."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.doc_id = "doc_1234567890123456789012345678901234567890123456789012345678901234"

    def test_verify_complete_index_returns_true(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1", f"{self.doc_id}_chunk_2"],
            "metadatas": [
                {"chunk_id": 1, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 2, "page_number": 1, "document_id": self.doc_id},
            ],
            "documents": ["Text chunk 1", "Text chunk 2"],
        }
        is_valid = verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=2)
        self.assertTrue(is_valid)

    def test_verify_partial_index_detected_returns_false(self):
        # Expected 3 chunks, but database only has 2 chunks (e.g. from an interrupted run)
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1", f"{self.doc_id}_chunk_2"],
            "metadatas": [
                {"chunk_id": 1, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 2, "page_number": 1, "document_id": self.doc_id},
            ],
            "documents": ["Text chunk 1", "Text chunk 2"],
        }
        is_valid = verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=3)
        self.assertFalse(is_valid)

    def test_verify_gaps_in_chunk_ids_returns_false(self):
        # Database has 2 chunks, but IDs are 1 and 3 (chunk 2 is missing)
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1", f"{self.doc_id}_chunk_3"],
            "metadatas": [
                {"chunk_id": 1, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 3, "page_number": 2, "document_id": self.doc_id},
            ],
            "documents": ["Text chunk 1", "Text chunk 3"],
        }
        is_valid = verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=2)
        self.assertFalse(is_valid)

    def test_verify_empty_or_corrupt_text_returns_false(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": self.doc_id}],
            "documents": ["   "],  # Whitespace only
        }
        is_valid = verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1)
        self.assertFalse(is_valid)

    def test_verify_db_error_raises_runtime_error(self):
        self.mock_col.get.side_effect = RuntimeError("SQLite database is locked")
        with self.assertRaises(RuntimeError) as ctx:
            verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1)
        self.assertIn("Database read failure while verifying index", str(ctx.exception))


class TestProcessUploadedPDF(unittest.TestCase):
    """Tests end-to-end PDF processing with mocks for extraction, embedding, and storage."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.mock_model = MagicMock()
        self.valid_bytes = b"%PDF-1.4 test valid binary stream"
        self.filename = "study_guide.pdf"

    @patch("streamlit_app.store_chunks")
    @patch("streamlit_app.embed_chunks")
    @patch("streamlit_app.chunk_pages")
    @patch("streamlit_app.extract_text_from_pdf")
    def test_process_new_document_success(
        self, mock_extract, mock_chunk, mock_embed, mock_store
    ):
        mock_extract.return_value = [{"page_number": 1, "text": "Valid extracted text."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Valid extracted text."}
        ]
        mock_embed.return_value = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "text": "Valid extracted text.",
                "embedding": [0.1] * 384,
            }
        ]
        mock_store.return_value = 1
        self.mock_col.get.return_value = {"ids": []}  # Not in DB yet

        res = process_uploaded_pdf(
            pdf_bytes=self.valid_bytes,
            filename=self.filename,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertFalse(res["reused_existing"])
        self.assertEqual(res["page_count"], 1)
        self.assertEqual(res["chunk_count"], 1)
        self.assertTrue(res["document_id"].startswith("doc_"))
        mock_embed.assert_called_once()
        mock_store.assert_called_once()

    @patch("streamlit_app.store_chunks")
    @patch("streamlit_app.embed_chunks")
    @patch("streamlit_app.chunk_pages")
    @patch("streamlit_app.extract_text_from_pdf")
    def test_process_reuses_verified_existing_index(
        self, mock_extract, mock_chunk, mock_embed, mock_store
    ):
        mock_extract.return_value = [{"page_number": 1, "text": "Cached document text."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Cached document text."}
        ]
        doc_id = compute_document_id(self.valid_bytes)

        # Database already has the full verified index
        self.mock_col.get.return_value = {
            "ids": [f"{doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": doc_id}],
            "documents": ["Cached document text."],
        }

        res = process_uploaded_pdf(
            pdf_bytes=self.valid_bytes,
            filename=self.filename,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertTrue(res["reused_existing"])
        self.assertEqual(res["chunk_count"], 1)
        # Assert expensive embedding and upsert steps were skipped
        mock_embed.assert_not_called()
        mock_store.assert_not_called()

    @patch("streamlit_app.store_chunks")
    @patch("streamlit_app.embed_chunks")
    @patch("streamlit_app.chunk_pages")
    @patch("streamlit_app.extract_text_from_pdf")
    def test_process_recovers_from_partial_existing_index(
        self, mock_extract, mock_chunk, mock_embed, mock_store
    ):
        """When an incomplete index exists, it must be cleared for that document before re-indexing."""
        mock_extract.return_value = [{"page_number": 1, "text": "Two chunks doc."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Chunk 1"},
            {"chunk_id": 2, "page_number": 1, "text": "Chunk 2"},
        ]
        doc_id = compute_document_id(self.valid_bytes)

        # DB only has 1 chunk (partial write from crash)
        self.mock_col.get.return_value = {
            "ids": [f"{doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": doc_id}],
            "documents": ["Chunk 1"],
        }
        mock_embed.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Chunk 1", "embedding": [0.1] * 384},
            {"chunk_id": 2, "page_number": 1, "text": "Chunk 2", "embedding": [0.2] * 384},
        ]
        mock_store.return_value = 2

        res = process_uploaded_pdf(
            pdf_bytes=self.valid_bytes,
            filename=self.filename,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertFalse(res["reused_existing"])
        self.assertEqual(res["chunk_count"], 2)
        # Verify collection.delete was called to purge partial records
        self.mock_col.delete.assert_called_with(where={"document_id": doc_id})
        mock_embed.assert_called_once()
        mock_store.assert_called_once()

    def test_empty_bytes_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            process_uploaded_pdf(
                pdf_bytes=b"",
                filename="empty.pdf",
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )
        self.assertIn("empty", str(ctx.exception).lower())

    @patch("streamlit_app.extract_text_from_pdf")
    def test_no_text_extracted_raises_and_cleans_up(self, mock_extract):
        mock_extract.side_effect = NoTextExtractedError("Scanned image only")
        with self.assertRaises(NoTextExtractedError):
            process_uploaded_pdf(
                pdf_bytes=self.valid_bytes,
                filename="scanned.pdf",
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

    @patch("streamlit_app.extract_text_from_pdf")
    def test_pdf_read_error_raises_and_cleans_up(self, mock_extract):
        mock_extract.side_effect = PdfReadError("Corrupt PDF header")
        with self.assertRaises(PdfReadError):
            process_uploaded_pdf(
                pdf_bytes=self.valid_bytes,
                filename="corrupt.pdf",
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

    @patch("streamlit_app.embed_chunks")
    @patch("streamlit_app.chunk_pages")
    @patch("streamlit_app.extract_text_from_pdf")
    def test_embedding_failure_cleans_up(self, mock_extract, mock_chunk, mock_embed):
        mock_extract.return_value = [{"page_number": 1, "text": "Hello world"}]
        mock_chunk.return_value = [{"chunk_id": 1, "page_number": 1, "text": "Hello world"}]
        self.mock_col.get.return_value = {"ids": []}
        mock_embed.side_effect = RuntimeError("PyTorch out of memory")

        with self.assertRaises(RuntimeError):
            process_uploaded_pdf(
                pdf_bytes=self.valid_bytes,
                filename="large.pdf",
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

    @patch("streamlit_app.store_chunks")
    @patch("streamlit_app.embed_chunks")
    @patch("streamlit_app.chunk_pages")
    @patch("streamlit_app.extract_text_from_pdf")
    def test_process_uploaded_pdf_database_read_failure_aborts_without_deletion_or_embedding(
        self, mock_extract, mock_chunk, mock_embed, mock_store
    ):
        """
        When ChromaDB fails during verification read, process_uploaded_pdf must abort
        with RuntimeError and NOT call delete, embed, or store.
        """
        mock_extract.return_value = [{"page_number": 1, "text": "Valid extracted text."}]
        mock_chunk.return_value = [{"chunk_id": 1, "page_number": 1, "text": "Valid extracted text."}]
        self.mock_col.get.side_effect = RuntimeError("Database read connection lost")

        with self.assertRaises(RuntimeError) as ctx:
            process_uploaded_pdf(
                pdf_bytes=self.valid_bytes,
                filename=self.filename,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

        self.assertIn("Database read failure while verifying index", str(ctx.exception))
        self.mock_col.delete.assert_not_called()
        mock_embed.assert_not_called()
        mock_store.assert_not_called()

    @patch("streamlit_app.store_chunks")
    @patch("streamlit_app.embed_chunks")
    @patch("streamlit_app.chunk_pages")
    @patch("streamlit_app.extract_text_from_pdf")
    def test_process_uploaded_pdf_deletion_failure_aborts_without_embedding_or_storage(
        self, mock_extract, mock_chunk, mock_embed, mock_store
    ):
        """
        When document-scoped deletion fails, process_uploaded_pdf must abort immediately
        with RuntimeError and NOT proceed to embedding or storage.
        """
        mock_extract.return_value = [{"page_number": 1, "text": "Valid text."}]
        mock_chunk.return_value = [{"chunk_id": 1, "page_number": 1, "text": "Valid text."}]
        self.mock_col.get.return_value = {"ids": []}  # Needs rebuild
        self.mock_col.delete.side_effect = RuntimeError("ChromaDB permission denied on delete")

        with self.assertRaises(RuntimeError) as ctx:
            process_uploaded_pdf(
                pdf_bytes=self.valid_bytes,
                filename=self.filename,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

        self.assertIn("Failed to delete records for document", str(ctx.exception))
        mock_embed.assert_not_called()
        mock_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
