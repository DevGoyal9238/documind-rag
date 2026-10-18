"""
DocuMind - Unit Tests for CLI Index Verification & Safe Ingestion
Tests shared index verification, document-scoped purging, safe index reuse,
stale chunk prevention, and deletion failure recovery for the CLI pipeline.
"""

import io
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.vector_store import (
    verify_existing_index,
    delete_document_chunks,
)
from app import process_cli_document, get_cli_document_id, main as cli_main, ChromaError


class TestVerifyExistingIndexShared(unittest.TestCase):
    """Tests shared ChromaDB index verification logic across entry points."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.doc_id = "doc_1234567890123456789012345678901234567890123456789012345678901234"

    def test_verify_valid_complete_index_returns_true(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1", f"{self.doc_id}_chunk_2"],
            "metadatas": [
                {"chunk_id": 1, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 2, "page_number": 1, "document_id": self.doc_id},
            ],
            "documents": ["Text chunk 1", "Text chunk 2"],
        }
        self.assertTrue(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=2))

    def test_verify_rejects_none_or_invalid_collection(self):
        self.assertFalse(verify_existing_index(None, self.doc_id, expected_chunk_count=2))

    def test_verify_rejects_invalid_document_id(self):
        self.assertFalse(verify_existing_index(self.mock_col, "", expected_chunk_count=2))
        self.assertFalse(verify_existing_index(self.mock_col, "   ", expected_chunk_count=2))
        self.assertFalse(verify_existing_index(self.mock_col, None, expected_chunk_count=2))  # type: ignore

    def test_verify_rejects_non_positive_expected_count(self):
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=0))
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=-1))
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=True))  # type: ignore

    def test_verify_rejects_count_mismatch_too_few(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": self.doc_id}],
            "documents": ["Text chunk 1"],
        }
        # Expected 2 chunks, but only 1 exists
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=2))

    def test_verify_rejects_count_mismatch_too_many(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1", f"{self.doc_id}_chunk_2", f"{self.doc_id}_chunk_3"],
            "metadatas": [
                {"chunk_id": 1, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 2, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 3, "page_number": 2, "document_id": self.doc_id},
            ],
            "documents": ["Text 1", "Text 2", "Text 3"],
        }
        # Expected 2 chunks, but 3 exist in DB
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=2))

    def test_verify_rejects_wrong_chunk_ids_gap(self):
        # Count matches (2), but chunk_ids are 1 and 3 (gap)
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1", f"{self.doc_id}_chunk_3"],
            "metadatas": [
                {"chunk_id": 1, "page_number": 1, "document_id": self.doc_id},
                {"chunk_id": 3, "page_number": 2, "document_id": self.doc_id},
            ],
            "documents": ["Text 1", "Text 3"],
        }
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=2))

    def test_verify_rejects_wrong_chunk_ids_bad_prefix(self):
        # Stored ID doesn't match f"{document_id}_chunk_{i}"
        self.mock_col.get.return_value = {
            "ids": ["wrong_prefix_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": self.doc_id}],
            "documents": ["Text 1"],
        }
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1))

    def test_verify_rejects_empty_or_whitespace_text(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": self.doc_id}],
            "documents": ["    \n\t  "],
        }
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1))

    def test_verify_rejects_wrong_document_id_in_metadata(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": "doc_other_polluted_id"}],
            "documents": ["Text 1"],
        }
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1))

    def test_verify_rejects_non_dict_metadata(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1"],
            "metadatas": ["not_a_dict"],
            "documents": ["Text 1"],
        }
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1))

    def test_verify_rejects_invalid_page_number(self):
        self.mock_col.get.return_value = {
            "ids": [f"{self.doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": -5, "document_id": self.doc_id}],
            "documents": ["Text 1"],
        }
        self.assertFalse(verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1))

    def test_verify_db_error_raises_runtime_error(self):
        self.mock_col.get.side_effect = RuntimeError("Disk I/O failure")
        with self.assertRaises(RuntimeError) as ctx:
            verify_existing_index(self.mock_col, self.doc_id, expected_chunk_count=1)
        self.assertIn("Database read failure while verifying index", str(ctx.exception))


class TestDeleteDocumentChunks(unittest.TestCase):
    """Tests document-scoped deletion behavior and error handling."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.doc_id = "doc_aabbccddeeff11223344556677889900aabbccddeeff11223344556677889900"

    def test_delete_document_chunks_scoped_by_document_id(self):
        delete_document_chunks(self.mock_col, self.doc_id)
        self.mock_col.delete.assert_called_once_with(where={"document_id": self.doc_id})

    def test_delete_rejects_none_collection(self):
        with self.assertRaises(TypeError):
            delete_document_chunks(None, self.doc_id)

    def test_delete_rejects_invalid_or_empty_document_id(self):
        with self.assertRaises(TypeError):
            delete_document_chunks(self.mock_col, 12345)  # type: ignore
        with self.assertRaises(ValueError):
            delete_document_chunks(self.mock_col, "")
        with self.assertRaises(ValueError):
            delete_document_chunks(self.mock_col, "   ")

    def test_delete_db_error_raises_runtime_error(self):
        self.mock_col.delete.side_effect = Exception("SQLite read-only database")
        with self.assertRaises(RuntimeError) as ctx:
            delete_document_chunks(self.mock_col, self.doc_id)
        self.assertIn(f"Failed to delete records for document '{self.doc_id}'", str(ctx.exception))


class TestProcessCliDocument(unittest.TestCase):
    """Tests CLI document ingestion orchestration, index reuse, and recovery."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.mock_model = MagicMock()
        self.pdf_path = "data/sample.pdf"
        self.dummy_bytes = b"%PDF-1.4 synthetic binary stream for cli testing"

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    @patch("app.embed_chunks")
    @patch("app.store_chunks")
    def test_cli_valid_existing_index_skips_embedding_and_storage(
        self, mock_store, mock_embed, mock_chunk, mock_extract, mock_get_id
    ):
        doc_id = "doc_test123456789012345678901234567890123456789012345678901234567890"
        mock_get_id.return_value = doc_id
        mock_extract.return_value = [{"page_number": 1, "text": "Extracted text content."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Extracted text content."}
        ]
        # Simulate verified index in ChromaDB
        self.mock_col.get.return_value = {
            "ids": [f"{doc_id}_chunk_1"],
            "metadatas": [{"chunk_id": 1, "page_number": 1, "document_id": doc_id}],
            "documents": ["Extracted text content."],
        }

        result = process_cli_document(
            pdf_path=self.pdf_path,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertTrue(result["reused_existing"])
        self.assertEqual(result["document_id"], doc_id)
        self.assertEqual(result["chunk_count"], 1)
        self.assertIsNone(result["embedded_chunks"])
        # Embedding and ChromaDB storage MUST be skipped
        mock_embed.assert_not_called()
        mock_store.assert_not_called()
        self.mock_col.delete.assert_not_called()

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    @patch("app.embed_chunks")
    @patch("app.store_chunks")
    def test_cli_missing_index_triggers_ingestion(
        self, mock_store, mock_embed, mock_chunk, mock_extract, mock_get_id
    ):
        doc_id = "doc_test123456789012345678901234567890123456789012345678901234567890"
        mock_get_id.return_value = doc_id
        mock_extract.return_value = [{"page_number": 1, "text": "New document content."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "New document content."}
        ]
        # Collection has no records for this document
        self.mock_col.get.return_value = {"ids": []}
        mock_embed.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "New document content.", "embedding": [0.1] * 384}
        ]
        mock_store.return_value = 1

        result = process_cli_document(
            pdf_path=self.pdf_path,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertFalse(result["reused_existing"])
        self.assertEqual(result["chunk_count"], 1)
        self.assertIsNotNone(result["embedded_chunks"])
        # Verifies document-scoped purge was called before writing
        self.mock_col.delete.assert_called_once_with(where={"document_id": doc_id})
        mock_embed.assert_called_once()
        mock_store.assert_called_once()

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    @patch("app.embed_chunks")
    @patch("app.store_chunks")
    def test_cli_partial_index_purged_before_rebuilding(
        self, mock_store, mock_embed, mock_chunk, mock_extract, mock_get_id
    ):
        doc_id = "doc_test123456789012345678901234567890123456789012345678901234567890"
        mock_get_id.return_value = doc_id
        mock_extract.return_value = [{"page_number": 1, "text": "Page text."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Chunk 1"},
            {"chunk_id": 2, "page_number": 1, "text": "Chunk 2"},
        ]
        # DB only has 1 chunk out of 2 (partial index from interrupted run)
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

        result = process_cli_document(
            pdf_path=self.pdf_path,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertFalse(result["reused_existing"])
        self.assertEqual(result["chunk_count"], 2)
        # Verify partial index was purged before re-embedding
        self.mock_col.delete.assert_called_once_with(where={"document_id": doc_id})
        mock_embed.assert_called_once()
        mock_store.assert_called_once()

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    @patch("app.embed_chunks")
    @patch("app.store_chunks")
    def test_cli_stale_chunks_purged_when_reindexing(
        self, mock_store, mock_embed, mock_chunk, mock_extract, mock_get_id
    ):
        """
        Simulates scenario where ChromaDB had 4 chunks from an older version,
        but the new chunking produces 2 chunks. Verification must fail (count mismatch 4 != 2),
        triggering document-scoped purge so that extra stale chunks (chunks 3 and 4) cannot remain.
        """
        doc_id = "doc_test123456789012345678901234567890123456789012345678901234567890"
        mock_get_id.return_value = doc_id
        mock_extract.return_value = [{"page_number": 1, "text": "Shorter text."}]
        mock_chunk.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Chunk 1"},
            {"chunk_id": 2, "page_number": 1, "text": "Chunk 2"},
        ]
        # DB has 4 stale chunks
        self.mock_col.get.return_value = {
            "ids": [f"{doc_id}_chunk_{i}" for i in range(1, 5)],
            "metadatas": [
                {"chunk_id": i, "page_number": 1, "document_id": doc_id} for i in range(1, 5)
            ],
            "documents": [f"Old chunk {i}" for i in range(1, 5)],
        }
        mock_embed.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Chunk 1", "embedding": [0.1] * 384},
            {"chunk_id": 2, "page_number": 1, "text": "Chunk 2", "embedding": [0.2] * 384},
        ]
        mock_store.return_value = 2

        result = process_cli_document(
            pdf_path=self.pdf_path,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertFalse(result["reused_existing"])
        # Assert delete was called to purge the 4 stale records
        self.mock_col.delete.assert_called_once_with(where={"document_id": doc_id})
        mock_embed.assert_called_once()
        mock_store.assert_called_once()

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    @patch("app.embed_chunks")
    @patch("app.store_chunks")
    def test_cli_deletion_failure_prevents_embedding_and_storage(
        self, mock_store, mock_embed, mock_chunk, mock_extract, mock_get_id
    ):
        """
        When document-scoped deletion fails, process_cli_document must abort immediately
        and NOT proceed to embedding or storage.
        """
        doc_id = "doc_test123456789012345678901234567890123456789012345678901234567890"
        mock_get_id.return_value = doc_id
        mock_extract.return_value = [{"page_number": 1, "text": "Text."}]
        mock_chunk.return_value = [{"chunk_id": 1, "page_number": 1, "text": "Text."}]
        self.mock_col.get.return_value = {"ids": []}  # Needs rebuild
        # Simulate ChromaDB deletion failure
        self.mock_col.delete.side_effect = RuntimeError("ChromaDB locked or read-only")

        with self.assertRaises(RuntimeError) as ctx:
            process_cli_document(
                pdf_path=self.pdf_path,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

        self.assertIn("Failed to delete records for document", str(ctx.exception))
        # Embedding and storage MUST NOT be called after failed purge
        mock_embed.assert_not_called()
        mock_store.assert_not_called()

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    def test_cli_zero_chunks_raises_value_error(
        self, mock_chunk, mock_extract, mock_get_id
    ):
        mock_get_id.return_value = "doc_test"
        mock_extract.return_value = [{"page_number": 1, "text": ""}]
        mock_chunk.return_value = []

        with self.assertRaises(ValueError) as ctx:
            process_cli_document(
                pdf_path=self.pdf_path,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )
        self.assertIn("0 chunks", str(ctx.exception))

    def test_cli_none_collection_raises_type_error(self):
        with self.assertRaises(TypeError):
            process_cli_document(
                pdf_path=self.pdf_path,
                collection=None,
                embedding_model=self.mock_model,
            )

    @patch("app.get_cli_document_id")
    @patch("app.extract_text_from_pdf")
    @patch("app.chunk_pages")
    @patch("app.embed_chunks")
    @patch("app.store_chunks")
    def test_cli_read_failure_aborts_without_deletion_or_embedding(
        self, mock_store, mock_embed, mock_chunk, mock_extract, mock_get_id
    ):
        """
        When ChromaDB fails during verification read, process_cli_document must abort
        immediately with RuntimeError and NOT call delete, embed, or store.
        """
        doc_id = "doc_test123456789012345678901234567890123456789012345678901234567890"
        mock_get_id.return_value = doc_id
        mock_extract.return_value = [{"page_number": 1, "text": "Some text."}]
        mock_chunk.return_value = [{"chunk_id": 1, "page_number": 1, "text": "Some text."}]
        self.mock_col.get.side_effect = RuntimeError("ChromaDB read connection timeout")

        with self.assertRaises(RuntimeError) as ctx:
            process_cli_document(
                pdf_path=self.pdf_path,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )

        self.assertIn("Database read failure while verifying index", str(ctx.exception))
        self.mock_col.delete.assert_not_called()
        mock_embed.assert_not_called()
        mock_store.assert_not_called()


class TestCliMainErrorHandling(unittest.TestCase):
    """Tests top-level CLI error handling, database error categorization, and sanitization."""

    @patch("app.get_chroma_client")
    def test_cli_main_catches_chroma_error_and_sanitizes(self, mock_get_client):
        secret_key = "AIzaSyFakeKey1234567890123456789012345"
        mock_get_client.side_effect = ChromaError(
            f"Connection failed for {secret_key} and token=query_secret_999"
        )

        captured = io.StringIO()
        with patch.object(sys, "argv", ["app.py", "dummy.pdf"]):
            with patch("sys.stdout", captured):
                cli_main()

        output = captured.getvalue()
        self.assertIn("[Database Error]", output)
        self.assertNotIn(secret_key, output)
        self.assertNotIn("query_secret_999", output)
        self.assertIn("[REDACTED_KEY]", output)
        self.assertIn("token=[REDACTED]", output)

    @patch("app.get_chroma_client")
    @patch("app.get_or_create_collection")
    @patch("app.process_cli_document")
    def test_cli_main_catches_runtime_db_error_and_sanitizes(
        self, mock_process, mock_get_col, mock_get_client
    ):
        mock_process.side_effect = RuntimeError(
            "Database read failure: query auth failed with bearer secret_token_xyz999"
        )

        captured = io.StringIO()
        with patch.object(sys, "argv", ["app.py", "dummy.pdf"]):
            with patch("sys.stdout", captured):
                cli_main()

        output = captured.getvalue()
        self.assertIn("[Database Error]", output)
        self.assertNotIn("secret_token_xyz999", output)
        self.assertIn("Bearer [REDACTED]", output)


if __name__ == "__main__":
    unittest.main()
