"""
DocuMind - Unit Tests for Document Utilities (Milestone 8, Phase 2, Step 1)
Tests content-addressable document ID derivation using SHA-256 cryptographic hashing,
format verification, input validation, and adoption across CLI and Streamlit entry points.
"""

import os
import re
import tempfile
import unittest
from pathlib import Path

from src.document_utils import compute_document_id, compute_document_id_from_file
import streamlit_app
import app


class TestComputeDocumentId(unittest.TestCase):
    """Tests compute_document_id hashing, validation, and formatting."""

    def test_identical_bytes_produce_identical_ids(self):
        """Identical byte payloads must always compute the exact same document ID (deterministic)."""
        bytes_1 = b"%PDF-1.4 fixed content for testing determinism"
        bytes_2 = b"%PDF-1.4 fixed content for testing determinism"
        id_1 = compute_document_id(bytes_1)
        id_2 = compute_document_id(bytes_2)
        self.assertEqual(id_1, id_2)

    def test_different_bytes_produce_different_ids(self):
        """Different byte payloads must produce distinct document IDs (collision resistance)."""
        bytes_a = b"%PDF-1.4 Document A content"
        bytes_b = b"%PDF-1.4 Document B content"
        id_a = compute_document_id(bytes_a)
        id_b = compute_document_id(bytes_b)
        self.assertNotEqual(id_a, id_b)

    def test_exact_format_and_length(self):
        """The document ID must follow 'doc_<64 lowercase hex characters>' (length 68)."""
        sample_bytes = b"%PDF-1.4 Sample document for format validation"
        doc_id = compute_document_id(sample_bytes)

        # Prefix check
        self.assertTrue(doc_id.startswith("doc_"))
        # Exact length: 4 ('doc_') + 64 (hex digest) = 68
        self.assertEqual(len(doc_id), 68)

        # Lowercase hexadecimal characters only
        hex_digest = doc_id[4:]
        self.assertRegex(hex_digest, r"^[0-9a-f]{64}$")
        self.assertEqual(hex_digest, hex_digest.lower())

    def test_invalid_types_raise_type_error(self):
        """Non-bytes inputs (str, int, list, dict, None, float) must raise TypeError."""
        invalid_inputs = [
            "string_instead_of_bytes",
            12345,
            None,
            ["sample", "bytes"],
            {"content": b"%PDF"},
            3.14159,
            True,
        ]
        for bad_input in invalid_inputs:
            with self.subTest(input_val=bad_input):
                with self.assertRaises(TypeError) as ctx:
                    compute_document_id(bad_input)  # type: ignore[arg-type]
                self.assertIn("must be bytes or bytearray", str(ctx.exception))

    def test_empty_bytes_raises_value_error(self):
        """Empty bytes or bytearray payloads must raise ValueError."""
        for empty_val in [b"", bytearray()]:
            with self.subTest(empty_input=empty_val):
                with self.assertRaises(ValueError) as ctx:
                    compute_document_id(empty_val)
                self.assertIn("empty byte payload", str(ctx.exception).lower())

    def test_bytearray_input_supported(self):
        """bytearray inputs are supported and yield identical output to bytes."""
        raw_bytes = b"%PDF-1.4 bytearray test stream"
        b_array = bytearray(raw_bytes)
        self.assertEqual(compute_document_id(raw_bytes), compute_document_id(b_array))


class TestComputeDocumentIdFromFile(unittest.TestCase):
    """Tests file-based document ID derivation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_computes_matching_id_from_disk_file(self):
        """Reading a file on disk matches compute_document_id called with the raw bytes."""
        test_file = os.path.join(self.temp_dir.name, "test_doc.pdf")
        content = b"%PDF-1.4 Binary content on disk"
        with open(test_file, "wb") as f:
            f.write(content)

        file_id = compute_document_id_from_file(test_file)
        direct_id = compute_document_id(content)
        self.assertEqual(file_id, direct_id)

    def test_accepts_path_object(self):
        """Accepts pathlib.Path instances in addition to strings."""
        test_file = Path(self.temp_dir.name) / "test_path.pdf"
        test_file.write_bytes(b"%PDF-1.4 Pathlib support test")

        file_id = compute_document_id_from_file(test_file)
        self.assertTrue(file_id.startswith("doc_"))

    def test_missing_file_raises_file_not_found(self):
        """Non-existent file path raises FileNotFoundError."""
        missing = os.path.join(self.temp_dir.name, "non_existent.pdf")
        with self.assertRaises(FileNotFoundError):
            compute_document_id_from_file(missing)

    def test_directory_path_raises_value_error(self):
        """Directory path raises ValueError."""
        with self.assertRaises(ValueError):
            compute_document_id_from_file(self.temp_dir.name)

    def test_invalid_path_type_raises_type_error(self):
        """Non-string/Path arguments raise TypeError."""
        for bad_path in [123, None, [self.temp_dir.name]]:
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(TypeError):
                    compute_document_id_from_file(bad_path)  # type: ignore[arg-type]


class TestCrossToolAdoption(unittest.TestCase):
    """Verifies CLI and Streamlit share the same helper and derive identical IDs."""

    def test_streamlit_app_uses_shared_helper(self):
        """streamlit_app.compute_document_id must reference the shared helper."""
        self.assertIs(streamlit_app.compute_document_id, compute_document_id)

    def test_cli_get_cli_document_id_matches_shared_helper(self):
        """app.get_cli_document_id must produce identical IDs to compute_document_id."""
        sample_path = os.path.join("data", "sample.pdf")
        if not os.path.exists(sample_path):
            self.skipTest("data/sample.pdf not available")

        with open(sample_path, "rb") as f:
            sample_bytes = f.read()

        cli_id = app.get_cli_document_id(sample_path)
        ui_id = streamlit_app.compute_document_id(sample_bytes)
        shared_id = compute_document_id(sample_bytes)

        self.assertEqual(cli_id, shared_id)
        self.assertEqual(ui_id, shared_id)
        self.assertEqual(cli_id, ui_id)

    def test_arbitrary_pdf_bytes_cross_tool_parity(self):
        """Any PDF file processed by CLI or Streamlit receives the exact same document ID."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_bytes = b"%PDF-1.4 Mock arbitrary cross-tool content"
            tmp.write(tmp_bytes)
            tmp_path = tmp.name

        try:
            cli_id = app.get_cli_document_id(tmp_path)
            ui_id = streamlit_app.compute_document_id(tmp_bytes)
            self.assertEqual(cli_id, ui_id)
            self.assertTrue(cli_id.startswith("doc_"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
