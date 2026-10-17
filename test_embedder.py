"""
DocuMind - Unit and Integration Tests for Text Embedding Module
Tests input validation, batch encoding, metadata preservation, model caching,
and real model inference.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.embedder import (
    embed_chunks,
    get_embedding_model,
    _MODEL_CACHE,
    EMBEDDING_DIMENSION,
    DEFAULT_MODEL_NAME,
)


class MockSentenceTransformer:
    """Mock model double that simulates SentenceTransformer without downloading weights."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.encode_calls = 0
        self.last_texts: list[str] | None = None

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        self.encode_calls += 1
        self.last_texts = texts
        # Return a deterministic 2D array of floats with shape (len(texts), dimension)
        return np.full((len(texts), self.dimension), 0.1234, dtype=np.float32)


class TestEmbedderWithMock(unittest.TestCase):
    """Unit tests using MockSentenceTransformer (fast, offline, zero network)."""

    def setUp(self):
        self.mock_model = MockSentenceTransformer(dimension=384)
        self.sample_chunks = [
            {"chunk_id": 1, "page_number": 1, "text": "First chunk text."},
            {"chunk_id": 2, "page_number": 1, "text": "Second chunk text."},
        ]

    def test_empty_chunks_returns_empty_list(self):
        """Empty input list should immediately return [] without invoking the model."""
        result = embed_chunks([], model=self.mock_model)
        self.assertEqual(result, [])
        self.assertEqual(self.mock_model.encode_calls, 0)

    def test_metadata_preservation(self):
        """Original chunk_id, page_number, and text must be preserved exactly."""
        result = embed_chunks(self.sample_chunks, model=self.mock_model)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chunk_id"], 1)
        self.assertEqual(result[0]["page_number"], 1)
        self.assertEqual(result[0]["text"], "First chunk text.")
        self.assertEqual(result[1]["chunk_id"], 2)
        self.assertEqual(result[1]["page_number"], 1)
        self.assertEqual(result[1]["text"], "Second chunk text.")

    def test_embedding_structure_and_dimensions(self):
        """Embedding field must be a standard Python list of floats with dimension 384."""
        result = embed_chunks(self.sample_chunks, model=self.mock_model)

        for chunk in result:
            self.assertIn("embedding", chunk)
            self.assertIsInstance(chunk["embedding"], list)
            self.assertEqual(len(chunk["embedding"]), EMBEDDING_DIMENSION)
            self.assertTrue(all(isinstance(x, float) for x in chunk["embedding"]))

    def test_batch_encoding_called_once(self):
        """Model encode must be called exactly once for the whole batch, not per chunk."""
        embed_chunks(self.sample_chunks, model=self.mock_model)

        self.assertEqual(self.mock_model.encode_calls, 1)
        self.assertEqual(
            self.mock_model.last_texts,
            ["First chunk text.", "Second chunk text."],
        )

    def test_malformed_input_type(self):
        """Passing a non-list as chunks must raise TypeError."""
        with self.assertRaises(TypeError):
            embed_chunks("not a list", model=self.mock_model)
        with self.assertRaises(TypeError):
            embed_chunks({"chunk_id": 1}, model=self.mock_model)

    def test_malformed_chunk_entries(self):
        """Chunks that are not dictionaries must raise TypeError."""
        with self.assertRaises(TypeError):
            embed_chunks(["string chunk"], model=self.mock_model)

    def test_missing_required_keys(self):
        """Missing required chunk keys must raise ValueError."""
        # Missing chunk_id
        with self.assertRaises(ValueError):
            embed_chunks([{"page_number": 1, "text": "abc"}], model=self.mock_model)

        # Missing page_number
        with self.assertRaises(ValueError):
            embed_chunks([{"chunk_id": 1, "text": "abc"}], model=self.mock_model)

        # Missing text
        with self.assertRaises(ValueError):
            embed_chunks([{"chunk_id": 1, "page_number": 1}], model=self.mock_model)

    def test_invalid_field_data_types(self):
        """Invalid types for chunk_id, page_number, or text must raise TypeError."""
        # Boolean chunk_id
        with self.assertRaises(TypeError):
            embed_chunks([{"chunk_id": True, "page_number": 1, "text": "abc"}], model=self.mock_model)

        # String page_number
        with self.assertRaises(TypeError):
            embed_chunks([{"chunk_id": 1, "page_number": "1", "text": "abc"}], model=self.mock_model)

        # Non-string text
        with self.assertRaises(TypeError):
            embed_chunks([{"chunk_id": 1, "page_number": 1, "text": 12345}], model=self.mock_model)

    def test_zero_and_negative_chunk_ids(self):
        """Chunk IDs <= 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            embed_chunks([{"chunk_id": 0, "page_number": 1, "text": "Valid text"}], model=self.mock_model)
        with self.assertRaises(ValueError):
            embed_chunks([{"chunk_id": -1, "page_number": 1, "text": "Valid text"}], model=self.mock_model)

    def test_zero_and_negative_page_numbers(self):
        """Page numbers <= 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            embed_chunks([{"chunk_id": 1, "page_number": 0, "text": "Valid text"}], model=self.mock_model)
        with self.assertRaises(ValueError):
            embed_chunks([{"chunk_id": 1, "page_number": -3, "text": "Valid text"}], model=self.mock_model)

    def test_fewer_model_outputs_than_inputs(self):
        """Model returning fewer embeddings than input chunks must raise ValueError."""
        class MismatchCountModel:
            def encode(self, texts, show_progress_bar=False):
                # Returns only 1 embedding for 2 inputs
                return np.full((len(texts) - 1, 384), 0.1, dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            embed_chunks(self.sample_chunks, model=MismatchCountModel())
        self.assertIn("Model output count mismatch", str(ctx.exception))

    def test_incorrect_embedding_dimension(self):
        """Model returning embeddings with dimension != 384 must raise ValueError."""
        bad_dim_model = MockSentenceTransformer(dimension=128)
        with self.assertRaises(ValueError) as ctx:
            embed_chunks(self.sample_chunks, model=bad_dim_model)
        self.assertIn("Embedding dimension mismatch", str(ctx.exception))

    def test_model_cache_reusability(self):
        """Calling get_embedding_model repeatedly with the same name returns the cached instance."""
        fake_model = MagicMock()
        model_name = "test-mock-cache-model"

        # Populate cache manually
        _MODEL_CACHE[model_name] = fake_model
        try:
            m1 = get_embedding_model(model_name)
            m2 = get_embedding_model(model_name)
            self.assertIs(m1, fake_model)
            self.assertIs(m1, m2)
        finally:
            _MODEL_CACHE.pop(model_name, None)

    @patch("src.embedder.SentenceTransformer")
    def test_model_constructor_called_once_on_cache_miss(self, mock_transformer_cls):
        """A cache miss must invoke SentenceTransformer constructor once; subsequent calls reuse it."""
        unique_name = "test-unique-cache-miss-model"
        _MODEL_CACHE.pop(unique_name, None)
        try:
            m1 = get_embedding_model(unique_name)
            m2 = get_embedding_model(unique_name)
            # Constructor was called only once despite two retrieval calls
            mock_transformer_cls.assert_called_once_with(unique_name)
            self.assertIs(m1, m2)
        finally:
            _MODEL_CACHE.pop(unique_name, None)


class TestEmbedderRealModelSmoke(unittest.TestCase):
    """
    Integration smoke test using the actual all-MiniLM-L6-v2 pretrained model.
    
    Requirements:
        - Requires model weights cached locally (~/.cache/huggingface/hub)
          or an active internet connection on the initial run to download ~80MB weights.
        - Executes genuine neural network inference on CPU to verify real-world pipeline integrity.
    """

    def test_real_model_inference(self):
        """Verify real sentence-transformers model produces 384-dimensional float embeddings."""
        sample_chunk = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "text": "DocuMind is an open-source Python RAG system for document intelligence.",
            }
        ]

        result = embed_chunks(sample_chunk, model_name=DEFAULT_MODEL_NAME)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chunk_id"], 1)
        self.assertEqual(result[0]["page_number"], 1)
        self.assertEqual(result[0]["text"], sample_chunk[0]["text"])

        embedding = result[0]["embedding"]
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

        # Real model embeddings should not be all zeros
        self.assertTrue(any(x != 0.0 for x in embedding))


if __name__ == "__main__":
    unittest.main()
