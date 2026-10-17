"""
DocuMind - Unit and Integration Tests for Semantic Search Module
Tests cosine similarity computation, query validation, top_k ranking,
immutability of input chunks, shape validation, and real-model semantic retrieval.
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Any
import numpy as np

from src.semantic_search import (
    search_chunks,
    compute_cosine_similarity,
)
from src.embedder import (
    _MODEL_CACHE,
    EMBEDDING_DIMENSION,
    DEFAULT_MODEL_NAME,
)


class MockSentenceTransformer:
    """Mock model double that produces deterministic embeddings for testing without network/weights."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.encode_calls = 0

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        self.encode_calls += 1
        # Generates deterministic unit vectors based on text content with shape (len(texts), dimension)
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            if "python" in text.lower():
                vec[0] = 1.0  # Points along dimension 0
            elif "architecture" in text.lower():
                vec[1] = 1.0  # Points along dimension 1
            else:
                vec[2] = 1.0  # Points along dimension 2
            vectors.append(vec)
        return np.array(vectors)


class CustomOutputMockModel:
    """Mock model that returns custom outputs for testing malformed shapes, NaNs, and types."""

    def __init__(self, output: Any):
        self.output = output

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> Any:
        return self.output


class TestSemanticSearchUnit(unittest.TestCase):
    """Deterministic unit tests using mock embeddings and known vectors (zero network)."""

    def setUp(self):
        self.mock_model = MockSentenceTransformer(dimension=384)

        # Vector 1: aligned with dimension 0 ("python")
        vec_python = [0.0] * 384
        vec_python[0] = 1.0

        # Vector 2: aligned with dimension 1 ("architecture")
        vec_arch = [0.0] * 384
        vec_arch[1] = 1.0

        # Vector 3: aligned with dimension 2 ("other")
        vec_other = [0.0] * 384
        vec_other[2] = 1.0

        self.sample_chunks = [
            {"chunk_id": 1, "page_number": 1, "text": "Python programming language.", "embedding": vec_python},
            {"chunk_id": 2, "page_number": 2, "text": "DocuMind software architecture.", "embedding": vec_arch},
            {"chunk_id": 3, "page_number": 2, "text": "General topics and notes.", "embedding": vec_other},
        ]

    def test_empty_chunk_list_returns_empty_results(self):
        """1. Empty chunk list returns an empty list without encoding or loading a model."""
        results = search_chunks("python", [], top_k=3, model=self.mock_model)
        self.assertEqual(results, [])
        self.assertEqual(self.mock_model.encode_calls, 0)

    def test_most_similar_chunk_ranks_first(self):
        """2. Query matching 'python' must return Chunk 1 with similarity 1.0 at the top rank."""
        results = search_chunks("Python questions", self.sample_chunks, top_k=3, model=self.mock_model)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["chunk_id"], 1)
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=4)
        # Chunks orthogonal to dimension 0 should score 0.0
        self.assertAlmostEqual(results[1]["similarity_score"], 0.0, places=4)
        self.assertAlmostEqual(results[2]["similarity_score"], 0.0, places=4)

    def test_top_k_limits_results(self):
        """3. top_k=2 must return exactly 2 results when 3 chunks are present."""
        results = search_chunks("architecture", self.sample_chunks, top_k=2, model=self.mock_model)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], 2)  # Architecture chunk matches best

    def test_cosine_scores_identical_orthogonal_opposite(self):
        """4. Correct cosine scores for identical (1.0), orthogonal (0.0), and opposite (-1.0) 384D vectors."""
        v1 = [0.0] * 384
        v1[0] = 1.0

        v_identical = [0.0] * 384
        v_identical[0] = 1.0
        self.assertAlmostEqual(compute_cosine_similarity(v1, v_identical), 1.0, places=5)

        v_orthogonal = [0.0] * 384
        v_orthogonal[1] = 1.0
        self.assertAlmostEqual(compute_cosine_similarity(v1, v_orthogonal), 0.0, places=5)

        v_opposite = [0.0] * 384
        v_opposite[0] = -1.0
        self.assertAlmostEqual(compute_cosine_similarity(v1, v_opposite), -1.0, places=5)

    def test_zero_magnitude_vectors_rejected(self):
        """5. Rejection of zero-magnitude vectors with a clear ValueError."""
        v_valid = [0.0] * 384
        v_valid[0] = 1.0

        v_zero = [0.0] * 384
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_zero)
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_zero, v_valid)
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_zero, v_zero)

    def test_malformed_and_2d_vectors_rejected(self):
        """6. Rejection of two-dimensional or otherwise malformed vectors."""
        v_valid = [0.0] * 384
        v_valid[0] = 1.0

        v_2d = np.zeros((1, 384))
        v_2d[0, 0] = 1.0
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_2d)

        v_matrix = np.zeros((2, 192))
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_matrix, v_valid)

    def test_mismatched_and_incorrect_dimensions_rejected(self):
        """7. Rejection of mismatched or incorrect dimensions."""
        v_valid = [0.0] * 384
        v_valid[0] = 1.0

        v_short = [1.0] * 383
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_short)

        v_100 = [1.0] * 100
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_100, v_100)

        v_long = [1.0] * 500
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_long)

    def test_non_numeric_and_boolean_vectors_rejected(self):
        """8. Rejection of non-numeric values and booleans in vectors."""
        v_valid = [0.0] * 384
        v_valid[0] = 1.0

        v_bool = [True] * 384
        with self.assertRaises(TypeError):
            compute_cosine_similarity(v_valid, v_bool)
        with self.assertRaises(TypeError):
            compute_cosine_similarity(v_bool, v_valid)

        v_bool_arr = np.ones(384, dtype=bool)
        with self.assertRaises(TypeError):
            compute_cosine_similarity(v_valid, v_bool_arr)

        v_str = ["hello"] * 384
        with self.assertRaises(TypeError):
            compute_cosine_similarity(v_valid, v_str)

        with self.assertRaises(TypeError):
            compute_cosine_similarity("not a list", v_valid)

    def test_nan_and_infinity_vectors_rejected(self):
        """9. Rejection of NaN and positive/negative infinity."""
        v_valid = [0.0] * 384
        v_valid[0] = 1.0

        v_nan = [0.0] * 384
        v_nan[0] = float("nan")
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_nan)

        v_inf = [0.0] * 384
        v_inf[0] = float("inf")
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_inf)

        v_neginf = [0.0] * 384
        v_neginf[0] = float("-inf")
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v_valid, v_neginf)

    def test_query_embedding_wrong_shapes_rejected(self):
        """10. Query embeddings with wrong shapes: 1D (384,), (1, 383), and (2, 384) rejected."""
        # 1-dimensional 384-element output: shape == (384,)
        model_1d = CustomOutputMockModel(np.ones(384, dtype=np.float32))
        with self.assertRaises(ValueError) as ctx1:
            search_chunks("test query", self.sample_chunks, model=model_1d)
        self.assertIn("Query embedding output shape mismatch", str(ctx1.exception))

        # (1, 383) output: wrong dimension
        model_short = CustomOutputMockModel(np.ones((1, 383), dtype=np.float32))
        with self.assertRaises(ValueError) as ctx2:
            search_chunks("test query", self.sample_chunks, model=model_short)
        self.assertIn("Query embedding output shape mismatch", str(ctx2.exception))

        # (2, 384) output: multiple vectors instead of single vector
        model_multi = CustomOutputMockModel(np.ones((2, 384), dtype=np.float32))
        with self.assertRaises(ValueError) as ctx3:
            search_chunks("test query", self.sample_chunks, model=model_multi)
        self.assertIn("Query embedding output shape mismatch", str(ctx3.exception))

    def test_query_embedding_nan_inf_zero_magnitude_rejected(self):
        """11. Query embeddings containing NaN, infinity, or zero magnitude are rejected."""
        # NaN in query embedding
        nan_arr = np.ones((1, 384), dtype=np.float32)
        nan_arr[0, 0] = float("nan")
        model_nan = CustomOutputMockModel(nan_arr)
        with self.assertRaises(ValueError):
            search_chunks("test query", self.sample_chunks, model=model_nan)

        # Inf in query embedding
        inf_arr = np.ones((1, 384), dtype=np.float32)
        inf_arr[0, 0] = float("inf")
        model_inf = CustomOutputMockModel(inf_arr)
        with self.assertRaises(ValueError):
            search_chunks("test query", self.sample_chunks, model=model_inf)

        # Zero magnitude query embedding
        zero_arr = np.zeros((1, 384), dtype=np.float32)
        model_zero = CustomOutputMockModel(zero_arr)
        with self.assertRaises(ValueError):
            search_chunks("test query", self.sample_chunks, model=model_zero)

        # Boolean query embedding
        bool_arr = np.ones((1, 384), dtype=bool)
        model_bool = CustomOutputMockModel(bool_arr)
        with self.assertRaises(TypeError):
            search_chunks("test query", self.sample_chunks, model=model_bool)

    def test_query_embedding_inference_error_propagates(self):
        """12. Genuine model loading and inference errors propagate uncaught."""
        class FailingModel:
            def encode(self, texts: list[str], show_progress_bar: bool = False):
                raise RuntimeError("Inference hardware out of memory")

        with self.assertRaises(RuntimeError) as ctx:
            search_chunks("test query", self.sample_chunks, model=FailingModel())
        self.assertIn("Inference hardware out of memory", str(ctx.exception))

    def test_empty_or_whitespace_query_rejected(self):
        """13. Empty or whitespace-only queries must raise ValueError; non-strings raise TypeError."""
        with self.assertRaises(ValueError):
            search_chunks("", self.sample_chunks, model=self.mock_model)
        with self.assertRaises(ValueError):
            search_chunks("   \n\t  ", self.sample_chunks, model=self.mock_model)
        with self.assertRaises(TypeError):
            search_chunks(None, self.sample_chunks, model=self.mock_model)
        with self.assertRaises(TypeError):
            search_chunks(12345, self.sample_chunks, model=self.mock_model)

    def test_invalid_top_k_rejected(self):
        """14. top_k <= 0, booleans, floats, or strings must raise exceptions."""
        with self.assertRaises(ValueError):
            search_chunks("query", self.sample_chunks, top_k=0, model=self.mock_model)
        with self.assertRaises(ValueError):
            search_chunks("query", self.sample_chunks, top_k=-2, model=self.mock_model)
        with self.assertRaises(TypeError):
            search_chunks("query", self.sample_chunks, top_k=True, model=self.mock_model)
        with self.assertRaises(TypeError):
            search_chunks("query", self.sample_chunks, top_k=False, model=self.mock_model)
        with self.assertRaises(TypeError):
            search_chunks("query", self.sample_chunks, top_k="3", model=self.mock_model)
        with self.assertRaises(TypeError):
            search_chunks("query", self.sample_chunks, top_k=2.5, model=self.mock_model)

    def test_malformed_chunk_records_rejected(self):
        """15. Chunks missing keys or having invalid identifier types must raise clear errors."""
        # Not a list
        with self.assertRaises(TypeError):
            search_chunks("query", "not a list", model=self.mock_model)

        # Item not a dict
        with self.assertRaises(TypeError):
            search_chunks("query", ["not a dict"], model=self.mock_model)

        # Missing required keys
        with self.assertRaises(ValueError):
            search_chunks("query", [{"chunk_id": 1, "text": "t", "embedding": [0.1]*384}], model=self.mock_model)

        # Non-positive chunk_id
        with self.assertRaises(ValueError):
            bad_chunk = {"chunk_id": 0, "page_number": 1, "text": "t", "embedding": [0.1]*384}
            search_chunks("query", [bad_chunk], model=self.mock_model)

        # Boolean chunk_id
        with self.assertRaises(TypeError):
            bad_chunk = {"chunk_id": True, "page_number": 1, "text": "t", "embedding": [0.1]*384}
            search_chunks("query", [bad_chunk], model=self.mock_model)

        # Non-positive page_number
        with self.assertRaises(ValueError):
            bad_chunk = {"chunk_id": 1, "page_number": -1, "text": "t", "embedding": [0.1]*384}
            search_chunks("query", [bad_chunk], model=self.mock_model)

        # Boolean page_number
        with self.assertRaises(TypeError):
            bad_chunk = {"chunk_id": 1, "page_number": False, "text": "t", "embedding": [0.1]*384}
            search_chunks("query", [bad_chunk], model=self.mock_model)

        # Non-string text
        with self.assertRaises(TypeError):
            bad_chunk = {"chunk_id": 1, "page_number": 1, "text": 12345, "embedding": [0.1]*384}
            search_chunks("query", [bad_chunk], model=self.mock_model)

    def test_invalid_chunk_embeddings_rejected(self):
        """16. Chunk embeddings with wrong dimension, non-finite, booleans, or non-numeric values rejected."""
        # Wrong dimension (e.g. 100 instead of 384)
        bad_dim_chunk = {"chunk_id": 1, "page_number": 1, "text": "t", "embedding": [0.1] * 100}
        with self.assertRaises(ValueError):
            search_chunks("query", [bad_dim_chunk], model=self.mock_model)

        # Embedding is not a list
        bad_type_chunk = {"chunk_id": 1, "page_number": 1, "text": "t", "embedding": "not a list"}
        with self.assertRaises(TypeError):
            search_chunks("query", [bad_type_chunk], model=self.mock_model)

        # Embedding contains NaN or Inf
        bad_val_chunk = {"chunk_id": 1, "page_number": 1, "text": "t", "embedding": [float("nan")] * 384}
        with self.assertRaises(ValueError):
            search_chunks("query", [bad_val_chunk], model=self.mock_model)

        # Embedding contains boolean
        bad_bool_chunk = {"chunk_id": 1, "page_number": 1, "text": "t", "embedding": [True] + [0.1] * 383}
        with self.assertRaises(TypeError):
            search_chunks("query", [bad_bool_chunk], model=self.mock_model)

        # Embedding contains string
        bad_str_chunk = {"chunk_id": 1, "page_number": 1, "text": "t", "embedding": ["0.1"] * 384}
        with self.assertRaises(TypeError):
            search_chunks("query", [bad_str_chunk], model=self.mock_model)

    def test_result_metadata_preserved_and_chunks_not_mutated(self):
        """17. Verify output dictionaries preserve all metadata via {**chunk, ...} and inputs are untouched."""
        chunk_with_extra = {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Python programming language.",
            "embedding": list(self.sample_chunks[0]["embedding"]),
            "author": "Guido van Rossum",
            "section": "Overview",
        }
        original_chunk_dict = chunk_with_extra.copy()
        original_chunk_dict["embedding"] = list(chunk_with_extra["embedding"])

        results = search_chunks("Python", [chunk_with_extra], top_k=1, model=self.mock_model)

        # Preserves existing and extra metadata fields
        self.assertEqual(results[0]["chunk_id"], 1)
        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["text"], "Python programming language.")
        self.assertEqual(results[0]["author"], "Guido van Rossum")
        self.assertEqual(results[0]["section"], "Overview")
        self.assertIn("similarity_score", results[0])

        # Result embedding must be a distinct copy, not the exact same reference
        self.assertIsNot(results[0]["embedding"], chunk_with_extra["embedding"])
        self.assertEqual(results[0]["embedding"], chunk_with_extra["embedding"])

        # Original chunk dictionary must NOT be mutated
        self.assertNotIn("similarity_score", chunk_with_extra)
        self.assertEqual(chunk_with_extra, original_chunk_dict)

    @patch("src.semantic_search.get_embedding_model")
    def test_model_loading_reused_from_cache(self, mock_get_model):
        """18. When model=None, search_chunks calls get_embedding_model once to reuse cached instance."""
        mock_get_model.return_value = self.mock_model

        search_chunks("query 1", self.sample_chunks, model=None)
        mock_get_model.assert_called_once_with(DEFAULT_MODEL_NAME)


class TestSemanticSearchRealModelSmoke(unittest.TestCase):
    """
    Integration smoke test using the real all-MiniLM-L6-v2 SentenceTransformer.

    Requirements:
        - Requires model weights cached locally (~/.cache/huggingface/hub) or network on initial download.
        - Verifies that real query embeddings score conceptually related chunks higher than unrelated ones.
    """

    def test_real_model_semantic_ranking(self):
        """Real model should rank a 'programming' query higher for a software chunk than a food chunk."""
        from src.embedder import embed_chunks

        raw_chunks = [
            {"chunk_id": 1, "page_number": 1, "text": "Python is a high-level programming language for software development."},
            {"chunk_id": 2, "page_number": 1, "text": "A pizza recipe with tomato sauce, mozzarella cheese, and basil."},
        ]

        # Embed using real model
        embedded = embed_chunks(raw_chunks, model_name=DEFAULT_MODEL_NAME)

        # Search with real model
        results = search_chunks("coding and software engineering", embedded, top_k=2, model_name=DEFAULT_MODEL_NAME)

        self.assertEqual(len(results), 2)
        # Programming chunk must rank #1
        self.assertEqual(results[0]["chunk_id"], 1)
        self.assertGreater(results[0]["similarity_score"], results[1]["similarity_score"])
        # Score must be between -1.0 and 1.0
        self.assertTrue(-1.0 <= results[0]["similarity_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
