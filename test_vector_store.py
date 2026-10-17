"""
DocuMind - Unit and Integration Tests for Vector Store Module (ChromaDB)
Tests client setup, collection creation, cosine distance to similarity conversion,
input validation, chunk upsert, query formatting, ID collision prevention, and real disk persistence.
"""

import gc
import tempfile
import unittest
from unittest.mock import MagicMock

from src.vector_store import (
    cosine_distance_to_similarity,
    get_chroma_client,
    get_or_create_collection,
    store_chunks,
    query_chunks,
)
from src.embedder import EMBEDDING_DIMENSION


def make_test_vector(
    first_val: float = 1.0,
    dimension: int = EMBEDDING_DIMENSION,
    target_dim: int = 0,
) -> list[float]:
    """Helper to generate a deterministic 384-dimensional test vector."""
    vec = [0.0] * dimension
    vec[target_dim] = first_val
    return vec


class TestCosineDistanceToSimilarity(unittest.TestCase):
    """Tests conversion of ChromaDB cosine distances to similarity scores."""

    def test_exact_conversions(self):
        # distance = 0.0 -> similarity = 1.0
        self.assertEqual(cosine_distance_to_similarity(0.0), 1.0)
        # distance = 1.0 -> similarity = 0.0
        self.assertEqual(cosine_distance_to_similarity(1.0), 0.0)
        # distance = 2.0 -> similarity = -1.0
        self.assertEqual(cosine_distance_to_similarity(2.0), -1.0)
        # distance = 0.4122 -> similarity = 0.5878
        self.assertAlmostEqual(cosine_distance_to_similarity(0.4122), 0.5878, places=4)

    def test_float_precision_clamping(self):
        # Slightly negative distance from floating-point noise clamped to 1.0
        self.assertEqual(cosine_distance_to_similarity(-1.5e-6), 1.0)
        # Distance > 2.0 clamped to -1.0
        self.assertEqual(cosine_distance_to_similarity(2.05), -1.0)

    def test_custom_round_digits(self):
        self.assertEqual(cosine_distance_to_similarity(0.333333, round_digits=2), 0.67)
        self.assertAlmostEqual(cosine_distance_to_similarity(0.333333, round_digits=None), 0.666667, places=5)

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            cosine_distance_to_similarity("0.5")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cosine_distance_to_similarity(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cosine_distance_to_similarity(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cosine_distance_to_similarity(0.5, round_digits="4")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cosine_distance_to_similarity(0.5, round_digits=True)  # type: ignore[arg-type]

    def test_value_errors(self):
        with self.assertRaises(ValueError):
            cosine_distance_to_similarity(float("nan"))
        with self.assertRaises(ValueError):
            cosine_distance_to_similarity(float("inf"))
        with self.assertRaises(ValueError):
            cosine_distance_to_similarity(float("-inf"))
        with self.assertRaises(ValueError):
            cosine_distance_to_similarity(0.5, round_digits=-1)


class TestClientAndCollectionSetup(unittest.TestCase):
    """Tests client initialization and collection configuration."""

    def test_get_chroma_client_validation(self):
        with self.assertRaises(TypeError):
            get_chroma_client(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            get_chroma_client("")
        with self.assertRaises(ValueError):
            get_chroma_client("   ")

    def test_get_or_create_collection_validation(self):
        with self.assertRaises(TypeError):
            get_or_create_collection(None, "my_collection")
        mock_client = MagicMock()
        with self.assertRaises(TypeError):
            get_or_create_collection(mock_client, 123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            get_or_create_collection(mock_client, "")
        with self.assertRaises(ValueError):
            get_or_create_collection(mock_client, "   ")

    def test_get_or_create_collection_configures_cosine_space(self):
        mock_client = MagicMock()
        get_or_create_collection(mock_client, "test_col")
        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_col",
            metadata={"hnsw:space": "cosine"},
        )


class TestStoreChunksValidation(unittest.TestCase):
    """Tests input validation and error handling for store_chunks."""

    def setUp(self):
        self.mock_collection = MagicMock()
        self.valid_chunks = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "text": "Valid chunk one text.",
                "embedding": make_test_vector(1.0),
            },
            {
                "chunk_id": 2,
                "page_number": 1,
                "text": "Valid chunk two text.",
                "embedding": make_test_vector(0.5),
            },
        ]

    def test_collection_and_document_id_validation(self):
        with self.assertRaises(TypeError):
            store_chunks(self.valid_chunks, None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            store_chunks(self.valid_chunks, self.mock_collection, document_id=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store_chunks(self.valid_chunks, self.mock_collection, document_id="")
        with self.assertRaises(ValueError):
            store_chunks(self.valid_chunks, self.mock_collection, document_id="   ")

    def test_embedded_chunks_type_and_empty(self):
        with self.assertRaises(TypeError):
            store_chunks("not a list", self.mock_collection)  # type: ignore[arg-type]
        # Empty list should return 0 without invoking upsert
        count = store_chunks([], self.mock_collection)
        self.assertEqual(count, 0)
        self.mock_collection.upsert.assert_not_called()

    def test_chunk_not_a_dict(self):
        with self.assertRaises(TypeError):
            store_chunks(["not a dict"], self.mock_collection)  # type: ignore[list-item]

    def test_chunk_missing_required_keys(self):
        bad_chunk = {"chunk_id": 1, "page_number": 1, "text": "hello"}  # missing 'embedding'
        with self.assertRaises(ValueError) as ctx:
            store_chunks([bad_chunk], self.mock_collection)
        self.assertIn("missing required keys", str(ctx.exception).lower())

    def test_chunk_id_validation(self):
        # Non-int or boolean
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "chunk_id": "1"}], self.mock_collection)
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "chunk_id": True}], self.mock_collection)
        # <= 0
        with self.assertRaises(ValueError):
            store_chunks([{**self.valid_chunks[0], "chunk_id": 0}], self.mock_collection)
        with self.assertRaises(ValueError):
            store_chunks([{**self.valid_chunks[0], "chunk_id": -1}], self.mock_collection)

    def test_page_number_validation(self):
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "page_number": "1"}], self.mock_collection)
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "page_number": False}], self.mock_collection)
        with self.assertRaises(ValueError):
            store_chunks([{**self.valid_chunks[0], "page_number": 0}], self.mock_collection)

    def test_text_validation(self):
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "text": 12345}], self.mock_collection)

    def test_embedding_validation(self):
        # Non-sequence
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "embedding": "bad"}], self.mock_collection)
        # Dimension mismatch
        with self.assertRaises(ValueError):
            store_chunks([{**self.valid_chunks[0], "embedding": [0.1] * 10}], self.mock_collection)
        with self.assertRaises(ValueError):
            store_chunks([{**self.valid_chunks[0], "embedding": [0.1] * 383}], self.mock_collection)
        # Non-numeric element or boolean
        bad_vec_bool = make_test_vector(1.0)
        bad_vec_bool[5] = True  # type: ignore[call-overload]
        with self.assertRaises(TypeError):
            store_chunks([{**self.valid_chunks[0], "embedding": bad_vec_bool}], self.mock_collection)
        # Non-finite elements (NaN/Inf)
        bad_vec_nan = make_test_vector(1.0)
        bad_vec_nan[5] = float("nan")
        with self.assertRaises(ValueError):
            store_chunks([{**self.valid_chunks[0], "embedding": bad_vec_nan}], self.mock_collection)

    def test_duplicate_chunk_ids_in_batch_rejected(self):
        """Verifies duplicate document-scoped chunk IDs in a single batch raise ValueError."""
        duplicate_chunks = [
            self.valid_chunks[0],
            {
                "chunk_id": 1,  # Duplicate chunk_id within the same document batch
                "page_number": 2,
                "text": "Another chunk with duplicate chunk_id.",
                "embedding": make_test_vector(0.3),
            },
        ]
        with self.assertRaises(ValueError) as ctx:
            store_chunks(duplicate_chunks, self.mock_collection, document_id="doc1")
        self.assertIn("duplicate document-scoped chunk id", str(ctx.exception).lower())
        self.mock_collection.upsert.assert_not_called()


class TestStoreChunksMocked(unittest.TestCase):
    """Tests store_chunks calls collection.upsert with correct scoped IDs and metadata."""

    def test_upsert_payload_formatting(self):
        mock_col = MagicMock()
        chunks = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "text": "Text 1",
                "embedding": make_test_vector(0.8),
            },
            {
                "chunk_id": 2,
                "page_number": 2,
                "text": "Text 2",
                "embedding": make_test_vector(0.4),
            },
        ]
        count = store_chunks(chunks, mock_col, document_id="sample_doc")
        self.assertEqual(count, 2)
        mock_col.upsert.assert_called_once()
        kwargs = mock_col.upsert.call_args[1]

        self.assertEqual(kwargs["ids"], ["sample_doc_chunk_1", "sample_doc_chunk_2"])
        self.assertEqual(kwargs["documents"], ["Text 1", "Text 2"])
        self.assertEqual(len(kwargs["embeddings"]), 2)
        self.assertEqual(kwargs["metadatas"], [
            {"chunk_id": 1, "page_number": 1, "document_id": "sample_doc"},
            {"chunk_id": 2, "page_number": 2, "document_id": "sample_doc"},
        ])


class TestQueryChunksValidation(unittest.TestCase):
    """Tests query_chunks argument validation and defensive checks."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.mock_col.count.return_value = 5
        self.valid_vec = make_test_vector(1.0)

    def test_collection_validation(self):
        with self.assertRaises(TypeError):
            query_chunks(self.valid_vec, None)  # type: ignore[arg-type]

    def test_top_k_validation(self):
        with self.assertRaises(TypeError):
            query_chunks(self.valid_vec, self.mock_col, top_k="3")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            query_chunks(self.valid_vec, self.mock_col, top_k=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            query_chunks(self.valid_vec, self.mock_col, top_k=0)
        with self.assertRaises(ValueError):
            query_chunks(self.valid_vec, self.mock_col, top_k=-1)

    def test_query_embedding_validation(self):
        # Non-sequence
        with self.assertRaises(TypeError):
            query_chunks("not a vector", self.mock_col)  # type: ignore[arg-type]
        # Wrong length
        with self.assertRaises(ValueError):
            query_chunks([0.1] * 100, self.mock_col)
        # Non-numeric element
        bad_vec = make_test_vector(1.0)
        bad_vec[0] = "invalid"  # type: ignore[call-overload]
        with self.assertRaises(TypeError):
            query_chunks(bad_vec, self.mock_col)
        # Non-finite element
        bad_nan_vec = make_test_vector(1.0)
        bad_nan_vec[0] = float("nan")
        with self.assertRaises(ValueError):
            query_chunks(bad_nan_vec, self.mock_col)
        # Zero norm
        with self.assertRaises(ValueError):
            query_chunks([0.0] * EMBEDDING_DIMENSION, self.mock_col)

    def test_document_id_validation(self):
        with self.assertRaises(TypeError):
            query_chunks(self.valid_vec, self.mock_col, document_id=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            query_chunks(self.valid_vec, self.mock_col, document_id="")
        with self.assertRaises(ValueError):
            query_chunks(self.valid_vec, self.mock_col, document_id="   ")

    def test_document_id_validation_with_empty_collection(self):
        """Verifies document_id is validated before the empty collection early return."""
        mock_empty_col = MagicMock()
        mock_empty_col.count.return_value = 0
        with self.assertRaises(ValueError):
            query_chunks(self.valid_vec, mock_empty_col, document_id="   ")
        with self.assertRaises(ValueError):
            query_chunks(self.valid_vec, mock_empty_col, document_id="")
        with self.assertRaises(TypeError):
            query_chunks(self.valid_vec, mock_empty_col, document_id=123)  # type: ignore[arg-type]


class TestQueryChunksMocked(unittest.TestCase):
    """Tests query_chunks mocked query processing and formatting."""

    def test_empty_collection_returns_empty_list(self):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        vec = make_test_vector(1.0)
        results = query_chunks(vec, mock_col, top_k=3)
        self.assertEqual(results, [])
        mock_col.query.assert_not_called()

    def test_result_formatting_and_sorting(self):
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {
            "ids": [["doc_chunk_2", "doc_chunk_1"]],
            "documents": [["Chunk 2 text", "Chunk 1 text"]],
            "metadatas": [[
                {"chunk_id": 2, "page_number": 2, "document_id": "doc"},
                {"chunk_id": 1, "page_number": 1, "document_id": "doc"},
            ]],
            "distances": [[0.8, 0.2]],
        }

        vec = make_test_vector(1.0)
        results = query_chunks(vec, mock_col, top_k=2)

        # Distance 0.2 -> similarity 0.8 (Rank 1)
        # Distance 0.8 -> similarity 0.2 (Rank 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc_chunk_1")
        self.assertEqual(results[0]["similarity_score"], 0.8)
        self.assertEqual(results[0]["cosine_distance"], 0.2)
        self.assertEqual(results[0]["page_number"], 1)

        self.assertEqual(results[1]["id"], "doc_chunk_2")
        self.assertEqual(results[1]["similarity_score"], 0.2)
        self.assertEqual(results[1]["cosine_distance"], 0.8)
        self.assertEqual(results[1]["page_number"], 2)


class TestChromaDBIntegration(unittest.TestCase):
    """Real integration tests using temporary directories and persistent ChromaDB."""

    def test_real_chroma_store_query_and_persistence(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            # Phase 1: Initialize client and store chunks
            client1 = get_chroma_client(tmp_dir)
            col1 = get_or_create_collection(client1, "test_persistence")

            chunks = [
                {
                    "chunk_id": 1,
                    "page_number": 1,
                    "text": "DocuMind is an AI PDF Study Assistant.",
                    "embedding": make_test_vector(1.0, target_dim=0),
                },
                {
                    "chunk_id": 2,
                    "page_number": 2,
                    "text": "Python supports object-oriented programming.",
                    "embedding": make_test_vector(1.0, target_dim=1),
                },
            ]
            count = store_chunks(chunks, col1, document_id="guide")
            self.assertEqual(count, 2)
            self.assertEqual(col1.count(), 2)

            # Phase 2: Create a NEW client pointing to the exact same directory (verifying disk persistence)
            client2 = get_chroma_client(tmp_dir)
            col2 = get_or_create_collection(client2, "test_persistence")
            self.assertEqual(col2.count(), 2)

            # Query with vector matching chunk 1 along dimension 0
            query_vec = make_test_vector(1.0, target_dim=0)
            results = query_chunks(query_vec, col2, top_k=2)

            self.assertEqual(len(results), 2)
            # Rank 1 should unambiguously be chunk 1 (collinear, similarity 1.0)
            self.assertEqual(results[0]["id"], "guide_chunk_1")
            self.assertEqual(results[0]["chunk_id"], 1)
            self.assertEqual(results[0]["page_number"], 1)
            self.assertEqual(results[0]["document_id"], "guide")
            self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=3)
            self.assertIn("DocuMind", results[0]["text"])

            # Rank 2 should unambiguously be chunk 2 (orthogonal, similarity 0.0)
            self.assertEqual(results[1]["id"], "guide_chunk_2")
            self.assertEqual(results[1]["chunk_id"], 2)
            self.assertEqual(results[1]["page_number"], 2)
            self.assertEqual(results[1]["document_id"], "guide")
            self.assertAlmostEqual(results[1]["similarity_score"], 0.0, places=3)
            self.assertIn("Python", results[1]["text"])

            del col1
            del client1
            del col2
            del client2
            gc.collect()

    def test_real_chroma_upsert_idempotence(self):
        """Verifies calling store_chunks repeatedly does not create duplicate entries."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            client = get_chroma_client(tmp_dir)
            col = get_or_create_collection(client, "test_upsert")

            chunks = [
                {
                    "chunk_id": 1,
                    "page_number": 1,
                    "text": "Original text.",
                    "embedding": make_test_vector(1.0),
                }
            ]
            store_chunks(chunks, col, document_id="doc1")
            self.assertEqual(col.count(), 1)

            # Re-upsert with updated text for the same chunk
            updated_chunks = [
                {
                    "chunk_id": 1,
                    "page_number": 1,
                    "text": "Updated text.",
                    "embedding": make_test_vector(1.0),
                }
            ]
            store_chunks(updated_chunks, col, document_id="doc1")
            self.assertEqual(col.count(), 1)  # Still 1 chunk

            # Query confirms text was updated
            results = query_chunks(make_test_vector(1.0), col, top_k=1)
            self.assertEqual(results[0]["text"], "Updated text.")

            del col
            del client
            gc.collect()

    def test_real_chroma_document_isolation_and_no_collision(self):
        """Verifies chunks from different documents have distinct IDs and do not collide."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            client = get_chroma_client(tmp_dir)
            col = get_or_create_collection(client, "multi_doc_col")

            doc_a_chunks = [
                {
                    "chunk_id": 1,
                    "page_number": 1,
                    "text": "Doc A Chunk 1",
                    "embedding": make_test_vector(1.0),
                }
            ]
            doc_b_chunks = [
                {
                    "chunk_id": 1,
                    "page_number": 1,
                    "text": "Doc B Chunk 1",
                    "embedding": make_test_vector(0.2),
                }
            ]

            store_chunks(doc_a_chunks, col, document_id="doc_a")
            store_chunks(doc_b_chunks, col, document_id="doc_b")

            self.assertEqual(col.count(), 2)

            # Filter query to doc_a
            results_a = query_chunks(make_test_vector(1.0), col, top_k=2, document_id="doc_a")
            self.assertEqual(len(results_a), 1)
            self.assertEqual(results_a[0]["id"], "doc_a_chunk_1")
            self.assertEqual(results_a[0]["document_id"], "doc_a")
            self.assertEqual(results_a[0]["text"], "Doc A Chunk 1")

            # Filter query to doc_b
            results_b = query_chunks(make_test_vector(1.0), col, top_k=2, document_id="doc_b")
            self.assertEqual(len(results_b), 1)
            self.assertEqual(results_b[0]["id"], "doc_b_chunk_1")
            self.assertEqual(results_b[0]["document_id"], "doc_b")
            self.assertEqual(results_b[0]["text"], "Doc B Chunk 1")

            del col
            del client
            gc.collect()

    def test_real_chroma_empty_collection_query(self):
        """Verifies querying an empty collection returns [] cleanly."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            client = get_chroma_client(tmp_dir)
            col = get_or_create_collection(client, "empty_col")
            results = query_chunks(make_test_vector(1.0), col, top_k=3)
            self.assertEqual(results, [])

            del col
            del client
            gc.collect()



if __name__ == "__main__":
    unittest.main()
