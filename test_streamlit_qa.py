"""
DocuMind - Unit Tests for Streamlit Question-Answering Logic (Milestone 7, Step 4)
Tests query validation, document-scoped vector retrieval, grounded Gemini answer synthesis,
source citation metadata integrity, empty query handling, zero retrieved chunks,
API and configuration failure handling, and document switching state isolation using mocks.
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Any

from streamlit_app import (
    answer_question_for_document,
    sanitize_error_message,
    init_session_state,
    reset_qa_session_state,
)


class TestAnswerQuestionForDocument(unittest.TestCase):
    """Tests the core scoped question-answering workflow in streamlit_app.py."""

    def setUp(self):
        self.mock_col = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = [[0.05] * 384]
        self.doc_id = "doc_abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        self.valid_query = "What is the primary contribution of this research?"

    # -------------------------------------------------------------------------
    # Input Validation Tests
    # -------------------------------------------------------------------------
    def test_empty_or_whitespace_query_raises_value_error(self):
        for invalid_q in ["", "   ", "\t\n"]:
            with self.subTest(query=invalid_q):
                with self.assertRaises(ValueError) as ctx:
                    answer_question_for_document(
                        query=invalid_q,
                        document_id=self.doc_id,
                        collection=self.mock_col,
                        embedding_model=self.mock_model,
                    )
                self.assertIn("empty", str(ctx.exception).lower())

    def test_non_string_query_raises_type_error(self):
        for invalid_q in [123, None, ["query"], True]:
            with self.subTest(query=invalid_q):
                with self.assertRaises(TypeError):
                    answer_question_for_document(
                        query=invalid_q,  # type: ignore
                        document_id=self.doc_id,
                        collection=self.mock_col,
                        embedding_model=self.mock_model,
                    )

    def test_empty_or_whitespace_document_id_raises_value_error(self):
        for invalid_id in ["", "   ", "\t"]:
            with self.subTest(doc_id=invalid_id):
                with self.assertRaises(ValueError) as ctx:
                    answer_question_for_document(
                        query=self.valid_query,
                        document_id=invalid_id,
                        collection=self.mock_col,
                        embedding_model=self.mock_model,
                    )
                self.assertIn("document_id", str(ctx.exception).lower())

    def test_non_string_document_id_raises_type_error(self):
        for invalid_id in [None, 999, False]:
            with self.subTest(doc_id=invalid_id):
                with self.assertRaises(TypeError):
                    answer_question_for_document(
                        query=self.valid_query,
                        document_id=invalid_id,  # type: ignore
                        collection=self.mock_col,
                        embedding_model=self.mock_model,
                    )

    def test_missing_collection_or_model_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            answer_question_for_document(
                query=self.valid_query,
                document_id=self.doc_id,
                collection=None,
                embedding_model=self.mock_model,
            )
        self.assertIn("collection", str(ctx.exception).lower())

        with self.assertRaises(RuntimeError) as ctx:
            answer_question_for_document(
                query=self.valid_query,
                document_id=self.doc_id,
                collection=self.mock_col,
                embedding_model=None,
            )
        self.assertIn("embedding model", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Retrieval Scoping & Document Isolation
    # -------------------------------------------------------------------------
    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_document_filtering_strictly_passed_to_query_chunks(
        self, mock_query_chunks, mock_generate_answer
    ):
        """Verifies that retrieval queries ChromaDB strictly with the active document ID."""
        mock_query_chunks.return_value = [
            {
                "id": f"{self.doc_id}_chunk_1",
                "chunk_id": 1,
                "page_number": 1,
                "document_id": self.doc_id,
                "text": "Relevant paragraph on neural attention mechanisms.",
                "similarity_score": 0.8842,
                "cosine_distance": 0.1158,
            }
        ]
        mock_generate_answer.return_value = {
            "answer": "The paper introduces self-attention mechanisms (Page 1).",
            "sources": [{"page_number": 1, "chunk_id": 1, "document_id": self.doc_id}],
            "model": "gemini-3.8-flash",
            "abstention": False,
        }

        res = answer_question_for_document(
            query=self.valid_query,
            document_id=self.doc_id,
            collection=self.mock_col,
            embedding_model=self.mock_model,
            top_k=3,
        )

        # Assert document_id was explicitly passed to query_chunks
        mock_query_chunks.assert_called_once()
        _, kwargs = mock_query_chunks.call_args
        self.assertEqual(kwargs.get("document_id"), self.doc_id)
        self.assertEqual(kwargs.get("top_k"), 3)

        # Assert result structure
        self.assertEqual(res["query"], self.valid_query)
        self.assertEqual(res["document_id"], self.doc_id)
        self.assertEqual(res["answer"], "The paper introduces self-attention mechanisms (Page 1).")
        self.assertFalse(res["abstention"])
        self.assertEqual(len(res["retrieved_chunks"]), 1)
        self.assertEqual(res["retrieved_chunks"][0]["similarity_score"], 0.8842)

    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_different_documents_receive_distinct_retrieval_filters(
        self, mock_query_chunks, mock_generate_answer
    ):
        """Verifies document isolation across distinct document IDs."""
        doc_a = "doc_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        doc_b = "doc_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        mock_query_chunks.return_value = []
        mock_generate_answer.return_value = {
            "answer": "No relevant context found.",
            "sources": [],
            "model": "gemini-3.8-flash",
            "abstention": True,
        }

        # Query Document A
        answer_question_for_document(
            query="Question A",
            document_id=doc_a,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )
        self.assertEqual(mock_query_chunks.call_args[1]["document_id"], doc_a)

        # Query Document B
        answer_question_for_document(
            query="Question B",
            document_id=doc_b,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )
        self.assertEqual(mock_query_chunks.call_args[1]["document_id"], doc_b)

    # -------------------------------------------------------------------------
    # Empty Retrieval & Abstention Handling
    # -------------------------------------------------------------------------
    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_zero_retrieved_chunks_triggers_abstention(
        self, mock_query_chunks, mock_generate_answer
    ):
        """When vector retrieval returns 0 chunks, abstention must be returned cleanly."""
        mock_query_chunks.return_value = []
        mock_generate_answer.return_value = {
            "answer": "No relevant document context found to answer the question.",
            "sources": [],
            "model": "gemini-3.8-flash",
            "abstention": True,
        }

        res = answer_question_for_document(
            query="What is quantum gravity?",
            document_id=self.doc_id,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertTrue(res["abstention"])
        self.assertEqual(len(res["retrieved_chunks"]), 0)
        self.assertEqual(len(res["sources"]), 0)
        self.assertIn("no relevant document context", res["answer"].lower())

    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_out_of_scope_query_returns_abstention_flag(
        self, mock_query_chunks, mock_generate_answer
    ):
        """When LLM emits the explicit fallback abstention message, abstention=True is set."""
        mock_query_chunks.return_value = [
            {
                "id": f"{self.doc_id}_chunk_1",
                "chunk_id": 1,
                "page_number": 1,
                "document_id": self.doc_id,
                "text": "The document discusses Python web architectures.",
                "similarity_score": 0.1200,
                "cosine_distance": 0.8800,
            }
        ]
        mock_generate_answer.return_value = {
            "answer": "I cannot find the answer to your question in the provided document.",
            "sources": [{"page_number": 1, "chunk_id": 1, "document_id": self.doc_id}],
            "model": "gemini-3.8-flash",
            "abstention": True,
        }

        res = answer_question_for_document(
            query="What is the price of oil in 2030?",
            document_id=self.doc_id,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertTrue(res["abstention"])
        self.assertEqual(
            res["answer"],
            "I cannot find the answer to your question in the provided document.",
        )

    # -------------------------------------------------------------------------
    # Failure Handling: API Errors, Quota & Keys
    # -------------------------------------------------------------------------
    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_missing_api_key_propagates_value_error(
        self, mock_query_chunks, mock_generate_answer
    ):
        """When GEMINI_API_KEY is unconfigured, ValueError is raised without leaking secrets."""
        mock_query_chunks.return_value = [
            {"id": "c1", "chunk_id": 1, "page_number": 1, "document_id": self.doc_id, "text": "Sample"}
        ]
        mock_generate_answer.side_effect = ValueError(
            "GEMINI_API_KEY is not configured. Please create a .env file..."
        )

        with self.assertRaises(ValueError) as ctx:
            answer_question_for_document(
                query=self.valid_query,
                document_id=self.doc_id,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )
        self.assertIn("GEMINI_API_KEY is not configured", str(ctx.exception))

    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_gemini_api_network_or_quota_failure_propagates_runtime_error(
        self, mock_query_chunks, mock_generate_answer
    ):
        """When Gemini API encounters 429 quota or connection drop, RuntimeError is raised."""
        mock_query_chunks.return_value = [
            {"id": "c1", "chunk_id": 1, "page_number": 1, "document_id": self.doc_id, "text": "Sample"}
        ]
        mock_generate_answer.side_effect = RuntimeError(
            "Gemini API generation failed (ClientError [HTTP 429 RESOURCE_EXHAUSTED])."
        )

        with self.assertRaises(RuntimeError) as ctx:
            answer_question_for_document(
                query=self.valid_query,
                document_id=self.doc_id,
                collection=self.mock_col,
                embedding_model=self.mock_model,
            )
        self.assertIn("RESOURCE_EXHAUSTED", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Source Citations Integrity
    # -------------------------------------------------------------------------
    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_source_passages_preserve_all_metadata_fields(
        self, mock_query_chunks, mock_generate_answer
    ):
        """Verifies rank, page number, chunk ID, similarity score, and text are preserved."""
        retrieved = [
            {
                "id": f"{self.doc_id}_chunk_3",
                "chunk_id": 3,
                "page_number": 2,
                "document_id": self.doc_id,
                "text": "Detailed methodology section explaining cross-validation.",
                "similarity_score": 0.8123,
                "cosine_distance": 0.1877,
            },
            {
                "id": f"{self.doc_id}_chunk_1",
                "chunk_id": 1,
                "page_number": 1,
                "document_id": self.doc_id,
                "text": "Introduction defining hyperparameter tuning.",
                "similarity_score": 0.7410,
                "cosine_distance": 0.2590,
            },
        ]
        mock_query_chunks.return_value = retrieved
        mock_generate_answer.return_value = {
            "answer": "The methodology uses 5-fold cross-validation (Page 2).",
            "sources": [
                {"page_number": 2, "chunk_id": 3, "document_id": self.doc_id},
                {"page_number": 1, "chunk_id": 1, "document_id": self.doc_id},
            ],
            "model": "gemini-3.8-flash",
            "abstention": False,
        }

        res = answer_question_for_document(
            query="Explain the validation methodology",
            document_id=self.doc_id,
            collection=self.mock_col,
            embedding_model=self.mock_model,
            top_k=2,
        )

        chunks = res["retrieved_chunks"]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], 3)
        self.assertEqual(chunks[0]["page_number"], 2)
        self.assertEqual(chunks[0]["similarity_score"], 0.8123)
        self.assertEqual(chunks[1]["chunk_id"], 1)
        self.assertEqual(chunks[1]["page_number"], 1)
        self.assertEqual(chunks[1]["similarity_score"], 0.7410)

    # -------------------------------------------------------------------------
    # Timing & Performance Instrumentation Tests
    # -------------------------------------------------------------------------
    @patch("streamlit_app.generate_answer")
    @patch("streamlit_app.query_chunks")
    def test_timing_breakdown_included_in_qa_result(
        self, mock_query_chunks, mock_generate_answer
    ):
        """Verifies that elapsed-time measurements are captured and returned in the timing dict."""
        mock_query_chunks.return_value = [
            {
                "id": f"{self.doc_id}_chunk_1",
                "chunk_id": 1,
                "page_number": 1,
                "document_id": self.doc_id,
                "text": "Neural network architecture details.",
                "similarity_score": 0.85,
            }
        ]
        mock_generate_answer.return_value = {
            "answer": "Architecture includes multiple layers.",
            "sources": [{"page_number": 1, "chunk_id": 1}],
            "model": "gemini-3.8-flash",
            "abstention": False,
        }

        res = answer_question_for_document(
            query=self.valid_query,
            document_id=self.doc_id,
            collection=self.mock_col,
            embedding_model=self.mock_model,
        )

        self.assertIn("timing", res)
        timing = res["timing"]
        self.assertIn("embedding_ms", timing)
        self.assertIn("retrieval_ms", timing)
        self.assertIn("generation_ms", timing)
        self.assertIn("total_seconds", timing)

        self.assertIsInstance(timing["embedding_ms"], float)
        self.assertIsInstance(timing["retrieval_ms"], float)
        self.assertIsInstance(timing["generation_ms"], float)
        self.assertIsInstance(timing["total_seconds"], float)
        self.assertGreaterEqual(timing["embedding_ms"], 0.0)
        self.assertGreaterEqual(timing["retrieval_ms"], 0.0)
        self.assertGreaterEqual(timing["generation_ms"], 0.0)
        self.assertGreaterEqual(timing["total_seconds"], 0.0)


class TestStreamlitQASessionStateIsolation(unittest.TestCase):
    """Tests session state invariants to verify cross-document contamination prevention and input reset."""

    def test_session_state_defaults_include_qa_fields(self):
        """Verifies init_session_state sets up clean Q&A state variables and empty question input."""
        mock_session_state: dict[str, Any] = {}
        with patch("streamlit_app.st.session_state", mock_session_state):
            init_session_state()
            self.assertIn("question_input", mock_session_state)
            self.assertIn("last_question", mock_session_state)
            self.assertIn("last_answer", mock_session_state)
            self.assertIn("last_sources", mock_session_state)
            self.assertIn("last_retrieved_chunks", mock_session_state)
            self.assertIn("last_abstention", mock_session_state)
            self.assertIn("last_timing", mock_session_state)
            self.assertIn("qa_error", mock_session_state)

            self.assertEqual(mock_session_state["question_input"], "")
            self.assertIsNone(mock_session_state["last_question"])
            self.assertIsNone(mock_session_state["last_answer"])
            self.assertIsNone(mock_session_state["last_sources"])
            self.assertIsNone(mock_session_state["last_retrieved_chunks"])
            self.assertFalse(mock_session_state["last_abstention"])
            self.assertIsNone(mock_session_state["last_timing"])
            self.assertIsNone(mock_session_state["qa_error"])

    def test_reset_qa_session_state_clears_input_when_clear_input_is_true(self):
        """Verifies reset_qa_session_state(clear_input=True) clears the text input and Q&A history."""
        mock_session_state = {
            "question_input": "What is the conclusion of this study?",
            "last_question": "What is the conclusion of this study?",
            "last_answer": "The study concludes that model fine-tuning improves accuracy.",
            "last_sources": [{"page_number": 5, "chunk_id": 12}],
            "last_retrieved_chunks": [{"chunk_id": 12, "text": "Conclusion text..."}],
            "last_abstention": False,
            "last_timing": {"embedding_ms": 10.0, "retrieval_ms": 15.0, "generation_ms": 800.0, "total_seconds": 0.85},
            "qa_error": None,
        }
        with patch("streamlit_app.st.session_state", mock_session_state):
            reset_qa_session_state(clear_input=True)

            self.assertEqual(mock_session_state["question_input"], "")
            self.assertIsNone(mock_session_state["last_question"])
            self.assertIsNone(mock_session_state["last_answer"])
            self.assertIsNone(mock_session_state["last_sources"])
            self.assertIsNone(mock_session_state["last_retrieved_chunks"])
            self.assertFalse(mock_session_state["last_abstention"])
            self.assertIsNone(mock_session_state["last_timing"])
            self.assertIsNone(mock_session_state["qa_error"])

    def test_reset_qa_session_state_preserves_input_when_clear_input_is_false(self):
        """Verifies reset_qa_session_state(clear_input=False) clears Q&A history while preserving input text."""
        mock_session_state = {
            "question_input": "Draft question typed by user",
            "last_question": "Old question",
            "last_answer": "Old answer",
            "last_sources": [{"page_number": 1}],
            "last_retrieved_chunks": [{"chunk_id": 1}],
            "last_abstention": False,
            "last_timing": {"total_seconds": 1.2},
            "qa_error": None,
        }
        with patch("streamlit_app.st.session_state", mock_session_state):
            reset_qa_session_state(clear_input=False)

            self.assertEqual(mock_session_state["question_input"], "Draft question typed by user")
            self.assertIsNone(mock_session_state["last_question"])
            self.assertIsNone(mock_session_state["last_answer"])
            self.assertIsNone(mock_session_state["last_sources"])
            self.assertIsNone(mock_session_state["last_retrieved_chunks"])
            self.assertFalse(mock_session_state["last_abstention"])
            self.assertIsNone(mock_session_state["last_timing"])
            self.assertIsNone(mock_session_state["qa_error"])

    def test_file_removal_wipes_active_document_and_question_input(self):
        """Simulates removing uploaded file and asserts Q&A state and widget input are wiped."""
        state = {
            "active_document_id": "doc_abc123456789",
            "uploaded_filename": "sample.pdf",
            "page_count": 5,
            "chunk_count": 20,
            "document_processed": True,
            "ingestion_success": True,
            "reused_index": True,
            "last_error": None,
            "question_input": "What were the findings on page 3?",
            "last_question": "What were the findings on page 3?",
            "last_answer": "Findings on page 3 show positive correlation.",
            "last_sources": [{"page_number": 3}],
            "last_retrieved_chunks": [{"chunk_id": 3}],
            "last_abstention": False,
            "last_timing": {"total_seconds": 0.5},
            "qa_error": None,
        }

        with patch("streamlit_app.st.session_state", state):
            # Simulate the file removal block in streamlit_app.py
            if state["active_document_id"] is not None or state["document_processed"]:
                state["active_document_id"] = None
                state["uploaded_filename"] = None
                state["page_count"] = 0
                state["chunk_count"] = 0
                state["document_processed"] = False
                state["ingestion_success"] = False
                state["reused_index"] = False
                state["last_error"] = None
                reset_qa_session_state(clear_input=True)

        self.assertFalse(state["document_processed"])
        self.assertIsNone(state["active_document_id"])
        self.assertEqual(state["question_input"], "")
        self.assertIsNone(state["last_question"])
        self.assertIsNone(state["last_answer"])
        self.assertIsNone(state["last_sources"])
        self.assertIsNone(state["last_retrieved_chunks"])
        self.assertIsNone(state["last_timing"])

    def test_file_replacement_wipes_active_document_and_question_input(self):
        """Simulates uploading a different file and asserts previous question and answer are wiped."""
        state = {
            "active_document_id": "doc_old1234567890",
            "uploaded_filename": "doc_a.pdf",
            "page_count": 2,
            "chunk_count": 10,
            "document_processed": True,
            "ingestion_success": True,
            "reused_index": False,
            "last_error": None,
            "question_input": "Previous query for doc A",
            "last_question": "Previous query for doc A",
            "last_answer": "Answer for doc A",
            "last_sources": [{"page_number": 1}],
            "last_retrieved_chunks": [{"chunk_id": 1}],
            "last_abstention": False,
            "last_timing": {"total_seconds": 1.1},
            "qa_error": None,
        }

        new_doc_id = "doc_new9876543210"
        with patch("streamlit_app.st.session_state", state):
            # Simulate state sync detection on file replacement
            if state["active_document_id"] is not None and state["active_document_id"] != new_doc_id:
                state["document_processed"] = False
                state["ingestion_success"] = False
                state["active_document_id"] = None
                state["uploaded_filename"] = "doc_b.pdf"
                state["page_count"] = 0
                state["chunk_count"] = 0
                state["reused_index"] = False
                state["last_error"] = None
                reset_qa_session_state(clear_input=True)

        self.assertFalse(state["document_processed"])
        self.assertIsNone(state["active_document_id"])
        self.assertEqual(state["uploaded_filename"], "doc_b.pdf")
        self.assertEqual(state["question_input"], "")
        self.assertIsNone(state["last_question"])
        self.assertIsNone(state["last_answer"])
        self.assertIsNone(state["last_sources"])
        self.assertIsNone(state["last_retrieved_chunks"])


class TestErrorSanitizationInQA(unittest.TestCase):
    """Tests redaction of secrets in QA error messages."""

    def test_sanitizes_gemini_api_key_in_qa_error(self):
        fake_diagnostic_key = "AIza" + "SyMockDiagnosticKeyPatternTest0000"
        err = RuntimeError(
            f"API request failed with {fake_diagnostic_key} on endpoint https://..."
        )
        clean = sanitize_error_message(err)
        self.assertNotIn(fake_diagnostic_key, clean)
        self.assertNotIn("AIza", clean)
        self.assertIn("[REDACTED_KEY]", clean)

    def test_sanitizes_bearer_tokens_in_qa_error(self):
        err = RuntimeError("Unauthorized: Bearer ya29.a0AfH6SMD... returned HTTP 401")
        clean = sanitize_error_message(err)
        self.assertNotIn("ya29", clean)
        self.assertIn("Bearer [REDACTED]", clean)


if __name__ == "__main__":
    unittest.main()
