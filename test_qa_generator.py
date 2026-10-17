"""
DocuMind - Unit and Integration Tests for QA Generator Module (Gemini RAG)
Tests secure API key loading, context block formatting, prompt construction,
empty-context guards, mocked LLM answer generation, abstention handling, and error wrapping.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.qa_generator import (
    DEFAULT_MODEL_NAME,
    FALLBACK_ABSTENTION_MESSAGE,
    FALLBACK_EMPTY_CONTEXT_MESSAGE,
    MAX_CONTEXT_CHARACTERS,
    load_gemini_api_key,
    get_gemini_client,
    format_context_block,
    build_rag_prompt,
    generate_answer,
)


class TestLoadApiKey(unittest.TestCase):
    """Tests secure environment variable retrieval for Gemini API key."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "[REDACTED_HISTORICAL_FIXTURE]"}, clear=True)
    def test_valid_api_key_loaded(self):
        key = load_gemini_api_key()
        self.assertEqual(key, "[REDACTED_HISTORICAL_FIXTURE]")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            load_gemini_api_key()
        self.assertIn("not configured", str(ctx.exception).lower())

    @patch.dict(os.environ, {"GEMINI_API_KEY": "   "}, clear=True)
    def test_whitespace_api_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_gemini_api_key()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "your_gemini_api_key_here"}, clear=True)
    def test_placeholder_api_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_gemini_api_key()


class TestFormatContextBlock(unittest.TestCase):
    """Tests chunk context assembly and source citation tracking."""

    def test_valid_chunks_formatting(self):
        chunks = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "document_id": "sample",
                "text": "First chunk content.",
            },
            {
                "chunk_id": 2,
                "page_number": 2,
                "document_id": "sample",
                "text": "Second chunk content.",
            },
        ]
        context_str, sources = format_context_block(chunks)

        self.assertIn("[Source: Page 1 | Chunk 1 | Doc: sample]", context_str)
        self.assertIn("First chunk content.", context_str)
        self.assertIn("[Source: Page 2 | Chunk 2 | Doc: sample]", context_str)
        self.assertIn("Second chunk content.", context_str)

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["page_number"], 1)
        self.assertEqual(sources[0]["chunk_id"], 1)
        self.assertEqual(sources[0]["document_id"], "sample")
        self.assertEqual(sources[1]["page_number"], 2)
        self.assertEqual(sources[1]["chunk_id"], 2)

    def test_empty_chunks_list(self):
        context_str, sources = format_context_block([])
        self.assertEqual(context_str, "")
        self.assertEqual(sources, [])

    def test_chunks_with_empty_text_skipped(self):
        chunks = [
            {"chunk_id": 1, "page_number": 1, "text": ""},
            {"chunk_id": 2, "page_number": 1, "text": "   "},
            {"chunk_id": 3, "page_number": 2, "text": "Valid content."},
        ]
        context_str, sources = format_context_block(chunks)
        self.assertNotIn("Chunk 1", context_str)
        self.assertNotIn("Chunk 2", context_str)
        self.assertIn("Chunk 3", context_str)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["chunk_id"], 3)

    def test_context_truncation_on_excessive_length(self):
        # Create chunk larger than max_chars threshold
        long_text = "A" * 500
        chunks = [
            {"chunk_id": 1, "page_number": 1, "text": long_text},
            {"chunk_id": 2, "page_number": 2, "text": long_text},
        ]
        # Restrict max_chars to 600
        context_str, sources = format_context_block(chunks, max_chars=600)
        self.assertIn("Chunk 1", context_str)
        self.assertNotIn("Chunk 2", context_str)
        self.assertEqual(len(sources), 1)
        self.assertLessEqual(len(context_str), 600)

    def test_oversized_first_chunk_truncated_strictly(self):
        # Even if the first chunk alone is larger than max_chars, it must never exceed max_chars
        long_text = "B" * 500
        chunks = [
            {"chunk_id": 1, "page_number": 1, "text": long_text},
            {"chunk_id": 2, "page_number": 2, "text": long_text},
        ]
        context_str, sources = format_context_block(chunks, max_chars=100)
        self.assertLessEqual(len(context_str), 100)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["chunk_id"], 1)
        self.assertNotIn("Chunk 2", context_str)

    def test_very_small_max_chars_limit(self):
        chunks = [
            {"chunk_id": 1, "page_number": 1, "text": "Some text content."},
        ]
        context_str, sources = format_context_block(chunks, max_chars=10)
        self.assertLessEqual(len(context_str), 10)
        self.assertTrue(len(context_str) > 0)
        self.assertEqual(len(sources), 1)

    def test_invalid_max_chars_limits_rejected(self):
        chunks = [{"chunk_id": 1, "page_number": 1, "text": "Valid text"}]
        with self.assertRaises(TypeError):
            format_context_block(chunks, max_chars="500")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            format_context_block(chunks, max_chars=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            format_context_block(chunks, max_chars=0)
        with self.assertRaises(ValueError):
            format_context_block(chunks, max_chars=-10)

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            format_context_block("not a list")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            format_context_block(["not a dict"])  # type: ignore[list-item]


class TestBuildRagPrompt(unittest.TestCase):
    """Tests prompt assembly and grounding constraints."""

    def test_prompt_contains_all_components(self):
        prompt = build_rag_prompt("What is RAG?", "[Source: Page 1]\nRAG stands for Retrieval-Augmented Generation.")
        self.assertIn("You are DocuMind", prompt)
        self.assertIn("STRICT CONSTRAINTS", prompt)
        self.assertIn(FALLBACK_ABSTENTION_MESSAGE, prompt)
        self.assertIn("RAG stands for Retrieval-Augmented Generation.", prompt)
        self.assertIn("User Question: What is RAG?", prompt)

    def test_invalid_question_validation(self):
        with self.assertRaises(TypeError):
            build_rag_prompt(123, "valid context")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_rag_prompt("", "valid context")
        with self.assertRaises(ValueError):
            build_rag_prompt("   ", "valid context")

    def test_invalid_context_validation(self):
        with self.assertRaises(TypeError):
            build_rag_prompt("valid question", 123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_rag_prompt("valid question", "")
        with self.assertRaises(ValueError):
            build_rag_prompt("valid question", "   ")


class TestGenerateAnswerOfflineMocks(unittest.TestCase):
    """Tests generate_answer logic using offline mocked Gemini clients (zero network calls)."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.valid_chunks = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "document_id": "sample",
                "text": "DocuMind is an AI PDF Study Assistant built in Python.",
            }
        ]

    def test_empty_chunks_returns_early_without_api_call(self):
        result = generate_answer("What is DocuMind?", [], client=self.mock_client)
        self.assertEqual(result["answer"], FALLBACK_EMPTY_CONTEXT_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["abstention"])
        # Crucial invariant: Never call the LLM API when there is no context
        self.mock_client.models.generate_content.assert_not_called()

    def test_successful_grounded_generation(self):
        mock_response = MagicMock()
        mock_response.text = "DocuMind is an AI PDF Study Assistant built in Python based on Page 1."
        self.mock_client.models.generate_content.return_value = mock_response

        result = generate_answer(
            question="What is DocuMind?",
            context_chunks=self.valid_chunks,
            client=self.mock_client,
            model_name="gemini-2.5-flash",
            temperature=0.0,
        )

        self.assertEqual(result["answer"], "DocuMind is an AI PDF Study Assistant built in Python based on Page 1.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["page_number"], 1)
        self.assertEqual(result["sources"][0]["chunk_id"], 1)
        self.assertEqual(result["model"], "gemini-2.5-flash")
        self.assertFalse(result["abstention"])

        self.mock_client.models.generate_content.assert_called_once()
        call_kwargs = self.mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(call_kwargs["config"].temperature, 0.0)
        self.assertIn("What is DocuMind?", call_kwargs["contents"])

    def test_abstention_response_detected(self):
        mock_response = MagicMock()
        mock_response.text = FALLBACK_ABSTENTION_MESSAGE
        self.mock_client.models.generate_content.return_value = mock_response

        result = generate_answer(
            question="What is quantum mechanics?",
            context_chunks=self.valid_chunks,
            client=self.mock_client,
        )
        self.assertEqual(result["answer"], FALLBACK_ABSTENTION_MESSAGE)
        self.assertTrue(result["abstention"])

    def test_empty_llm_response_falls_back_to_abstention(self):
        mock_response = MagicMock()
        mock_response.text = ""
        self.mock_client.models.generate_content.return_value = mock_response

        result = generate_answer(
            question="What is DocuMind?",
            context_chunks=self.valid_chunks,
            client=self.mock_client,
        )
        self.assertEqual(result["answer"], FALLBACK_ABSTENTION_MESSAGE)
        self.assertTrue(result["abstention"])

    def test_api_error_wrapped_in_runtime_error(self):
        sensitive_payload = "https://generativelanguage.googleapis.com?key=[REDACTED_HISTORICAL_FIXTURE]"
        self.mock_client.models.generate_content.side_effect = ConnectionError(
            f"Quota exceeded with sensitive URL: {sensitive_payload}"
        )

        with self.assertRaises(RuntimeError) as ctx:
            generate_answer(
                question="What is DocuMind?",
                context_chunks=self.valid_chunks,
                client=self.mock_client,
            )
        err_msg = str(ctx.exception)
        self.assertIn("Gemini API generation failed", err_msg)
        self.assertIn("ConnectionError", err_msg)
        # Verify sensitive payload from original exception is NOT exposed in RuntimeError string
        self.assertNotIn("[REDACTED_HISTORICAL_FIXTURE]", err_msg)
        self.assertNotIn("sensitive URL", err_msg)
        # Verify exception chaining (__cause__) is preserved for debugging
        self.assertIsInstance(ctx.exception.__cause__, ConnectionError)

    def test_invalid_input_types_rejected(self):
        with self.assertRaises(TypeError):
            generate_answer(12345, self.valid_chunks, client=self.mock_client)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generate_answer("", self.valid_chunks, client=self.mock_client)
        with self.assertRaises(ValueError):
            generate_answer("   ", self.valid_chunks, client=self.mock_client)
        with self.assertRaises(TypeError):
            generate_answer("What is DocuMind?", "not a list", client=self.mock_client)  # type: ignore[arg-type]


class TestGeminiRealApiSmoke(unittest.TestCase):
    """Optional smoke test against live Gemini API; automatically skipped unless GEMINI_API_KEY is configured."""

    @unittest.skipUnless(
        os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here",
        "GEMINI_API_KEY is not set in environment or is default placeholder; skipping live API smoke test.",
    )
    def test_live_gemini_api_generation(self):
        chunks = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "text": "DocuMind is an open-source AI PDF Study Assistant developed in Python.",
            }
        ]
        result = generate_answer("What is DocuMind?", chunks)
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertTrue(len(result["answer"]) > 0)
        self.assertEqual(len(result["sources"]), 1)


if __name__ == "__main__":
    unittest.main()
