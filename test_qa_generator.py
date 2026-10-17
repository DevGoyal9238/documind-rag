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
    format_api_error_diagnostic,
)


class TestLoadApiKey(unittest.TestCase):
    """Tests secure environment variable retrieval for Gemini API key."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "mock_valid_gemini_api_key_12345"}, clear=True)
    def test_valid_api_key_loaded(self):
        key = load_gemini_api_key()
        self.assertEqual(key, "mock_valid_gemini_api_key_12345")

    @patch("src.qa_generator.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self, mock_load_dotenv):
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
            model_name="gemini-3.8-flash",
            temperature=0.0,
        )

        self.assertEqual(result["answer"], "DocuMind is an AI PDF Study Assistant built in Python based on Page 1.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["page_number"], 1)
        self.assertEqual(result["sources"][0]["chunk_id"], 1)
        self.assertEqual(result["model"], "gemini-3.8-flash")
        self.assertFalse(result["abstention"])

        self.mock_client.models.generate_content.assert_called_once()
        call_kwargs = self.mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash")
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
        mock_err = ConnectionError(
            "Quota exceeded with sensitive URL: https://generativelanguage.googleapis.com?key=mock_secret_query_key_12345"
        )
        setattr(mock_err, "code", 429)
        setattr(mock_err, "status", "RESOURCE_EXHAUSTED")
        setattr(mock_err, "message", "Quota exceeded with sensitive URL: https://generativelanguage.googleapis.com?key=mock_secret_query_key_12345")
        self.mock_client.models.generate_content.side_effect = mock_err

        with self.assertRaises(RuntimeError) as ctx:
            generate_answer(
                question="What is DocuMind?",
                context_chunks=self.valid_chunks,
                client=self.mock_client,
            )
        err_msg = str(ctx.exception)
        self.assertIn("Gemini API generation failed", err_msg)
        self.assertIn("ConnectionError", err_msg)
        # Verify diagnostic includes HTTP code and status
        self.assertIn("HTTP 429 RESOURCE_EXHAUSTED", err_msg)
        # Verify sensitive payload from original exception is NOT exposed in RuntimeError string
        self.assertNotIn("mock_secret_query_key_12345", err_msg)
        self.assertIn("key=[REDACTED]", err_msg)
        # Verify exception chaining (__cause__) is preserved for debugging
        self.assertIs(ctx.exception.__cause__, mock_err)

    def test_invalid_input_types_rejected(self):
        with self.assertRaises(TypeError):
            generate_answer(12345, self.valid_chunks, client=self.mock_client)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generate_answer("", self.valid_chunks, client=self.mock_client)
        with self.assertRaises(ValueError):
            generate_answer("   ", self.valid_chunks, client=self.mock_client)
        with self.assertRaises(TypeError):
            generate_answer("What is DocuMind?", "not a list", client=self.mock_client)  # type: ignore[arg-type]


class TestErrorDiagnosticSanitization(unittest.TestCase):
    """Tests diagnostic extraction and secret redacting from API exceptions."""

    def test_http_status_code_and_status_in_diagnostic(self):
        mock_err = Exception("Model not found")
        setattr(mock_err, "code", 404)
        setattr(mock_err, "status", "NOT_FOUND")
        setattr(mock_err, "message", "This model models/gemini-2.5-flash is no longer available.")

        diag = format_api_error_diagnostic(mock_err)
        self.assertIn("Exception [HTTP 404 NOT_FOUND]", diag)
        self.assertIn("models/gemini-2.5-flash is no longer available", diag)

    def test_secrets_redacted_from_diagnostic_message(self):
        # Dynamically build a synthetic 39-character AIza token at runtime to test AIza regex redaction
        # without hardcoding a static token that triggers automated repository secret scanners.
        synthetic_aiza_key = "AIza" + "SyMockDiagnosticKeyPatternTest" + "0" * 5
        mock_err = Exception(
            f"Unauthorized: https://api.google.com/v1beta?key={synthetic_aiza_key}&token=secret_pass_123 with Bearer my_jwt_token_456"
        )
        diag = format_api_error_diagnostic(mock_err)

        self.assertNotIn(synthetic_aiza_key, diag)
        self.assertNotIn("secret_pass_123", diag)
        self.assertNotIn("my_jwt_token_456", diag)
        self.assertIn("key=[REDACTED]", diag)
        self.assertIn("token=[REDACTED]", diag)
        self.assertIn("Bearer [REDACTED]", diag)

    def test_aiza_pattern_redacted_standalone(self):
        synthetic_key = "AIza" + "SyStandaloneRedactionPatternTest" + "0" * 3
        mock_err = Exception(f"Internal failure for key {synthetic_key} in request body")
        diag = format_api_error_diagnostic(mock_err)

        self.assertNotIn(synthetic_key, diag)
        self.assertIn("[REDACTED_KEY]", diag)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "mock_custom_env_key_for_redaction_test_12345"}, clear=True)
    def test_loaded_env_api_key_redacted(self):
        mock_err = Exception("Failed call for key mock_custom_env_key_for_redaction_test_12345 in request header")
        diag = format_api_error_diagnostic(mock_err)

        self.assertNotIn("mock_custom_env_key_for_redaction_test_12345", diag)
        self.assertIn("[REDACTED_KEY]", diag)

    def test_generic_exception_without_http_code(self):
        err = TimeoutError("Connection timed out waiting for gateway")
        diag = format_api_error_diagnostic(err)
        self.assertEqual(diag, "TimeoutError: Connection timed out waiting for gateway")


def _is_valid_live_gemini_key(key: str | None) -> bool:
    """Helper to detect if a valid, non-placeholder, non-mock API key is explicitly configured."""
    if not key:
        return False
    stripped = key.strip()
    if not stripped or stripped == "your_gemini_api_key_here":
        return False
    if stripped.lower().startswith(("mock", "dummy", "test", "fake", "placeholder")):
        return False
    return True


class TestGeminiRealApiSmoke(unittest.TestCase):
    """Optional smoke test against live Gemini API; automatically skipped unless GEMINI_API_KEY is configured."""

    def setUp(self):
        key = os.getenv("GEMINI_API_KEY")
        if not _is_valid_live_gemini_key(key):
            self.skipTest(
                "GEMINI_API_KEY is not set in environment or is a placeholder/mock value; skipping live API smoke test."
            )

    @unittest.skipUnless(
        _is_valid_live_gemini_key(os.getenv("GEMINI_API_KEY")),
        "GEMINI_API_KEY is not set in environment or is a placeholder/mock value; skipping live API smoke test.",
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
