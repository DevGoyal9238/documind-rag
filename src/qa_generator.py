"""
DocuMind - QA Generator Module (Gemini RAG Integration)
Orchestrates grounded question-answering using Google's official google-genai SDK,
retrieved context chunks, strict anti-hallucination prompting, and citation tracking.
"""

import os
import re
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL_NAME = "gemini-3.8-flash"
FALLBACK_ABSTENTION_MESSAGE = "I cannot find the answer to your question in the provided document."
FALLBACK_EMPTY_CONTEXT_MESSAGE = "No relevant document context found to answer the question."
MAX_CONTEXT_CHARACTERS = 40000


def load_gemini_api_key(env_var_name: str = "GEMINI_API_KEY") -> str:
    """
    Lazily loads the Gemini API key from the environment or a local .env file.

    Guarantees:
        - Never prints, logs, or leaks the raw API key in error messages.
        - Raises a clean, descriptive ValueError if the key is missing or is the default placeholder.

    Args:
        env_var_name: Environment variable name to look up (default: GEMINI_API_KEY).

    Returns:
        The validated API key string.

    Raises:
        ValueError: If the environment variable is unset, empty, or set to placeholder text.
    """
    load_dotenv()
    api_key = os.environ.get(env_var_name, "").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            f"{env_var_name} is not configured. "
            "Please create a .env file from .env.example and supply a valid API key from Google AI Studio."
        )

    return api_key


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    """
    Initializes and returns a Google GenAI client instance.
    Supports dependency injection by accepting an explicit api_key or client.

    Args:
        api_key: Optional Gemini API key. If None, loaded via load_gemini_api_key().

    Returns:
        A google.genai.Client instance.
    """
    if api_key is None:
        api_key = load_gemini_api_key()

    return genai.Client(api_key=api_key)


def format_context_block(
    context_chunks: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARACTERS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Formats a list of retrieved chunk dictionaries into a structured text context block
    with unambiguous source attribution headers.

    Strictly enforces max_chars: the returned formatted context string is guaranteed
    never to exceed max_chars characters. If an individual chunk (including the first chunk)
    exceeds max_chars, it is strictly truncated to fit within the limit.

    Preserves actual metadata from retrieved chunks (page number, chunk ID, document ID).
    Never invents or parses citations from LLM output.

    Args:
        context_chunks: List of chunk dictionaries from retrieval.
        max_chars: Safety threshold to prevent prompt context overflow (positive integer > 0).

    Returns:
        A tuple of (formatted_context_string, list_of_source_metadata_dicts).

    Raises:
        TypeError: If context_chunks is not a list, contains non-dict elements,
                   or if max_chars is not an integer.
        ValueError: If max_chars <= 0.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError(f"max_chars must be an integer, got {type(max_chars).__name__}")
    if max_chars <= 0:
        raise ValueError(f"max_chars must be a positive integer greater than 0, got {max_chars}")

    if not isinstance(context_chunks, list):
        raise TypeError(f"context_chunks must be a list, got {type(context_chunks).__name__}")

    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    total_chars = 0
    separator = "\n\n"

    for idx, chunk in enumerate(context_chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Context chunk at index {idx} must be a dictionary, got {type(chunk).__name__}")

        page_number = chunk.get("page_number")
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("document_id")
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        header_parts: list[str] = []
        if page_number is not None:
            header_parts.append(f"Page {page_number}")
        if chunk_id is not None:
            header_parts.append(f"Chunk {chunk_id}")
        if doc_id is not None:
            header_parts.append(f"Doc: {doc_id}")

        header = f"[Source: {' | '.join(header_parts)}]" if header_parts else "[Source: Document]"
        chunk_block = f"{header}\n{text}"

        needed_chars = len(chunk_block) if not lines else len(separator) + len(chunk_block)

        if total_chars + needed_chars > max_chars:
            if not lines:
                # First chunk exceeds max_chars: strictly truncate to fit within max_chars
                truncated_block = chunk_block[:max_chars].rstrip()
                if truncated_block:
                    lines.append(truncated_block)
                    sources.append({
                        "page_number": page_number,
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                    })
            # Safety ceiling reached; stop accumulating further chunks
            break

        lines.append(chunk_block)
        total_chars += needed_chars

        sources.append({
            "page_number": page_number,
            "chunk_id": chunk_id,
            "document_id": doc_id,
        })

    return separator.join(lines).strip(), sources


def build_rag_prompt(question: str, context_block: str) -> str:
    """
    Constructs a grounded RAG prompt enforcing strict non-hallucination constraints
    and explicit abstention instructions.

    Args:
        question: User query string.
        context_block: Formatted context text containing source snippets.

    Returns:
        The formatted prompt string for Gemini.

    Raises:
        TypeError: If question or context_block are not strings.
        ValueError: If question or context_block are empty/whitespace.
    """
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question cannot be empty or whitespace-only")

    if not isinstance(context_block, str):
        raise TypeError(f"context_block must be a string, got {type(context_block).__name__}")
    clean_context = context_block.strip()
    if not clean_context:
        raise ValueError("context_block cannot be empty or whitespace-only")

    return f"""You are DocuMind, an accurate, concise, and helpful AI PDF Study Assistant.
Your task is to answer the user's question using ONLY the provided document context below.

STRICT CONSTRAINTS:
1. Rely strictly and solely on the provided Context. Do NOT assume, extrapolate, or use outside knowledge.
2. If the Context does NOT contain the answer, or is insufficient or irrelevant, you MUST reply with exactly:
   "{FALLBACK_ABSTENTION_MESSAGE}"
3. Cite the relevant Page number(s) from the source tags where applicable.
4. Keep the answer clear, professional, and factual.

Context:
--------------------
{clean_context}
--------------------

User Question: {clean_question}

Answer:"""


def sanitize_error_message(err: Any) -> str:
    """
    Sanitizes raw error strings or exception instances to prevent accidental leakage
    of API keys, bearer tokens, and sensitive query parameters in UI banners, CLI output,
    or logs.

    Args:
        err: Exception instance, string, or arbitrary error representation.

    Returns:
        A sanitized string with credentials redacted and normalized whitespace.
        For empty exceptions (e.g. ValueError("")), returns the exception class name.
    """
    if isinstance(err, BaseException):
        raw = str(err).strip()
        if not raw:
            raw = type(err).__name__
    elif isinstance(err, str):
        raw = err.strip()
        if not raw:
            return ""
    elif err is None:
        return ""
    else:
        raw = str(err).strip()
        if not raw:
            return type(err).__name__

    sanitized = raw

    # 1. Redact active GEMINI_API_KEY from environment if present
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and len(env_key) > 5:
        sanitized = sanitized.replace(env_key, "[REDACTED_KEY]")

    # 2. Redact variable-length Google API key patterns (e.g. AIzaSy... between 26 and 60 chars)
    sanitized = re.sub(r"AIza[0-9A-Za-z_-]{26,60}", "[REDACTED_KEY]", sanitized)

    # 3. Redact Bearer tokens (standard and base64/JWT)
    sanitized = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\-\.\+\=]+", "Bearer [REDACTED]", sanitized)

    # 4. Redact sensitive query parameters (e.g. ?key=..., &auth=..., &token=..., &secret=..., &password=...)
    sanitized = re.sub(
        r"(?i)(\b[\w\-]*(?:key|token|auth|secret|password)[\w\-]*)=[^&\s'\"`]+",
        r"\1=[REDACTED]",
        sanitized,
    )

    # 5. Normalize multiple whitespace and newlines to a clean single space
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    return sanitized


_sanitize_error_text = sanitize_error_message


def format_api_error_diagnostic(err: Exception) -> str:
    """
    Extracts safe, non-sensitive diagnostic details from an API exception.
    Includes the exception class name, HTTP status code (if available),
    status enum (if available), and sanitized message.
    """
    err_cls = type(err).__name__
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    status = getattr(err, "status", None)

    # Prefer specific message attribute if present, otherwise fallback to str(err)
    raw_msg = getattr(err, "message", None)
    if not raw_msg or not isinstance(raw_msg, str):
        raw_msg = str(err)

    clean_msg = _sanitize_error_text(raw_msg)

    # Build structured diagnostic tokens: e.g. "ClientError [HTTP 404 NOT_FOUND]"
    diag_parts: list[str] = [err_cls]
    if code is not None and status is not None:
        diag_parts.append(f"[HTTP {code} {status}]")
    elif code is not None:
        diag_parts.append(f"[HTTP {code}]")
    elif status is not None:
        diag_parts.append(f"[{status}]")

    diagnostic_header = " ".join(diag_parts)

    if clean_msg:
        return f"{diagnostic_header}: {clean_msg}"
    return diagnostic_header


def generate_answer(
    question: str,
    context_chunks: list[dict[str, Any]],
    client: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Coordinates grounded answer generation using Google Gemini.

    Temperature & Determinism Note:
        temperature=0.0 selects greedy (argmax) token decoding, substantially reducing
        sampling variance compared to higher temperatures. However, setting temperature to 0.0
        does NOT guarantee 100% deterministic outputs across distributed GPU kernels, floating-point
        hardware non-associativity, or API version updates, nor does it guarantee an output
        is completely free of hallucinations. Strict prompt grounding and fallback abstention
        rules are required in tandem.

    Args:
        question: User search query.
        context_chunks: Retrieved document chunks with text and page metadata.
        client: Optional pre-configured genai.Client instance (for testing/dependency injection).
        model_name: Gemini model identifier (default: gemini-3.8-flash).
        temperature: Sampling temperature (default: 0.0 for minimum variance).

    Returns:
        Dictionary containing:
            - 'answer': Generated response text or fallback abstention string.
            - 'sources': List of source metadata dicts (page_number, chunk_id, document_id).
            - 'model': Model name used.
            - 'abstention': Boolean flag indicating whether the system abstained from answering.

    Raises:
        TypeError: If question or context_chunks have invalid types.
        ValueError: If question is empty or whitespace-only.
        RuntimeError: If Gemini API generation fails.
    """
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")

    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question cannot be empty or whitespace-only")

    if not isinstance(context_chunks, list):
        raise TypeError(f"context_chunks must be a list, got {type(context_chunks).__name__}")

    # 1. Pre-LLM Guard: If context is empty, return fallback immediately without calling API
    if not context_chunks:
        return {
            "answer": FALLBACK_EMPTY_CONTEXT_MESSAGE,
            "sources": [],
            "model": model_name,
            "abstention": True,
        }

    # 2. Format context block and preserve actual chunk citation metadata
    context_block, sources = format_context_block(context_chunks)

    if not context_block:
        return {
            "answer": FALLBACK_EMPTY_CONTEXT_MESSAGE,
            "sources": [],
            "model": model_name,
            "abstention": True,
        }

    # 3. Assemble grounded prompt
    prompt = build_rag_prompt(clean_question, context_block)

    # 4. Resolve client lazily if not provided
    if client is None:
        client = get_gemini_client()

    # 5. Call Gemini API
    try:
        config = types.GenerateContentConfig(temperature=temperature)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
    except Exception as err:
        diagnostic = format_api_error_diagnostic(err)
        raise RuntimeError(
            f"Gemini API generation failed ({diagnostic}). "
            "Please check your API key, model availability, network connection, or quota."
        ) from err

    # 6. Extract generated text safely
    answer_text = ""
    if hasattr(response, "text") and response.text:
        answer_text = response.text.strip()

    if not answer_text:
        answer_text = FALLBACK_ABSTENTION_MESSAGE

    is_abstention = FALLBACK_ABSTENTION_MESSAGE.lower() in answer_text.lower()

    return {
        "answer": answer_text,
        "sources": sources,
        "model": model_name,
        "abstention": is_abstention,
    }
