"""Fallback mechanisms for transcription and fact-checking using Groq API."""

from collections.abc import Callable
import logging
import os
from typing import Any, Optional

from pipeline.factchecker import (
    ACTION_KIT_PROMPT,
    SYNTHESIS_PROMPT,
    Verdict,
    _extract_urls,
    parse_verdict_response,
)
from pipeline.transcriber import (
    TranscriptionResult,
    parse_gemini_response,
)

logger = logging.getLogger(__name__)

DEFAULT_GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
DEFAULT_GROQ_LLM_MODEL = "openai/gpt-oss-120b"

GROQ_EXTRACTION_PROMPT = """Analyze this transcript from an Instagram Reel:

{transcript}

Perform the following tasks:
1. Provide a concise summary of the content (3 to 5 lines).
2. Identify and extract ONLY verifiable factual claims (e.g. statistics, percentages, scientific studies, historical events, named facts, technical specs). Do NOT include subjective opinions, feelings, or vague advice.
3. Determine whether the video is technical/AI/software-related (true or false).

Output your response using EXACTLY this format:
SUMMARY:
<3 to 5 line summary>

CLAIMS:
["claim 1", "claim 2"]

IS_TECH:
<true or false>
"""


def with_fallback(
    primary_fn: Callable[..., Any],
    fallback_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute primary_fn; if an exception is raised, log a warning and execute fallback_fn.

    Args:
        primary_fn: Primary function to attempt.
        fallback_fn: Fallback function to invoke on primary failure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Result from primary_fn if successful, else result from fallback_fn.
    """
    try:
        return primary_fn(*args, **kwargs)
    except Exception as exc:
        primary_name = getattr(primary_fn, "__name__", str(primary_fn))
        fallback_name = getattr(fallback_fn, "__name__", str(fallback_fn))
        logger.warning(
            "Primary function %s failed (%s), invoking fallback %s",
            primary_name,
            exc,
            fallback_name,
        )
        return fallback_fn(*args, **kwargs)


def _get_groq_client(client: Optional[Any] = None) -> tuple[Optional[Any], Optional[str]]:
    """Retrieve or initialize Groq client."""
    if client is not None:
        return client, None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY environment variable not set"

    try:
        from groq import Groq

        return Groq(api_key=api_key), None
    except Exception as err:
        return None, f"Failed to initialize Groq client: {err}"


def groq_transcribe(
    audio_path: str,
    client: Optional[Any] = None,
) -> TranscriptionResult:
    """Fallback audio transcription + claim extraction using Groq Free Tier.

    Uses whisper-large-v3-turbo for audio transcription and llama-3.3-70b-versatile
    for claim extraction.

    Args:
        audio_path: Local filesystem path to the audio file.
        client: Optional pre-configured Groq client instance.

    Returns:
        TranscriptionResult containing transcript, summary, claims, is_tech, and error.
    """
    if not audio_path or not os.path.exists(audio_path):
        return TranscriptionResult(error=f"Audio file not found: {audio_path}")

    groq_client, err = _get_groq_client(client)
    if err or groq_client is None:
        return TranscriptionResult(error=err)

    whisper_model = os.environ.get("GROQ_WHISPER_MODEL", DEFAULT_GROQ_WHISPER_MODEL)
    llm_model = os.environ.get("GROQ_LLM_MODEL", DEFAULT_GROQ_LLM_MODEL)

    valid_groq_extensions = (
        ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".opus", ".wav", ".webm"
    )
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in valid_groq_extensions:
        return TranscriptionResult(
            error=f"Unsupported audio format for Groq Whisper: '{ext}' ({audio_path})"
        )

    try:
        # 1. Transcribe audio using Whisper
        logger.info("Transcribing audio %s with Groq Whisper model %s...", audio_path, whisper_model)
        with open(audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                model=whisper_model,
                file=(os.path.basename(audio_path), f.read()),
            )
        raw_transcript = getattr(transcription, "text", "") or ""
    except Exception as exc:
        logger.error("Groq Whisper transcription failed: %s", exc, exc_info=True)
        return TranscriptionResult(error=f"Groq Whisper transcription failed: {exc}")

    try:
        # 2. Extract claims and metadata using Llama 3.3
        logger.info("Extracting claims using Groq LLM model %s...", llm_model)
        prompt = GROQ_EXTRACTION_PROMPT.format(transcript=raw_transcript)
        chat_completion = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        chat_text = ""
        if hasattr(chat_completion, "choices") and chat_completion.choices:
            chat_text = getattr(chat_completion.choices[0].message, "content", "") or ""

        parsed = parse_gemini_response(chat_text)
        if not parsed.transcript:
            parsed.transcript = raw_transcript
        return parsed
    except Exception as exc:
        logger.error("Groq claim extraction failed: %s", exc, exc_info=True)
        return TranscriptionResult(
            transcript=raw_transcript,
            error=f"Groq claim extraction failed: {exc}",
        )


def groq_synthesize(
    claim: str,
    search_context: str,
    client: Optional[Any] = None,
) -> Verdict:
    """Fallback verdict synthesis using Groq llama-3.3-70b-versatile.

    Args:
        claim: The factual claim to verify.
        search_context: Web search evidence context.
        client: Optional pre-configured Groq client instance.

    Returns:
        Structured Verdict object.
    """
    clean_claim = claim.strip() if claim else ""
    groq_client, err = _get_groq_client(client)
    if err or groq_client is None:
        return Verdict(
            claim=clean_claim,
            verdict="PARTIELLEMENT VRAI",
            explanation=f"Erreur Groq: {err}",
            sources=_extract_urls(search_context),
        )

    llm_model = os.environ.get("GROQ_LLM_MODEL", DEFAULT_GROQ_LLM_MODEL)
    prompt = SYNTHESIS_PROMPT.format(
        claim=clean_claim,
        search_context=search_context or "Aucun résultat trouvé sur le web.",
    )

    try:
        logger.info("Synthesizing verdict with Groq model %s for claim '%s'...", llm_model, clean_claim)
        chat_completion = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = ""
        if hasattr(chat_completion, "choices") and chat_completion.choices:
            raw_text = getattr(chat_completion.choices[0].message, "content", "") or ""

        verdict = parse_verdict_response(raw_text, clean_claim)
        if not verdict.sources and search_context:
            verdict.sources = _extract_urls(search_context)
        return verdict
    except Exception as exc:
        logger.error("Groq verdict synthesis error for claim '%s': %s", clean_claim, exc, exc_info=True)
        return Verdict(
            claim=clean_claim,
            verdict="PARTIELLEMENT VRAI",
            explanation=f"Erreur lors de la synthèse Groq: {exc}",
            sources=_extract_urls(search_context),
        )


def groq_action_kit(
    claims: list[str],
    search_contexts: list[str],
    client: Optional[Any] = None,
) -> Optional[str]:
    """Fallback action kit generation using Groq llama-3.3-70b-versatile.

    Args:
        claims: List of technical claims.
        search_contexts: Context/search results gathered for the claims.
        client: Optional pre-configured Groq client instance.

    Returns:
        Markdown string containing the action kit, or None if skipped/failed.
    """
    valid_claims = [c.strip() for c in claims if c and c.strip()]
    if not valid_claims:
        return None

    groq_client, err = _get_groq_client(client)
    if err or groq_client is None:
        logger.warning("Cannot generate action kit via Groq: %s", err)
        return None

    llm_model = os.environ.get("GROQ_LLM_MODEL", DEFAULT_GROQ_LLM_MODEL)
    claims_text = "\n".join(f"- {c}" for c in valid_claims)
    search_text = "\n\n".join(
        f"--- Contexte affirmation {i+1} ---\n{ctx}"
        for i, ctx in enumerate(search_contexts)
        if ctx
    )

    prompt = ACTION_KIT_PROMPT.format(
        claims_text=claims_text,
        search_text=search_text or "Aucune recherche supplémentaire.",
    )

    try:
        logger.info("Generating action kit with Groq model %s...", llm_model)
        chat_completion = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        if hasattr(chat_completion, "choices") and chat_completion.choices:
            text = getattr(chat_completion.choices[0].message, "content", "")
            if text:
                return text.strip()
        return None
    except Exception as exc:
        logger.error("Groq action kit generation error: %s", exc, exc_info=True)
        return None
