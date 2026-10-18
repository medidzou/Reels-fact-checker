"""Audio transcription and claim extraction pipeline using Google Gemini API."""

import ast
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

EXTRACTION_PROMPT = """Analyze this audio recording from an Instagram Reel.
Perform the following tasks:
1. Provide an accurate, verbatim transcript of everything spoken in the audio.
2. Provide a concise summary of the content (3 to 5 lines).
3. Identify and extract ONLY verifiable factual claims (e.g. statistics, percentages, scientific studies, historical events, named facts, technical specs). Extract up to the 3 or 4 most important claims. Do NOT include subjective opinions, feelings, or vague advice.
4. Determine whether the video is technical/AI/software-related (true or false).

Output your response using EXACTLY this format:
TRANSCRIPT:
<full transcript text>

SUMMARY:
<3 to 5 line summary>

CLAIMS:
["claim 1", "claim 2"]

IS_TECH:
<true or false>
"""


@dataclass
class TranscriptionResult:
    """Represents the output of audio transcription and claim extraction."""

    transcript: str = ""
    summary: str = ""
    claims: list[str] = field(default_factory=list)
    is_tech: bool = False
    error: Optional[str] = None


def _parse_bool(val: str) -> bool:
    """Parse boolean strings such as 'true', 'yes', 'oui', '1'."""
    clean = val.strip().lower()
    return clean in {"true", "yes", "oui", "1", "vrai"}


def _parse_claims_block(claims_raw: str) -> list[str]:
    """Parse claims from JSON array, literal eval, or bullet/numbered lists."""
    claims_raw = claims_raw.strip()
    if not claims_raw:
        return []

    # Attempt 1: Standard JSON or Python literal parsing (e.g. ["claim 1", "claim 2"])
    json_match = re.search(r"\[[\s\S]*\]", claims_raw)
    if json_match:
        bracketed = json_match.group(0)
        try:
            parsed = json.loads(bracketed)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except Exception:
            try:
                parsed = ast.literal_eval(bracketed)
                if isinstance(parsed, list):
                    return [str(c).strip() for c in parsed if str(c).strip()]
            except Exception:
                pass

    # Attempt 2: Bullet points or numbered lists (- claim, * claim, 1. claim)
    claims: list[str] = []
    lines = claims_raw.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullets (- , * , • ) or numbering (1. , 1) )
        cleaned = re.sub(r"^(?:[-*•]|\d+[\.)])\s*", "", line).strip()
        # Remove surrounding quotes if present
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        if cleaned:
            claims.append(cleaned)

    return claims


def parse_gemini_response(raw_text: str) -> TranscriptionResult:
    """Parse raw text from Gemini into a structured TranscriptionResult.

    Supports:
    - Structured text with TRANSCRIPT, SUMMARY, CLAIMS, IS_TECH headers
    - Pure JSON or JSON inside markdown code fences
    - Fallback unstructured text
    """
    if not raw_text or not raw_text.strip():
        return TranscriptionResult()

    text = raw_text.strip()

    # 1. Check for JSON block (either full text or in ```json ... ``` fence)
    json_candidates = []
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        json_candidates.append(fence_match.group(1).strip())
    json_candidates.append(text)

    for candidate in json_candidates:
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    transcript = str(data.get("transcript", "")).strip()
                    summary = str(data.get("summary", "")).strip()
                    raw_claims = data.get("claims", [])
                    if isinstance(raw_claims, list):
                        claims = [str(c).strip() for c in raw_claims if str(c).strip()]
                    elif isinstance(raw_claims, str):
                        claims = _parse_claims_block(raw_claims)
                    else:
                        claims = []

                    raw_is_tech = data.get("is_tech", False)
                    if isinstance(raw_is_tech, bool):
                        is_tech = raw_is_tech
                    else:
                        is_tech = _parse_bool(str(raw_is_tech))

                    return TranscriptionResult(
                        transcript=transcript,
                        summary=summary,
                        claims=claims,
                        is_tech=is_tech,
                        error=None,
                    )
            except Exception:
                pass

    # 2. Check for section-based text format
    # Patterns for section headers
    section_patterns = {
        "transcript": r"(?:^|\n)\s*(?:\*\*)?(?:TRANSCRIPT|TRANSCRIPTION)(?:\*\*)?\s*:\s*",
        "summary": r"(?:^|\n)\s*(?:\*\*)?(?:SUMMARY|RÉSUMÉ)(?:\*\*)?\s*:\s*",
        "claims": r"(?:^|\n)\s*(?:\*\*)?(?:CLAIMS|AFFIRMATIONS)(?:\*\*)?\s*:\s*",
        "is_tech": r"(?:^|\n)\s*(?:\*\*)?(?:IS_TECH|TECH)(?:\*\*)?\s*:\s*",
    }

    # Find start indices of all present sections
    matches: list[tuple[str, int, int]] = []
    for sec_name, pattern in section_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            matches.append((sec_name, m.start(), m.end()))

    if matches:
        # Sort matches by start position in text
        matches.sort(key=lambda x: x[1])

        sections: dict[str, str] = {}
        for i, (name, start, end) in enumerate(matches):
            next_start = matches[i + 1][1] if i + 1 < len(matches) else len(text)
            sections[name] = text[end:next_start].strip()

        transcript = sections.get("transcript", "").strip()
        summary = sections.get("summary", "").strip()
        claims_raw = sections.get("claims", "")
        claims = _parse_claims_block(claims_raw)
        is_tech = _parse_bool(sections.get("is_tech", "false"))

        return TranscriptionResult(
            transcript=transcript,
            summary=summary,
            claims=claims,
            is_tech=is_tech,
            error=None,
        )

    # 3. Fallback: unstructured text treated as transcript
    return TranscriptionResult(
        transcript=text,
        summary="",
        claims=[],
        is_tech=False,
        error=None,
    )


def transcribe_and_extract(
    audio_path: str,
    client: Optional[Any] = None,
    groq_client: Optional[Any] = None,
) -> TranscriptionResult:
    """Transcribe an audio file and extract factual claims using Google Gemini API with Groq fallback.

    Args:
        audio_path: Local filesystem path to the extracted audio file.
        client: Optional pre-configured genai.Client instance for dependency injection.
        groq_client: Optional pre-configured Groq client instance for fallback.

    Returns:
        TranscriptionResult containing transcript, summary, claims, is_tech, and error.
    """
    if not audio_path or not os.path.exists(audio_path):
        return TranscriptionResult(
            error=f"Audio file not found: {audio_path}",
        )

    valid_audio_extensions = (
        ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".opus", ".wav", ".webm", ".aac"
    )
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in valid_audio_extensions:
        return TranscriptionResult(
            error=f"Unsupported media format for transcription: '{ext}' ({audio_path})",
        )

    def _fallback_or_error(err_msg: str) -> TranscriptionResult:
        if os.environ.get("GROQ_API_KEY") or groq_client is not None:
            logger.info("Gemini failed (%s), attempting Groq fallback for transcription...", err_msg)
            from pipeline.fallback import groq_transcribe

            return groq_transcribe(audio_path, client=groq_client)
        return TranscriptionResult(error=err_msg)

    # Setup Google GenAI Client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return _fallback_or_error("GEMINI_API_KEY environment variable not set")
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
        except Exception as init_err:
            return _fallback_or_error(f"Failed to initialize Gemini client: {init_err}")

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    uploaded_file = None

    try:
        # Upload audio file via Files API
        logger.info(f"Uploading audio file {audio_path} to Gemini...")
        uploaded_file = client.files.upload(file=audio_path)

        # Generate multimodal transcription and claim extraction
        logger.info(f"Requesting transcription and claim extraction with model {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents=[uploaded_file, EXTRACTION_PROMPT],
        )

        raw_text = response.text if hasattr(response, "text") and response.text else ""
        result = parse_gemini_response(raw_text)
        if result.error:
            return _fallback_or_error(result.error)
        return result

    except Exception as e:
        logger.error(f"Error during Gemini transcription: {e}", exc_info=True)
        return _fallback_or_error(f"Gemini transcription failed: {str(e)}")

    finally:
        # Clean up uploaded audio on Gemini Free Tier to avoid storage accumulation
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as del_err:
                logger.debug(f"Could not delete uploaded file {uploaded_file.name}: {del_err}")
