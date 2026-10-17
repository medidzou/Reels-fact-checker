"""ReelCheck processing pipeline."""

from .downloader import (
    DownloadResult,
    download_reel_audio,
    save_telegram_audio,
)
from .factchecker import (
    FactCheckResult,
    Verdict,
    fact_check_claims,
    generate_action_kit,
    search_claim,
    synthesize_verdict,
)
from .fallback import (
    groq_action_kit,
    groq_synthesize,
    groq_transcribe,
    with_fallback,
)
from .transcriber import (
    TranscriptionResult,
    parse_gemini_response,
    transcribe_and_extract,
)

__all__ = [
    "DownloadResult",
    "download_reel_audio",
    "save_telegram_audio",
    "TranscriptionResult",
    "parse_gemini_response",
    "transcribe_and_extract",
    "Verdict",
    "FactCheckResult",
    "search_claim",
    "synthesize_verdict",
    "generate_action_kit",
    "fact_check_claims",
    "with_fallback",
    "groq_transcribe",
    "groq_synthesize",
    "groq_action_kit",
]
