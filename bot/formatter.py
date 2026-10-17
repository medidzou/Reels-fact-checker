"""Telegram response formatting, message chunking, and user mode management."""

import logging
from typing import Optional

from pipeline.factchecker import FactCheckResult
from pipeline.transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

SUPPORTED_MODES = {"full", "fact", "action"}
DEFAULT_MODE = "full"

# In-memory user modes dictionary: {chat_id: mode}
_user_modes: dict[int, str] = {}

VERDICT_EMOJIS = {
    "VRAI": "✅",
    "FAUX": "❌",
    "PARTIELLEMENT VRAI": "⚠️",
}


def get_user_mode(chat_id: int) -> str:
    """Return current analysis mode for a user/chat. Defaults to 'full'."""
    return _user_modes.get(chat_id, DEFAULT_MODE)


def set_user_mode(chat_id: int, mode: str) -> None:
    """Set analysis mode for a user/chat.

    Args:
        chat_id: Telegram chat ID.
        mode: One of 'full', 'fact', or 'action'.

    Raises:
        ValueError: If mode is not one of the supported modes.
    """
    clean_mode = mode.strip().lower() if mode else ""
    if clean_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Supported modes are: {', '.join(sorted(SUPPORTED_MODES))}"
        )
    _user_modes[chat_id] = clean_mode


def split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks not exceeding max_len characters.

    Splits preferentially at double newlines (paragraphs), single newlines,
    or spaces to avoid breaking sentences or words. If no whitespace boundary
    exists within max_len, a hard break at max_len is used.

    Args:
        text: The string to split.
        max_len: Maximum length of each chunk (Telegram limit is 4096).

    Returns:
        List of chunks, each with length <= max_len.
    """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text.strip()

    while len(remaining) > max_len:
        split_idx = -1

        # 1. Double newline (paragraph break)
        p_idx = remaining.rfind("\n\n", 0, max_len)
        if p_idx > 0:
            split_idx = p_idx + 2
        else:
            # 2. Single newline
            nl_idx = remaining.rfind("\n", 0, max_len)
            if nl_idx > 0:
                split_idx = nl_idx + 1
            else:
                # 3. Space
                sp_idx = remaining.rfind(" ", 0, max_len)
                if sp_idx > 0:
                    split_idx = sp_idx + 1
                else:
                    # 4. Hard break at max_len
                    split_idx = max_len

        if split_idx <= 0:
            split_idx = max_len

        chunk = remaining[:split_idx].rstrip()
        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_idx:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _format_transcription_section(transcription: TranscriptionResult) -> str:
    """Format the transcription section."""
    transcript_text = transcription.transcript.strip() if transcription.transcript else ""
    if not transcript_text:
        transcript_text = "(Aucune transcription disponible)"
    return f"📝 **TRANSCRIPTION**\n\n{transcript_text}"


def _format_summary_section(transcription: TranscriptionResult) -> str:
    """Format the summary section."""
    summary_text = transcription.summary.strip() if transcription.summary else ""
    if not summary_text:
        summary_text = "(Aucun résumé disponible)"
    return f"📋 **RÉSUMÉ**\n\n{summary_text}"


def _format_factcheck_section(factcheck: FactCheckResult) -> str:
    """Format the fact-check verdicts section."""
    lines = ["🔍 **FACT-CHECK**"]
    if not factcheck.verdicts:
        lines.append("\n• Aucune affirmation factuelle détectée à vérifier.")
        return "\n".join(lines)

    lines.append("")
    for i, v in enumerate(factcheck.verdicts, 1):
        emoji = VERDICT_EMOJIS.get(v.verdict.upper(), "⚠️")
        claim_block = [
            f"• **Affirmation {i}** : {v.claim}",
            f"  **Verdict** : {emoji} {v.verdict}",
            f"  **Explication** : {v.explanation}",
        ]
        if v.sources:
            claim_block.append("  **Sources** :")
            for src in v.sources:
                claim_block.append(f"    - {src}")
        lines.append("\n".join(claim_block))

    return "\n\n".join(lines)


def _format_reliability_section(factcheck: FactCheckResult) -> str:
    """Format the reliability score section."""
    score = factcheck.reliability_score
    if score >= 8:
        indicator = "🟢 Fiable"
    elif score >= 5:
        indicator = "🟡 Partiellement fiable / À nuancer"
    else:
        indicator = "🔴 Peu fiable / Trompeur"

    return f"📊 **SCORE DE FIABILITÉ** : {score}/10 ({indicator})"


def _format_action_kit_section(factcheck: FactCheckResult, fallback_if_empty: bool = False) -> Optional[str]:
    """Format the action kit section if present or required."""
    if factcheck.action_kit and factcheck.action_kit.strip():
        return f"🛠️ **KIT D'ACTION**\n\n{factcheck.action_kit.strip()}"
    if fallback_if_empty:
        return "🛠️ **KIT D'ACTION**\n\nAucun kit d'action disponible pour ce contenu (sujet non technique ou aucune affirmation exploitable)."
    return None


def format_response(
    transcription: TranscriptionResult,
    factcheck: FactCheckResult,
    mode: str = "full",
) -> list[str]:
    """Format transcription and fact-check results into a list of messages.

    Args:
        transcription: Result from transcription and claim extraction.
        factcheck: Result from fact-check search and synthesis.
        mode: Display mode ('full', 'fact', or 'action').

    Returns:
        List of formatted strings, each <= 4096 characters.
    """
    clean_mode = mode.strip().lower() if mode else DEFAULT_MODE
    if clean_mode not in SUPPORTED_MODES:
        clean_mode = DEFAULT_MODE

    sections: list[str] = []

    if clean_mode == "full":
        # Full mode: Transcription, Summary, Fact-check, Score, Action Kit (if present)
        sections.append(_format_transcription_section(transcription))
        sections.append(_format_summary_section(transcription))
        sections.append(_format_factcheck_section(factcheck))
        sections.append(_format_reliability_section(factcheck))
        action_kit = _format_action_kit_section(factcheck, fallback_if_empty=False)
        if action_kit:
            sections.append(action_kit)

    elif clean_mode == "fact":
        # Fact mode: Summary, Fact-check, Score
        sections.append(_format_summary_section(transcription))
        sections.append(_format_factcheck_section(factcheck))
        sections.append(_format_reliability_section(factcheck))

    elif clean_mode == "action":
        # Action mode: Summary, Action Kit (with fallback message if absent)
        sections.append(_format_summary_section(transcription))
        action_kit = _format_action_kit_section(factcheck, fallback_if_empty=True)
        if action_kit:
            sections.append(action_kit)

    full_text = "\n\n".join(sections)
    return split_message(full_text, max_len=4096)
