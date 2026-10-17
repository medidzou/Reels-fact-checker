"""Tests for response formatter, message splitting, and user mode management."""

import pytest

from bot.formatter import (
    DEFAULT_MODE,
    SUPPORTED_MODES,
    format_response,
    get_user_mode,
    set_user_mode,
    split_message,
)
from pipeline.factchecker import FactCheckResult, Verdict
from pipeline.transcriber import TranscriptionResult


def test_format_response_full_mode():
    """Test full mode formatting includes all sections."""
    tr = TranscriptionResult(
        transcript="Test transcript spoken in video.",
        summary="Test summary of the reel.",
        claims=["Claim 1", "Claim 2"],
        is_tech=True,
        error=None,
    )
    fc = FactCheckResult(
        verdicts=[
            Verdict("Claim 1", "VRAI", "Confirmed by official documentation.", ["https://example.com/doc"]),
            Verdict("Claim 2", "FAUX", "Refuted by benchmark studies.", ["https://example.com/study"]),
        ],
        reliability_score=5,
        action_kit="### Installation\n`pip install cool-lib`",
    )

    messages = format_response(tr, fc, mode="full")

    assert len(messages) >= 1
    full_output = "\n".join(messages)
    assert "TRANSCRIPTION" in full_output
    assert "RÉSUMÉ" in full_output
    assert "FACT-CHECK" in full_output
    assert "SCORE DE FIABILITÉ" in full_output
    assert "KIT D'ACTION" in full_output
    assert "Claim 1" in full_output
    assert "Claim 2" in full_output
    assert "5/10" in full_output
    assert "pip install cool-lib" in full_output
    assert "TRANSCRIPTION" in messages[0]
    assert "FACT-CHECK" in messages[0] or "FACT-CHECK" in messages[-1]


def test_format_response_fact_mode():
    """Test fact mode contains summary, verdicts, score, but no transcription or action kit."""
    tr = TranscriptionResult(
        transcript="Sensitive transcript that shouldn't appear in fact mode.",
        summary="Short summary.",
        claims=["Earth is flat"],
        is_tech=True,
    )
    fc = FactCheckResult(
        verdicts=[
            Verdict("Earth is flat", "FAUX", "Earth is an oblate spheroid.", ["https://nasa.gov"]),
        ],
        reliability_score=0,
        action_kit="Do not attempt to walk off edge.",
    )

    messages = format_response(tr, fc, mode="fact")
    full_output = "\n".join(messages)

    assert "RÉSUMÉ" in full_output
    assert "FACT-CHECK" in full_output
    assert "SCORE DE FIABILITÉ" in full_output
    assert "0/10" in full_output
    assert "TRANSCRIPTION" not in full_output
    assert "KIT D'ACTION" not in full_output


def test_format_response_action_mode():
    """Test action mode contains summary and action kit, but no transcription or verdicts."""
    tr = TranscriptionResult(
        transcript="Full speech transcript.",
        summary="A tutorial on Docker.",
        claims=["Docker is lightweight"],
        is_tech=True,
    )
    fc = FactCheckResult(
        verdicts=[
            Verdict("Docker is lightweight", "VRAI", "Containerization uses shared kernel.", []),
        ],
        reliability_score=10,
        action_kit="Step 1: docker build -t myapp .",
    )

    messages = format_response(tr, fc, mode="action")
    full_output = "\n".join(messages)

    assert "RÉSUMÉ" in full_output
    assert "KIT D'ACTION" in full_output
    assert "Step 1: docker build" in full_output
    assert "TRANSCRIPTION" not in full_output
    assert "FACT-CHECK" not in full_output
    assert "SCORE DE FIABILITÉ" not in full_output


def test_format_response_action_mode_no_action_kit():
    """Test action mode handles absence of action kit gracefully."""
    tr = TranscriptionResult(
        transcript="A video about cooking pasta.",
        summary="Pasta cooking tips.",
        claims=["Salt makes water boil faster"],
        is_tech=False,
    )
    fc = FactCheckResult(
        verdicts=[
            Verdict("Salt makes water boil faster", "FAUX", "Effect is negligible.", []),
        ],
        reliability_score=0,
        action_kit=None,
    )

    messages = format_response(tr, fc, mode="action")
    full_output = "\n".join(messages)

    assert "RÉSUMÉ" in full_output
    assert "KIT D'ACTION" in full_output
    assert "Aucun kit d'action" in full_output


def test_format_response_verdict_emojis():
    """Test that verdicts use correct visual emojis (✅, ❌, ⚠️)."""
    tr = TranscriptionResult(
        transcript="Test",
        summary="Summary",
        claims=["C1", "C2", "C3"],
    )
    fc = FactCheckResult(
        verdicts=[
            Verdict("C1", "VRAI", "True claim", []),
            Verdict("C2", "FAUX", "False claim", []),
            Verdict("C3", "PARTIELLEMENT VRAI", "Partial claim", []),
        ],
        reliability_score=5,
    )

    messages = format_response(tr, fc, mode="full")
    full_output = "\n".join(messages)

    assert "✅" in full_output
    assert "❌" in full_output
    assert "⚠️" in full_output


def test_format_response_empty_claims():
    """Test response formatting when video has no factual claims."""
    tr = TranscriptionResult(
        transcript="Hello beautiful world! Have a wonderful day!",
        summary="A motivational greeting.",
        claims=[],
        is_tech=False,
    )
    fc = FactCheckResult(
        verdicts=[],
        reliability_score=10,
        action_kit=None,
    )

    messages = format_response(tr, fc, mode="full")
    full_output = "\n".join(messages)

    assert "RÉSUMÉ" in full_output
    assert "10/10" in full_output
    assert "Aucune affirmation factuelle" in full_output


def test_format_response_splits_when_exceeding_limit():
    """Test format_response automatically splits into multiple messages if content exceeds 4096 chars."""
    tr = TranscriptionResult(
        transcript="Very long transcript sentence. " * 200,  # ~6200 chars
        summary="A long summary.",
        claims=["Claim 1"],
    )
    fc = FactCheckResult(
        verdicts=[Verdict("Claim 1", "VRAI", "Explanation.", [])],
        reliability_score=10,
    )

    messages = format_response(tr, fc, mode="full")
    assert len(messages) >= 2
    for m in messages:
        assert len(m) <= 4096


def test_split_message_respects_limit():
    """Test split_message handles exact limit and splits large unbroken texts."""
    long_text = "A" * 5000
    parts = split_message(long_text, max_len=4096)
    assert all(len(p) <= 4096 for p in parts)
    assert len(parts) == 2


def test_split_message_preserves_paragraphs():
    """Test split_message prefers splitting on double newlines."""
    p1 = "Paragraph 1 content. " * 100  # ~2100 chars
    p2 = "Paragraph 2 content. " * 100  # ~2100 chars
    combined = f"{p1}\n\n{p2}"

    parts = split_message(combined, max_len=3000)
    assert len(parts) == 2
    assert parts[0].strip() == p1.strip()
    assert parts[1].strip() == p2.strip()


def test_split_message_preserves_single_newlines():
    """Test split_message falls back to single newlines if double newlines are absent."""
    line1 = "Line 1 words here. " * 60  # ~1140 chars
    line2 = "Line 2 words here. " * 60  # ~1140 chars
    combined = f"{line1}\n{line2}"

    parts = split_message(combined, max_len=1500)
    assert len(parts) == 2
    assert parts[0].strip() == line1.strip()
    assert parts[1].strip() == line2.strip()


def test_split_message_preserves_words():
    """Test split_message splits on space without breaking words."""
    words = "word " * 600  # 3000 chars
    parts = split_message(words, max_len=1600)
    assert len(parts) == 2
    for p in parts:
        assert len(p) <= 1600
        assert not p.endswith("wor")  # Should not cut mid-word


def test_split_message_empty():
    """Test split_message on empty or small string."""
    assert split_message("") == []
    assert split_message("Short message") == ["Short message"]


def test_user_mode_management():
    """Test get_user_mode, set_user_mode, and invalid mode validation."""
    chat_id = 998877

    # Default is "full"
    assert get_user_mode(chat_id) == DEFAULT_MODE

    # Set valid modes
    set_user_mode(chat_id, "fact")
    assert get_user_mode(chat_id) == "fact"

    set_user_mode(chat_id, "action")
    assert get_user_mode(chat_id) == "action"

    set_user_mode(chat_id, "full")
    assert get_user_mode(chat_id) == "full"

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid mode"):
        set_user_mode(chat_id, "invalid_mode")
