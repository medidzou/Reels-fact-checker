"""Tests for Groq fallback mechanism and error handling in ReelCheck."""

import os
from unittest.mock import MagicMock, patch
import pytest

from pipeline.fallback import (
    groq_action_kit,
    groq_synthesize,
    groq_transcribe,
    with_fallback,
)
from pipeline.factchecker import Verdict, synthesize_verdict
from pipeline.transcriber import TranscriptionResult, transcribe_and_extract


def test_with_fallback_success():
    """Returns primary result when primary succeeds."""
    primary_mock = MagicMock(return_value="primary_success")
    fallback_mock = MagicMock(return_value="fallback_success")

    result = with_fallback(primary_mock, fallback_mock, "arg1", kw="kw1")

    assert result == "primary_success"
    primary_mock.assert_called_once_with("arg1", kw="kw1")
    fallback_mock.assert_not_called()


def test_with_fallback_failure():
    """Invokes fallback when primary raises an Exception."""
    primary_mock = MagicMock(side_effect=RuntimeError("Primary service down"))
    fallback_mock = MagicMock(return_value="fallback_success")

    result = with_fallback(primary_mock, fallback_mock, "arg1", kw="kw1")

    assert result == "fallback_success"
    primary_mock.assert_called_once_with("arg1", kw="kw1")
    fallback_mock.assert_called_once_with("arg1", kw="kw1")


def test_with_fallback_both_fail():
    """Propagates error when both primary and fallback fail."""
    primary_mock = MagicMock(side_effect=RuntimeError("Primary failed"))
    fallback_mock = MagicMock(side_effect=ValueError("Fallback also failed"))

    with pytest.raises(ValueError, match="Fallback also failed"):
        with_fallback(primary_mock, fallback_mock)


def test_groq_transcribe_missing_api_key(tmp_path):
    """Returns TranscriptionResult with error when GROQ_API_KEY is not set."""
    fake_audio = tmp_path / "audio.mp3"
    fake_audio.write_bytes(b"dummy audio")

    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        result = groq_transcribe(str(fake_audio), client=None)
        assert result.error is not None
        assert "GROQ_API_KEY" in result.error


def test_groq_transcribe_file_not_found():
    """Returns TranscriptionResult with error when audio file does not exist."""
    result = groq_transcribe("/non/existent/path/audio.mp3")
    assert result.error is not None
    assert "not found" in result.error.lower()


def test_groq_transcribe_success_with_mock(tmp_path):
    """Mocks Groq client (audio.transcriptions.create + chat.completions.create) returning parsed TranscriptionResult."""
    fake_audio = tmp_path / "valid.mp3"
    fake_audio.write_bytes(b"dummy audio data")

    mock_client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "L'intelligence artificielle générative transforme le développement logiciel."
    mock_client.audio.transcriptions.create.return_value = mock_transcription

    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """SUMMARY:
Une présentation sur l'impact de l'IA générative dans le code.

CLAIMS:
["L'IA transforme le développement logiciel", "Les gains de productivité dépassent 30%"]

IS_TECH:
true"""
    mock_chat_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    result = groq_transcribe(str(fake_audio), client=mock_client)

    assert result.error is None
    assert "intelligence artificielle" in result.transcript
    assert "impact de l'IA" in result.summary
    assert len(result.claims) == 2
    assert "L'IA transforme le développement logiciel" in result.claims
    assert result.is_tech is True

    # Verify calls
    mock_client.audio.transcriptions.create.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()


def test_groq_synthesize_success_with_mock():
    """Mocks Groq chat completions returning Verdict."""
    mock_client = MagicMock()
    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """VERDICT: VRAI
EXPLANATION: L'affirmation est confirmée par de nombreuses études empiriques.
SOURCES:
- https://example.com/study-ai
- https://mit.edu/research/ai-productivity"""
    mock_chat_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    verdict = groq_synthesize(
        claim="L'IA augmente la productivité",
        search_context="Source: https://example.com/study-ai",
        client=mock_client,
    )

    assert isinstance(verdict, Verdict)
    assert verdict.verdict == "VRAI"
    assert "confirmée" in verdict.explanation
    assert len(verdict.sources) == 2
    assert "https://example.com/study-ai" in verdict.sources
    mock_client.chat.completions.create.assert_called_once()


def test_groq_synthesize_missing_api_key():
    """Returns fallback Verdict when GROQ_API_KEY is not set and no client is passed."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        verdict = groq_synthesize(
            claim="Affirmation test",
            search_context="Context with https://example.com/info",
            client=None,
        )
        assert isinstance(verdict, Verdict)
        assert "GROQ_API_KEY" in verdict.explanation
        assert "https://example.com/info" in verdict.sources


def test_groq_action_kit_success_with_mock():
    """Mocks Groq chat completions returning action kit markdown."""
    mock_client = MagicMock()
    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """# Kit d'Action : Mise en place LLM
1. Installer `pip install groq`
2. Configurer la clé API
3. Lancer l'inférence"""
    mock_chat_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    action_kit = groq_action_kit(
        claims=["Claim tech 1"],
        search_contexts=["Context 1"],
        client=mock_client,
    )

    assert action_kit is not None
    assert "Kit d'Action" in action_kit
    assert "pip install groq" in action_kit
    mock_client.chat.completions.create.assert_called_once()


def test_groq_action_kit_empty_claims():
    """Returns None when claims list is empty."""
    mock_client = MagicMock()
    result = groq_action_kit(claims=[], search_contexts=[], client=mock_client)
    assert result is None
    mock_client.chat.completions.create.assert_not_called()


def test_transcribe_and_extract_gemini_fails_groq_succeeds(tmp_path):
    """Verify transcribe_and_extract seamlessly falls back to Groq when Gemini fails."""
    fake_audio = tmp_path / "valid.mp3"
    fake_audio.write_bytes(b"dummy audio data")

    # Mock failing Gemini client
    mock_gemini_client = MagicMock()
    mock_gemini_client.files.upload.side_effect = RuntimeError("Gemini Quota Exceeded 429")

    # Mock working Groq client
    mock_groq_client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "Texte transcrit via Groq Whisper."
    mock_groq_client.audio.transcriptions.create.return_value = mock_transcription

    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """SUMMARY: Résumé via Groq Llama 3.3.
CLAIMS: ["Affirmation Groq"]
IS_TECH: false"""
    mock_chat_completion.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_chat_completion

    result = transcribe_and_extract(
        str(fake_audio),
        client=mock_gemini_client,
        groq_client=mock_groq_client,
    )

    assert result.error is None
    assert "transcrit via Groq Whisper" in result.transcript
    assert "Affirmation Groq" in result.claims
    mock_gemini_client.files.upload.assert_called_once()
    mock_groq_client.audio.transcriptions.create.assert_called_once()


def test_synthesize_verdict_gemini_fails_groq_succeeds():
    """Verify synthesize_verdict seamlessly falls back to Groq when Gemini fails."""
    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.side_effect = RuntimeError("Gemini Quota Exceeded 429")

    mock_groq_client = MagicMock()
    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """VERDICT: FAUX
EXPLANATION: Totalement contredit selon Groq Llama.
SOURCES:
- https://groq-source.com"""
    mock_chat_completion.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_chat_completion

    verdict = synthesize_verdict(
        claim="La Terre est plate",
        search_context="Recherche web: https://groq-source.com",
        gemini_client=mock_gemini_client,
        groq_client=mock_groq_client,
    )

    assert verdict.verdict == "FAUX"
    assert "Totalement contredit selon Groq" in verdict.explanation
    assert "https://groq-source.com" in verdict.sources
    mock_gemini_client.models.generate_content.assert_called_once()
    mock_groq_client.chat.completions.create.assert_called_once()


def test_groq_transcribe_unsupported_audio_format(tmp_path):
    """Verify Groq Whisper rejects non-audio files (like cookies.txt) with a clean error."""
    txt_file = tmp_path / "cookies.txt"
    txt_file.write_text("sessionid=fake123")

    mock_groq = MagicMock()
    result = groq_transcribe(str(txt_file), client=mock_groq)

    assert result.error is not None
    assert "Unsupported audio format" in result.error
    mock_groq.audio.transcriptions.create.assert_not_called()


def test_groq_chat_completion_candidate_fallback():
    """Verify that if the primary model fails with model_not_found, it falls back to the next model."""
    from pipeline.fallback import _call_groq_chat_completion

    mock_client = MagicMock()
    # First call fails with model_not_found, second call succeeds
    mock_client.chat.completions.create.side_effect = [
        Exception("The model `invalid-model` does not exist (model_not_found)"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Success with candidate"))]),
    ]

    res = _call_groq_chat_completion(
        mock_client,
        [{"role": "user", "content": "hi"}],
        preferred_model="invalid-model",
    )
    assert res is not None
    assert mock_client.chat.completions.create.call_count == 2


