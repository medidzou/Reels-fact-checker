"""Unit tests for the Gemini audio transcription and claim extraction module."""

import os
from unittest.mock import MagicMock, patch
import pytest

from pipeline.transcriber import (
    TranscriptionResult,
    parse_gemini_response,
    transcribe_and_extract,
)


def test_transcription_result_dataclass():
    """Test TranscriptionResult default fields and initialization."""
    res = TranscriptionResult(
        transcript="Test transcript",
        summary="Test summary",
        claims=["Claim 1", "Claim 2"],
        is_tech=False,
        error=None,
    )
    assert res.transcript == "Test transcript"
    assert res.summary == "Test summary"
    assert len(res.claims) == 2
    assert res.is_tech is False
    assert res.error is None


def test_transcription_result_defaults():
    """Test TranscriptionResult default values."""
    res = TranscriptionResult()
    assert res.transcript == ""
    assert res.summary == ""
    assert res.claims == []
    assert res.is_tech is False
    assert res.error is None


def test_parse_gemini_response_standard_format():
    """Test parsing standard key-value format as in task specification."""
    raw = """TRANSCRIPT: Selon une étude de Harvard, 90% des gens...
SUMMARY: Une vidéo qui parle d'une étude.
CLAIMS: ["90% des gens préfèrent X selon Harvard"]
IS_TECH: false"""

    result = parse_gemini_response(raw)
    assert result.transcript.startswith("Selon")
    assert result.summary == "Une vidéo qui parle d'une étude."
    assert len(result.claims) == 1
    assert result.claims[0] == "90% des gens préfèrent X selon Harvard"
    assert result.is_tech is False
    assert result.error is None


def test_parse_gemini_response_multiline_transcript_and_summary():
    """Test parsing when transcript and summary contain multiple lines."""
    raw = """TRANSCRIPT: Première ligne de transcription.
Deuxième ligne avec des détails sur l'algorithme.
SUMMARY: C'est un résumé complet.
Sur deux lignes.
CLAIMS: ["L'algorithme traite 1000 requêtes par seconde", "La latence est de 5ms"]
IS_TECH: true"""

    result = parse_gemini_response(raw)
    assert "Première ligne" in result.transcript
    assert "Deuxième ligne" in result.transcript
    assert "Sur deux lignes" in result.summary
    assert len(result.claims) == 2
    assert result.is_tech is True
    assert result.error is None


def test_parse_gemini_response_bullet_point_claims():
    """Test parsing claims formatted as markdown bullet list."""
    raw = """TRANSCRIPT: Texte parlé de la vidéo.
SUMMARY: Résumé court.
CLAIMS:
- 75% des entreprises utilisent l'IA en 2025
- Le coût moyen a baissé de 30%
IS_TECH: true"""

    result = parse_gemini_response(raw)
    assert len(result.claims) == 2
    assert "75% des entreprises" in result.claims[0]
    assert "30%" in result.claims[1]
    assert result.is_tech is True


def test_parse_gemini_response_numbered_claims():
    """Test parsing claims formatted as numbered list."""
    raw = """TRANSCRIPT: Texte parlé.
SUMMARY: Résumé.
CLAIMS:
1. Premier fait vérifiable
2. Deuxième fait chiffré: 10 millions d'utilisateurs
IS_TECH: false"""

    result = parse_gemini_response(raw)
    assert len(result.claims) == 2
    assert result.claims[0] == "Premier fait vérifiable"
    assert "10 millions" in result.claims[1]
    assert result.is_tech is False


def test_parse_gemini_response_json_in_markdown_fence():
    """Test parsing when Gemini outputs a JSON block inside markdown code fences."""
    raw = """Voici l'analyse demandée:
```json
{
  "transcript": "Transcription complète du reel.",
  "summary": "Résumé de l'outil présenté.",
  "claims": ["L'outil est 10x plus rapide", "Open source sous licence MIT"],
  "is_tech": true
}
```"""

    result = parse_gemini_response(raw)
    assert result.transcript == "Transcription complète du reel."
    assert result.summary == "Résumé de l'outil présenté."
    assert len(result.claims) == 2
    assert result.claims[0] == "L'outil est 10x plus rapide"
    assert result.is_tech is True
    assert result.error is None


def test_parse_gemini_response_raw_json():
    """Test parsing when response is pure JSON."""
    raw = """{
  "transcript": "Reel audio text",
  "summary": "Short summary",
  "claims": ["Fact 1"],
  "is_tech": false
}"""
    result = parse_gemini_response(raw)
    assert result.transcript == "Reel audio text"
    assert result.summary == "Short summary"
    assert result.claims == ["Fact 1"]
    assert result.is_tech is False


def test_parse_gemini_response_fallback_unstructured():
    """Test fallback when response is unstructured text without expected sections."""
    raw = "Ceci est un texte brut sans format particulier qui décrit une vidéo."
    result = parse_gemini_response(raw)
    assert result.transcript == raw
    assert result.claims == []
    assert result.is_tech is False
    assert result.error is None


def test_parse_gemini_response_empty():
    """Test parsing an empty string."""
    result = parse_gemini_response("")
    assert result.transcript == ""
    assert result.summary == ""
    assert result.claims == []
    assert result.is_tech is False


def test_transcribe_and_extract_missing_file():
    """Test transcribe_and_extract with a non-existent file path."""
    result = transcribe_and_extract("/non/existent/path/to/audio.mp3")
    assert result.error is not None
    assert "not found" in result.error.lower()
    assert result.transcript == ""
    assert result.claims == []


def test_transcribe_and_extract_missing_api_key_no_client(tmp_path):
    """Test transcribe_and_extract with no client and missing GEMINI_API_KEY."""
    fake_audio = tmp_path / "test.mp3"
    fake_audio.write_bytes(b"dummy audio content")

    with patch.dict(os.environ, {}, clear=True):
        # Ensure GEMINI_API_KEY is not in env
        os.environ.pop("GEMINI_API_KEY", None)
        result = transcribe_and_extract(str(fake_audio), client=None)
        assert result.error is not None
        assert "GEMINI_API_KEY" in result.error


def test_transcribe_and_extract_success_with_mock_client(tmp_path):
    """Test successful audio upload, Gemini call, and parsing using injected mock client."""
    audio_file = tmp_path / "valid_reel.mp3"
    audio_file.write_bytes(b"fake mp3 audio data")

    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "files/test-file-123"

    mock_response = MagicMock()
    mock_response.text = """TRANSCRIPT: Selon une étude du MIT, l'IA générative augmente la productivité de 37%.
SUMMARY: Une étude sur l'impact de l'IA sur les développeurs.
CLAIMS: ["L'IA augmente la productivité de 37% selon le MIT"]
IS_TECH: true"""

    mock_client = MagicMock()
    mock_client.files.upload.return_value = mock_uploaded_file
    mock_client.models.generate_content.return_value = mock_response

    result = transcribe_and_extract(str(audio_file), client=mock_client)

    assert result.error is None
    assert "37%" in result.transcript
    assert len(result.claims) == 1
    assert result.is_tech is True

    # Verify client interactions
    mock_client.files.upload.assert_called_once_with(file=str(audio_file))
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "model" in call_kwargs
    assert "contents" in call_kwargs
    # Check that contents contains the uploaded file
    assert mock_uploaded_file in call_kwargs["contents"]


def test_transcribe_and_extract_client_exception_handling(tmp_path):
    """Test graceful error handling when Gemini API raises an exception."""
    audio_file = tmp_path / "valid_reel.mp3"
    audio_file.write_bytes(b"fake mp3 audio data")

    mock_client = MagicMock()
    mock_client.files.upload.side_effect = RuntimeError("Quota exceeded (429)")

    result = transcribe_and_extract(str(audio_file), client=mock_client)
    assert result.error is not None
    assert "Quota exceeded" in result.error
    assert result.transcript == ""
