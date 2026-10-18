"""Tests for Google Fact Check Tools API (ClaimReview) fast-check engine."""

import io
import json
from unittest.mock import MagicMock, patch
import urllib.request
import pytest

from pipeline.claim_review import query_claim_review
from pipeline.factchecker import Verdict


def test_query_claim_review_no_api_key(monkeypatch):
    """Returns None when no API key is configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert query_claim_review("Le citron guérit le cancer", api_key=None) is None


def test_query_claim_review_empty_claim():
    """Returns None for empty or whitespace-only claims."""
    assert query_claim_review("   ", api_key="dummy-key") is None


def test_query_claim_review_success(monkeypatch):
    """Parses official ClaimReview response into a structured Verdict."""
    mock_payload = {
        "claims": [
            {
                "text": "Boire du jus de citron à jeun guérit le cancer.",
                "claimant": "Réseaux sociaux",
                "claimReview": [
                    {
                        "publisher": {"name": "AFP Factuel", "site": "factuel.afp.com"},
                        "url": "https://factuel.afp.com/le-citron-et-le-cancer",
                        "title": "Non, le citron ne remplace pas une chimiothérapie",
                        "textualRating": "Faux",
                        "languageCode": "fr",
                    }
                ],
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.object(urllib.request, "urlopen", return_value=mock_resp):
        verdict = query_claim_review("Le jus de citron guérit le cancer", api_key="test-key")

    assert verdict is not None
    assert isinstance(verdict, Verdict)
    assert verdict.verdict == "FAUX"
    assert "AFP Factuel" in verdict.explanation
    assert "https://factuel.afp.com/le-citron-et-le-cancer" in verdict.sources


def test_query_claim_review_no_matches():
    """Returns None when ClaimReview finds no matches."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"claims": []}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.object(urllib.request, "urlopen", return_value=mock_resp):
        verdict = query_claim_review("Un fait obscur jamais fact-checké", api_key="test-key")

    assert verdict is None


def test_query_claim_review_handles_network_error():
    """Handles network errors gracefully and returns None."""
    with patch.object(urllib.request, "urlopen", side_effect=Exception("Connection timeout")):
        verdict = query_claim_review("Affirmation test", api_key="test-key")

    assert verdict is None
