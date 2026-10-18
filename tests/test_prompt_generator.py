"""Tests for prompt generation and deep link dispatching across all Reel categories."""

from types import SimpleNamespace
import pytest
from pipeline.prompt_generator import (
    generate_prompt_for_category,
    generate_deep_links,
    build_dispatch_report,
)


def test_generate_prompt_for_all_categories():
    """Each category produces a distinct, high-value prompt containing the transcript."""
    sample_text = "Voici mon test de transcription verbatim."

    for cat in ["tech", "cooking", "health", "debate", "general"]:
        prompt = generate_prompt_for_category(sample_text, cat)
        assert sample_text in prompt
        assert len(prompt) > 100

        if cat == "tech":
            assert "i5-1035G1" in prompt
            assert "cybersécurité" in prompt or "cyber" in prompt
        elif cat == "cooking":
            assert "ingrédients" in prompt.lower()
            assert "calories" in prompt.lower()
        elif cat == "health":
            assert "consensus" in prompt.lower() or "scientifique" in prompt.lower()
            assert "risques" in prompt.lower() or "dangers" in prompt.lower()
        elif cat == "debate":
            assert "sophisme" in prompt.lower() or "fallac" in prompt.lower()
            assert "arguments" in prompt.lower()
        elif cat == "general":
            assert "vérification" in prompt.lower() or "factuel" in prompt.lower()


def test_generate_deep_links():
    """Generates valid 1-click links for ChatGPT, Claude, and Gemini."""
    prompt = "Court prompt de test"
    links = generate_deep_links(prompt)

    assert "chatgpt" in links
    assert "https://chatgpt.com/?q=" in links["chatgpt"]
    assert "claude" in links
    assert "https://claude.ai/new?q=" in links["claude"]
    assert "gemini" in links
    assert links["gemini"] == "https://gemini.google.com/app"


def test_generate_deep_links_truncates_long_query():
    """Ensures query length avoids HTTP 414 errors for URL query params."""
    long_prompt = "A" * 5000
    links = generate_deep_links(long_prompt)

    # URL query string must be safely bounded
    assert len(links["chatgpt"]) < 4000
    assert len(links["claude"]) < 4000


def test_build_dispatch_report():
    """Builds a standardized dispatch report dict with category metadata."""
    trans = SimpleNamespace(
        transcript="Ajouter 200g de sucre et 3 oeufs pour cette recette de cuisine",
        summary="Recette de cuisine facile",
        claims=["200g de sucre nécessaires"],
        is_tech=False,
    )
    report = build_dispatch_report(trans)

    assert report["category"] == "cooking"
    assert "Cuisine" in report["category_label"]
    assert report["is_tech"] is False
    assert report["transcript"] == trans.transcript
    assert report["summary"] == trans.summary
    assert "tech_prompt" in report  # Backward compatibility
    assert "deep_links" in report
    assert "chatgpt" in report["deep_links"]
    assert report["verdict"] == "Cuisine & Recette Détectée"


def test_tech_prompt_backward_compatibility():
    """Ensures legacy imports from pipeline.tech_prompt still work."""
    from pipeline.tech_prompt import generate_audit_prompt, build_tech_report_dict

    prompt = generate_audit_prompt("MCP server github")
    assert "i5-1035G1" in prompt

    trans = SimpleNamespace(transcript="MCP server github", summary="Tech tool", claims=[], is_tech=True)
    report = build_tech_report_dict(trans)
    assert report["is_tech"] is True
    assert "deep_links" in report
