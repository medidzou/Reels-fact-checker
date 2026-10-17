"""Tests for tech reel heuristic classifier."""

import pytest
from pipeline.tech_classifier import is_tech_content


def test_detects_mcp_and_repo_keywords():
    text = "Arrêtez d'utiliser Claude avant d'avoir installé ces 4 extensions MCP : Firecrawl, Playwright, Perplexity."
    assert is_tech_content(text) is True


def test_detects_github_and_prompt_engineering():
    text = "Voici un repo github incroyable avec un prompt pour automatiser vos tests avec docker et python."
    assert is_tech_content(text) is True


def test_ignores_non_tech_news_and_health():
    text = "Le président a annoncé aujourd'hui une réforme des retraites votée à l'assemblée nationale."
    assert is_tech_content(text) is False


def test_handles_empty_and_none_text():
    assert is_tech_content("") is False
    assert is_tech_content(None) is False


def test_avoids_false_positives_isolated_words():
    text = "Claude François chantait le lundi au soleil en espérant un prompt rétablissement."
    assert is_tech_content(text) is False


def test_detects_ollama_and_local_ai_tools():
    text = "Installez Ollama en local avec ce modèle open-source pour coder sans payer d'API."
    assert is_tech_content(text) is True
