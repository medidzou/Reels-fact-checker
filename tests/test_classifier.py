"""Tests for multi-category reel content classification."""

import pytest
from pipeline.classifier import classify_content, CATEGORIES


def test_classify_content_empty_or_none():
    """Empty or None input defaults safely to general category."""
    cat, label = classify_content(None)
    assert cat == "general"
    assert label == CATEGORIES["general"]["label"]

    cat_empty, _ = classify_content("   ")
    assert cat_empty == "general"


def test_classify_content_tech():
    """Technical and dev reels are properly classified."""
    samples = [
        "Découvrez ce nouveau serveur MCP sur GitHub avec Claude Code et Docker",
        "Voici comment configurer Ollama et HuggingFace en local avec python pip install",
        "Ce prompt ChatGPT et Gemini permet de générer du code propre dans votre terminal",
    ]
    for s in samples:
        cat, label = classify_content(s)
        assert cat == "tech", f"Failed on: {s}"
        assert "Tech" in label


def test_classify_content_cooking():
    """Cooking, recipe, and food reels are properly classified."""
    samples = [
        "Pour cette recette, ajoutez 200g de farine, 3 œufs et enfournez à 180 degrés",
        "Voici ma recette de pâtes crémeuses : faites cuire les pâtes, faites revenir l'ail dans la poêle",
        "Un repas sain et rapide avec seulement 400 calories et 35g de protéines pour votre dîner",
    ]
    for s in samples:
        cat, label = classify_content(s)
        assert cat == "cooking", f"Failed on: {s}"
        assert "Cuisine" in label


def test_classify_content_health():
    """Health, medical, and wellness reels are properly classified."""
    samples = [
        "Cette étude montre que le jeûne intermittent réduit la glycémie et le cholestérol",
        "Attention si vous prenez ce médicament contre l'hypertension ou cette maladie",
        "Voici 3 symptômes d'un manque de magnésium et comment booster votre système immunitaire",
    ]
    for s in samples:
        cat, label = classify_content(s)
        assert cat == "health", f"Failed on: {s}"
        assert "Santé" in label


def test_classify_content_debate():
    """Debates, rhetoric, and logic fallacy reels are properly classified."""
    samples = [
        "Dans ce débat télévisé, l'argument est un pur sophisme ad hominem et une fausse dichotomie",
        "Regardez cette joute oratoire où l'orateur démonte l'argument de son adversaire",
        "Voici une erreur de logique flagrante dans ce débat politique avec un homme de paille",
    ]
    for s in samples:
        cat, label = classify_content(s)
        assert cat == "debate", f"Failed on: {s}"
        assert "Débat" in label


def test_classify_content_general_fallback():
    """Unclassified or general reels default to general fact-check category."""
    samples = [
        "Voici les faits marquants de l'actualité sportive et du championnat du monde",
        "Saviez-vous que la tour Eiffel a été construite en 1889 pour l'exposition universelle ?",
        "Une découverte historique fascinante dans les archives nationales de France",
    ]
    for s in samples:
        cat, label = classify_content(s)
        assert cat == "general", f"Failed on: {s}"
        assert "Vérification" in label


def test_tech_classifier_backward_compatibility():
    """Verify pipeline/tech_classifier.is_tech_content still works transparently."""
    from pipeline.tech_classifier import is_tech_content

    assert is_tech_content("Installer un serveur MCP github avec Docker") is True
    assert is_tech_content("Faire une tarte aux pommes au four") is False
    assert is_tech_content(None) is False
