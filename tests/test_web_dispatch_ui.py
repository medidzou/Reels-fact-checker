"""Tests for web UI rendering of multi-category AI dispatch."""

import pytest
from app import create_app
from storage import save_report


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_renders_cooking_dispatch_report(client):
    """Renders cooking category report with badge and AI action buttons."""
    report_id = save_report({
        "url": "https://instagram.com/reel/cooking123",
        "category": "cooking",
        "category_label": "Cuisine & Recette",
        "is_tech": False,
        "transcript": "Ajouter 200g de farine et cuire au four",
        "summary": "Recette de gâteau",
        "tech_prompt": "Prompt Cuisine",
        "verdict": "Cuisine & Recette Détectée",
        "deep_links": {
            "chatgpt": "https://chatgpt.com/?q=test",
            "claude": "https://claude.ai/new?q=test",
            "gemini": "https://gemini.google.com/app",
        },
    })

    response = client.get(f"/r/{report_id}")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Cuisine" in html or "CUISINE" in html
    assert "ChatGPT" in html
    assert "Claude" in html
    assert "Gemini" in html


def test_renders_debate_dispatch_report(client):
    """Renders debate category report with badge and AI action buttons."""
    report_id = save_report({
        "url": "https://instagram.com/reel/debate123",
        "category": "debate",
        "category_label": "Débat & Analyse Rhétorique",
        "is_tech": False,
        "transcript": "Sophisme ad hominem dans ce débat",
        "summary": "Débat politique",
        "tech_prompt": "Prompt Débat",
        "verdict": "Débat & Rhétorique Détecté",
        "deep_links": {
            "chatgpt": "https://chatgpt.com/?q=test",
            "claude": "https://claude.ai/new?q=test",
            "gemini": "https://gemini.google.com/app",
        },
    })

    response = client.get(f"/r/{report_id}")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Débat" in html or "DÉBAT" in html
