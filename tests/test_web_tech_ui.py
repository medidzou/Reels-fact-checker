"""Tests for web UI rendering of Tech Reel Audit panel."""

import pytest
from app import create_app
from storage import save_report


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_renders_tech_banner_and_action_buttons(client):
    report_id = save_report({
        "url": "https://instagram.com/reel/123",
        "is_tech": True,
        "transcript": "Test tech transcript avec MCP et Playwright",
        "summary": "Serveurs MCP pour Claude Desktop",
        "tech_prompt": "Audit Prompt standardisé",
        "deep_links": {
            "chatgpt": "https://chatgpt.com/?q=test",
            "claude": "https://claude.ai/new?q=test",
            "gemini": "https://gemini.google.com/app",
        },
    })

    response = client.get(f"/r/{report_id}")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "REEL TECH" in html
    assert "ChatGPT" in html
    assert "Claude" in html
    assert "Gemini" in html
    assert "Antigravity" in html or "Terminal" in html
