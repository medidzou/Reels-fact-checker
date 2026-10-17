"""Tests for ReelCheck frontend routes, template rendering, and PWA manifest."""

import json
import pytest
from reelcheck.storage import save_report


def test_index_route_returns_html(client):
    """Test GET / returns 200 HTML with ReelCheck branding and input elements."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"ReelCheck" in res.data
    assert b"Instagram" in res.data
    assert "text/html" in res.content_type


def test_manifest_json(client):
    """Test GET /static/manifest.json returns valid PWA manifest."""
    res = client.get("/static/manifest.json")
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json()
    assert data["name"] == "ReelCheck"
    assert data["display"] == "standalone"
    assert data["theme_color"] == "#0f172a"


def test_view_shared_report_renders_html_with_data(client, tmp_path, monkeypatch):
    """Test GET /r/<report_id> returns HTML pre-injected with report data."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    sample = {
        "score": 9.0,
        "verdict": "VRAI",
        "analysis": "Factuel et vérifié",
        "summary": "Résumé de vérification",
        "verdicts": [
            {
                "claim": "La Terre est ronde",
                "verdict": "VRAI",
                "explanation": "Prouvé scientifiquement",
                "sources": ["https://fr.wikipedia.org/wiki/Terre"],
            }
        ],
    }
    report_id = save_report(sample)
    res = client.get(f"/r/{report_id}")
    assert res.status_code == 200
    assert "text/html" in res.content_type
    assert b"Factuel et" in res.data or b"9.0" in res.data
    assert b"ReelCheck" in res.data


def test_view_shared_report_not_found(client, tmp_path, monkeypatch):
    """Test GET /r/<nonexistent> returns 404."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    res = client.get("/r/does_not_exist_123")
    assert res.status_code == 404


def test_frontend_has_pwa_tags_and_tailwind(client):
    """Test index page contains PWA manifest link and Tailwind CDN."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "manifest.json" in html
    assert "cdn.tailwindcss.com" in html
    assert "bg-slate-900" in html or "bg-obsidian" in html
