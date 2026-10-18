"""Tests for universal AI dispatch in Telegram bot."""

import pytest
from types import SimpleNamespace
from bot.formatter import format_dispatch_response


def test_telegram_dispatch_formatter_all_categories():
    """Verify telegram formatting produces correct headers and share links for each category."""
    trans = SimpleNamespace(
        transcript="Test transcript",
        summary="Ceci est un résumé clair",
        claims=[],
    )

    categories = [
        ("cooking", "Cuisine & Recette", "🍳"),
        ("health", "Santé & Bien-être", "🌿"),
        ("debate", "Débat & Analyse Rhétorique", "⚖️"),
        ("tech", "Tech, IA & Outils Développeur", "⚡"),
        ("general", "Vérification Factuelle (Général)", "🔍"),
    ]

    for cat_key, cat_label, emoji in categories:
        msgs = format_dispatch_response(
            trans,
            category=cat_key,
            category_label=cat_label,
            share_url="https://reelcheck.app/r/abc123",
        )
        assert len(msgs) >= 1
        full_msg = "\n".join(msgs)
        assert emoji in full_msg
        assert "Ceci est un résumé clair" in full_msg
        assert "https://reelcheck.app/r/abc123" in full_msg
        assert len(full_msg) <= 4096
