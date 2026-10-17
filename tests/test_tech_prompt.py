"""Tests for tech reel audit prompt generation and deep links."""

import pytest
from pipeline.tech_prompt import generate_audit_prompt, generate_deep_links


def test_prompt_contains_machine_specs_and_cyber_rules():
    transcript = "Installez ce repo github pour avoir un bot de trading autonome."
    prompt = generate_audit_prompt(transcript)
    assert "Windows 11" in prompt
    assert "Intel Core i5-1035G1" in prompt
    assert "24 Go" in prompt
    assert "GPU" in prompt
    assert "clés API" in prompt or "cles API" in prompt
    assert "Audit Cyber" in prompt
    assert transcript in prompt


def test_deep_links_generation():
    prompt = "Test audit prompt"
    links = generate_deep_links(prompt)
    assert "chatgpt" in links
    assert "https://chatgpt.com/?q=" in links["chatgpt"]
    assert "claude" in links
    assert "https://claude.ai/new?q=" in links["claude"]
    assert "gemini" in links
    assert "https://gemini.google.com/app" in links["gemini"]


def test_prompt_escapes_or_handles_empty_transcript():
    prompt = generate_audit_prompt("")
    assert "Windows 11" in prompt
    assert "[Aucune transcription disponible]" in prompt
