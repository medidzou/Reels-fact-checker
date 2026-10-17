"""Pytest fixtures for ReelCheck test suite."""

import pytest
from app import create_app


@pytest.fixture
def client(monkeypatch):
    """Flask test client fixture with testing configuration."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_bot_token")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com")
    app = create_app({"TESTING": True})
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_rate_limiter_fixture():
    """Reset the in-memory rate limiter between tests to ensure test isolation."""
    from job_manager import rate_limiter
    rate_limiter.reset()

