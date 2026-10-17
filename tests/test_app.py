"""Tests for Flask app, health check, and webhook endpoints."""

import os
from unittest.mock import AsyncMock, patch
import pytest

from app import create_app, setup_webhook


@pytest.fixture
def test_client(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_bot_token")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com")
    app = create_app({"TESTING": True})
    return app.test_client()


def test_health_check(test_client):
    """Test GET / health check returns 200 OK."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "reelcheck"


def test_health_check_alias(test_client):
    """Test GET /health returns 200 OK."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "reelcheck"


def test_webhook_wrong_token(test_client):
    """Test POST /<token> and /webhook/<token> with wrong token returns 403."""
    response = test_client.post("/wrong_token", json={})
    assert response.status_code == 403
    data = response.get_json()
    assert data["status"] == "forbidden"

    response_alias = test_client.post("/webhook/wrong_token", json={})
    assert response_alias.status_code == 403
    assert response_alias.get_json()["status"] == "forbidden"


def test_webhook_empty_payload(test_client):
    """Test POST /<token> without payload returns 400."""
    response = test_client.post("/test_bot_token", data="")
    assert response.status_code == 400

    response_alias = test_client.post("/webhook/test_bot_token", data="")
    assert response_alias.status_code == 400


def test_webhook_valid_update(test_client):
    """Test POST /<token> with valid update payload returns 200."""
    update_payload = {
        "update_id": 999999,
        "message": {
            "message_id": 1,
            "date": 1700000000,
            "chat": {"id": 12345, "type": "private"},
            "text": "/help",
            "entities": [{"type": "bot_command", "offset": 0, "length": 5}],
        },
    }
    with patch("telegram.ext.Application.process_update", new_callable=AsyncMock) as mock_process:
        response = test_client.post("/test_bot_token", json=update_payload)
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        mock_process.assert_awaited_once()

    with patch("telegram.ext.Application.process_update", new_callable=AsyncMock) as mock_process_alias:
        update_payload_alias = dict(update_payload, update_id=999998)
        response_alias = test_client.post("/webhook/test_bot_token", json=update_payload_alias)
        assert response_alias.status_code == 200
        assert response_alias.get_json()["status"] == "ok"
        mock_process_alias.assert_awaited_once()


def test_webhook_duplicate_update_ignored(test_client):
    """Test duplicate update_id is acknowledged with 200 OK without re-processing."""
    payload = {
        "update_id": 112233,
        "message": {
            "message_id": 2,
            "date": 1700000000,
            "chat": {"id": 12345, "type": "private"},
            "text": "/help",
        },
    }
    with patch("telegram.ext.Application.process_update", new_callable=AsyncMock) as mock_process:
        res1 = test_client.post("/test_bot_token", json=payload)
        assert res1.status_code == 200
        mock_process.assert_awaited_once()

    # Second call with identical update_id should be ignored
    with patch("telegram.ext.Application.process_update", new_callable=AsyncMock) as mock_process_dup:
        res2 = test_client.post("/test_bot_token", json=payload)
        assert res2.status_code == 200
        assert res2.get_json().get("duplicate") is True
        mock_process_dup.assert_not_called()




def test_setup_webhook_calls_bot():
    """Test setup_webhook configures Telegram webhook."""
    with patch("telegram.Bot.delete_webhook", new_callable=AsyncMock) as mock_del, \
         patch("telegram.Bot.set_webhook", new_callable=AsyncMock) as mock_set, \
         patch("telegram.Bot.shutdown", new_callable=AsyncMock):
        success = setup_webhook("token123", "https://mybot.onrender.com")
        assert success is True
        mock_del.assert_awaited_once()
        mock_set.assert_awaited_once_with(url="https://mybot.onrender.com/token123")


def test_setup_webhook_missing_args():
    """Test setup_webhook returns False when args are missing."""
    assert setup_webhook("", "https://example.com") is False
    assert setup_webhook("token", "") is False


def test_async_worker_lifecycle():
    """Test AsyncWorker persistent loop executes multiple sequential coroutines."""
    from app import AsyncWorker

    worker = AsyncWorker()
    assert worker.loop.is_running()

    async def _add(a, b):
        return a + b

    assert worker.run(_add(2, 3)) == 5
    assert worker.run(_add(10, 20)) == 30

    worker.stop()

