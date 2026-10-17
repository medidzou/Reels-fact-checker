"""Tests for ReelCheck Web API endpoints, JobManager, and security headers."""

import io
from unittest.mock import MagicMock, patch
import pytest

from job_manager import JobManager, RateLimiter


def test_check_url_validation_rejects_ssrf(client):
    """Test POST /api/check-url rejects localhost, private IPs and non-Instagram URLs."""
    # Localhost SSRF
    res1 = client.post("/api/check-url", json={"url": "http://127.0.0.1:5000/health"})
    assert res1.status_code == 400
    assert "invalide" in res1.get_json()["error"].lower()

    # AWS metadata SSRF
    res2 = client.post("/api/check-url", json={"url": "http://169.254.169.254/latest/meta-data"})
    assert res2.status_code == 400
    assert "invalide" in res2.get_json()["error"].lower()

    # Malicious scheme
    res3 = client.post("/api/check-url", json={"url": "file:///etc/passwd"})
    assert res3.status_code == 400
    assert "invalide" in res3.get_json()["error"].lower()

    # Arbitrary domain
    res4 = client.post("/api/check-url", json={"url": "https://evil.com/reel/12345"})
    assert res4.status_code == 400
    assert "invalide" in res4.get_json()["error"].lower()

    # Missing or empty body
    res5 = client.post("/api/check-url", json={})
    assert res5.status_code == 400


def test_check_url_valid_creates_job(client, monkeypatch):
    """Test POST /api/check-url with valid Instagram Reel creates an async job."""
    monkeypatch.setattr("job_manager.JobManager.run_job", lambda self, *a, **k: None)
    res = client.post("/api/check-url", json={"url": "https://www.instagram.com/reel/DehQF9lohPI/"})
    assert res.status_code == 202
    data = res.get_json()
    assert "job_id" in data
    assert data["status"] == "processing"


def test_check_url_valid_post_format(client, monkeypatch):
    """Test POST /api/check-url accepts instagram.com/p/ format."""
    monkeypatch.setattr("job_manager.JobManager.run_job", lambda self, *a, **k: None)
    res = client.post("/api/check-url", json={"url": "https://instagram.com/p/C_123xyz/?igsh=test"})
    assert res.status_code == 202
    data = res.get_json()
    assert "job_id" in data
    assert data["status"] == "processing"


def test_get_job_status_not_found(client):
    """Test GET /api/status/<job_id> returns 404 for unknown job ID."""
    res = client.get("/api/status/non-existent-id")
    assert res.status_code == 404
    assert "non trouvé" in res.get_json()["error"].lower()


def test_get_job_status_existing(client):
    """Test GET /api/status/<job_id> retrieves job status."""
    from app import job_manager
    job_id = job_manager.create_job(initial_stage="downloading")
    res = client.get(f"/api/status/{job_id}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["job_id"] == job_id
    assert data["status"] == "processing"
    assert data["stage"] == "downloading"


def test_security_headers_present(client):
    """Test security headers are injected in responses."""
    res = client.get("/health")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert "Content-Security-Policy" in res.headers


def test_check_upload_no_file(client):
    """Test POST /api/check-upload without file returns 400."""
    res = client.post("/api/check-upload", data={})
    assert res.status_code == 400


def test_check_upload_invalid_extension(client):
    """Test POST /api/check-upload rejects dangerous or unsupported extensions."""
    data = {
        "file": (io.BytesIO(b"echo evil"), "script.sh")
    }
    res = client.post("/api/check-upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "supporté" in res.get_json()["error"].lower() or "extension" in res.get_json()["error"].lower()


def test_check_upload_file_too_large(client):
    """Test POST /api/check-upload rejects files larger than 25MB."""
    oversized = io.BytesIO(b"0" * (26 * 1024 * 1024))
    data = {
        "file": (oversized, "huge_video.mp4")
    }
    res = client.post("/api/check-upload", data=data, content_type="multipart/form-data")
    assert res.status_code in (400, 413)


def test_check_upload_valid(client, monkeypatch):
    """Test POST /api/check-upload creates job with valid media file."""
    monkeypatch.setattr("job_manager.JobManager.run_job", lambda self, *a, **k: None)
    valid_file = io.BytesIO(b"mock video data")
    data = {
        "file": (valid_file, "reel_sample.mp4")
    }
    res = client.post("/api/check-upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 202
    res_data = res.get_json()
    assert "job_id" in res_data
    assert res_data["status"] == "processing"


def test_rate_limiter_blocks_excessive_requests(client):
    """Test IP rate limiting rejects more than 5 requests per minute."""
    from app import rate_limiter
    rate_limiter.reset()

    # Send 5 valid requests with a unique test IP
    headers = {"X-Forwarded-For": "203.0.113.195"}
    for _ in range(5):
        res = client.post(
            "/api/check-url",
            json={"url": "https://www.instagram.com/reel/DehQF9lohPI/"},
            headers=headers,
        )
        assert res.status_code in (202, 400)

    # 6th request from same IP should receive 429 Too Many Requests
    res_blocked = client.post(
        "/api/check-url",
        json={"url": "https://www.instagram.com/reel/DehQF9lohPI/"},
        headers=headers,
    )
    assert res_blocked.status_code == 429
    assert "trop de requêtes" in res_blocked.get_json()["error"].lower()


def test_job_manager_lifecycle():
    """Test JobManager create, update, and get lifecycle."""
    jm = JobManager(max_concurrent=1)
    jid = jm.create_job()
    job = jm.get_job(jid)
    assert job is not None
    assert job["status"] == "processing"
    assert job["stage"] == "queued"

    jm.update_job(jid, status="completed", stage="completed", result={"score": 9})
    updated = jm.get_job(jid)
    assert updated["status"] == "completed"
    assert updated["stage"] == "completed"
    assert updated["result"]["score"] == 9
