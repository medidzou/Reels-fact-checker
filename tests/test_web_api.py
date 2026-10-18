"""Tests for ReelCheck Web API endpoints, JobManager, and security headers."""

import io
import time
from unittest.mock import MagicMock, patch
import pytest

from job_manager import (
    JobManager,
    RateLimiter,
    process_upload_job,
    process_url_job,
)
from pipeline.downloader import DownloadResult
from pipeline.factchecker import FactCheckResult, Verdict
from pipeline.transcriber import TranscriptionResult



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
    """Test GET /api/status/<job_id> and /api/jobs/<job_id> retrieve job status and report payload."""
    from app import job_manager
    job_id = job_manager.create_job(initial_stage="downloading")
    job_manager.update_job(
        job_id,
        status="completed",
        stage="completed",
        result={"summary": "Fact-checked", "score": 8.5, "is_tech": False},
    )

    # Test canonical /api/status/<job_id>
    res1 = client.get(f"/api/status/{job_id}")
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert data1["job_id"] == job_id
    assert data1["status"] == "completed"
    assert data1["stage"] == "completed"
    assert data1["result"]["summary"] == "Fact-checked"
    assert data1["report"]["summary"] == "Fact-checked"
    assert "report_id" in data1

    # Test alias /api/jobs/<job_id> used by web frontend
    res2 = client.get(f"/api/jobs/{job_id}")
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["job_id"] == job_id
    assert data2["report"]["score"] == 8.5
    assert data2["report_id"] == data1["report_id"]


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


def test_check_file_alias_valid(client, monkeypatch):
    """Test POST /api/check-file creates job with valid media file (frontend alias)."""
    monkeypatch.setattr("job_manager.JobManager.run_job", lambda self, *a, **k: None)
    valid_file = io.BytesIO(b"mock video data")
    data = {
        "file": (valid_file, "reel_sample.mp4")
    }
    res = client.post("/api/check-file", data=data, content_type="multipart/form-data")
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


def test_job_manager_auto_prune_old_jobs():
    """Test that create_job automatically prunes jobs older than 2 hours."""
    jm = JobManager(max_concurrent=1)
    old_id = jm.create_job()
    # Artificially age the job by 3 hours
    jm._jobs[old_id]["created_at"] = time.time() - 10000

    # Creating a new job should auto-prune the expired one
    new_id = jm.create_job()
    assert jm.get_job(old_id) is None
    assert jm.get_job(new_id) is not None


def test_rate_limiter_prunes_stale_ips():
    """Test that RateLimiter prunes inactive IP keys from memory."""
    rl = RateLimiter(limits=[(5, 10)])
    rl._requests["old_ip"] = [time.time() - 20]
    allowed, retry_after = rl.is_allowed("new_ip")
    assert allowed is True
    assert "old_ip" not in rl._requests


def test_run_job_integration_process_url_job(tmp_path):
    """Integration test verifying run_job executes process_url_job to completion without TypeError."""
    dummy_audio = tmp_path / "dummy.mp3"
    dummy_audio.write_bytes(b"dummy audio content")

    mock_dl = DownloadResult(audio_path=str(dummy_audio), success=True, error=None)
    mock_transcription = TranscriptionResult(
        transcript="Test transcript",
        summary="Test summary",
        claims=["Claim 1"],
        is_tech=False,
    )
    mock_factcheck = FactCheckResult(
        verdicts=[
            Verdict(
                claim="Claim 1",
                verdict="VRAI",
                explanation="Explication vérifiée",
                sources=["https://trusted-source.org"],
            )
        ],
        reliability_score=8,
        action_kit="Action kit content",
    )

    jm = JobManager(max_concurrent=1)
    jid = jm.create_job()

    with patch("job_manager.download_reel_audio", return_value=mock_dl), \
         patch("job_manager.transcribe_and_extract", return_value=mock_transcription), \
         patch("job_manager.fact_check_claims", return_value=mock_factcheck):
        t = jm.run_job(
            jid,
            process_url_job,
            jm,
            jid,
            "https://www.instagram.com/reel/DehQF9lohPI/",
            force_full=True,
        )
        t.join(timeout=5)

    job = jm.get_job(jid)
    assert job is not None
    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["result"]["score"] == 8
    assert job["result"]["verdict"] == "VRAI"
    assert len(job["result"]["verdicts"]) == 1
    assert job["result"]["verdicts"][0]["claim"] == "Claim 1"


def test_run_job_integration_process_upload_job(tmp_path):
    """Integration test verifying run_job executes process_upload_job to completion without TypeError."""
    dummy_audio = tmp_path / "dummy_upload.mp3"
    dummy_audio.write_bytes(b"dummy audio upload")

    mock_transcription = TranscriptionResult(
        transcript="Transcript from upload",
        summary="Summary from upload",
        claims=["Claim upload"],
        is_tech=False,
    )
    mock_factcheck = FactCheckResult(
        verdicts=[
            Verdict(
                claim="Claim upload",
                verdict="FAUX",
                explanation="Explication faux",
                sources=["https://debunk.org"],
            )
        ],
        reliability_score=2,
        action_kit=None,
    )

    jm = JobManager(max_concurrent=1)
    jid = jm.create_job()

    with patch("job_manager.save_telegram_audio", return_value=str(dummy_audio)), \
         patch("job_manager.transcribe_and_extract", return_value=mock_transcription), \
         patch("job_manager.fact_check_claims", return_value=mock_factcheck):
        t = jm.run_job(
            jid,
            process_upload_job,
            jm,
            jid,
            b"raw video bytes",
            "video.mp4",
            force_full=True,
        )
        t.join(timeout=5)

    job = jm.get_job(jid)
    assert job is not None
    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["result"]["score"] == 2
    assert job["result"]["verdict"] == "FAUX"
    assert job["result"]["summary"] == "Summary from upload"

