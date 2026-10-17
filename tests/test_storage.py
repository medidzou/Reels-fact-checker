"""Tests for ephemeral report storage and sharing (/r/<id>)."""

import time
import pytest
from reelcheck.storage import (
    save_report,
    get_report,
    cleanup_expired_reports,
)


def test_save_and_retrieve_report(tmp_path, monkeypatch):
    """Test saving a report persists to disk and can be retrieved."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    sample = {"score": 8.5, "verdict": "VRAI", "analysis": "Tout est vérifié"}
    report_id = save_report(sample)
    assert isinstance(report_id, str)
    assert len(report_id) >= 6

    retrieved = get_report(report_id)
    assert retrieved is not None
    assert retrieved["score"] == 8.5
    assert retrieved["verdict"] == "VRAI"
    assert retrieved["analysis"] == "Tout est vérifié"


def test_get_nonexistent_report(tmp_path, monkeypatch):
    """Test retrieving an unknown report returns None."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    assert get_report("inexistant") is None


def test_get_expired_report(tmp_path, monkeypatch):
    """Test that expired reports return None and are not returned."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    sample = {"score": 5.0, "verdict": "PARTIELLEMENT VRAI"}
    # Save with negative TTL so it is immediately expired
    report_id = save_report(sample, ttl_hours=-1)
    retrieved = get_report(report_id)
    assert retrieved is None


def test_cleanup_expired_reports(tmp_path, monkeypatch):
    """Test cleanup_expired_reports removes only expired files."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)

    # 1 expired report
    expired_id = save_report({"score": 1}, ttl_hours=-2)
    # 1 valid report
    valid_id = save_report({"score": 10}, ttl_hours=24)

    assert (tmp_path / f"{expired_id}.json").exists()
    assert (tmp_path / f"{valid_id}.json").exists()

    removed = cleanup_expired_reports()
    assert removed >= 1

    assert not (tmp_path / f"{expired_id}.json").exists()
    assert (tmp_path / f"{valid_id}.json").exists()
    assert get_report(valid_id) is not None


def test_report_id_traversal_protection(tmp_path, monkeypatch):
    """Test path traversal attempts via report_id return None safely."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    assert get_report("../outside") is None
    assert get_report("..\\outside") is None
    assert get_report("sub/nested") is None


def test_api_get_report_endpoint(client, tmp_path, monkeypatch):
    """Test GET /api/report/<report_id> returns 200 with report payload."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    sample = {"score": 9.0, "verdict": "VRAI", "summary": "Super test"}
    report_id = save_report(sample)

    res = client.get(f"/api/report/{report_id}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["id"] == report_id
    assert data["report"]["score"] == 9.0
    assert data["report"]["verdict"] == "VRAI"


def test_api_get_report_not_found(client, tmp_path, monkeypatch):
    """Test GET /api/report/<report_id> returns 404 for unknown report."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    res = client.get("/api/report/does_not_exist")
    assert res.status_code == 404
    data = res.get_json()
    assert "non trouvé" in data["error"].lower() or "not found" in data["error"].lower()


def test_api_get_report_expired(client, tmp_path, monkeypatch):
    """Test GET /api/report/<report_id> returns 404 when expired."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    report_id = save_report({"score": 3.0}, ttl_hours=-1)

    res = client.get(f"/api/report/{report_id}")
    assert res.status_code == 404


def test_job_completed_hook_saves_report(tmp_path, monkeypatch):
    """Test that updating a job to completed automatically persists report and adds report_id + share_url."""
    from job_manager import JobManager
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    jm = JobManager()
    jid = jm.create_job()
    raw_result = {"score": 9.5, "verdict": "VRAI"}
    jm.update_job(jid, status="completed", stage="completed", result=raw_result)

    job = jm.get_job(jid)
    assert job["result"] is not None
    assert "report_id" in job["result"]
    assert "share_url" in job["result"]
    assert job["result"]["share_url"] == f"/r/{job['result']['report_id']}"

    # Verify report is actually saved and retrievable via get_report
    saved = get_report(job["result"]["report_id"])
    assert saved is not None
    assert saved["score"] == 9.5
    assert saved["report_id"] == job["result"]["report_id"]


def test_view_shared_report_route(client, tmp_path, monkeypatch):
    """Test GET /r/<report_id> serves report or fallback."""
    monkeypatch.setattr("reelcheck.storage.STORAGE_DIR", tmp_path)
    sample = {"score": 7.5, "verdict": "PARTIELLEMENT VRAI"}
    report_id = save_report(sample)

    res = client.get(f"/r/{report_id}")
    assert res.status_code == 200

    # Non-existent report on /r/<id> returns 404
    res_404 = client.get("/r/nonexistent")
    assert res_404.status_code == 404
