"""Tests for universal AI dispatch in JobManager."""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from job_manager import JobManager, process_upload_job, process_url_job


@patch("job_manager.save_telegram_audio")
@patch("job_manager.transcribe_and_extract")
@patch("job_manager.fact_check_claims")
def test_upload_job_dispatches_directly_without_factcheck(mock_factcheck, mock_transcribe, mock_save):
    """Every upload job must complete immediately after transcription with dispatch report, bypassing factcheck."""
    mock_save.return_value = "fake_audio.mp3"
    mock_transcribe.return_value = SimpleNamespace(
        transcript="Voici comment faire cette recette de gâteau au chocolat au four",
        summary="Recette de gâteau",
        claims=["Farine et chocolat"],
        is_tech=False,
        error=None,
    )

    with patch("os.path.exists", return_value=True):
        jm = JobManager()
        job_id = jm.create_job("upload")
        process_upload_job(jm, job_id, b"fake_bytes", filename="video.mp4")

        job = jm.get_job(job_id)
        assert job["status"] == "completed"
        assert job["stage"] == "completed"
        res = job["result"]
        assert res["category"] == "cooking"
        assert "deep_links" in res
        assert "chatgpt" in res["deep_links"]
        assert "claude" in res["deep_links"]
        assert "gemini" in res["deep_links"]

        # Factcheck must NOT have been called
        mock_factcheck.assert_not_called()


@patch("job_manager.download_reel_audio")
@patch("job_manager.transcribe_and_extract")
@patch("job_manager.fact_check_claims")
def test_url_job_dispatches_directly_without_factcheck(mock_factcheck, mock_transcribe, mock_dl):
    """Every URL job must complete immediately after transcription with dispatch report, bypassing factcheck."""
    dl_mock = MagicMock()
    dl_mock.success = True
    dl_mock.audio_path = "fake_reel.mp3"
    dl_mock.title = "Débat politique"
    dl_mock.description = "Un débat sur l'économie"
    mock_dl.return_value = dl_mock

    mock_transcribe.return_value = SimpleNamespace(
        transcript="Ce débat télévisé oppose deux thèses contradictoires avec des sophismes",
        summary="Débat",
        claims=["Thèse A contre Thèse B"],
        is_tech=False,
        error=None,
    )

    with patch("os.path.exists", return_value=True):
        jm = JobManager()
        job_id = jm.create_job("url")
        process_url_job(jm, job_id, "https://instagram.com/reel/123")

        job = jm.get_job(job_id)
        assert job["status"] == "completed"
        res = job["result"]
        assert res["category"] == "debate"
        assert "deep_links" in res

        # Factcheck must NOT have been called
        mock_factcheck.assert_not_called()
