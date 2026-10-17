"""Tests for tech reel short-circuiting in job_manager."""

import os
from unittest.mock import MagicMock, patch
import pytest

from job_manager import JobManager, process_url_job, process_upload_job
from pipeline.transcriber import TranscriptionResult
from pipeline.downloader import DownloadResult


def test_process_url_job_short_circuits_for_tech_reel(tmp_path):
    jm = JobManager()
    job_id = jm.create_job()

    tech_transcription = TranscriptionResult(
        transcript="Arrêtez tout ! Installez ces 4 serveurs MCP pour Claude Desktop : Firecrawl et Playwright.",
        summary="Extensions MCP pour Claude.",
        claims=["Firecrawl permet de scraper"],
        is_tech=True,
    )

    with patch("job_manager.download_reel_audio") as mock_dl, \
         patch("job_manager.transcribe_and_extract", return_value=tech_transcription), \
         patch("job_manager.fact_check_claims") as mock_factcheck:

        # Create dummy audio file
        dummy_audio = tmp_path / "dummy.mp3"
        dummy_audio.write_text("audio content")
        mock_dl.return_value = DownloadResult(success=True, audio_path=str(dummy_audio))

        process_url_job(jm, job_id, "https://instagram.com/reel/123")

        job_data = jm.get_job(job_id)
        assert job_data is not None
        assert job_data["status"] == "completed"
        assert job_data["stage"] == "completed"
        
        result = job_data["result"]
        assert result["is_tech"] is True
        assert "tech_prompt" in result
        assert "deep_links" in result
        assert "chatgpt" in result["deep_links"]
        assert "claude" in result["deep_links"]
        assert "gemini" in result["deep_links"]
        
        # Verify fact_check_claims was NOT called (short-circuit confirmed)
        assert mock_factcheck.called is False


def test_process_upload_job_short_circuits_for_tech_reel(tmp_path):
    jm = JobManager()
    job_id = jm.create_job()

    tech_transcription = TranscriptionResult(
        transcript="Voici un repo github incroyable pour coder avec docker et python.",
        summary="Repo Github pour coder.",
        claims=["Repo Github"],
        is_tech=False,  # Evaluated by is_tech_content fallback
    )

    with patch("job_manager.save_telegram_audio") as mock_save, \
         patch("job_manager.transcribe_and_extract", return_value=tech_transcription), \
         patch("job_manager.fact_check_claims") as mock_factcheck:

        dummy_audio = tmp_path / "dummy.mp3"
        dummy_audio.write_text("audio content")
        mock_save.return_value = str(dummy_audio)

        process_upload_job(jm, job_id, b"fake_audio_bytes")

        job_data = jm.get_job(job_id)
        assert job_data is not None
        assert job_data["status"] == "completed"
        
        result = job_data["result"]
        assert result["is_tech"] is True
        assert "tech_prompt" in result
        assert mock_factcheck.called is False
