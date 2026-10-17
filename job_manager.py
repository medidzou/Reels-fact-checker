"""Asynchronous Job Manager, Rate Limiter, and Background Pipeline Workers for ReelCheck."""

from collections import deque
import logging
import os
import tempfile
import threading
import time
from typing import Any, Callable, Optional
import uuid

from pipeline.downloader import download_reel_audio, save_telegram_audio
from pipeline.factchecker import fact_check_claims
from pipeline.transcriber import transcribe_and_extract

logger = logging.getLogger(__name__)


class JobManager:
    """Thread-safe in-memory job manager for asynchronous video processing tasks."""

    def __init__(self, max_concurrent: int = 1) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.semaphore = threading.Semaphore(max_concurrent)

    def create_job(self, initial_stage: str = "queued") -> str:
        """Create a new job and return its unique identifier."""
        job_id = uuid.uuid4().hex
        now = time.time()
        job_data = {
            "job_id": job_id,
            "status": "processing",
            "stage": initial_stage,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._jobs[job_id] = job_data
        return job_id

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve job data by id."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return dict(job)
            return None

    def update_job(
        self,
        job_id: str,
        status: str,
        stage: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the status, stage, result, or error of a job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                self._jobs[job_id]["stage"] = stage
                if result is not None:
                    self._jobs[job_id]["result"] = result
                if error is not None:
                    self._jobs[job_id]["error"] = error
                self._jobs[job_id]["updated_at"] = time.time()

    def run_job(
        self,
        job_id: str,
        target: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> threading.Thread:
        """Run a job target function in a background thread under the concurrency semaphore."""
        def _worker() -> None:
            try:
                with self.semaphore:
                    target(job_id, *args, **kwargs)
            except Exception as exc:
                logger.error("Job %s execution failed: %s", job_id, exc, exc_info=True)
                self.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=str(exc),
                )

        t = threading.Thread(target=_worker, daemon=True, name=f"JobWorker-{job_id}")
        t.start()
        return t

    def cleanup_old_jobs(self, max_age_seconds: int = 86400) -> int:
        """Remove jobs older than max_age_seconds."""
        now = time.time()
        removed = 0
        with self._lock:
            expired = [
                jid for jid, data in self._jobs.items()
                if now - data.get("created_at", now) > max_age_seconds
            ]
            for jid in expired:
                del self._jobs[jid]
                removed += 1
        return removed


class RateLimiter:
    """Thread-safe sliding-window in-memory IP rate limiter."""

    def __init__(self, limits: Optional[list[tuple[int, int]]] = None) -> None:
        # Default limits: 5 req/min (60s), 30 req/h (3600s)
        self.limits = limits or [(5, 60), (30, 3600)]
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> tuple[bool, Optional[int]]:
        """Check if an IP address is allowed under the configured rate limits.

        Returns:
            (True, None) if allowed.
            (False, retry_after_seconds) if limit exceeded.
        """
        now = time.time()
        with self._lock:
            timestamps = self._requests.get(ip, [])
            max_window = max(w for _, w in self.limits)
            timestamps = [t for t in timestamps if now - t < max_window]
            self._requests[ip] = timestamps

            for max_count, window_seconds in self.limits:
                window_timestamps = [t for t in timestamps if now - t < window_seconds]
                if len(window_timestamps) >= max_count:
                    oldest_in_window = min(window_timestamps)
                    retry_after = int(window_seconds - (now - oldest_in_window)) + 1
                    return False, max(1, retry_after)

            timestamps.append(now)
            return True, None

    def reset(self) -> None:
        """Clear all stored rate limit timestamps."""
        with self._lock:
            self._requests.clear()


def _build_result_dict(transcription: Any, factcheck: Any) -> dict[str, Any]:
    """Helper to structure pipeline output into a clean JSON serializable dictionary."""
    verdict_summary = "VRAI"
    if any(v.verdict == "FAUX" for v in factcheck.verdicts):
        verdict_summary = "FAUX"
    elif any(v.verdict == "PARTIELLEMENT VRAI" for v in factcheck.verdicts):
        verdict_summary = "PARTIELLEMENT VRAI"

    return {
        "summary": transcription.summary,
        "transcript": transcription.transcript,
        "is_tech": transcription.is_tech,
        "score": factcheck.reliability_score,
        "verdict": verdict_summary,
        "explanation": getattr(factcheck, "explanation", None),
        "claims": transcription.claims,
        "verdicts": [
            {
                "claim": v.claim,
                "verdict": v.verdict,
                "explanation": v.explanation,
                "sources": v.sources,
            }
            for v in factcheck.verdicts
        ],
        "action_kit": factcheck.action_kit,
    }


def process_url_job(job_mgr: JobManager, job_id: str, url: str) -> None:
    """Execute download, transcription, and factchecking for an Instagram URL."""
    job_mgr.update_job(job_id, status="processing", stage="downloading")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dl_res = download_reel_audio(url, tmpdir)
            if not dl_res.success or not dl_res.audio_path or not os.path.exists(dl_res.audio_path):
                raw_err = dl_res.error or "Échec du téléchargement du Reel Instagram"
                if any(x in raw_err.lower() for x in ["login", "401", "restricted", "private", "connexion"]):
                    err_msg = "INSTAGRAM_RESTRICTED: Connexion requise par Instagram. Téléchargez la vidéo et utilisez l'upload."
                else:
                    err_msg = raw_err
                job_mgr.update_job(job_id, status="failed", stage="downloading", error=err_msg)
                return

            audio_path = dl_res.audio_path
            job_mgr.update_job(job_id, status="processing", stage="transcribing")
            transcription = transcribe_and_extract(audio_path)
            if transcription.error:
                job_mgr.update_job(
                    job_id,
                    status="failed",
                    stage="transcribing",
                    error=f"Erreur de transcription : {transcription.error}",
                )
                return

            job_mgr.update_job(job_id, status="processing", stage="factchecking")
            factcheck = fact_check_claims(transcription.claims, transcription.is_tech)

            result = _build_result_dict(transcription, factcheck)
            job_mgr.update_job(job_id, status="completed", stage="completed", result=result)
    except Exception as exc:
        logger.error("Error in process_url_job for job %s: %s", job_id, exc, exc_info=True)
        job_mgr.update_job(job_id, status="failed", stage="failed", error=str(exc))


def process_upload_job(
    job_mgr: JobManager,
    job_id: str,
    file_bytes: bytes,
    filename: Optional[str] = None,
) -> None:
    """Execute transcription and factchecking for an uploaded audio/video file."""
    job_mgr.update_job(job_id, status="processing", stage="transcribing")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = save_telegram_audio(file_bytes, tmpdir, filename=filename)
            if not audio_path or not os.path.exists(audio_path):
                job_mgr.update_job(
                    job_id,
                    status="failed",
                    stage="transcribing",
                    error="Impossible d'enregistrer le fichier pour analyse",
                )
                return

            transcription = transcribe_and_extract(audio_path)
            if transcription.error:
                job_mgr.update_job(
                    job_id,
                    status="failed",
                    stage="transcribing",
                    error=f"Erreur de transcription : {transcription.error}",
                )
                return

            job_mgr.update_job(job_id, status="processing", stage="factchecking")
            factcheck = fact_check_claims(transcription.claims, transcription.is_tech)

            result = _build_result_dict(transcription, factcheck)
            job_mgr.update_job(job_id, status="completed", stage="completed", result=result)
    except Exception as exc:
        logger.error("Error in process_upload_job for job %s: %s", job_id, exc, exc_info=True)
        job_mgr.update_job(job_id, status="failed", stage="failed", error=str(exc))


# Global singletons for app-level usage
job_manager = JobManager(max_concurrent=1)
rate_limiter = RateLimiter(limits=[(5, 60), (30, 3600)])
