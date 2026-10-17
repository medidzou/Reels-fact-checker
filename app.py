"""Flask application for ReelCheck Telegram Bot webhook."""

import asyncio
import logging
import os
import re
import threading
from typing import Any, Optional

from flask import Flask, jsonify, request
import telegram
from telegram import Update

from bot.handlers import setup_application, setup_bot
from job_manager import (
    JobManager,
    RateLimiter,
    job_manager,
    process_upload_job,
    process_url_job,
    rate_limiter,
)
from storage import get_report

logger = logging.getLogger(__name__)

INSTAGRAM_WEB_URL_REGEX = re.compile(
    r"^https?://(?:www\.)?instagram\.com/(?:reel|reels|p)/[a-zA-Z0-9_-]+/?(?:\?.*)?$",
    re.IGNORECASE,
)
ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".m4a",
    ".wav",
    ".aac",
    ".ogg",
}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB



class AsyncWorker:
    """Dedicated asyncio worker thread with a persistent event loop.

    Ensures python-telegram-bot's ExtBot and HTTPXRequest maintain active,
    unclosed connection pools across multiple webhook requests without
    hitting 'RuntimeError: Event loop is closed'.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AsyncTelegramWorker"
        )
        self.thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


def setup_webhook(
    token: str, webhook_url: str, worker: Optional[AsyncWorker] = None
) -> bool:
    """Remove previous Telegram webhook and set new webhook URL."""
    if not token or not webhook_url:
        logger.info("Webhook setup skipped: token or WEBHOOK_URL not configured")
        return False

    full_url = (
        webhook_url
        if webhook_url.endswith(f"/{token}")
        else f"{webhook_url.rstrip('/')}/{token}"
    )

    async def _configure() -> None:
        bot = telegram.Bot(token=token)
        try:
            await bot.delete_webhook(drop_pending_updates=False)
            logger.info("Old Telegram webhook removed")
            await bot.set_webhook(url=full_url)
            logger.info("Telegram webhook configured: %s", full_url)
        finally:
            await bot.shutdown()

    try:
        if worker:
            worker.run(_configure())
        else:
            asyncio.run(_configure())
        return True
    except Exception as exc:
        logger.warning("Failed to configure webhook with Telegram: %s", exc)
        return False


def create_app(test_config: Optional[dict] = None) -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(__name__)

    if test_config:
        app.config.update(test_config)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    webhook_url = render_url if render_url else os.environ.get("WEBHOOK_URL", "").strip()

    # Dedicated worker with persistent asyncio event loop
    worker = AsyncWorker()
    app.config["ASYNC_WORKER"] = worker

    # Configure Telegram bot & application
    tg_app = None
    bot = None
    if token:
        try:
            tg_app = setup_application(token)
            bot = tg_app.bot

            async def _init_app() -> None:
                await tg_app.initialize()
                await tg_app.start()

            worker.run(_init_app())
        except Exception as exc:
            logger.warning("Failed to initialize Telegram application: %s", exc)

    app.config["TG_APPLICATION"] = tg_app
    app.config["TG_BOT"] = bot

    # Remove old webhook and set new one on startup (if not testing)
    if not app.config.get("TESTING") and token and webhook_url:
        logger.info("Initializing Telegram webhook registration for: %s", webhook_url)
        setup_webhook(token, webhook_url, worker=worker)

    app.config.setdefault("MAX_CONTENT_LENGTH", MAX_UPLOAD_BYTES)
    app.config["JOB_MANAGER"] = job_manager
    app.config["RATE_LIMITER"] = rate_limiter

    @app.after_request
    def set_security_headers(response):
        """Inject robust security headers to protect against clickjacking, sniffing, and XSS."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self';"
        )
        return response

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle oversized uploads with explicit JSON error."""
        return jsonify({"error": "Fichier trop volumineux. La taille maximale est de 25 Mo."}), 413

    def _get_client_ip() -> str:
        """Extract client IP address accounting for reverse proxies."""
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.remote_addr or "127.0.0.1"

    @app.route("/", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint for Render and monitoring."""
        return jsonify({
            "status": "ok",
            "service": "reelcheck",
        }), 200

    @app.route("/api/check-url", methods=["POST"])
    def api_check_url():
        """Validate Instagram Reel URL and enqueue async processing job."""
        client_ip = _get_client_ip()
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            res = jsonify({
                "error": f"Trop de requêtes. Veuillez réessayer dans {retry_after} secondes.",
                "retry_after": retry_after,
            })
            res.status_code = 429
            res.headers["Retry-After"] = str(retry_after)
            return res

        payload = request.get_json(silent=True)
        if not payload or not isinstance(payload, dict):
            return jsonify({"error": "Requête invalide, JSON attendu"}), 400

        url = payload.get("url")
        if not url or not isinstance(url, str):
            return jsonify({"error": "Paramètre 'url' manquant ou invalide"}), 400

        url = url.strip()
        if not INSTAGRAM_WEB_URL_REGEX.match(url):
            return jsonify({"error": "URL invalide. Seuls les liens Instagram Reels sont acceptés."}), 400

        job_id = job_manager.create_job(initial_stage="queued")
        job_manager.run_job(job_id, process_url_job, job_manager, job_id, url)
        return jsonify({"job_id": job_id, "status": "processing"}), 202

    @app.route("/api/check-upload", methods=["POST"])
    def api_check_upload():
        """Validate uploaded video/audio file and enqueue async processing job."""
        client_ip = _get_client_ip()
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            res = jsonify({
                "error": f"Trop de requêtes. Veuillez réessayer dans {retry_after} secondes.",
                "retry_after": retry_after,
            })
            res.status_code = 429
            res.headers["Retry-After"] = str(retry_after)
            return res

        if not request.files:
            return jsonify({"error": "Aucun fichier fourni"}), 400

        file = request.files.get("file") or request.files.get("video")
        if not file:
            file = next(iter(request.files.values()), None)

        if not file or not file.filename:
            return jsonify({"error": "Fichier manquant ou invalide"}), 400

        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Extension '{ext}' non supportée. Formats acceptés : MP4, MOV, MP3, etc."
            }), 400

        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Le fichier envoyé est vide"}), 400

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "Fichier trop volumineux. La taille maximale est de 25 Mo."}), 413

        job_id = job_manager.create_job(initial_stage="queued")
        job_manager.run_job(
            job_id,
            process_upload_job,
            job_manager,
            job_id,
            file_bytes,
            file.filename,
        )
        return jsonify({"job_id": job_id, "status": "processing"}), 202

    @app.route("/api/status/<job_id>", methods=["GET"])
    def api_job_status(job_id: str):
        """Retrieve real-time status and stage of an async processing job."""
        job = job_manager.get_job(job_id)
        if not job:
            return jsonify({"error": f"Job '{job_id}' non trouvé"}), 404

        return jsonify({
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "result": job.get("result"),
            "error": job.get("error"),
        }), 200

    @app.route("/api/report/<report_id>", methods=["GET"])
    def api_get_report(report_id: str):
        """Retrieve an ephemeral fact-check report by its unique ID."""
        report = get_report(report_id)
        if not report:
            return jsonify({"error": f"Rapport '{report_id}' non trouvé ou expiré"}), 404

        return jsonify({
            "id": report_id,
            "report": report,
        }), 200

    @app.route("/r/<report_id>", methods=["GET"])
    def view_shared_report(report_id: str):
        """Serve shared report page or JSON fallback."""
        report = get_report(report_id)
        if not report:
            return jsonify({"error": f"Rapport '{report_id}' non trouvé ou expiré"}), 404

        try:
            from flask import render_template
            return render_template("index.html", report_id=report_id, report=report)
        except Exception:
            return jsonify({"id": report_id, "report": report}), 200


    @app.route("/<token_param>", methods=["POST"])
    @app.route("/webhook/<token_param>", methods=["POST"])
    def telegram_webhook(token_param: str):
        """Telegram webhook endpoint."""
        expected_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if expected_token and token_param != expected_token:
            return jsonify({"status": "forbidden", "error": "Invalid token"}), 403

        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"status": "bad request", "error": "Missing JSON payload"}), 400

        current_tg_app = app.config.get("TG_APPLICATION")
        current_bot = app.config.get("TG_BOT")
        if not current_bot and token_param:
            current_bot = telegram.Bot(token=token_param)

        try:
            update = Update.de_json(payload, current_bot)
        except Exception as exc:
            logger.error("Failed to parse Telegram update: %s", exc)
            return jsonify({"status": "bad request", "error": str(exc)}), 400

        # Deduplication to prevent processing the same update multiple times
        processed_updates = app.config.setdefault("_PROCESSED_UPDATE_IDS", set())
        if update and getattr(update, "update_id", None):
            if update.update_id in processed_updates:
                logger.info("Ignoring duplicate Telegram update ID: %s", update.update_id)
                return jsonify({"status": "ok", "duplicate": True}), 200
            processed_updates.add(update.update_id)
            if len(processed_updates) > 2000:
                processed_updates.clear()

        if current_tg_app:
            try:
                worker_instance = app.config.get("ASYNC_WORKER")
                if app.config.get("TESTING"):
                    # Synchronous execution in tests for deterministic assertions
                    if worker_instance:
                        worker_instance.run(current_tg_app.process_update(update))
                    else:
                        asyncio.run(current_tg_app.process_update(update))
                else:
                    # Asynchronous execution in production: immediately return 200 OK
                    # to prevent Telegram webhook 60s timeout and duplicate retries
                    if worker_instance:
                        asyncio.run_coroutine_threadsafe(
                            current_tg_app.process_update(update), worker_instance.loop
                        )
                    else:
                        asyncio.run(current_tg_app.process_update(update))
            except Exception as exc:
                logger.error("Error processing Telegram update: %s", exc)
                return jsonify({"status": "error", "error": str(exc)}), 500

        return jsonify({"status": "ok"}), 200

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    )
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting ReelCheck Telegram Webhook server on port %d...", port)
    app = create_app()
    app.run(host="0.0.0.0", port=port)

