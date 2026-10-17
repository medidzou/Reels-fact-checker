"""Flask application for ReelCheck Telegram Bot webhook."""

import asyncio
import logging
import os
from typing import Any, Optional

from flask import Flask, jsonify, request
import telegram
from telegram import Update

from bot.handlers import setup_application, setup_bot

import threading

logger = logging.getLogger(__name__)


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

    @app.route("/", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint for Render and monitoring."""
        return jsonify({
            "status": "ok",
            "service": "reelcheck",
        }), 200

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

