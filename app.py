"""Flask application for ReelCheck Telegram Bot webhook."""

import asyncio
import logging
import os
from typing import Optional

from flask import Flask, jsonify, request
import telegram
from telegram import Update

from bot.handlers import setup_application, setup_bot

logger = logging.getLogger(__name__)


def setup_webhook(token: str, webhook_url: str) -> bool:
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

    # Configure Telegram bot & application
    tg_app = None
    bot = None
    if token:
        try:
            tg_app = setup_application(token)
            bot = tg_app.bot
        except Exception as exc:
            logger.warning("Failed to initialize Telegram application: %s", exc)

    app.config["TG_APPLICATION"] = tg_app
    app.config["TG_BOT"] = bot

    # Remove old webhook and set new one on startup (if not testing)
    if not app.config.get("TESTING") and token and webhook_url:
        logger.info("Initializing Telegram webhook registration for: %s", webhook_url)
        setup_webhook(token, webhook_url)

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

        if current_tg_app:
            try:
                # Ensure application and bot are marked initialized
                if not getattr(current_tg_app, "_initialized", False):
                    current_tg_app._initialized = True
                    if hasattr(current_tg_app, "bot") and not getattr(current_tg_app.bot, "_initialized", False):
                        current_tg_app.bot._initialized = True

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

