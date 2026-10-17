"""Run ReelCheck bot locally using long polling (ideal for local testing without public URL)."""

import logging
import os
import sys
from dotenv import load_dotenv

# Load local .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

try:
    import static_ffmpeg

    static_ffmpeg.add_paths()
except Exception:
    pass

from bot.handlers import setup_application

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("reelcheck.polling")


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN manquant dans l'environnement ou .env")
        sys.exit(1)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        logger.warning("GEMINI_API_KEY non configurée dans .env")

    logger.info("Démarrage du bot ReelCheck en mode polling local...")
    logger.info("Vous pouvez tester le bot directement sur Telegram !")

    app = setup_application(token)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
