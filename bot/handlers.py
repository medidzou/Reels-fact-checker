"""Telegram bot handlers for ReelCheck."""

import logging
import os
import re
import tempfile
from typing import Optional

import telegram
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

# Regex pattern for Instagram Reel / Post URLs
INSTAGRAM_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:reel|reels|p)/([a-zA-Z0-9_-]+)/?",
    re.IGNORECASE,
)


def is_instagram_url(text: str) -> bool:
    """Validate whether the given text contains or is an Instagram Reel/Post URL."""
    if not text or not isinstance(text, str):
        return False
    return bool(INSTAGRAM_URL_PATTERN.search(text.strip()))


def extract_instagram_url(text: str) -> Optional[str]:
    """Extract first Instagram URL found in text, if any."""
    if not text or not isinstance(text, str):
        return None
    match = INSTAGRAM_URL_PATTERN.search(text.strip())
    return match.group(0) if match else None


# Import pipeline and formatter components after is_instagram_url to avoid circular import with downloader
from bot.formatter import (  # noqa: E402
    DEFAULT_MODE,
    SUPPORTED_MODES,
    format_response,
    get_user_mode,
    set_user_mode,
)
from pipeline.downloader import download_reel_audio, save_telegram_audio  # noqa: E402
from pipeline.factchecker import fact_check_claims  # noqa: E402
from pipeline.transcriber import transcribe_and_extract  # noqa: E402


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    welcome_text = (
        "👋 Bienvenue sur ReelCheck !\n\n"
        "Partage-moi un lien de Reel Instagram (ou envoie directement une vidéo) et je vais :\n"
        "1. 📝 Transcrire la vidéo\n"
        "2. ✅ Fact-checker les affirmations avec des sources réelles\n"
        "3. 🛠️ Générer un kit d'action prêt à l'emploi si applicable\n\n"
        "Commandes disponibles :\n"
        "/start - Démarrer le bot et voir ce message\n"
        "/help - Aide et instructions détaillées\n"
        "/mode - Changer le mode d'analyse (full, fact, action)"
    )
    if update.message:
        await update.message.reply_text(welcome_text)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    help_text = (
        "ℹ️ ReelCheck - Aide & Utilisation :\n\n"
        "1. Partagez simplement le lien d'un Reel Instagram :\n"
        "   Exemple : https://www.instagram.com/reel/ABC123/\n"
        "2. Ou envoyez directement un fichier vidéo / audio en pièce jointe.\n\n"
        "Modes disponibles via /mode :\n"
        "• full : Analyse complète (Transcription, Résumé, Fact-Check, Kit d'action)\n"
        "• fact : Fact-check uniquement\n"
        "• action : Kit d'action uniquement\n\n"
        "Commandes :\n"
        "/start : Message d'accueil\n"
        "/help : Afficher ce guide\n"
        "/mode [full|fact|action] : Changer de mode d'analyse"
    )
    if update.message:
        await update.message.reply_text(help_text)


async def handle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /mode command to inspect or switch user analysis mode."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id

    if context.args and len(context.args) > 0:
        requested_mode = context.args[0].strip().lower()
        if requested_mode in SUPPORTED_MODES:
            set_user_mode(chat_id, requested_mode)
            await update.message.reply_text(
                f"✅ Mode configuré sur : **{requested_mode}**\n\n"
                f"Vos prochaines analyses utiliseront ce mode."
            )
        else:
            valid_modes_str = ", ".join(sorted(SUPPORTED_MODES))
            await update.message.reply_text(
                f"❌ Mode '{requested_mode}' non reconnu.\n"
                f"Modes disponibles : {valid_modes_str}\n\n"
                f"Exemple : `/mode fact`"
            )
    else:
        current_mode = get_user_mode(chat_id)
        await update.message.reply_text(
            f"⚙️ **Mode d'analyse actuel** : `{current_mode}`\n\n"
            f"Pour changer de mode :\n"
            f"• `/mode full` : Analyse complète (Transcription, Résumé, Fact-Check, Kit d'action)\n"
            f"• `/mode fact` : Fact-check uniquement (Résumé, Vérifications, Score)\n"
            f"• `/mode action` : Kit d'action uniquement (Résumé, Kit d'action)\n\n"
            f"Exemple : tapez `/mode fact`"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route incoming messages: Instagram URL, direct video upload, or help prompt."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
    text = update.message.text or update.message.caption or ""

    instagram_url = extract_instagram_url(text)
    has_media = bool(
        update.message.video
        or update.message.document
        or update.message.audio
        or update.message.voice
    )

    if not instagram_url and not has_media:
        await update.message.reply_text(
            "Veuillez m'envoyer un lien de Reel Instagram valide (ex: https://www.instagram.com/reel/...) "
            "ou uploader directement une vidéo.\n\nTapez /help pour en savoir plus."
        )
        return

    # 1. Immediate typing indicator
    if context.bot and chat_id:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception as e:
            logger.debug("Failed to send typing chat action: %s", e)

    # Send acknowledgment
    if instagram_url:
        await update.message.reply_text(
            "⏳ Reel détecté ! Traitement en cours...\n"
            "Téléchargement et analyse en cours, veuillez patienter."
        )
    else:
        await update.message.reply_text(
            "⏳ Fichier média reçu ! Traitement en cours...\n"
            "Analyse de la vidéo/audio, veuillez patienter."
        )

    # 2. Process in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path: Optional[str] = None

        if instagram_url:
            dl_res = download_reel_audio(instagram_url, tmpdir)
            if not dl_res.success or not dl_res.audio_path or not os.path.exists(dl_res.audio_path):
                logger.warning("Download failed for URL %s: %s", instagram_url, dl_res.error)
                await update.message.reply_text(
                    "❌ Impossible de télécharger la vidéo depuis ce lien Instagram "
                    "(compte privé, reel indisponible ou protection Instagram).\n\n"
                    "💡 Astuce : Téléchargez la vidéo sur votre téléphone et envoyez-la moi "
                    "directement ici en pièce jointe !"
                )
                return
            audio_path = dl_res.audio_path

        elif has_media:
            media = (
                update.message.video
                or update.message.document
                or update.message.audio
                or update.message.voice
            )
            try:
                if hasattr(media, "get_file"):
                    tg_file = await media.get_file()
                elif context.bot and hasattr(media, "file_id"):
                    tg_file = await context.bot.get_file(media.file_id)
                else:
                    raise ValueError("Cannot access media file")

                byte_arr = await tg_file.download_as_bytearray()
                raw_filename = getattr(media, "file_name", None)
                audio_path = save_telegram_audio(bytes(byte_arr), tmpdir, filename=raw_filename)
            except Exception as dl_err:
                logger.error("Failed to retrieve Telegram media: %s", dl_err, exc_info=True)
                await update.message.reply_text(
                    f"❌ Erreur lors de la récupération du fichier média : {dl_err}"
                )
                return

        if not audio_path or not os.path.exists(audio_path):
            await update.message.reply_text(
                "❌ Impossible d'accéder au fichier pour l'analyse."
            )
            return

        # 3. Transcribe and extract claims
        transcription = transcribe_and_extract(audio_path)
        if transcription.error:
            logger.error("Transcription error: %s", transcription.error)
            await update.message.reply_text(
                f"❌ Erreur lors de la transcription : {transcription.error}"
            )
            return

        # Universal AI Dispatch: Instant categorized response with 1-click links
        from pipeline.prompt_generator import build_dispatch_report
        from storage import save_report
        from bot.formatter import format_dispatch_response

        dispatch_report = build_dispatch_report(transcription)
        report_id = save_report(dispatch_report)
        raw_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("WEBHOOK_URL", "")
        base_url = raw_url.rstrip("/") if raw_url else ""
        share_url = f"{base_url}/r/{report_id}" if base_url else f"/r/{report_id}"

        msgs = format_dispatch_response(
            transcription,
            category=dispatch_report.get("category", "general"),
            category_label=dispatch_report.get("category_label"),
            share_url=share_url,
        )
        for msg in msgs:
            await update.message.reply_text(msg)
        return


def register_handlers(application: Application) -> None:
    """Register command and message handlers on the Application."""
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("mode", handle_mode))
    application.add_handler(
        MessageHandler(filters.ALL & ~filters.COMMAND, handle_message)
    )


def setup_application(token: str) -> Application:
    """Create and configure Telegram Application with all handlers."""
    application = Application.builder().token(token).build()
    register_handlers(application)
    return application


def setup_bot(token: str) -> telegram.Bot:
    """Create and return a configured Telegram Bot instance."""
    return telegram.Bot(token=token)
