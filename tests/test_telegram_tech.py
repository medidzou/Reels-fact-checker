"""Tests for Telegram bot handling of Tech Reels."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from bot.formatter import format_tech_response
from bot.handlers import handle_message
from pipeline.downloader import DownloadResult
from pipeline.transcriber import TranscriptionResult


def test_format_tech_response_contains_headers_and_link():
    transcription = TranscriptionResult(
        transcript="Test MCP Firecrawl Playwright",
        summary="Extensions MCP pour Claude",
        claims=["Claim 1"],
        is_tech=True,
    )
    share_url = "https://reelcheck-app.onrender.com/r/rep123"
    msgs = format_tech_response(transcription, share_url=share_url)
    assert len(msgs) > 0
    full_msg = "\n".join(msgs)
    assert "REEL TECH" in full_msg
    assert "Audit Machine" in full_msg
    assert share_url in full_msg


def test_handle_message_short_circuits_tech_reel(tmp_path):
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 99999
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.text = "https://www.instagram.com/reel/TECH123/"
    update.message.chat_id = 99999
    update.message.video = None
    update.message.document = None
    update.message.audio = None
    update.message.voice = None

    context = MagicMock()

    tech_transcription = TranscriptionResult(
        transcript="Arrêtez tout ! Installez ces extensions MCP Firecrawl et Playwright dans Claude Desktop.",
        summary="Extensions MCP pour Claude.",
        claims=["Firecrawl scrape le web"],
        is_tech=True,
    )

    with patch("bot.handlers.download_reel_audio") as mock_dl, \
         patch("bot.handlers.transcribe_and_extract", return_value=tech_transcription), \
         patch("bot.handlers.fact_check_claims") as mock_factcheck, \
         patch("storage.save_report", return_value="rep-test-999"):

        dummy_audio = tmp_path / "dummy.mp3"
        dummy_audio.write_text("audio")
        mock_dl.return_value = DownloadResult(success=True, audio_path=str(dummy_audio))

        asyncio.run(handle_message(update, context))

        # Ensure fact_check_claims was NOT called on Telegram either!
        assert mock_factcheck.called is False
        assert update.message.reply_text.called
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        combined = "\n".join(replies)
        assert "REEL TECH" in combined or "Outil" in combined
