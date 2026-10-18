"""Tests for bot handlers, mode switching, and full pipeline orchestration."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import telegram
from telegram.constants import ChatAction
from telegram.ext import Application

from bot.formatter import get_user_mode, set_user_mode
from bot.handlers import (
    extract_instagram_url,
    handle_help,
    handle_message,
    handle_mode,
    handle_start,
    is_instagram_url,
    setup_application,
    setup_bot,
)
from pipeline.downloader import DownloadResult
from pipeline.factchecker import FactCheckResult, Verdict
from pipeline.transcriber import TranscriptionResult


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.instagram.com/reel/ABC123/", True),
        ("https://instagram.com/reel/ABC123", True),
        ("https://www.instagram.com/p/ABC123/", True),
        ("https://www.instagram.com/reels/ABC123/", True),
        ("not a url", False),
        ("https://youtube.com/watch?v=123", False),
        ("", False),
        (None, False),
        ("Check this reel: https://www.instagram.com/reel/XYZ789/ super cool", True),
    ],
)
def test_is_instagram_url(url, expected):
    """Test is_instagram_url against various valid and invalid URLs."""
    assert is_instagram_url(url) == expected


def test_extract_instagram_url():
    """Test extracting Instagram URL from text."""
    text = "Regarde ce reel https://www.instagram.com/reel/ABC12345/ incroyable!"
    extracted = extract_instagram_url(text)
    assert extracted is not None
    assert "https://www.instagram.com/reel/ABC12345/" in extracted

    assert extract_instagram_url("no url here") is None
    assert extract_instagram_url("") is None


def test_handle_start():
    """Test /start command response."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    asyncio.run(handle_start(update, context))

    update.message.reply_text.assert_awaited_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "ReelCheck" in call_args
    assert "/start" in call_args
    assert "/help" in call_args
    assert "/mode" in call_args


def test_handle_help():
    """Test /help command response."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    asyncio.run(handle_help(update, context))

    update.message.reply_text.assert_awaited_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "ReelCheck" in call_args
    assert "/mode" in call_args


def test_handle_mode_shows_current_mode():
    """Test /mode with no arguments displays current mode and options."""
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 112233
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.args = []

    asyncio.run(handle_mode(update, context))

    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.call_args[0][0]
    assert "full" in text
    assert "/mode full" in text
    assert "/mode fact" in text
    assert "/mode action" in text


def test_handle_mode_sets_valid_mode():
    """Test /mode with valid mode argument updates mode."""
    chat_id = 445566
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.args = ["fact"]

    asyncio.run(handle_mode(update, context))

    assert get_user_mode(chat_id) == "fact"
    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.call_args[0][0]
    assert "fact" in text
    assert "✅" in text


def test_handle_mode_rejects_invalid_mode():
    """Test /mode with invalid argument returns error message."""
    chat_id = 778899
    set_user_mode(chat_id, "full")

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.args = ["invalid_xyz"]

    asyncio.run(handle_mode(update, context))

    # User mode should remain unchanged
    assert get_user_mode(chat_id) == "full"
    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.call_args[0][0]
    assert "❌" in text
    assert "non reconnu" in text


@patch("bot.handlers.download_reel_audio")
@patch("bot.handlers.transcribe_and_extract")
@patch("bot.handlers.fact_check_claims")
def test_handle_message_with_instagram_url_success(
    mock_factcheck,
    mock_transcribe,
    mock_download,
    tmp_path,
):
    """Test end-to-end pipeline execution with an Instagram URL."""
    fake_audio = tmp_path / "test.mp3"
    fake_audio.write_bytes(b"dummy audio content")

    mock_download.return_value = DownloadResult(
        audio_path=str(fake_audio), success=True, error=None
    )
    mock_transcribe.return_value = TranscriptionResult(
        transcript="Spoken words here",
        summary="Concise summary",
        claims=["Claim A"],
        is_tech=False,
    )
    mock_factcheck.return_value = FactCheckResult(
        verdicts=[Verdict("Claim A", "VRAI", "Verified fact", ["https://source.com"])],
        reliability_score=10,
    )

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    update.message = MagicMock()
    update.message.text = "https://www.instagram.com/reel/ABC123/"
    update.message.caption = None
    update.message.video = None
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    # 1. Typing action sent
    context.bot.send_chat_action.assert_awaited_once_with(
        chat_id=12345, action=ChatAction.TYPING
    )

    # 2. Pipeline invoked
    mock_download.assert_called_once()
    mock_transcribe.assert_called_once()
    mock_factcheck.assert_not_called()

    # 3. Replies sent: acknowledgment + formatted output
    assert update.message.reply_text.await_count >= 2
    all_replies = [call[0][0] for call in update.message.reply_text.call_args_list]
    assert any("Reel détecté" in r or "Traitement" in r for r in all_replies)
    assert any("/r/" in r for r in all_replies)


@patch("bot.handlers.download_reel_audio")
def test_handle_message_with_instagram_url_download_failure(mock_download):
    """Test handle_message returns clear error with upload tip when Instagram download fails."""
    mock_download.return_value = DownloadResult(
        audio_path=None, success=False, error="Private account"
    )

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    update.message = MagicMock()
    update.message.text = "https://www.instagram.com/reel/PRIVATE123/"
    update.message.caption = None
    update.message.video = None
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    assert update.message.reply_text.await_count == 2
    err_reply = update.message.reply_text.call_args_list[1][0][0]
    assert "Impossible de télécharger" in err_reply
    assert "Astuce" in err_reply
    assert "pièce jointe" in err_reply


@patch("bot.handlers.transcribe_and_extract")
@patch("bot.handlers.fact_check_claims")
def test_handle_message_with_video_file_success(
    mock_factcheck,
    mock_transcribe,
):
    """Test end-to-end pipeline execution with uploaded video media file."""
    mock_transcribe.return_value = TranscriptionResult(
        transcript="Video spoken audio",
        summary="Video summary",
        claims=["Claim X"],
        is_tech=False,
    )
    mock_factcheck.return_value = FactCheckResult(
        verdicts=[Verdict("Claim X", "VRAI", "True", [])],
        reliability_score=10,
    )

    tg_file_mock = MagicMock()
    tg_file_mock.download_as_bytearray = AsyncMock(return_value=bytearray(b"raw-video-bytes"))

    video_mock = MagicMock()
    video_mock.file_name = "reel.mp4"
    video_mock.get_file = AsyncMock(return_value=tg_file_mock)

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    update.message = MagicMock()
    update.message.text = None
    update.message.caption = None
    update.message.video = video_mock
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    # Typing indicator
    context.bot.send_chat_action.assert_awaited_once_with(
        chat_id=67890, action=ChatAction.TYPING
    )

    # Pipeline called
    mock_transcribe.assert_called_once()
    mock_factcheck.assert_not_called()

    # Results delivered
    assert update.message.reply_text.await_count >= 2
    all_replies = [call[0][0] for call in update.message.reply_text.call_args_list]
    assert any("média reçu" in r or "Traitement" in r for r in all_replies)
    assert any("/r/" in r for r in all_replies)


def test_handle_message_with_video_file_media_error():
    """Test handle_message handles Telegram media download exceptions gracefully."""
    video_mock = MagicMock()
    video_mock.get_file = AsyncMock(side_effect=Exception("Telegram connection lost"))

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    update.message = MagicMock()
    update.message.text = None
    update.message.caption = None
    update.message.video = video_mock
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    assert update.message.reply_text.await_count == 2
    err_reply = update.message.reply_text.call_args_list[1][0][0]
    assert "Erreur lors de la récupération" in err_reply


@patch("bot.handlers.download_reel_audio")
@patch("bot.handlers.transcribe_and_extract")
def test_handle_message_transcription_error(
    mock_transcribe,
    mock_download,
    tmp_path,
):
    """Test handle_message sends error when transcription fails."""
    fake_audio = tmp_path / "audio.mp3"
    fake_audio.write_bytes(b"content")
    mock_download.return_value = DownloadResult(
        audio_path=str(fake_audio), success=True, error=None
    )
    mock_transcribe.return_value = TranscriptionResult(
        error="Gemini rate limit exceeded"
    )

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    update.message = MagicMock()
    update.message.text = "https://www.instagram.com/reel/XYZ/"
    update.message.caption = None
    update.message.video = None
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    assert update.message.reply_text.await_count == 2
    err_reply = update.message.reply_text.call_args_list[1][0][0]
    assert "Erreur lors de la transcription" in err_reply
    assert "rate limit" in err_reply


@patch("bot.handlers.download_reel_audio")
@patch("bot.handlers.transcribe_and_extract")
@patch("storage.save_report")
def test_handle_message_downstream_factcheck_error(
    mock_save_report,
    mock_transcribe,
    mock_download,
    tmp_path,
):
    """Test handle_message returns friendly error when downstream processing raises exception."""
    fake_audio = tmp_path / "audio.mp3"
    fake_audio.write_bytes(b"content")
    mock_download.return_value = DownloadResult(
        audio_path=str(fake_audio), success=True, error=None
    )
    mock_transcribe.return_value = TranscriptionResult(
        transcript="Spoken content",
        claims=["Claim 1"],
        is_tech=False,
    )
    mock_save_report.side_effect = RuntimeError("All LLM providers unavailable")

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    update.message = MagicMock()
    update.message.text = "https://www.instagram.com/reel/XYZ/"
    update.message.caption = None
    update.message.video = None
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    assert update.message.reply_text.await_count == 2
    err_reply = update.message.reply_text.call_args_list[1][0][0]
    assert "Une erreur est survenue lors de l'analyse" in err_reply



def test_handle_message_fallback_prompt():
    """Test handle_message with non-Instagram text provides guidance."""
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    update.message = MagicMock()
    update.message.text = "Bonjour tout le monde"
    update.message.caption = None
    update.message.video = None
    update.message.document = None
    update.message.audio = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    asyncio.run(handle_message(update, context))

    context.bot.send_chat_action.assert_not_awaited()
    update.message.reply_text.assert_awaited_once()
    reply_text = update.message.reply_text.call_args[0][0]
    assert "Instagram" in reply_text or "/help" in reply_text


def test_setup_bot():
    """Test setup_bot returns configured telegram.Bot instance."""
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    bot = setup_bot(token)
    assert isinstance(bot, telegram.Bot)
    assert bot.token == token


def test_setup_application():
    """Test setup_application creates Application with handlers including mode."""
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    app = setup_application(token)
    assert isinstance(app, Application)
    assert len(app.handlers) >= 1
