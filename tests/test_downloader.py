"""Tests for video download pipeline and audio extraction."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from pipeline.downloader import (
    DownloadResult,
    download_reel_audio,
    save_telegram_audio,
)


def test_download_result_dataclass():
    result = DownloadResult(audio_path="/tmp/test.mp3", success=True, error=None)
    assert result.success is True
    assert result.audio_path == "/tmp/test.mp3"
    assert result.error is None


def test_download_result_defaults():
    result = DownloadResult()
    assert result.audio_path is None
    assert result.success is False
    assert result.error is None


def test_download_invalid_url():
    result = download_reel_audio("https://notinstagram.com/fake", "/tmp")
    assert result.success is False
    assert result.error is not None
    assert "Instagram" in result.error or "URL" in result.error


def test_download_empty_url():
    result = download_reel_audio("", "/tmp")
    assert result.success is False
    assert result.error is not None


@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_reel_audio_success(mock_ydl_cls, tmp_path):
    output_dir = str(tmp_path / "downloads")
    test_id = "C-xyz123"
    fake_mp3 = tmp_path / "downloads" / f"{test_id}.mp3"
    
    mock_instance = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    def fake_extract_info(url, download=True):
        fake_mp3.parent.mkdir(parents=True, exist_ok=True)
        fake_mp3.write_bytes(b"fake mp3 audio data")
        return {"id": test_id, "ext": "mp4"}

    mock_instance.extract_info.side_effect = fake_extract_info

    valid_url = f"https://www.instagram.com/reel/{test_id}/"
    result = download_reel_audio(valid_url, output_dir)

    assert result.success is True
    assert result.error is None
    assert result.audio_path == str(fake_mp3)
    assert os.path.exists(result.audio_path)

    # Verify options passed to YoutubeDL
    args, kwargs = mock_ydl_cls.call_args
    ydl_opts = args[0] if args else kwargs.get("params", {})
    assert "bestaudio/best" in ydl_opts.get("format", "")
    assert "Mobile" in ydl_opts.get("http_headers", {}).get("User-Agent", "")
    assert ydl_opts.get("max_filesize") == 50 * 1024 * 1024
    assert any(
        pp.get("key") == "FFmpegExtractAudio" and pp.get("preferredcodec") == "mp3"
        for pp in ydl_opts.get("postprocessors", [])
    )


@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_reel_audio_yt_dlp_failure(mock_ydl_cls, tmp_path):
    output_dir = str(tmp_path / "downloads")
    mock_instance = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.side_effect = Exception("Video unavailable or private")

    valid_url = "https://www.instagram.com/reel/C-private/"
    result = download_reel_audio(valid_url, output_dir)

    assert result.success is False
    assert result.audio_path is None
    assert "Video unavailable or private" in result.error


@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_reel_audio_file_not_found(mock_ydl_cls, tmp_path):
    output_dir = str(tmp_path / "downloads")
    mock_instance = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {"id": "no_file", "ext": "mp4"}

    valid_url = "https://www.instagram.com/reel/no_file/"
    result = download_reel_audio(valid_url, output_dir)

    assert result.success is False
    assert result.audio_path is None
    assert result.error is not None


def test_save_telegram_audio_with_filename(tmp_path):
    output_dir = str(tmp_path / "telegram_media")
    raw_data = b"ID3\x03\x00\x00\x00some raw mp3 bytes"
    saved_path = save_telegram_audio(raw_data, output_dir, filename="custom_voice.mp3")

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "custom_voice.mp3"
    assert open(saved_path, "rb").read() == raw_data


def test_save_telegram_audio_generates_filename(tmp_path):
    output_dir = str(tmp_path / "telegram_media")
    raw_data = b"video data here"
    saved_path = save_telegram_audio(raw_data, output_dir)

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path).startswith("audio_")
    assert saved_path.endswith(".mp3")
    assert open(saved_path, "rb").read() == raw_data


def test_save_telegram_audio_creates_directory(tmp_path):
    nested_dir = str(tmp_path / "nested" / "sub" / "folder")
    assert not os.path.exists(nested_dir)
    saved_path = save_telegram_audio(b"sample bytes", nested_dir, "test.mp3")
    assert os.path.exists(nested_dir)
    assert os.path.exists(saved_path)


@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_reel_audio_with_cookie_env(mock_ydl_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("INSTAGRAM_COOKIES", "fake_session_123")
    output_dir = str(tmp_path / "downloads")
    test_id = "C-cookieTest"
    fake_mp3 = tmp_path / "downloads" / f"{test_id}.mp3"

    mock_instance = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_instance

    def fake_extract(url, download=True):
        fake_mp3.parent.mkdir(parents=True, exist_ok=True)
        fake_mp3.write_bytes(b"audio")
        return {"id": test_id}

    mock_instance.extract_info.side_effect = fake_extract

    valid_url = f"https://www.instagram.com/reel/{test_id}/"
    res = download_reel_audio(valid_url, output_dir)
    assert res.success is True

    # Verify cookiefile was created and configured
    args, kwargs = mock_ydl_cls.call_args
    ydl_opts = args[0] if args else kwargs.get("params", {})
    cookiefile = ydl_opts.get("cookiefile")
    assert cookiefile is not None
    assert os.path.exists(cookiefile)
    cookie_content = open(cookiefile, "r", encoding="utf-8").read()
    assert "fake_session_123" in cookie_content
    assert "# Netscape" in cookie_content

