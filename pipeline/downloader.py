"""Video and audio download pipeline for ReelCheck using yt-dlp."""

from dataclasses import dataclass
import logging
import os
from typing import Optional
import uuid

import yt_dlp

try:
    import static_ffmpeg

    static_ffmpeg.add_paths()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Mobile user agent to minimize Instagram bot blocks
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; SM-G991B) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36"
)

# 50 MB maximum file size limit
MAX_FILESIZE_BYTES = 50 * 1024 * 1024


@dataclass
class DownloadResult:
    """Represents the outcome of an audio download / extraction attempt."""

    audio_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


def _generate_code_variations(code: str) -> list[str]:
    """Generate common font/OCR disambiguation variations for Instagram shortcodes (I vs l)."""
    variations: set[str] = set()
    for idx, ch in enumerate(code):
        if ch == "I":
            variations.add(code[:idx] + "l" + code[idx + 1 :])
        elif ch == "l":
            variations.add(code[:idx] + "I" + code[idx + 1 :])
    variations.add(code.replace("I", "l"))
    variations.add(code.replace("l", "I"))
    variations.discard(code)
    return list(variations)


def download_reel_audio(url: str, output_dir: str) -> DownloadResult:
    """Download an Instagram Reel and extract its audio as an MP3 file.

    Args:
        url: The Instagram Reel/post URL.
        output_dir: Directory where the extracted audio file will be saved.

    Returns:
        DownloadResult containing the path to the MP3 file on success,
        or error details on failure.
    """
    from bot.handlers import is_instagram_url
    if not is_instagram_url(url):
        return DownloadResult(
            audio_path=None,
            success=False,
            error=f"Invalid Instagram URL: '{url}'",
        )

    try:
        os.makedirs(output_dir, exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "http_headers": {
                "User-Agent": MOBILE_USER_AGENT,
            },
            "max_filesize": MAX_FILESIZE_BYTES,
            "quiet": True,
            "no_warnings": True,
        }

        cookie_path = os.environ.get("INSTAGRAM_COOKIES_FILE")
        cookie_content = os.environ.get("INSTAGRAM_COOKIES")
        if not cookie_path and cookie_content:
            cookie_path = os.path.join(output_dir, "cookies.txt")
            with open(cookie_path, "w", encoding="utf-8") as f:
                f.write(cookie_content)
        if cookie_path and os.path.exists(cookie_path):
            ydl_opts["cookiefile"] = cookie_path

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        video_id = info.get("id") if isinstance(info, dict) else None
        if not video_id:
            return DownloadResult(
                audio_path=None,
                success=False,
                error="Could not extract video ID from Instagram URL",
            )

        expected_mp3 = os.path.join(output_dir, f"{video_id}.mp3")
        if os.path.exists(expected_mp3):
            return DownloadResult(audio_path=expected_mp3, success=True, error=None)

        # Fallback check for any file matching video_id in output_dir
        for fname in os.listdir(output_dir):
            if fname.startswith(str(video_id)):
                candidate = os.path.join(output_dir, fname)
                if os.path.isfile(candidate):
                    return DownloadResult(audio_path=candidate, success=True, error=None)

        return DownloadResult(
            audio_path=None,
            success=False,
            error=f"Audio extraction completed but file not found for ID: {video_id}",
        )

    except Exception as e:
        logger.warning("yt-dlp error: %s. Checking for downloaded files in %s...", e, output_dir)
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if not fname.endswith((".part", ".ytdl")) and os.path.isfile(os.path.join(output_dir, fname)):
                    candidate = os.path.join(output_dir, fname)
                    if os.path.getsize(candidate) > 0:
                        logger.info("Found downloaded media file: %s", candidate)
                        return DownloadResult(audio_path=candidate, success=True, error=None)

        # Check if shortcode contains ambiguous characters (e.g. I vs l) or try alternate /p/ endpoint
        import re

        m = re.search(r"/(?:reel|reels|p|tv)/([a-zA-Z0-9_-]+)", url)
        if m:
            orig_code = m.group(1)
            codes_to_try = [orig_code] + _generate_code_variations(orig_code)
            urls_to_try: list[str] = []
            for c in codes_to_try:
                urls_to_try.append(f"https://www.instagram.com/reel/{c}/")
                urls_to_try.append(f"https://www.instagram.com/p/{c}/")

            # Remove original input URL if already tried
            candidate_urls = [u for u in urls_to_try if u != url]

            # Options fallback: first with audio postprocessor, then direct best without postprocessors
            fallback_opts_list = [
                ydl_opts,
                {
                    "format": "best",
                    "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
                    "http_headers": {"User-Agent": MOBILE_USER_AGENT},
                    "max_filesize": MAX_FILESIZE_BYTES,
                    "quiet": True,
                    "no_warnings": True,
                },
            ]

            for alt_url in candidate_urls:
                for opts in fallback_opts_list:
                    try:
                        logger.info("Attempting Instagram download fallback: %s", alt_url)
                        with yt_dlp.YoutubeDL(opts) as alt_ydl:
                            alt_info = alt_ydl.extract_info(alt_url, download=True)
                        alt_id = alt_info.get("id") if isinstance(alt_info, dict) else orig_code
                        alt_mp3 = os.path.join(output_dir, f"{alt_id}.mp3")
                        if os.path.exists(alt_mp3) and os.path.getsize(alt_mp3) > 0:
                            logger.info("Successfully recovered audio with fallback: %s", alt_url)
                            return DownloadResult(audio_path=alt_mp3, success=True, error=None)
                        for fname in os.listdir(output_dir):
                            if not fname.endswith((".part", ".ytdl")):
                                candidate = os.path.join(output_dir, fname)
                                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                                    logger.info("Found downloaded fallback media: %s", candidate)
                                    return DownloadResult(audio_path=candidate, success=True, error=None)
                    except Exception as alt_exc:
                        logger.debug("Fallback %s failed: %s", alt_url, alt_exc)

        logger.error(f"Error downloading reel audio from {url}: {e}", exc_info=True)
        return DownloadResult(
            audio_path=None,
            success=False,
            error=str(e),
        )


def save_telegram_audio(
    file_bytes: bytes, output_dir: str, filename: Optional[str] = None
) -> str:
    """Save raw audio or video bytes uploaded via Telegram to a local file.

    Args:
        file_bytes: Binary contents of the media file.
        output_dir: Directory where the media file will be stored.
        filename: Optional explicit filename. If not provided, a random
                  filename with .mp3 extension is generated.

    Returns:
        Absolute or relative path to the saved media file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"audio_{uuid.uuid4().hex[:10]}.mp3"
    else:
        filename = os.path.basename(filename)

    target_path = os.path.join(output_dir, filename)
    with open(target_path, "wb") as f:
        f.write(file_bytes)

    return target_path
