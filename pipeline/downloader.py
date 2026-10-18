"""Video and audio download pipeline for ReelCheck using yt-dlp."""

from dataclasses import dataclass
import logging
import os
from typing import Optional
import uuid

import yt_dlp

try:
    import shutil

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
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

# Valid media extensions for audio extraction and transcription
VALID_MEDIA_EXTENSIONS = (
    ".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".opus", ".webm", ".flac", ".aac"
)


@dataclass
class DownloadResult:
    """Represents the outcome of an audio download / extraction attempt."""

    audio_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


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


def _download_via_embed_scrape(shortcode: str, output_dir: str) -> Optional[str]:
    """Fallback scraper for public Instagram embeds when yt-dlp encounters GraphQL JSONDecodeError or IP blocks.

    Fetches the public embed page, extracts direct CDN video stream URL, downloads it,
    and converts it to MP3.
    """
    import re
    import shutil
    import subprocess

    embed_urls = [
        f"https://www.instagram.com/p/{shortcode}/embed/captioned/",
        f"https://www.instagram.com/p/{shortcode}/embed/",
    ]

    html = ""
    for embed_url in embed_urls:
        try:
            from curl_cffi import requests as cffi_requests

            resp = cffi_requests.get(embed_url, impersonate="chrome", timeout=12)
            if resp.status_code == 200 and ".mp4" in resp.text:
                html = resp.text
                break
        except Exception as exc:
            logger.debug("curl_cffi embed scrape failed for %s: %s", embed_url, exc)

        if not html:
            try:
                import urllib.request

                req = urllib.request.Request(
                    embed_url,
                    headers={"User-Agent": MOBILE_USER_AGENT},
                )
                with urllib.request.urlopen(req, timeout=12) as r:
                    content = r.read().decode("utf-8", errors="ignore")
                    if ".mp4" in content:
                        html = content
                        break
            except Exception as exc:
                logger.debug("urllib embed scrape failed for %s: %s", embed_url, exc)

    if not html:
        return None

    mp4_matches = re.findall(
        r'https?:[\\/A-Za-z0-9_\.\-~%]+\.mp4[\\/A-Za-z0-9_\.\-~%&=?#]*', html
    )
    if not mp4_matches:
        return None

    clean_mp4_url = mp4_matches[0].replace("\\", "").replace("&amp;", "&")
    os.makedirs(output_dir, exist_ok=True)
    temp_mp4 = os.path.join(output_dir, f"{shortcode}.mp4")
    target_mp3 = os.path.join(output_dir, f"{shortcode}.mp3")

    downloaded = False
    try:
        from curl_cffi import requests as cffi_requests

        with cffi_requests.get(
            clean_mp4_url, impersonate="chrome", stream=True, timeout=25
        ) as r:
            if r.status_code == 200:
                with open(temp_mp4, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                downloaded = True
    except Exception as exc:
        logger.debug("curl_cffi stream download failed: %s", exc)

    if not downloaded:
        try:
            import urllib.request

            urllib.request.urlretrieve(clean_mp4_url, temp_mp4)
            downloaded = True
        except Exception as exc:
            logger.debug("urllib stream download failed: %s", exc)

    if not downloaded or not os.path.exists(temp_mp4) or os.path.getsize(temp_mp4) == 0:
        return None

    try:
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            temp_mp4,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            target_mp3,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if (
            res.returncode == 0
            and os.path.exists(target_mp3)
            and os.path.getsize(target_mp3) > 0
        ):
            try:
                os.remove(temp_mp4)
            except Exception:
                pass
            return target_mp3
    except Exception as exc:
        logger.debug("ffmpeg conversion failed: %s", exc)

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(temp_mp4)
        audio.export(target_mp3, format="mp3", bitrate="192k")
        try:
            os.remove(temp_mp4)
        except Exception:
            pass
        return target_mp3
    except Exception as exc:
        logger.debug("pydub conversion failed: %s", exc)

    return temp_mp4


class YtdlpQuietLogger:
    """Redirect yt-dlp internal messages to Python logging instead of raw stderr."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        logger.debug("yt-dlp warning: %s", msg)

    def error(self, msg: str) -> None:
        logger.debug("yt-dlp error: %s", msg)


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
            "extractor_args": {
                "instagram": {
                    "app_id": ["ios"],
                }
            },
            "logger": YtdlpQuietLogger(),
            "max_filesize": MAX_FILESIZE_BYTES,
            "quiet": True,
            "no_warnings": True,
        }

        cookie_path = os.environ.get("INSTAGRAM_COOKIES_FILE")
        cookie_content = os.environ.get("INSTAGRAM_COOKIES", "").strip()
        if not cookie_path and cookie_content:
            cookie_path = os.path.join(output_dir, "cookies.txt")
            if not cookie_content.startswith("# Netscape"):
                # Automatically format raw sessionid to valid Netscape cookies file
                raw_sessionid = cookie_content.replace("sessionid=", "").strip()
                cookie_content = (
                    "# Netscape HTTP Cookie File\n"
                    f".instagram.com\tTRUE\t/\tTRUE\t2147483647\tsessionid\t{raw_sessionid}\n"
                )
            with open(cookie_path, "w", encoding="utf-8") as f:
                f.write(cookie_content)
        if cookie_path and os.path.exists(cookie_path):
            ydl_opts["cookiefile"] = cookie_path

        import re
        m = re.search(r"/(?:reel|reels|p|tv)/([a-zA-Z0-9_-]+)", url)
        shortcode = m.group(1) if m else None

        # Prioritize public embed endpoint for anonymous requests on datacenter IPs
        primary_url = url
        if shortcode and not ydl_opts.get("cookiefile"):
            primary_url = f"https://www.instagram.com/p/{shortcode}/embed/"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(primary_url, download=True)
            except Exception:
                if primary_url != url:
                    info = ydl.extract_info(url, download=True)
                else:
                    raise

        video_id = info.get("id") if isinstance(info, dict) else (shortcode or None)
        title = info.get("title") if isinstance(info, dict) else None
        description = info.get("description") if isinstance(info, dict) else None
        if not video_id:
            return DownloadResult(
                audio_path=None,
                success=False,
                error="Could not extract video ID from Instagram URL",
                title=title,
                description=description,
            )

        expected_mp3 = os.path.join(output_dir, f"{video_id}.mp3")
        if os.path.exists(expected_mp3):
            return DownloadResult(audio_path=expected_mp3, success=True, error=None, title=title, description=description)

        # Fallback check for any file matching video_id in output_dir
        for fname in os.listdir(output_dir):
            if fname.startswith(str(video_id)) and fname.lower().endswith(VALID_MEDIA_EXTENSIONS):
                candidate = os.path.join(output_dir, fname)
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    return DownloadResult(audio_path=candidate, success=True, error=None, title=title, description=description)

        return DownloadResult(
            audio_path=None,
            success=False,
            error=f"Audio extraction completed but file not found for ID: {video_id}",
            title=title,
            description=description,
        )

    except Exception as e:
        logger.warning("yt-dlp error: %s. Checking for downloaded files in %s...", e, output_dir)
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if fname.lower().endswith(VALID_MEDIA_EXTENSIONS) and os.path.isfile(os.path.join(output_dir, fname)):
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
                urls_to_try.append(f"https://www.instagram.com/p/{c}/embed/")
                urls_to_try.append(f"https://www.instagram.com/reel/{c}/embed/")
                urls_to_try.append(f"https://www.instagram.com/p/{c}/embed/captioned/")
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
                    "extractor_args": {
                        "instagram": {
                            "app_id": ["ios"],
                        }
                    },
                    "max_filesize": MAX_FILESIZE_BYTES,
                    "quiet": True,
                    "no_warnings": True,
                    **({"cookiefile": cookie_path} if cookie_path and os.path.exists(cookie_path) else {}),
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
                            if fname.lower().endswith(VALID_MEDIA_EXTENSIONS):
                                candidate = os.path.join(output_dir, fname)
                                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                                    logger.info("Found downloaded fallback media: %s", candidate)
                                    return DownloadResult(audio_path=candidate, success=True, error=None)
                    except Exception as alt_exc:
                        logger.debug("Fallback %s failed: %s", alt_url, alt_exc)

            # Direct embed scrape fallback if all yt-dlp attempts failed
            try:
                direct_file = _download_via_embed_scrape(orig_code, output_dir)
                if direct_file and os.path.exists(direct_file) and os.path.getsize(direct_file) > 0:
                    logger.info("Successfully recovered audio via direct embed scrape: %s", direct_file)
                    return DownloadResult(audio_path=direct_file, success=True, error=None)
            except Exception as direct_exc:
                logger.debug("Direct embed scrape fallback failed for %s: %s", orig_code, direct_exc)

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
