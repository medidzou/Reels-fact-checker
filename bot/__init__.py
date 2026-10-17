"""ReelCheck Telegram Bot package."""

from bot.handlers import (
    extract_instagram_url,
    handle_help,
    handle_message,
    handle_start,
    is_instagram_url,
    register_handlers,
    setup_application,
    setup_bot,
)

__all__ = [
    "is_instagram_url",
    "extract_instagram_url",
    "handle_start",
    "handle_help",
    "handle_message",
    "setup_bot",
    "setup_application",
    "register_handlers",
]
