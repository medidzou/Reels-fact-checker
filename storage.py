"""Lightweight ephemeral storage for ReelCheck reports (/r/<id>).

Stores JSON reports on disk with automatic TTL expiration (default 24h)
to enable instant report retrieval and sharing without re-running heavy AI pipelines.
"""

import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)

# Default directory where report JSON files are stored on disk
STORAGE_DIR = Path(os.environ.get("REELCHECK_STORAGE_DIR", "data/reports"))

# Valid ID pattern: alphanumeric characters, underscores, and hyphens only
_REPORT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")


def save_report(report_data: dict[str, Any], ttl_hours: int = 24) -> str:
    """Save a report dictionary to disk as a JSON file with TTL expiration.

    Args:
        report_data: The dictionary containing fact-check results and metadata.
        ttl_hours: Expiration time-to-live in hours (defaults to 24).

    Returns:
        The generated report_id (str).
    """
    if not isinstance(report_data, dict):
        raise TypeError("report_data must be a dictionary")

    storage_path = Path(STORAGE_DIR)
    storage_path.mkdir(parents=True, exist_ok=True)

    report_id = uuid.uuid4().hex[:12]
    now = time.time()
    expires_at = now + (ttl_hours * 3600)

    # Attach report_id and share_url if not already present
    stored_report = dict(report_data)
    stored_report.setdefault("report_id", report_id)
    stored_report.setdefault("share_url", f"/r/{report_id}")

    envelope = {
        "id": report_id,
        "created_at": now,
        "expires_at": expires_at,
        "report": stored_report,
    }

    file_path = storage_path / f"{report_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)

    return report_id


def get_report(report_id: str) -> Optional[dict[str, Any]]:
    """Retrieve a report dictionary by its ID if it exists and has not expired.

    Args:
        report_id: The unique report identifier.

    Returns:
        The report dictionary, or None if not found, invalid ID, or expired.
    """
    if not report_id or not isinstance(report_id, str):
        return None

    # Protect against path traversal and invalid characters
    if not _REPORT_ID_REGEX.match(report_id):
        return None

    storage_path = Path(STORAGE_DIR)
    file_path = storage_path / f"{report_id}.json"

    if not file_path.is_file():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        expires_at = data.get("expires_at")
        if expires_at is not None and time.time() > float(expires_at):
            return None

        return data.get("report", data)
    except Exception as exc:
        logger.warning("Failed to read report %s: %s", report_id, exc)
        return None


def cleanup_expired_reports() -> int:
    """Delete expired report JSON files from the storage directory.

    Returns:
        The number of expired files removed.
    """
    storage_path = Path(STORAGE_DIR)
    if not storage_path.exists() or not storage_path.is_dir():
        return 0

    now = time.time()
    removed_count = 0

    for file_path in storage_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                expires_at = data.get("expires_at")
                if expires_at is not None and now > float(expires_at):
                    file_path.unlink(missing_ok=True)
                    removed_count += 1
        except Exception as exc:
            logger.warning("Error processing %s during cleanup: %s", file_path, exc)

    return removed_count


# Module alias synchronization:
# Ensure both `import storage` and `from reelcheck.storage import ...` resolve to this same module
sys.modules["reelcheck.storage"] = sys.modules[__name__]
sys.modules["storage"] = sys.modules[__name__]
try:
    import reelcheck
    setattr(reelcheck, "storage", sys.modules[__name__])
except Exception:
    pass
