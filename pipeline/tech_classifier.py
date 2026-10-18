"""Heuristic classifier for detecting tech/tools/repos/AI-related Reels.

Maintained for backward compatibility. Wraps the unified classifier in pipeline.classifier.
"""

from typing import Optional
from pipeline.classifier import classify_content


def is_tech_content(text: Optional[str]) -> bool:
    """Determine whether the input transcription is technical or software/AI-related.

    Args:
        text: Plain-text audio transcription or caption.

    Returns:
        True if the text is classified as a tech/repo/AI reel, False otherwise.
    """
    category, _ = classify_content(text)
    return category == "tech"
