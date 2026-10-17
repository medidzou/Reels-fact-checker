"""Heuristic classifier for detecting tech/tools/repos/AI-related Reels."""

import re
from typing import Optional

# High-confidence indicators: specialized developer/AI tools and concepts (weight 2)
HIGH_TECH_KEYWORDS = [
    r"\bmcp\b",
    r"model\s+context\s+protocol",
    r"\bgithub\b",
    r"\brepo\b",
    r"\brepository\b",
    r"\bollama\b",
    r"\bfirecrawl\b",
    r"\bplaywright\b",
    r"\bdocker\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+(?:i|install)\b",
    r"\bapi\s+key\b",
    r"\bclaude\s+(?:ai|desktop|code|sonnet|opus|haiku|3|3\.5)\b",
    r"\bantigravity\b",
    r"\bhugging\s*face\b",
    r"\bopen[- ]?source\b",
]

# Medium-confidence indicators: general tech/AI terms requiring context (weight 1)
# Note: Negative lookahead avoids common non-tech French idioms (e.g. "prompt rétablissement", "Claude François")
MEDIUM_TECH_KEYWORDS = [
    r"\bprompts?\b(?!\s+r[eé]tablissement\b)",
    r"\bextension[s]?\b",
    r"\bclaude\b(?!\s+(?:fran[cç]ois|monet|lelouch|barzotti)\b)",
    r"\bchatgpt\b",
    r"\bgemini\b(?!\s+(?:horoscope|astrologie)\b)",
    r"\bpython\b",
    r"\bcode\b(?!\s+(?:de\s+la\s+route|civil|penal|pénal)\b)",
    r"\bcoder\b",
    r"\bterminal\b(?!\s+(?:d'aeroport|d'aéroport|ferroviaire)\b)",
    r"\bscript\b",
    r"\bframework\b",
    r"\bperplexity\b",
]

COMPILED_HIGH = [re.compile(pattern, re.IGNORECASE) for pattern in HIGH_TECH_KEYWORDS]
COMPILED_MEDIUM = [re.compile(pattern, re.IGNORECASE) for pattern in MEDIUM_TECH_KEYWORDS]


def is_tech_content(text: Optional[str]) -> bool:
    """Determine whether the input transcription is technical or software/AI-related.

    Uses a weighted regex matching mechanism:
    - High-confidence keywords add 2 points (alone enough to qualify).
    - Medium-confidence keywords add 1 point (requires at least 2 distinct tech concepts).
    - Threshold for qualification: score >= 2.

    Args:
        text: Plain-text audio transcription or caption.

    Returns:
        True if the text is classified as a tech/repo/AI reel, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    cleaned = text.strip()
    if not cleaned:
        return False

    score = 0

    for pattern in COMPILED_HIGH:
        if pattern.search(cleaned):
            score += 2
            return True

    for pattern in COMPILED_MEDIUM:
        if pattern.search(cleaned):
            score += 1
            if score >= 2:
                return True

    return False
