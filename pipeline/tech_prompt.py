"""Standardized audit prompt generator and deep link formatter for Tech Reels.

Maintained for backward compatibility. Wraps pipeline.prompt_generator.
"""

from typing import Any, Optional
from pipeline.prompt_generator import (
    generate_prompt_for_category,
    generate_deep_links,
    build_dispatch_report,
)


def generate_audit_prompt(transcript: Optional[str]) -> str:
    """Generate a rigorous, standardized technical and cybersecurity audit prompt."""
    return generate_prompt_for_category(transcript, "tech")


def build_tech_report_dict(transcription: Any) -> dict[str, Any]:
    """Assemble a standardized tech audit report dict shared across Web & Telegram pipelines."""
    return build_dispatch_report(transcription, category="tech")
