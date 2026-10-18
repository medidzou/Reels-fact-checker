"""Fast-Check engine querying Google Fact Check Tools API (ClaimReview standard)."""

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Optional

from pipeline.factchecker import Verdict, _normalize_verdict

logger = logging.getLogger(__name__)

FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def query_claim_review(
    claim: str,
    api_key: Optional[str] = None,
    language_code: Optional[str] = None,
    timeout: float = 3.0,
) -> Optional[Verdict]:
    """Query Google Fact Check Tools API for existing IFCN-certified fact-checks.

    Args:
        claim: The factual claim to search.
        api_key: Optional Google API key (falls back to GEMINI_API_KEY).
        language_code: Language filter (e.g. 'fr' or None for all languages).
        timeout: Request timeout in seconds.

    Returns:
        Structured Verdict if an existing fact check is found, else None.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    clean_claim = claim.strip() if claim else ""
    if not clean_claim or not key:
        return None

    # Keep query clean and concise
    query_str = clean_claim[:150]
    params = {
        "query": query_str,
        "key": key,
    }
    if language_code:
        params["languageCode"] = language_code

    url = f"{FACT_CHECK_API_URL}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ReelCheck-FactChecker/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = getattr(resp, "status", getattr(resp, "code", 200))
            if status_code != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))

        claims_list = data.get("claims", [])
        if not claims_list:
            return None

        # Take best matching claim review
        best_claim = claims_list[0]
        reviews = best_claim.get("claimReview", [])
        if not reviews:
            return None

        review = reviews[0]
        publisher = review.get("publisher", {}).get("name", "Fact-Checker certifié")
        raw_rating = review.get("textualRating", "Fact-checké")
        review_url = review.get("url", "")
        review_title = review.get("title", best_claim.get("text", clean_claim))

        normalized_v = _normalize_verdict(raw_rating)
        sources = [review_url] if review_url else []

        explanation = (
            f"Vérifié officiellement par {publisher} : « {review_title} » (Évaluation : {raw_rating})."
        )

        logger.info(
            "Found official ClaimReview from %s for claim '%s': %s",
            publisher,
            clean_claim,
            normalized_v,
        )
        return Verdict(
            claim=clean_claim,
            verdict=normalized_v,
            explanation=explanation,
            sources=sources,
        )
    except Exception as exc:
        logger.debug("ClaimReview search skipped or failed for '%s': %s", clean_claim, exc)
        return None
