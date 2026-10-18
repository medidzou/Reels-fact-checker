"""Fact-check engine combining DuckDuckGo search with Google Gemini synthesis."""

import concurrent.futures
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SYNTHESIS_PROMPT = """Tu es un expert vérificateur de faits (Fact-Checker).
Analyse l'affirmation suivante à la lumière des résultats de recherche web fournis.

Affirmation : {claim}

Résultats de recherche :
{search_context}

Consignes :
1. Détermine le VERDICT exact parmi ces 4 options strictes :
   - VRAI : L'affirmation est factuellement exacte et vérifiée par des sources concordantes.
   - FAUX : L'affirmation est fausse, trompeuse ou directement contredite.
   - PARTIELLEMENT VRAI : L'affirmation contient une part de vérité mais manque de contexte crucial, exagère ou mélange vrai et faux.
   - NON VÉRIFIABLE : Les résultats de recherche sont absents, insuffisants ou hors-sujet pour trancher avec certitude.
2. Rédige une EXPLICATION claire, objective et nuancée en français expliquant pourquoi ce verdict est attribué. Si l'information est non vérifiable, indique ce qui manque.
3. Liste les URL des SOURCES fiables confirmant ou réfutant l'affirmation.

Format de réponse requis :
VERDICT: <VRAI | FAUX | PARTIELLEMENT VRAI | NON VÉRIFIABLE>
EXPLANATION: <ton explication>
SOURCES:
- <url 1>
- <url 2>
"""

ACTION_KIT_PROMPT = """Tu es un expert en ingénierie logicielle et IA.
Les affirmations suivantes proviennent d'une vidéo technique ou IA qui a été analysée :
Affirmations :
{claims_text}

Recherches et contexte :
{search_text}

Génère un Kit d'Action concret, ultra-pratique et prêt à l'emploi (en Markdown) pour un développeur ou utilisateur souhaitant mettre en œuvre ou tester ce sujet :
1. 📋 Prérequis et outils nécessaires (avec commandes d'installation si applicable)
2. 🚀 Guide pas-à-pas d'implémentation
3. 💻 Snippets de code fonctionnels ou commandes terminal
4. 🤖 Prompts recommandés ou configurations types
5. ⚠️ Pièges à éviter et conseils pratiques

Sois direct, technique et structuré.
"""


@dataclass
class Verdict:
    """Fact-check verdict for an individual claim."""

    claim: str
    verdict: str  # "VRAI", "FAUX", or "PARTIELLEMENT VRAI"
    explanation: str
    sources: list[str] = field(default_factory=list)


@dataclass
class FactCheckResult:
    """Aggregated fact-check results for a set of claims."""

    verdicts: list[Verdict] = field(default_factory=list)
    reliability_score: int = 10
    action_kit: Optional[str] = None


def _get_gemini_client(client: Optional[Any] = None) -> tuple[Optional[Any], Optional[str]]:
    """Retrieve or initialize Gemini client."""
    if client is not None:
        return client, None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY environment variable not set"

    try:
        from google import genai

        return genai.Client(api_key=api_key), None
    except Exception as err:
        return None, f"Failed to initialize Gemini client: {err}"


def _normalize_verdict(raw_verdict: str) -> str:
    """Normalize arbitrary verdict text into one of four canonical values."""
    cleaned = raw_verdict.strip().upper()
    if any(k in cleaned for k in ["NON VÉRIFIABLE", "NON VERIFIABLE", "UNVERIFIABLE", "INSUFFISANT", "INSUFFICIENT"]):
        return "NON VÉRIFIABLE"
    if any(k in cleaned for k in ["FAUX", "FALSE", "INCORRECT"]):
        return "FAUX"
    if any(k in cleaned for k in ["PARTIEL", "PARTIAL", "NUANC", "TROMPEUR", "TROMPEUSE", "CHERRY-PICK"]):
        return "PARTIELLEMENT VRAI"
    if any(k in cleaned for k in ["VRAI", "TRUE", "EXACT", "CORRECT"]):
        return "VRAI"
    return "NON VÉRIFIABLE"


def _extract_urls(text: str) -> list[str]:
    """Extract and sanitize unique HTTP/HTTPS URLs from text."""
    urls = re.findall(r"https?://[^\s<>\"'()]+", text)
    cleaned_urls = []
    seen = set()
    for url in urls:
        # Strip trailing punctuation often caught in sentences
        cleaned = re.sub(r"[\.,;:!\?\)\]]+$", "", url).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            cleaned_urls.append(cleaned)
    return cleaned_urls


def search_claim(claim: str, ddgs_client: Optional[Any] = None) -> str:
    """Search DuckDuckGo for evidence related to the claim.

    Args:
        claim: The factual claim to search.
        ddgs_client: Optional DuckDuckGo search client for dependency injection.

    Returns:
        Formatted context string containing search results, or empty string if none.
    """
    clean_claim = claim.strip() if claim else ""
    if not clean_claim:
        return ""

    try:
        if ddgs_client is None:
            try:
                from ddgs import DDGS
            except ImportError:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
                    from duckduckgo_search import DDGS

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
                ddgs_client = DDGS(timeout=4)

        raw_results = ddgs_client.text(clean_claim, max_results=3)
        if raw_results:
            formatted_items = []
            for i, item in enumerate(raw_results, 1):
                title = str(item.get("title", "")).strip()
                href = str(item.get("href", "")).strip()
                body = str(item.get("body", "")).strip()
                formatted_items.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}")
            return "\n\n".join(formatted_items)

    except Exception as exc:
        logger.warning("DuckDuckGo search error for claim '%s': %s", clean_claim, exc)

    # Wikipedia Full-Text Search Fallback for factual context
    try:
        import html
        import json
        import urllib.parse
        import urllib.request

        for lang in ("fr", "en"):
            encoded_q = urllib.parse.quote(clean_claim[:100])
            wiki_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&format=json&srlimit=3"
            req = urllib.request.Request(
                wiki_url,
                headers={"User-Agent": "ReelCheckBot/1.0 (https://t.me/ReelsFactCheckBot; contact@reelcheck.app)"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_results = data.get("query", {}).get("search", [])
                if search_results:
                    wiki_items = []
                    for idx, item in enumerate(search_results, 1):
                        title = str(item.get("title", "")).strip()
                        raw_snippet = str(item.get("snippet", "")).strip()
                        clean_snippet = html.unescape(re.sub(r"<[^>]+>", "", raw_snippet)).strip()
                        link = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                        wiki_items.append(f"[{idx}] {title}\nURL: {link}\nSnippet: {clean_snippet or title}")
                    if wiki_items:
                        logger.info("Retrieved %d Wikipedia (%s) fallback results for claim '%s'", len(wiki_items), lang, clean_claim)
                        return "\n\n".join(wiki_items)
    except Exception as wiki_err:
        logger.debug("Wikipedia fallback error for claim '%s': %s", clean_claim, wiki_err)

    return ""


def parse_verdict_response(raw_text: str, claim: str) -> Verdict:
    """Parse Gemini response into a structured Verdict object."""
    if not raw_text or not raw_text.strip():
        return Verdict(
            claim=claim,
            verdict="NON VÉRIFIABLE",
            explanation="Aucune réponse d'analyse fournie par l'IA.",
            sources=[],
        )

    text = raw_text.strip()

    # 1. Check for JSON block (either full text or inside markdown fences)
    json_candidates = []
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        json_candidates.append(fence_match.group(1).strip())
    json_candidates.append(text)

    for candidate in json_candidates:
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    v_raw = str(data.get("verdict", "")).strip()
                    explanation = str(data.get("explanation", "")).strip()
                    raw_sources = data.get("sources", [])
                    sources: list[str] = []
                    if isinstance(raw_sources, list):
                        for s in raw_sources:
                            extracted = _extract_urls(str(s))
                            if extracted:
                                sources.extend(extracted)
                            elif str(s).strip().startswith("http"):
                                sources.append(str(s).strip())
                    elif isinstance(raw_sources, str):
                        sources = _extract_urls(raw_sources)

                    return Verdict(
                        claim=claim,
                        verdict=_normalize_verdict(v_raw),
                        explanation=explanation,
                        sources=sources,
                    )
            except Exception:
                pass

    # 2. Check for section-based text format (VERDICT, EXPLANATION, SOURCES)
    section_patterns = {
        "verdict": r"(?:^|\n)\s*(?:\*\*)?VERDICT(?:\*\*)?\s*:\s*",
        "explanation": r"(?:^|\n)\s*(?:\*\*)?EXPLANATION(?:\*\*)?\s*:\s*",
        "sources": r"(?:^|\n)\s*(?:\*\*)?SOURCES(?:\*\*)?\s*:\s*",
    }

    matches: list[tuple[str, int, int]] = []
    for sec_name, pattern in section_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            matches.append((sec_name, m.start(), m.end()))

    if matches:
        matches.sort(key=lambda x: x[1])
        sections: dict[str, str] = {}
        for i, (name, start, end) in enumerate(matches):
            next_start = matches[i + 1][1] if i + 1 < len(matches) else len(text)
            sections[name] = text[end:next_start].strip()

        raw_verdict = sections.get("verdict", "")
        explanation = sections.get("explanation", "")
        raw_sources = sections.get("sources", "")

        sources = _extract_urls(raw_sources)
        if not sources:
            # Fallback to search all text for URLs
            sources = _extract_urls(text)

        return Verdict(
            claim=claim,
            verdict=_normalize_verdict(raw_verdict),
            explanation=explanation,
            sources=sources,
        )

    # 3. Fallback: unformatted text
    urls = _extract_urls(text)
    return Verdict(
        claim=claim,
        verdict=_normalize_verdict(text),
        explanation=text,
        sources=urls,
    )


def synthesize_verdict(
    claim: str,
    search_context: str,
    gemini_client: Optional[Any] = None,
    groq_client: Optional[Any] = None,
) -> Verdict:
    """Synthesize web search evidence with Gemini to produce a fact-check verdict, with Groq fallback.

    Args:
        claim: The claim being verified.
        search_context: Evidence extracted from web search.
        gemini_client: Optional Gemini client instance.
        groq_client: Optional Groq client instance for fallback.

    Returns:
        Structured Verdict object.
    """
    clean_claim = claim.strip() if claim else ""

    # Étage 1 : Fast-check via Google Fact Check Tools API (ClaimReview IFCN)
    try:
        from pipeline.claim_review import query_claim_review

        cr_verdict = query_claim_review(clean_claim)
        if cr_verdict is not None:
            return cr_verdict
    except Exception as cr_err:
        logger.debug("ClaimReview check skipped: %s", cr_err)

    def _fallback_or_verdict(err_msg: str) -> Verdict:
        if os.environ.get("GROQ_API_KEY") or groq_client is not None:
            logger.info("Gemini failed (%s), falling back to Groq synthesis...", err_msg)
            from pipeline.fallback import groq_synthesize

            return groq_synthesize(clean_claim, search_context, client=groq_client)
        return Verdict(
            claim=clean_claim,
            verdict="NON VÉRIFIABLE",
            explanation=f"Vérification impossible : {err_msg}",
            sources=_extract_urls(search_context),
        )

    client, err = _get_gemini_client(gemini_client)
    if err or client is None:
        return _fallback_or_verdict(f"Erreur d'initialisation Gemini: {err}")

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    prompt = SYNTHESIS_PROMPT.format(
        claim=clean_claim,
        search_context=search_context or "Aucun résultat trouvé sur le web.",
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
        )
        raw_text = response.text if hasattr(response, "text") and response.text else ""
        verdict = parse_verdict_response(raw_text, clean_claim)
        # If no sources were extracted from the LLM response, pull URLs from the search context
        if not verdict.sources and search_context:
            verdict.sources = _extract_urls(search_context)
        return verdict
    except Exception as exc:
        logger.error("Error synthesizing verdict for claim '%s': %s", clean_claim, exc)
        return _fallback_or_verdict(f"Erreur lors de la synthèse IA: {exc}")


def generate_action_kit(
    claims: list[str],
    search_contexts: list[str],
    gemini_client: Optional[Any] = None,
    groq_client: Optional[Any] = None,
) -> Optional[str]:
    """Generate a practical implementation action kit for technical claims, with Groq fallback.

    Args:
        claims: List of technical claims from the video.
        search_contexts: Context/search results gathered for the claims.
        gemini_client: Optional Gemini client instance.
        groq_client: Optional Groq client instance for fallback.

    Returns:
        Markdown string containing the action kit, or None if skipped/failed.
    """
    valid_claims = [c.strip() for c in claims if c and c.strip()]
    if not valid_claims:
        return None

    def _fallback_or_none(err_msg: str) -> Optional[str]:
        if os.environ.get("GROQ_API_KEY") or groq_client is not None:
            logger.info("Gemini failed (%s), falling back to Groq action kit...", err_msg)
            from pipeline.fallback import groq_action_kit

            return groq_action_kit(valid_claims, search_contexts, client=groq_client)
        return None

    client, err = _get_gemini_client(gemini_client)
    if err or client is None:
        logger.warning("Cannot generate action kit via Gemini: %s", err)
        return _fallback_or_none(err or "Client not available")

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    claims_text = "\n".join(f"- {c}" for c in valid_claims)
    search_text = "\n\n".join(
        f"--- Contexte affirmation {i+1} ---\n{ctx}"
        for i, ctx in enumerate(search_contexts)
        if ctx
    )

    prompt = ACTION_KIT_PROMPT.format(
        claims_text=claims_text,
        search_text=search_text or "Aucune recherche supplémentaire.",
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return None
    except Exception as exc:
        logger.error("Error generating action kit via Gemini: %s", exc)
        return _fallback_or_none(str(exc))


def fact_check_claims(
    claims: list[str],
    is_tech: bool,
    gemini_client: Optional[Any] = None,
    ddgs_client: Optional[Any] = None,
    groq_client: Optional[Any] = None,
    on_stage_change: Optional[Callable[[str], None]] = None,
    max_claims: int = 4,
) -> FactCheckResult:
    """Orchestrate search, verdict synthesis, scoring, and optional action kit generation.

    Args:
        claims: List of extracted factual claims.
        is_tech: Whether the video is technical/AI/software related.
        gemini_client: Optional Gemini client instance.
        ddgs_client: Optional DuckDuckGo search client.
        groq_client: Optional Groq client instance for fallback.
        on_stage_change: Optional callback invoked when stage progresses (e.g. 'formatting').
        max_claims: Maximum number of claims to verify to bound latency (default 4).

    Returns:
        FactCheckResult containing verdicts, reliability score (0-10), and action kit.
    """
    cleaned_claims = [c.strip() for c in claims if c and c.strip()][:max_claims]
    if not cleaned_claims:
        return FactCheckResult(verdicts=[], reliability_score=10, action_kit=None)

    def _process_single_claim(claim: str) -> tuple[Verdict, str]:
        context = search_claim(claim, ddgs_client=ddgs_client)
        verdict = synthesize_verdict(
            claim,
            context,
            gemini_client=gemini_client,
            groq_client=groq_client,
        )
        return verdict, context

    # Run in parallel when multiple claims and not using a mock ddgs_client
    max_workers = min(4, len(cleaned_claims))
    if max_workers > 1 and ddgs_client is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            claim_results = list(executor.map(_process_single_claim, cleaned_claims))
    else:
        claim_results = [_process_single_claim(c) for c in cleaned_claims]

    verdicts = [v for v, _ in claim_results]
    search_contexts = [ctx for _, ctx in claim_results]

    if on_stage_change:
        try:
            on_stage_change("formatting")
        except Exception as stage_err:
            logger.debug("on_stage_change callback error: %s", stage_err)

    # Compute reliability score (0 to 10 scale) with Poison Rule
    score_weights = {
        "VRAI": 10,
        "PARTIELLEMENT VRAI": 5,
        "NON VÉRIFIABLE": 5,
        "FAUX": 0,
    }
    scores = [score_weights.get(v.verdict, 5) for v in verdicts]
    avg_score = round(sum(scores) / len(scores)) if scores else 10
    clamped_score = max(0, min(10, avg_score))

    # POISON RULE: If any claim is FAUX, the overall reliability score cannot exceed 3 / 10
    has_faux = any(v.verdict == "FAUX" for v in verdicts)
    has_partiel = any(v.verdict == "PARTIELLEMENT VRAI" for v in verdicts)
    if has_faux:
        clamped_score = min(clamped_score, 3)
    elif has_partiel:
        clamped_score = min(clamped_score, 6)

    action_kit = None
    if is_tech:
        action_kit = generate_action_kit(
            cleaned_claims,
            search_contexts,
            gemini_client=gemini_client,
            groq_client=groq_client,
        )

    return FactCheckResult(
        verdicts=verdicts,
        reliability_score=clamped_score,
        action_kit=action_kit,
    )
