"""Unit tests for the Fact-Check Engine (DuckDuckGo + LLM Synthesis)."""

from unittest.mock import MagicMock, patch
import pytest

from pipeline.factchecker import (
    FactCheckResult,
    Verdict,
    fact_check_claims,
    generate_action_kit,
    search_claim,
    synthesize_verdict,
)


def test_verdict_dataclass():
    """Test Verdict dataclass fields and types."""
    v = Verdict(
        claim="Le ciel est bleu à cause de la diffusion de Rayleigh",
        verdict="VRAI",
        explanation="Phénomène physique confirmé par la science.",
        sources=["https://fr.wikipedia.org/wiki/Diffusion_Rayleigh"],
    )
    assert v.claim == "Le ciel est bleu à cause de la diffusion de Rayleigh"
    assert v.verdict == "VRAI"
    assert "physique" in v.explanation
    assert v.sources == ["https://fr.wikipedia.org/wiki/Diffusion_Rayleigh"]


def test_fact_check_result_dataclass():
    """Test FactCheckResult dataclass defaults and values."""
    r = FactCheckResult(verdicts=[], reliability_score=10, action_kit=None)
    assert r.verdicts == []
    assert r.reliability_score == 10
    assert r.action_kit is None


def test_search_claim_success():
    """Test searching a claim with a mock DDGS client returning results."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {
            "title": "Diffusion Rayleigh",
            "href": "https://example.com/rayleigh",
            "body": "La diffusion Rayleigh explique la couleur bleue du ciel.",
        },
        {
            "title": "Atmosphère et lumière",
            "href": "https://example.com/atmosphere",
            "body": "Les ondes courtes bleues sont plus diffusées.",
        },
    ]

    context = search_claim("Pourquoi le ciel est bleu", ddgs_client=mock_ddgs)
    assert "Diffusion Rayleigh" in context
    assert "https://example.com/rayleigh" in context
    assert "Atmosphère et lumière" in context
    mock_ddgs.text.assert_called_once_with("Pourquoi le ciel est bleu", max_results=3)


def test_detect_institutional_query():
    """Verify that institutional keywords generate targeted queries."""
    from pipeline.factchecker import _detect_institutional_query

    assert "Cour des comptes" in (_detect_institutional_query("Selon la Cour des comptes le coût est de 1 milliard") or "")
    assert "Légifrance" in (_detect_institutional_query("Le nouveau décret augmente les droits") or "")
    assert "Insee" in (_detect_institutional_query("Le taux de pauvreté selon l'Insee est en hausse") or "")
    assert _detect_institutional_query("La Terre tourne autour du Soleil") is None


def test_search_claim_empty_or_no_results():
    """Test search with empty claim or no results returned."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = []

    # Empty query should return empty string without querying
    assert search_claim("", ddgs_client=mock_ddgs) == ""
    assert mock_ddgs.text.call_count == 0

    # Query with no results from DDGS or fallback
    with patch("urllib.request.urlopen") as mock_url:
        fake_empty = MagicMock()
        fake_empty.read.return_value = b'{"query": {"search": []}}'
        mock_url.return_value.__enter__.return_value = fake_empty
        assert search_claim("nonexistent_rare_query_xyz", ddgs_client=mock_ddgs) == ""


@patch("urllib.request.urlopen", side_effect=Exception("Wikipedia unreachable"))
def test_search_claim_handles_exception(mock_url):
    """Test that search_claim handles DDGS and fallback exceptions gracefully."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = Exception("Rate limit reached")

    context = search_claim("Un claim quelconque", ddgs_client=mock_ddgs)
    assert context == ""


@patch("urllib.request.urlopen")
def test_search_claim_wikipedia_fallback(mock_url):
    """Test that search_claim falls back to Wikipedia when DDGS fails."""
    import json

    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = Exception("Rate limit reached")

    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({
        "query": {
            "search": [
                {
                    "title": "Pakistan",
                    "snippet": "Le <span class=\"searchmatch\">Pakistan</span> est un pays d'Asie du Sud.",
                }
            ]
        }
    }).encode("utf-8")
    mock_url.return_value.__enter__.return_value = fake_resp

    context = search_claim("Pakistan population", ddgs_client=mock_ddgs)
    assert "[1] Pakistan" in context
    assert "https://fr.wikipedia.org/wiki/Pakistan" in context
    assert "Le Pakistan est un pays d'Asie du Sud." in context


def test_synthesize_verdict_vrai():
    """Test LLM synthesis yielding a VRAI verdict with structured text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "VERDICT: VRAI\n"
        "EXPLANATION: La claim est parfaitement corroborée par les données scientifiques.\n"
        "SOURCES:\n"
        "- https://science.org/study\n"
        "- https://nature.com/article"
    )
    mock_client.models.generate_content.return_value = mock_response

    verdict = synthesize_verdict(
        claim="L'eau gèle à 0°C à pression normale",
        search_context="Source: 0 degrés Celsius est le point de congélation.",
        gemini_client=mock_client,
    )

    assert verdict.verdict == "VRAI"
    assert "parfaitement corroborée" in verdict.explanation
    assert "https://science.org/study" in verdict.sources
    assert "https://nature.com/article" in verdict.sources


def test_synthesize_verdict_faux():
    """Test LLM synthesis yielding a FAUX verdict."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "VERDICT: FAUX\n"
        "EXPLANATION: Cette affirmation est totalement fausse et contredite par les faits.\n"
        "SOURCES:\n"
        "- https://hoaxbuster.com/check"
    )
    mock_client.models.generate_content.return_value = mock_response

    verdict = synthesize_verdict(
        claim="La Terre est plate",
        search_context="La Terre est un sphéroïde.",
        gemini_client=mock_client,
    )

    assert verdict.verdict == "FAUX"
    assert "totalement fausse" in verdict.explanation
    assert "https://hoaxbuster.com/check" in verdict.sources


def test_synthesize_verdict_partiellement_vrai():
    """Test LLM synthesis yielding a PARTIELLEMENT VRAI verdict."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "VERDICT: PARTIELLEMENT VRAI\n"
        "EXPLANATION: L'effet existe mais l'étude originale ne portait que sur 10 participants.\n"
        "SOURCES:\n"
        "- https://pubmed.gov/12345"
    )
    mock_client.models.generate_content.return_value = mock_response

    verdict = synthesize_verdict(
        claim="Prendre de la vitamine C guérit le rhume instantanément",
        search_context="La vitamine C peut réduire la durée mais ne guérit pas instantanément.",
        gemini_client=mock_client,
    )

    assert verdict.verdict == "PARTIELLEMENT VRAI"
    assert "10 participants" in verdict.explanation
    assert "https://pubmed.gov/12345" in verdict.sources


def test_synthesize_verdict_json_format():
    """Test LLM synthesis when response is returned in JSON format."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """```json
    {
        "verdict": "VRAI",
        "explanation": "Confirmé par les documentations officielles de Python.",
        "sources": ["https://docs.python.org/3/whatsnew/3.11.html"]
    }
    ```"""
    mock_client.models.generate_content.return_value = mock_response

    verdict = synthesize_verdict(
        claim="Python 3.11 est 10-60% plus rapide que Python 3.10",
        search_context="Python 3.11 speeds up Python by up to 60%.",
        gemini_client=mock_client,
    )

    assert verdict.verdict == "VRAI"
    assert "documentations officielles" in verdict.explanation
    assert "https://docs.python.org/3/whatsnew/3.11.html" in verdict.sources


def test_synthesize_verdict_handles_exception():
    """Test synthesize_verdict handles client exception gracefully."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API quota exceeded")

    verdict = synthesize_verdict(
        claim="Une affirmation quelconque",
        search_context="Contexte",
        gemini_client=mock_client,
    )

    assert verdict.claim == "Une affirmation quelconque"
    assert verdict.verdict == "NON VÉRIFIABLE"
    assert "API quota exceeded" in verdict.explanation or "Erreur" in verdict.explanation


def test_generate_action_kit_tech():
    """Test generating action kit for technical claims."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "### 🛠️ Kit d'action technique\n\n"
        "1. Installer la librairie: `pip install duckduckgo-search`\n"
        "2. Code d'exemple:\n"
        "```python\n"
        "from duckduckgo_search import DDGS\n"
        "```"
    )
    mock_client.models.generate_content.return_value = mock_response

    action_kit = generate_action_kit(
        claims=["DuckDuckGo Search est gratuit sans clé API"],
        search_contexts=["DDGS can be used without an API key."],
        gemini_client=mock_client,
    )

    assert action_kit is not None
    assert "Kit d'action technique" in action_kit
    assert "pip install" in action_kit


def test_generate_action_kit_empty_claims():
    """Test generating action kit when no claims provided returns None."""
    mock_client = MagicMock()
    assert generate_action_kit([], [], gemini_client=mock_client) is None
    assert mock_client.models.generate_content.call_count == 0


def test_generate_action_kit_handles_exception():
    """Test generate_action_kit handles client errors gracefully."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error")

    action_kit = generate_action_kit(
        claims=["Un outil tech"],
        search_contexts=["Contexte"],
        gemini_client=mock_client,
    )
    assert action_kit is None


def test_fact_check_claims_non_tech():
    """Test fact_check_claims on non-tech claims (no action kit, score computed)."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [{"title": "T", "href": "https://u.org", "body": "B"}]

    mock_gemini = MagicMock()
    # First claim -> VRAI, Second claim -> FAUX
    res_vrai = MagicMock(text="VERDICT: VRAI\nEXPLANATION: Vrai\nSOURCES:\n- https://u.org")
    res_faux = MagicMock(text="VERDICT: FAUX\nEXPLANATION: Faux\nSOURCES:\n- https://u.org")
    mock_gemini.models.generate_content.side_effect = [res_vrai, res_faux]

    result = fact_check_claims(
        claims=["Affirmation vraie", "Affirmation fausse"],
        is_tech=False,
        gemini_client=mock_gemini,
        ddgs_client=mock_ddgs,
    )

    assert len(result.verdicts) == 2
    assert result.verdicts[0].verdict == "VRAI"
    assert result.verdicts[1].verdict == "FAUX"
    # Poison Rule: presence of FAUX caps score to <= 3 (instead of naive average (10+0)/2 = 5)
    assert result.reliability_score <= 3
    assert result.action_kit is None


def test_fact_check_claims_all_vrai_and_tech():
    """Test fact_check_claims with tech flag and all claims true."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [{"title": "T", "href": "https://u.org", "body": "B"}]

    mock_gemini = MagicMock()
    res_vrai = MagicMock(text="VERDICT: VRAI\nEXPLANATION: Validé\nSOURCES:\n- https://u.org")
    res_action = MagicMock(text="### Guide d'implémentation\nCode here...")
    mock_gemini.models.generate_content.side_effect = [res_vrai, res_action]

    result = fact_check_claims(
        claims=["Python 3.13 introduit un mode no-GIL"],
        is_tech=True,
        gemini_client=mock_gemini,
        ddgs_client=mock_ddgs,
    )

    assert len(result.verdicts) == 1
    assert result.verdicts[0].verdict == "VRAI"
    assert result.reliability_score == 10
    assert result.action_kit is not None
    assert "Guide d'implémentation" in result.action_kit


def test_fact_check_claims_partially_vrai_score():
    """Test score calculation with partially true claims."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = []

    mock_gemini = MagicMock()
    res_partiel = MagicMock(text="VERDICT: PARTIELLEMENT VRAI\nEXPLANATION: Nuancé\nSOURCES:\n")
    mock_gemini.models.generate_content.return_value = res_partiel

    result = fact_check_claims(
        claims=["Affirmation mi-vraie"],
        is_tech=False,
        gemini_client=mock_gemini,
        ddgs_client=mock_ddgs,
    )

    assert result.verdicts[0].verdict == "PARTIELLEMENT VRAI"
    assert result.reliability_score == 5


def test_fact_check_claims_empty():
    """Test fact_check_claims with empty claims list."""
    result = fact_check_claims(claims=[], is_tech=False)
    assert result.verdicts == []
    assert result.reliability_score == 10
    assert result.action_kit is None


def test_normalize_verdict_variants():
    """Test various formats are normalized into standard verdicts."""
    from pipeline.factchecker import _normalize_verdict

    assert _normalize_verdict("VRAI") == "VRAI"
    assert _normalize_verdict("C'est vrai et exact") == "VRAI"
    assert _normalize_verdict("TRUE") == "VRAI"
    assert _normalize_verdict("FAUX") == "FAUX"
    assert _normalize_verdict("Totalement faux et trompeur") == "FAUX"
    assert _normalize_verdict("FALSE") == "FAUX"
    assert _normalize_verdict("PARTIELLEMENT VRAI") == "PARTIELLEMENT VRAI"
    assert _normalize_verdict("Nuancé / Partiel") == "PARTIELLEMENT VRAI"
    assert _normalize_verdict("NON VÉRIFIABLE") == "NON VÉRIFIABLE"
    assert _normalize_verdict("INCONNU / INDETERMINE") == "NON VÉRIFIABLE"


def test_extract_urls_sanitization():
    """Test URL extraction strips trailing punctuation and deduplicates."""
    from pipeline.factchecker import _extract_urls

    sample_text = (
        "Consultez https://example.com/page1, ainsi que https://example.com/page2. "
        "Encore https://example.com/page1! Et (https://example.com/page3)."
    )
    urls = _extract_urls(sample_text)
    assert urls == [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]


def test_synthesize_verdict_no_api_key(monkeypatch):
    """Test synthesize_verdict when GEMINI_API_KEY is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    verdict = synthesize_verdict("Une claim sans API key", "Contexte")
    assert verdict.verdict == "NON VÉRIFIABLE"
    assert "GEMINI_API_KEY" in verdict.explanation


def test_synthesize_verdict_fallback_sources_from_search():
    """Test fallback to search context URLs when LLM doesn't output source URLs."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "VERDICT: VRAI\n"
        "EXPLANATION: Confirmé par la recherche."
    )
    mock_client.models.generate_content.return_value = mock_response

    search_ctx = "Tiré de https://trusted-source.org/fact"
    verdict = synthesize_verdict(
        claim="Fait avéré",
        search_context=search_ctx,
        gemini_client=mock_client,
    )
    assert verdict.verdict == "VRAI"
    assert "https://trusted-source.org/fact" in verdict.sources


def test_parse_verdict_response_empty():
    """Test parsing empty response returns graceful fallback."""
    from pipeline.factchecker import parse_verdict_response

    verdict = parse_verdict_response("", "Claim test")
    assert verdict.verdict == "NON VÉRIFIABLE"
    assert "Aucune réponse" in verdict.explanation


def test_poison_rule_caps_score(monkeypatch):
    """Test that a single FAUX claim caps reliability score to at most 3/10 even with 3 VRAI."""
    from pipeline.factchecker import fact_check_claims, Verdict

    monkeypatch.setattr(
        "pipeline.factchecker.search_claim",
        lambda *a, **k: "Context",
    )
    # Simulate 3 VRAI and 1 FAUX
    responses = [
        Verdict("Claim 1", "VRAI", "Vrai", []),
        Verdict("Claim 2", "VRAI", "Vrai", []),
        Verdict("Claim 3", "VRAI", "Vrai", []),
        Verdict("Claim 4", "FAUX", "Faux toxique", []),
    ]
    resp_iter = iter(responses)
    monkeypatch.setattr(
        "pipeline.factchecker.synthesize_verdict",
        lambda *a, **k: next(resp_iter),
    )

    result = fact_check_claims(["c1", "c2", "c3", "c4"], is_tech=False)
    # Without poison rule, average would be (10+10+10+0)/4 = 7.5 -> 8
    # With poison rule, it must be capped at 3
    assert result.reliability_score <= 3


def test_generate_action_kit_no_api_key(monkeypatch):
    """Test generate_action_kit returns None when GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = generate_action_kit(["Claim"], ["Context"])
    assert result is None


def test_parse_verdict_response_cleans_markdown_prefixes():
    """Verify that parse_verdict_response strips orphan markdown and explanation prefixes."""
    from pipeline.factchecker import parse_verdict_response

    raw_text = (
        "VERDICT: FAUX\n"
        "EXPLANATION: **Explication : Les chiffres sont inexacts et réfutés.\n"
        "SOURCES:\n"
        "- https://example.com/source1"
    )
    verdict = parse_verdict_response(raw_text, "Affirmation test")
    assert verdict.explanation == "Les chiffres sont inexacts et réfutés."
    assert not verdict.explanation.startswith("**")
    assert not verdict.explanation.lower().startswith("explication")


def test_synthesize_verdict_filters_hallucinated_sources():
    """Verify that synthesize_verdict drops hallucinated URLs when valid search context exists."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "VERDICT: VRAI\n"
        "EXPLANATION: Analyse confirmée.\n"
        "SOURCES:\n"
        "- https://legifrance.gouv.fr/decret/2026-385\n"
        "- https://hallucinated-fake-site.com/fake-doc.pdf"
    )
    mock_client.models.generate_content.return_value = mock_response

    verdict = synthesize_verdict(
        claim="Les droits sont encadrés par décret",
        search_context="Source officielle: https://legifrance.gouv.fr/decret/2026-385",
        gemini_client=mock_client,
    )

    assert "https://legifrance.gouv.fr/decret/2026-385" in verdict.sources
    assert "https://hallucinated-fake-site.com/fake-doc.pdf" not in verdict.sources


def test_synthesize_verdict_empty_context_drops_hallucinated_sources():
    """Verify that synthesize_verdict eliminates sources completely when search context has no results."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "VERDICT: NON VÉRIFIABLE\n"
        "EXPLANATION: Impossible de confirmer sans sources externes.\n"
        "SOURCES:\n"
        "- https://hallucinated-fake-site.com/fake-doc.pdf"
    )
    mock_client.models.generate_content.return_value = mock_response

    verdict = synthesize_verdict(
        claim="Fait non vérifiable",
        search_context="Aucun résultat trouvé sur le web.",
        gemini_client=mock_client,
    )

    assert verdict.sources == []

