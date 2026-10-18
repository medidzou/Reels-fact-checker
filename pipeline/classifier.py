"""Multi-category heuristic classifier for Instagram Reels content.

Categorizes reels into:
- 'tech' : Tools, AI models, GitHub repos, MCP servers, developer workflows.
- 'cooking': Recipes, ingredients, kitchen techniques, nutrition/macros.
- 'health': Medical claims, wellness, symptoms, pathology, clinical studies.
- 'debate': Rhetoric, arguments, debate fallacies, logic errors, reasoning.
- 'general': General factual claims, news, history, sports, or other content.
"""

import re
from typing import Optional

CATEGORIES: dict[str, dict[str, str]] = {
    "tech": {
        "label": "Tech, IA & Outils Développeur",
        "description": "Audit de pertinence, compatibilité machine, coûts réels et cybersécurité.",
        "badge_color": "violet",
    },
    "cooking": {
        "label": "Cuisine & Recette",
        "description": "Extraction des ingrédients, déroulé de la recette et estimation des calories/macros.",
        "badge_color": "amber",
    },
    "health": {
        "label": "Santé & Bien-être",
        "description": "Fact-checking scientifique, consensus médical et détection des risques.",
        "badge_color": "emerald",
    },
    "debate": {
        "label": "Débat & Analyse Rhétorique",
        "description": "Détection des sophismes (fallacies), solidité des arguments et arbitrage impartial.",
        "badge_color": "indigo",
    },
    "general": {
        "label": "Vérification Factuelle (Général)",
        "description": "Fact-checking méthodique et sourcé des affirmations clés du Reel.",
        "badge_color": "neutral",
    },
}

# --- TECH PATTERNS ---
TECH_HIGH = [
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

TECH_MED = [
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

# --- COOKING PATTERNS ---
COOKING_HIGH = [
    r"\brecette[s]?\b",
    r"\bingr[eé]dients?\b",
    r"\benfourn(?:er|ez)\b",
    r"\bcuisson\b",
    r"\bcuire\b",
    r"\bfarine\b",
    r"\bcalories?\b",
    r"\bmacronutriments?\b|\bmacros\b",
    r"\bprot[eé]ines?\b",
    r"\bpo[eê]le\b",
    r"\bcasserole\b",
    r"\bcuill[eè]re[s]?\b",
    r"\bassaisonn(?:er|ement)\b",
]

COOKING_MED = [
    r"\bcuisine\b",
    r"\bplat\b",
    r"\bp[aâ]tes\b",
    r"\bg[aâ]teau\b",
    r"\bd[iî]ner\b",
    r"\bd[eé]jeuner\b",
    r"\bviande\b",
    r"\bl[eé]gumes?\b",
    r"\bsauce\b",
    r"\bfour\b",
    r"\bgrammes?\b|\b\d+\s*g\b",
]

# --- HEALTH PATTERNS ---
HEALTH_HIGH = [
    r"\bm[eé]dicament[s]?\b",
    r"\bpathologie[s]?\b",
    r"\bsympt[oô]mes?\b",
    r"\bglyc[eé]mie\b",
    r"\bcholest[eé]rol\b",
    r"\bhypertension\b",
    r"\bdiab[eè]te\b",
    r"\bje[uû]ne\s+intermittent\b",
    r"\bsyst[eè]me\s+immunitaire\b",
    r"\bposologie\b",
    r"\beffets?\s+secondaires?\b",
    r"\bmaladie[s]?\b",
]

HEALTH_MED = [
    r"\bsant[eé]\b",
    r"\bm[eé]decin\b",
    r"\bdocteur\b",
    r"\b[eé]tude\s+scientifique\b|\b[eé]tude\b",
    r"\bvitalit[eé]\b",
    r"\bmagn[eé]sium\b",
    r"\bvitamine[s]?\b",
    r"\bcareance[s]?\b",
    r"\btraitement\b",
    r"\binflammation\b",
]

# --- DEBATE & RHETORIC PATTERNS ---
DEBATE_HIGH = [
    r"\bsophisme[s]?\b",
    r"\bfallac(?:y|ies)\b",
    r"\bad\s+hominem\b",
    r"\bhomme\s+de\s+paille\b|\bstraw\s*man\b",
    r"\bfausse\s+dichotomie\b",
    r"\bpente\s+glissante\b",
    r"\brh[eé]torique\b",
    r"\bjoute\s+oratoire\b",
    r"\bd[eé]bat\b",
    r"\br[eé]futer\b|\br[eé]futation\b",
    r"\berreur\s+de\s+logique\b",
    r"\bbiais\s+cognitif\b",
]

DEBATE_MED = [
    r"\bargument[s]?\b",
    r"\bargumentaire\b",
    r"\bcontradiction\b",
    r"\bavoir\s+raison\b|\bavoir\s+tort\b",
    r"\bd[eé]batteur[s]?\b",
    r"\badversaire\b",
    r"\bd[eé]monter\b",
    r"\bconvaincre\b",
    r"\braisonnement\b",
]

COMPILED_RULES = {
    "tech": (
        [re.compile(p, re.IGNORECASE) for p in TECH_HIGH],
        [re.compile(p, re.IGNORECASE) for p in TECH_MED],
    ),
    "cooking": (
        [re.compile(p, re.IGNORECASE) for p in COOKING_HIGH],
        [re.compile(p, re.IGNORECASE) for p in COOKING_MED],
    ),
    "health": (
        [re.compile(p, re.IGNORECASE) for p in HEALTH_HIGH],
        [re.compile(p, re.IGNORECASE) for p in HEALTH_MED],
    ),
    "debate": (
        [re.compile(p, re.IGNORECASE) for p in DEBATE_HIGH],
        [re.compile(p, re.IGNORECASE) for p in DEBATE_MED],
    ),
}


def classify_content(text: Optional[str]) -> tuple[str, str]:
    """Classify input text into one of the 5 supported categories.

    Uses weighted regex heuristics:
    - High-confidence keywords: 2 points.
    - Medium-confidence keywords: 1 point.
    - Minimum threshold to qualify for a specialized category: 2 points.
    - If no category reaches threshold, returns ('general', label).

    Args:
        text: Plain-text transcription or caption.

    Returns:
        tuple[str, str]: (category_key, category_label)
    """
    if not text or not isinstance(text, str):
        return "general", CATEGORIES["general"]["label"]

    cleaned = text.strip()
    if not cleaned:
        return "general", CATEGORIES["general"]["label"]

    scores: dict[str, int] = {}

    for cat, (high_patterns, med_patterns) in COMPILED_RULES.items():
        score = 0
        for pat in high_patterns:
            if pat.search(cleaned):
                score += 2
        for pat in med_patterns:
            if pat.search(cleaned):
                score += 1
        scores[cat] = score

    # Find highest score
    best_cat, max_score = max(scores.items(), key=lambda item: item[1])

    if max_score >= 2:
        return best_cat, CATEGORIES[best_cat]["label"]

    return "general", CATEGORIES["general"]["label"]
