"""Specialized prompt generator and deep link dispatcher for Instagram Reels.

Generates tailored analysis prompts according to the detected content category:
- 'tech' : Software, AI, GitHub repos, MCP servers (hardware & cybersecurity audit).
- 'cooking': Recipes, ingredient extraction, step-by-step directions, calories/macros.
- 'health': Medical claims, clinical consensus, health risks, counter-indications.
- 'debate': Rhetoric, logical fallacies, argument strength, unbiased arbitration.
- 'general': Objective fact-checking, verifiable facts, claim verification.
"""

from typing import Any, Optional
import urllib.parse
from pipeline.classifier import classify_content, CATEGORIES


def generate_tech_prompt(clean_transcript: str) -> str:
    """Rigorous technical and cybersecurity audit prompt with user hardware context."""
    return f"""Tu es un ingénieur logiciel et expert en cybersécurité. Voici la transcription d'un Reel présentant un outil, repo ou prompt :

---
{clean_transcript}
---

Contexte de mon environnement :
- OS : Windows 11
- Processeur : Intel Core i5-1035G1 (4C/8T)
- RAM : 24 Go
- Carte graphique : Aucun GPU dédié (Intel UHD Graphics, pas de CUDA)
- Profil : Développeur étudiant en cybersécurité, budget 0 €, privilégie l'open-source local ou les API gratuites sans carte bancaire.

Réponds de façon concise et sans complaisance aux 4 points suivants :
1. **Promesse vs Réalité** : Que fait réellement cet outil ? Est-ce du pur sensationnalisme d'engagement ("commente X pour le prompt") ou y a-t-il une vraie valeur technique ?
2. **Faisabilité Machine** : Est-ce que cela peut tourner efficacement sur ma machine (CPU/RAM/GPU) ? Quels sont les prérequis réels (Docker, Node.js, Python, abonnements payants masqués) ?
3. **Audit Cyber & Sécurité** : Quels sont les risques potentiels (packages suspects, vol de clés API stockées en clair, injection de prompt dissimulée, exfiltration de données) ?
4. **Verdict & Recommandation** : Est-ce que cela vaut la peine d'être testé ou est-ce à ignorer ? Si oui, quelle est la commande exacte ou l'alternative open-source recommandée ?"""


def generate_cooking_prompt(clean_transcript: str) -> str:
    """Cooking, recipe extraction, directions and calorie/macro analysis prompt."""
    return f"""Tu es un chef cuisinier et nutritionniste expert. Voici la transcription d'un Reel de cuisine / recette :

---
{clean_transcript}
---

Fournis une analyse claire, structurée et directement exploitable :
1. **Liste complète des ingrédients** : Liste tous les ingrédients mentionnés ou nécessaires, avec proportions estimées et substitutions possibles.
2. **Recette étape par étape** : Déroulé chronologique clair et concis (préparation, temps de cuisson, température du four, astuces de chef).
3. **Estimation nutritionnelle & Calories** : Estimation réaliste du total calorique par portion et répartition des macronutriments (protéines, glucides, lipides).
4. **Verdict & Conseils** : Niveau de difficulté réel, temps total estimé et pièges à éviter lors de la préparation."""


def generate_health_prompt(clean_transcript: str) -> str:
    """Scientific and medical fact-checking prompt for health/wellness content."""
    return f"""Tu es un médecin et chercheur scientifique expert en méthodologie et santé publique. Voici la transcription d'un Reel abordant des questions de santé ou de bien-être :

---
{clean_transcript}
---

Analyse ces affirmations avec rigueur, impartialité et sens critique :
1. **Consensus Scientifique & Fact-Checking** : Que dit la littérature médicale et le consensus scientifique actuel sur ces affirmations ? Sont-elles prouvées, controversées ou infondées ?
2. **Risques & Dangers Potentiels** : Y a-t-il des effets indésirables, contre-indications, risques d'interactions ou dérives (retard de consultation médicale, carences, surdosage) ?
3. **Nuances & Contexte Manquant** : Qu'est-ce que la vidéo simplifie à l'excès, dramatise ou omet délibérément pour faire le buzz ?
4. **Recommandations & Précautions** : Quel est le bon réflexe à adopter et quand est-il impératif de consulter un professionnel de santé ?"""


def generate_debate_prompt(clean_transcript: str) -> str:
    """Rhetoric, fallacy detection, and logical argumentation audit prompt."""
    return f"""Tu es un logicien, philosophe et expert en argumentation et rhétorique. Voici la transcription d'un Reel présentant un débat ou une argumentation contradictoire :

---
{clean_transcript}
---

Examine cette intervention avec neutralité et rigueur logique :
1. **Thèse(s) & Positions** : Quels sont les points de vue et arguments principaux avancés par chaque partie ?
2. **Détection des Sophismes & Erreurs de Logique (Fallacies)** : Identifie précisément chaque sophisme ou biais utilisé (ad hominem, homme de paille / strawman, fausse dichotomie, appel à l'émotion, pente glissante...).
3. **Solidité Argumentative & Arbitrage** : Qui avance les arguments les plus solides, factuels et rationnels, indépendamment de l'éloquence ou de la forme ?
4. **Synthèse Critique** : Quel serait le point de vue le plus équilibré et rigoureux sur cette question d'après les données disponibles ?"""


def generate_general_prompt(clean_transcript: str) -> str:
    """General fact-checking prompt for news, claims, history, and miscellaneous content."""
    return f"""Tu es un journaliste d'investigation et fact-checker senior. Voici la transcription d'un Reel :

---
{clean_transcript}
---

Effectue une vérification factuelle complète, sourcée et sans complaisance :
1. **Affirmations Vérifiables** : Isole les faits précis avancés (statistiques, dates, citations, faits historiques ou d'actualité).
2. **Vérification Point par Point** : Indique pour chaque fait s'il est VRAI, FAUX, EXAGÉRÉ ou TROMPEUR, avec éléments de preuve et sources fiables.
3. **Contexte & Manipulation** : Y a-t-il manipulation d'images/chiffres, raccourci trompeur ou fait sorti de son contexte ?
4. **Verdict Final Synthétique** : Quel est le degré de fiabilité global de ce contenu (Fiable, Douteux, ou Désinformation) ?"""


PROMPT_GENERATORS = {
    "tech": generate_tech_prompt,
    "cooking": generate_cooking_prompt,
    "health": generate_health_prompt,
    "debate": generate_debate_prompt,
    "general": generate_general_prompt,
}


def generate_prompt_for_category(transcript: Optional[str], category: str) -> str:
    """Generate the specialized prompt for the requested category.

    Args:
        transcript: Verbatim text from the Reel.
        category: Category key ('tech', 'cooking', 'health', 'debate', 'general').

    Returns:
        Structured Markdown prompt ready for LLM injection.
    """
    clean_transcript = (
        transcript.strip()
        if transcript and transcript.strip()
        else "[Aucune transcription disponible]"
    )

    generator = PROMPT_GENERATORS.get(category, generate_general_prompt)
    return generator(clean_transcript)


def generate_deep_links(prompt: str) -> dict[str, str]:
    """Generate 1-click deep links for major AI platforms.

    Truncates long query strings to 3500 chars to avoid HTTP 414 URI Too Long errors
    while keeping the full prompt accessible via clipboard copy.

    Args:
        prompt: The full prompt text.

    Returns:
        Dict mapping platform keys ('chatgpt', 'claude', 'gemini') to deep-link URLs.
    """
    url_prompt = prompt if len(prompt) <= 3500 else prompt[:3450] + "\n... [texte tronqué pour l'URL, utilisez Copier]"
    encoded_query = urllib.parse.quote(url_prompt)

    return {
        "chatgpt": f"https://chatgpt.com/?q={encoded_query}",
        "claude": f"https://claude.ai/new?q={encoded_query}",
        "gemini": "https://gemini.google.com/app",
    }


def build_dispatch_report(transcription: Any, category: Optional[str] = None) -> dict[str, Any]:
    """Assemble a standardized dispatch report dict shared across Web & Telegram pipelines."""
    transcript_text = getattr(transcription, "transcript", "") or ""
    summary_text = getattr(transcription, "summary", "") or ""
    claims_list = getattr(transcription, "claims", []) or []

    # Detect category if not explicitly provided
    if not category:
        # Check if transcription flagged tech
        if getattr(transcription, "is_tech", False):
            detected_cat = "tech"
            detected_label = CATEGORIES["tech"]["label"]
        else:
            detected_cat, detected_label = classify_content(transcript_text)
    else:
        detected_cat = category
        detected_label = CATEGORIES.get(category, {}).get("label", CATEGORIES["general"]["label"])

    prompt = generate_prompt_for_category(transcript_text, detected_cat)
    links = generate_deep_links(prompt)

    verdict_labels = {
        "tech": "Outil ou Dépôt Détecté",
        "cooking": "Cuisine & Recette Détectée",
        "health": "Sujet Santé & Bien-être Détecté",
        "debate": "Débat & Rhétorique Détecté",
        "general": "Affirmations à Vérifier Détectées",
    }

    return {
        "summary": summary_text,
        "transcript": transcript_text,
        "category": detected_cat,
        "category_label": detected_label,
        "is_tech": (detected_cat == "tech"),
        "tech_prompt": prompt,  # Backward compatibility for UI & legacy callers
        "deep_links": links,
        "score": None,
        "verdict": verdict_labels.get(detected_cat, "Contenu Détecté"),
        "explanation": (
            f"Ce contenu a été classé en : {detected_label}. "
            "Utilisez les boutons ci-dessous pour ouvrir l'analyse ciblée sur votre IA préférée."
        ),
        "claims": claims_list,
        "verdicts": [],
        "action_kit": None,
    }
