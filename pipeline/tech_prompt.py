"""Standardized audit prompt generator and deep link formatter for Tech Reels."""

from typing import Any, Optional
import urllib.parse


def generate_audit_prompt(transcript: Optional[str]) -> str:
    """Generate a rigorous, standardized technical and cybersecurity audit prompt.

    Contextualized with the user's specific hardware specs:
    - OS: Windows 11
    - CPU: Intel Core i5-1035G1 (4C/8T)
    - RAM: 24 Go
    - GPU: None (Intel UHD Graphics, no CUDA)
    - Budget: 0 €, strictly free tier / open-source.

    Args:
        transcript: Verbatim transcription of the Reel.

    Returns:
        Structured Markdown prompt ready for injection into LLMs.
    """
    clean_transcript = transcript.strip() if transcript and transcript.strip() else "[Aucune transcription disponible]"

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


def generate_deep_links(prompt: str) -> dict[str, str]:
    """Generate 1-click deep links for major AI platforms.

    Truncates long query strings to 3500 chars to avoid HTTP 414 URI Too Long errors
    while keeping the full prompt accessible via clipboard copy.

    Args:
        prompt: The full audit prompt text.

    Returns:
        Dict mapping platform keys ('chatgpt', 'claude', 'gemini') to deep-link URLs.
    """
    # Safe truncation for URL query parameters to avoid browser/server length limits
    url_prompt = prompt if len(prompt) <= 3500 else prompt[:3450] + "\n... [texte tronqué pour l'URL, utilisez Copier]"
    encoded_query = urllib.parse.quote(url_prompt)

    return {
        "chatgpt": f"https://chatgpt.com/?q={encoded_query}",
        "claude": f"https://claude.ai/new?q={encoded_query}",
        "gemini": "https://gemini.google.com/app",
    }


def build_tech_report_dict(transcription: Any) -> dict[str, Any]:
    """Assemble a standardized tech audit report dict shared across Web & Telegram pipelines."""
    transcript_text = getattr(transcription, "transcript", "")
    summary_text = getattr(transcription, "summary", "")
    claims_list = getattr(transcription, "claims", [])

    prompt = generate_audit_prompt(transcript_text)
    links = generate_deep_links(prompt)

    return {
        "summary": summary_text,
        "transcript": transcript_text,
        "is_tech": True,
        "tech_prompt": prompt,
        "deep_links": links,
        "score": None,
        "verdict": "REEL TECH DÉTECTÉ",
        "explanation": (
            "Ce contenu a été identifié comme un Reel présentant un outil, repo ou prompt. "
            "Utilisez les boutons ci-dessous pour lancer l'audit technique et cybersécurité sur votre IA préférée."
        ),
        "claims": claims_list,
        "verdicts": [],
        "action_kit": None,
    }
