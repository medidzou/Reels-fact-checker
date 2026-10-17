"""Standardized audit prompt generator and deep link formatter for Tech Reels."""

import urllib.parse
from typing import Optional


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

    Args:
        prompt: The full audit prompt text.

    Returns:
        Dict mapping platform keys ('chatgpt', 'claude', 'gemini') to deep-link URLs.
    """
    encoded_query = urllib.parse.quote(prompt)

    return {
        "chatgpt": f"https://chatgpt.com/?q={encoded_query}",
        "claude": f"https://claude.ai/new?q={encoded_query}",
        "gemini": "https://gemini.google.com/app",
    }
