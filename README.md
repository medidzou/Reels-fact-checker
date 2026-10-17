# ReelCheck 🎯📱

> **Bot Telegram de Fact-Checking Automatisé d'Instagram Reels — 0€, 100% Gratuit, Zéro Carte Bancaire.**  
> *Automated Instagram Reels Fact-Checking Telegram Bot — 100% Free, Zero Card Required.*

---

## 🇫🇷 Français

### 1. Présentation & Fonctionnalités

**ReelCheck** est un bot Telegram conçu pour être utilisé directement depuis votre lit ou en déplacement sur votre smartphone (iOS / Android). Vous tombez sur un Reel Instagram aux affirmations sensationnelles, sur une astuce tech douteuse ou sur une « méthode miracle » ?  
Partagez simplement le lien (ou envoyez directement la vidéo) au bot Telegram ReelCheck : il analyse, vérifie et vous répond en quelques secondes !

#### Fonctionnalités Clés :
- 🔍 **Téléchargement & Extraction Automatisés** : Récupère l'audio du Reel via `yt-dlp` ou traite directement les fichiers vidéo/audio MP4/MOV/MP3/M4A envoyés (idéal pour les comptes privés).
- 🎙️ **Transcription Audio & Extraction d'Affirmations (Claims)** :
  - Pipeline principal ultra-rapide propulsé par **Google Gemini 2.0 Flash** (multimodal direct sur audio).
  - Fallback automatique transparent sur **Groq Whisper Large v3** et **Llama 3.3 70B** en cas de quota ou indisponibilité.
- ✅ **Fact-Checking Rigoureux avec Sources Web Réelles** :
  - Recherche web ciblée et instantanée via DuckDuckGo.
  - Attribution d'un verdict transparent par affirmation :
    - 🟢 **VRAI**
    - 🔴 **FAUX**
    - 🟡 **PARTIELLEMENT VRAI**
    - ⚪ **NON VÉRIFIABLE**
  - Synthèse globale avec niveau de confiance (0 à 100%) et liens directs vers les sources web.
- 🛠️ **Kits d'Action Techniques** : Pour tout Reel à teneur pratique ou technologique (outils, tutoriels, code, installation), le bot génère un plan d'action concret étape par étape avec commandes shell, prérequis et outils recommandés.
- ⚙️ **Modes d'Affichage Personnalisables** : `/mode full` (complet), `/mode fact` (synthèse & vérification seule), `/mode action` (kit technique seul).
- 💰 **100% Gratuit à Vie (0€)** : Aucun abonnement payant, aucune carte bancaire requise (Free Tier Render, Free Tier Gemini AI Studio, Free Tier Groq Cloud).

---

### 2. Architecture du Pipeline

```text
[Utilisateur iOS / Android]
           │
           ▼ (Partage URL Instagram ou Envoi Vidéo/Audio)
[Bot Telegram / Webhook Flask sur Render (Docker)]
           │
           ├─► yt-dlp (Extraction Audio MP3/WAV)
           │
           ├─► Extraction des Affirmations :
           │     1️⃣ Gemini 2.0 Flash (Audio direct)
           │     2️⃣ [Fallback] Groq Whisper v3 + Llama 3.3 70B
           │
           ├─► Recherche Web & Fact-Checking :
           │     • DuckDuckGo Search API
           │     • Synthèse des Verdicts & Sources
           │
           ├─► Génération du Kit d'Action Technique (si pertinent)
           │
           ▼
[Réponse formatée et structurée envoyée sur Telegram]
```

---

### 3. Prérequis 100% Gratuits (Sans Carte Bancaire)

Pour héberger et faire fonctionner votre propre instance ReelCheck gratuitement, vous avez besoin de 3 éléments :

1. **Token Bot Telegram (Gratuit)** :
   - Ouvrez Telegram et cherchez `@BotFather`.
   - Tapez `/newbot` et suivez les instructions pour donner un nom et un pseudo à votre bot (ex: `MonReelCheckBot`).
   - Copiez le token d'API fourni (format : `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).

2. **Clé API Google Gemini (Gratuit)** :
   - Rendez-vous sur [Google AI Studio](https://aistudio.google.com/).
   - Connectez-vous avec votre compte Google personnel.
   - Cliquez sur **Get API Key** → **Create API Key**.
   - Copiez votre clé API Gemini (Free Tier avec quotas généreux inclus sans carte de crédit).

3. **Clé API Groq (Gratuit)** :
   - Rendez-vous sur [Groq Cloud Console](https://console.groq.com/).
   - Connectez-vous (GitHub ou Google).
   - Allez dans **API Keys** → **Create API Key**.
   - Copiez votre clé API Groq (accès ultra-rapide à Whisper Large v3 et Llama 3.3 gratuit).

---

### 4. Déploiement en 1 Clic sur Render

Le projet inclut un Blueprint `render.yaml` prêt pour le déploiement Docker sur le Free Tier de [Render.com](https://render.com).

#### Étape par étape :
1. **GitHub** :
   - Poussez ce dépôt sur votre compte GitHub (public ou privé).
2. **Création du Service sur Render** :
   - Connectez-vous à [Render.com](https://render.com).
   - Cliquez sur **New +** → **Blueprint** (ou **Web Service**).
   - Sélectionnez votre dépôt GitHub.
   - Si vous utilisez **Blueprint**, Render détecte automatiquement `render.yaml`.
   - Si vous créez un **Web Service Docker** manuellement :
     - Runtime : `Docker`
     - Plan : `Free`
     - Région : `Frankfurt (EU)` ou `Oregon (US)`
     - Health Check Path : `/health`
3. **Variables d'Environnement** :
   Dans l'onglet **Environment** de votre Web Service sur Render, ajoutez :

   | Variable | Description | Exemple |
   | :--- | :--- | :--- |
   | `TELEGRAM_BOT_TOKEN` | Token obtenu auprès de @BotFather | `123456789:ABCdefGh...` |
   | `GEMINI_API_KEY` | Clé API Google AI Studio | `AIzaSyD...` |
   | `GROQ_API_KEY` | Clé API Groq Cloud | `gsk_...` |
   | `WEBHOOK_URL` | URL publique Render de votre webhook | `https://mon-reelcheck.onrender.com/webhook/123456789:ABCdefGh...` |
   | `PYTHONUNBUFFERED` | Forcer l'affichage immédiat des logs | `1` |

   > **Astuce Format `WEBHOOK_URL`** :  
   > L'URL de webhook peut être renseignée sous la forme :  
   > `https://<votre-service>.onrender.com/webhook/<VOTRE_TELEGRAM_BOT_TOKEN>`  
   > ou simplement `https://<votre-service>.onrender.com`.  
   > ReelCheck enregistre automatiquement le webhook auprès des serveurs Telegram au démarrage de l'application !

4. **Déploiement** :
   - Cliquez sur **Deploy**.
   - En 2 à 3 minutes, le conteneur Docker est construit, ffmpeg est installé, et votre bot est en ligne !

---

### 5. Utilisation Quotidienne (Mobile / iPhone / Android)

Le bot est pensé pour une utilisation ultra-fluide depuis son canapé ou son lit :

#### A. Depuis Instagram (Partage natif) :
1. Sur l'application Instagram, appuyez sur l'icône de partage ✈️ du Reel.
2. Choisissez **Partager via...** puis sélectionnez **Telegram**.
3. Choisissez votre bot **ReelCheck**.
4. Le bot accuse réception et commence l'analyse immédiatement !

#### B. Comptes Privés ou Reels Restreints :
Si Instagram restreint l'accès public au Reel :
1. Enregistrez l'écran ou téléchargez le Reel via votre application Instagram.
2. Envoyez directement le fichier vidéo (`.mp4`, `.mov`) ou audio (`.mp3`, `.m4a`) dans la discussion Telegram avec le bot.
3. ReelCheck traite directement le fichier média sans avoir besoin d'accéder aux serveurs d'Instagram !

#### C. Commandes Disponibles :
- `/start` : Message d'accueil et instructions de démarrage.
- `/help` : Aide détaillée et conseils d'utilisation.
- `/mode` : Affiche le mode d'affichage actuel.
- `/mode full` : Fact-check complet + Kit d'action technique.
- `/mode fact` : Uniquement la vérification des affirmations et verdicts.
- `/mode action` : Uniquement le kit pratique et les étapes techniques.

---

### 6. Dépannage & FAQ

#### Q : Pourquoi le bot met-il 30 à 50 secondes à répondre à la première requête après une période d'inactivité ?
> **R :** C'est le comportement normal du **Free Tier de Render**. Après 15 minutes sans requête, Render met le conteneur en sommeil pour économiser les ressources. Dès que Telegram envoie le webhook d'un message, Render réveille le conteneur (environ 30 à 50 secondes). Les requêtes suivantes répondent immédiatement en quelques secondes.

#### Q : Comment vérifier que le bot fonctionne ?
> **R :** Ouvrez l'URL de votre service dans un navigateur web : `https://<votre-app>.onrender.com/health`. Vous devez recevoir la réponse JSON :  
> `{"service": "reelcheck", "status": "ok"}`.

#### Q : Que faire si Instagram bloque le téléchargement (`yt-dlp`) ?
> **R :** Envoyez simplement la vidéo sous forme de fichier directement dans Telegram. ReelCheck possède un extracteur local d'audio via `ffmpeg` qui contourne tout blocage réseau d'Instagram.

---
---

## 🇬🇧 English

### 1. Overview & Features

**ReelCheck** is an automated Telegram fact-checking bot designed for effortless everyday mobile usage. Encounter a sensational Instagram Reel, suspicious life-hack, or dubious tech tutorial?  
Simply share the Reel link (or forward the video/audio file) to your ReelCheck Telegram bot: it analyzes claims, cross-references credible web sources, and replies with a complete, referenced verdict.

#### Highlights:
- 🔍 **Automated Ingestion** : Extracts audio from public Reels via `yt-dlp` or processes uploaded video/audio files directly (`.mp4`, `.mov`, `.mp3`, `.m4a`).
- 🎙️ **Audio Transcription & Claim Extraction** :
  - Primary pipeline using **Google Gemini 2.0 Flash** directly on audio.
  - Transparent automated fallback to **Groq Whisper Large v3** and **Llama 3.3 70B** if quotas or limits are reached.
- ✅ **Web Verification with Live Sources** :
  - Targeted web queries via DuckDuckGo Search API.
  - Structured verdicts per statement:
    - 🟢 **TRUE (VRAI)**
    - 🔴 **FALSE (FAUX)**
    - 🟡 **PARTIALLY TRUE (PARTIELLEMENT VRAI)**
    - ⚪ **UNVERIFIABLE**
  - Synthesized summary with confidence score (0-100%) and direct markdown citation links.
- 🛠️ **Actionable Tech Kits** : For technical and tutorial Reels, generates step-by-step implementation instructions, prerequisites, and shell commands.
- ⚙️ **Configurable Output Modes** : `/mode full`, `/mode fact`, `/mode action`.
- 💰 **Zero Cost (0€)** : No subscriptions, no credit card required (Render Free tier, Google Gemini free tier, Groq free tier).

---

### 2. Architecture Diagram

```text
[Mobile User (iOS / Android)]
           │
           ▼ (Share Instagram URL or upload video/audio)
[Flask Webhook Application on Render (Docker container)]
           │
           ├─► Audio Ingestion (yt-dlp / ffmpeg)
           │
           ├─► Audio Transcription & Claim Extraction:
           │     1️⃣ Gemini 2.0 Flash
           │     2️⃣ Fallback: Groq Whisper Large v3 + Llama 3.3
           │
           ├─► Live Web Search & Fact-Checking:
           │     • DuckDuckGo Search API
           │     • Synthesis & Source Citations
           │
           ├─► Tech Action Kit Generator (if applicable)
           │
           ▼
[Formatted telegram response sent to user]
```

---

### 3. Free Prerequisites (No Credit Card Needed)

1. **Telegram Bot Token** :
   - Open Telegram and message `@BotFather`.
   - Run `/newbot` and follow the prompts to name your bot.
   - Save your API Token (e.g. `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).

2. **Google Gemini API Key** :
   - Go to [Google AI Studio](https://aistudio.google.com/).
   - Sign in with your Google account.
   - Click **Get API Key** → **Create API Key**.
   - Free tier includes generous quotas without billing details.

3. **Groq Cloud API Key** :
   - Visit [Groq Cloud Console](https://console.groq.com/).
   - Sign in and navigate to **API Keys** → **Create API Key**.
   - Enjoy ultra-fast Whisper Large v3 audio transcription as a fallback.

---

### 4. 1-Click Render Deployment

The repository includes a ready-to-use Render Blueprint (`render.yaml`).

#### Step-by-Step Deployment:
1. **Push to GitHub**: Fork or push this repository to GitHub.
2. **Create Render Service**:
   - Log into [Render.com](https://render.com).
   - Click **New +** → **Blueprint** and select your GitHub repo.
   - Alternatively, choose **Web Service** → **Docker Runtime** (Free Plan).
3. **Configure Environment Variables**:
   Under **Environment** tab on Render:
   - `TELEGRAM_BOT_TOKEN` : Your BotFather token.
   - `GEMINI_API_KEY` : Your Google Gemini API key.
   - `GROQ_API_KEY` : Your Groq API key.
   - `WEBHOOK_URL` : `https://<your-service>.onrender.com/webhook/<TELEGRAM_BOT_TOKEN>`
   - `PYTHONUNBUFFERED` : `1`
4. **Deploy**:
   - Render builds the Docker image, installs `ffmpeg`, and automatically configures your Telegram webhook upon boot!

---

### 5. Daily Mobile Usage

- **From Instagram**: Tap the share icon ✈️ on any Reel → **Share via...** → **Telegram** → select your bot.
- **Direct Upload**: If the Instagram account is private or content is region-locked, screen-record or save the clip and send it directly as a video/audio message.
- **Commands**:
  - `/start` : Welcome message and usage instructions.
  - `/help` : Help and supported formats.
  - `/mode` : View or change response format (`full`, `fact`, `action`).

---

### 6. Troubleshooting & FAQ

- **Initial latency (30-50s)**: On Render's Free Tier, services spin down after 15 minutes of inactivity. When a new message triggers the webhook, Render automatically wakes up the container within 30-50 seconds at zero cost. Subsequent requests are instant.
- **Health check**: Verify service availability anytime by visiting `https://<your-app>.onrender.com/health`. Response will be `{"service": "reelcheck", "status": "ok"}`.
- **Logs**: Real-time webhook and pipeline logs can be inspected directly in the Render dashboard under the **Logs** tab.

---

## 📄 License
MIT License. Free for personal and educational use.
