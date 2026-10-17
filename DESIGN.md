# ReelCheck Design System — Swiss Investigative & Intelligence Desk

> Inspiré des principes suisses de clarté radicale (SBB `digital.sbb.ch`) et du framework Impeccable (`impeccable.style`).
> Banni tout cliché de "vibecoder" : zéro dégradé violet néon, zéro carte arrondie générique imbriquée, zéro halo flou.

---

## 1. Direction Artistique & Identité

- **Archétype :** *Desk de Fact-Checking & Journalisme d'Enquête d'élite*.
- **Atmosphère :** Sérieuse, chirurgicale, institutionnelle, sans compromis sur la lisibilité. Chaque bordure, typographie et chiffre a une utilité factuelle.
- **Support :** Mobile-first réactif, optimisé pour une consultation rapide au smartphone (Safari / Chrome) et un usage desktop précis.

---

## 2. Typographie & Échelle

### Polices
1. **Titres & Identité :** `Space Grotesk` (700 / 600) — géométrie technique, contemporaine et incisive.
2. **Citations & Affirmations analysées :** `Newsreader` (Italic, 500) — serif d'investigation de presse apportant une humanité et une solennité aux déclarations vérifiées.
3. **Interface & Synthèse :** `Plus Jakarta Sans` (400 / 500 / 600) — lisibilité parfaite à toutes les tailles sans fatigue visuelle.
4. **Données, Scores & Métadonnées :** `JetBrains Mono` (500) — alignement tabulaire des chiffres, scores et URL.

### Échelle Typographique
- **Hero Title :** 28px - 34px, line-height 1.15, tracking -0.03em.
- **Section Headers :** 14px, uppercase, tracking +0.08em, font-weight 600.
- **Claims Quotes :** 15px, line-height 1.5, serif italic.
- **Body & Explications :** 13px - 14px, line-height 1.6, max measure 68ch.
- **Badges & Métadonnées :** 11px, monospace ou sans-serif compact.

---

## 3. Palette Chromatique & Contraste

### Surfaces & Fonds
- **Fond principal :** Matte Obsidian `#080b11` (noir profond mat, neutre, sans dominante bleutée criarde).
- **Surface des blocs d'analyse :** Slate Navy `#0f141f` avec bordure hairline `#1e2638`.
- **Champs de saisie & zones actives :** `#131926` avec bordure active `#334155`.

### Hiérarchie du Texte
- **Texte Primaire (Contraste 14:1) :** `#f8fafc` (blanc cassé pur).
- **Texte Secondaire (Contraste 7:1) :** `#94a3b8` (gris neutre lisible).
- **Texte Tertiaire / Métadonnées :** `#64748b`.

### Verdicts Factuels
- **VRAI / VÉRIFIÉ :** Vert Émeraude chirurgical `#10b981` (Fond badge `#064e3b` 20%, bordure `#059669` 40%).
- **FAUX / INTOX :** Rouge Carmin `#ef4444` (Fond badge `#7f1d1d` 20%, bordure `#dc2626` 40%).
- **PARTIELLEMENT VRAI / TROMPEUR :** Ambre Doré `#f59e0b` (Fond badge `#78350f` 20%, bordure `#d97706` 40%).

---

## 4. Composants & Structure

### Boutons & Contrôles
- **Bouton d'Action Primaire :** `#f8fafc` (texte noir `#080b11`, font-semibold, hover `#e2e8f0`) — impact visuel maximal et contraste franc.
- **Boutons Secondaires :** Surface `#131926`, texte `#e2e8f0`, bordure `#1e2638`.
- **Coins :** Rayon maîtrisé de `6px` à `8px` (suppression des formes pilules exagérées de 24px).

### Jauge de Fiabilité
- Jauge circulaire ou cadran millimétré avec graduation typographique monospace (`8.5 / 10`).
- Indicateur de fiabilité avec mention explicite (*HAUTE FIABILITÉ*, *CONTENU TROMPEUR*, *DÉSINFORMATION*).

### Affichage des Affirmations
- Chaque affirmation est présentée comme une **« Pièce à conviction »** :
  - L'affirmation originale en italique éditorial `« ... »`.
  - Le tampon de verdict.
  - L'explication concise et argumentée.
  - Les sources sous forme d'étiquettes discrètes avec nom de domaine clair et flèche `↗`.
