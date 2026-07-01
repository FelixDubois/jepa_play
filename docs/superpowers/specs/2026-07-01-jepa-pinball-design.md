# JEPA Pinball — Spécification de design

**Date** : 2026-07-01
**Statut** : validé par l'utilisateur (design conversationnel du 2026-07-01)

## 1. Objectif

Projet **pédagogique** : apprendre l'architecture JEPA (Joint-Embedding Predictive
Architecture) en la codant de bout en bout sur un cas concret — un agent qui apprend à
jouer au flipper à partir d'images.

- **Entrée du modèle** : uniquement l'image du plateau (état du flipper et de la balle).
- **Sortie de l'agent** : le contrôle des deux batteurs (flippers).
- **Physique** : la balle suit un vrai modèle de corps rigides 2D (moteur physique).
- **Plateau paramétrable** : les éléments du flipper sont définis par une configuration.
- **Exécution** : Python, notebooks Google Colab **gratuit** (GPU T4, sessions courtes).

Priorités : clarté du code, explications pédagogiques en français dans les notebooks,
entraînements courts (15-45 min par étape), reprise après déconnexion Colab.

Ce que le projet n'est **pas** : pas de recherche de performance maximale, pas de RL
(pas de DQN/PPO), pas d'ablations de recherche.

## 2. Critères de succès

- **V1 (survie)** : l'agent garde la balle en jeu significativement plus longtemps que
  deux baselines — la politique aléatoire et la politique « boutons toujours appuyés » —
  mesuré sur 50 épisodes (temps de survie moyen).
- **V2 (score)** : avec des bumpers scoreurs ajoutés au plateau, l'agent obtient un
  score moyen supérieur aux mêmes baselines.
- **Pédagogique** : chaque notebook explique les concepts JEPA rencontrés (prédiction
  latente, encodeur cible EMA, collapse, planification dans l'espace latent) avec
  visualisations.

## 3. Architecture retenue

**World model JEPA conditionné par l'action + planification MPC** (dans l'esprit de
V-JEPA 2-AC et DINO-WM). Pas de RL : le contrôle émerge de la planification dans
l'espace latent appris.

```
o_t (image 64×64×2) ──▶ Encodeur CNN ──▶ z_t ∈ R²⁵⁶
                                           │
              action a_t ──▶ Prédicteur (MLP) ──▶ ẑ_{t+1}, ẑ_{t+2}, ... (rollout)
                                           │
o_{t+1} ──▶ Encodeur cible (EMA, sans gradient) ──▶ z̄_{t+1}   ← cible

Perte = distance(ẑ_{t+1}, z̄_{t+1}) dans l'espace latent — jamais en pixels.

Contrôle (MPC) : énumérer des séquences d'actions, les dérouler en latent avec le
prédicteur, sommer un coût appris (danger de perdre la balle, − score en V2),
exécuter la première action de la meilleure séquence, replanifier à chaque pas.
```

Alternative écartée : encodeur JEPA + politique RL (DQN) — écartée car elle ajoute
toute la complexité du RL (hors sujet pédagogique) et le RL est instable sur le budget
compute d'un Colab gratuit.

## 4. Phases et livrables

| Phase | Contenu | Notebook | Livrable / résultat visible |
|---|---|---|---|
| 0 | Simulateur physique + environnement | `01_simulateur.ipynb` | Vidéo de la balle, flippers actionnables, tests physiques verts |
| 1 | Collecte de données (politique aléatoire) | `02_collecte.ipynb` | Dataset de transitions shardé sur Google Drive |
| 2 | Entraînement JEPA action-conditionné | `03_jepa.ipynb` | Checkpoint + courbes de perte + diagnostics anti-collapse |
| 3 | Tête danger + planificateur MPC → **V1** | `04_controle.ipynb` | Vidéos de l'agent, courbe de survie vs baselines |
| 4 | Bumpers scoreurs + tête score → **V2** | `05_score.ipynb` | Score moyen vs baselines |

Chaque phase est validée avant de passer à la suivante.

## 5. Structure du code

Le code réutilisable vit dans des modules `.py` versionnés git ; les notebooks restent
légers (orchestration + pédagogie en français). Les notebooks obtiennent le code par
`git clone` (ou upload) et installent avec `pip`. Code en anglais (identifiants),
commentaires et notebooks en français.

```
jepa_play/
├── pinball/
│   ├── config.py      # BoardConfig (dataclass) : tous les paramètres du plateau
│   ├── sim.py         # Simulateur pymunk : table, flippers, balle, collisions
│   ├── render.py      # Rendu image 64×64 niveaux de gris (PIL, headless)
│   ├── env.py         # Environnement type Gymnasium (reset/step, frames empilées)
│   └── collect.py     # Politiques de collecte + écriture du dataset (.npz shardés)
├── jepa/
│   ├── model.py       # Encodeur CNN, prédicteur action-conditionné, EMA target
│   ├── train.py       # Boucle d'entraînement JEPA (multi-pas), checkpoints Drive
│   ├── heads.py       # Tête danger (V1), tête score (V2)
│   └── planner.py     # MPC : génération de séquences, rollout latent, coût
├── notebooks/         # 01 à 05, voir tableau des phases
├── tests/             # pytest : physique, env, modèle (overfit), planner
└── docs/superpowers/specs/   # ce document
```

## 6. Simulateur (Phase 0)

- **Moteur** : [pymunk](https://www.pymunk.org) (bindings Chipmunk2D). Corps rigides
  2D : gravité, restitution, friction, moments d'inertie. Installation en une ligne
  sur Colab, exécution CPU rapide.
- **Table paramétrable** : dataclass `BoardConfig` — dimensions du plateau, inclinaison
  (norme du vecteur gravité), murs latéraux et dôme supérieur, position/longueur/angle
  de repos/vitesse angulaire des flippers, largeur du drain (couloir de sortie entre
  les flippers), coefficients de restitution/friction, et en V2 : liste de bumpers
  (position, rayon, valeur en points, impulsion de rebond). Changer la table = changer
  la config, zéro code.
- **Flippers** : corps physiques attachés par pivot, actionnés par moteur angulaire
  avec butées (rotary limit joint). La balle est réellement frappée par transfert de
  moment — pas de téléportation ni de vélocité scriptée.
- **Fréquences** : physique interne à 120 Hz, décision de l'agent à 15 Hz (frame skip
  de 8). Parade anti-tunneling : petit pas de temps, murs épais, plafond de vitesse de
  la balle.
- **Épisode** : la balle est lancée depuis le haut (position/vitesse légèrement
  aléatoires). Fin d'épisode : balle dans le drain (perdue), stagnation détectée
  (balle quasi immobile hors flippers pendant N pas), ou durée max atteinte.

## 7. Environnement et observation

- API type Gymnasium : `reset() -> obs` ; `step(action) -> obs, info`.
- **Observation** : 2 frames consécutives empilées, 64×64 niveaux de gris, uint8 →
  tenseur (2, 64, 64). Justification pédagogique : une image seule ne contient pas la
  vitesse de la balle ; l'état ne serait pas Markovien et le futur serait imprévisible.
- **Actions** : 4 actions discrètes = produit de 2 boutons binaires
  (flipper gauche appuyé/relâché × flipper droit appuyé/relâché).
- **`info`** (jamais montré au modèle ; sert aux labels et au debug) : position et
  vitesse exactes de la balle, `ball_lost`, `stuck`, score (V2).
- **Rendu** : PIL ImageDraw (cercles, segments, polygones), headless, sans pygame.
  Un rendu « debug » plus grand et annoté existe pour les vidéos des notebooks.

## 8. Collecte de données (Phase 1)

- **Politique aléatoire à actions collantes** : chaque action tirée est maintenue
  pendant une durée aléatoire (plusieurs pas). Sans cela, les flippers vibrent sans
  jamais frapper et le dataset ne contient aucun exemple de frappe réussie — le modèle
  ne pourrait pas apprendre l'effet des actions.
- **Volume** : 50k à 100k transitions (o_t, a_t, o_{t+1}, info_t). Stockage en `.npz`
  shardés sur Google Drive, **par épisodes de frames** (chaque frame stockée une seule
  fois, les transitions sont indexées à la lecture — pas de duplication o_t/o_{t+1}).
  À 64×64 uint8, ~100k frames ≈ 400 Mo — acceptable sur Drive.
- Les épisodes de la politique aléatoire perdent la balle très souvent : c'est voulu,
  cela fournit les exemples positifs de la tête danger.

## 9. Modèle JEPA (Phase 2)

- **Encodeur** : CNN ~1-2 M paramètres (4-5 blocs conv, stride 2, GroupNorm, SiLU),
  sortie z ∈ R²⁵⁶ avec LayerNorm. (Un ViT serait fidèle à I-JEPA mais n'apporte rien à
  cette échelle — expliqué dans le notebook.)
- **Prédicteur** : MLP prenant (z_t, embedding de l'action a_t) → ẑ_{t+1}. Entraîné en
  **rollout multi-pas** (dérouler 4 à 8 pas en chaîne, perte à chaque pas) car c'est
  l'usage qu'en fera le planificateur.
- **Encodeur cible** : copie EMA de l'encodeur (momentum 0,99-0,999), cibles sans
  gradient (stop-gradient). C'est le mécanisme anti-collapse central de JEPA.
- **Perte** : L2 (ou distance cosinus) entre prédiction et cible dans l'espace latent.
  Garde-fou optionnel si collapse observé : régularisation de variance style VICReg.
- **Diagnostics obligatoires dans le notebook** :
  - variance des dimensions latentes au fil de l'entraînement (détecteur de collapse) ;
  - erreur de prédiction comparée à la baseline naïve « ẑ_{t+1} = z_t » ;
  - PCA 2D des trajectoires latentes, colorée par la position réelle de la balle.
- **Budget** : 20-40 min sur T4, batch ~256, AMP (mixed precision), checkpoint sur
  Drive à chaque epoch avec reprise automatique.

## 10. Contrôle (Phase 3 = V1, Phase 4 = V2)

- **Tête danger** : MLP sur z (encodeur gelé) → P(balle perdue dans les 10 prochains
  pas). Labels binaires calculés depuis `info` du dataset de la Phase 1. Entraînement
  supervisé rapide (< 5 min). Qualité vérifiée par AUC sur un split de validation et
  comparée à une heuristique « hauteur de la balle ».
- **Planificateur MPC** (aucun entraînement) :
  1. générer des séquences d'actions candidates sur un horizon H = 8 pas
     (~0,5 s à 15 Hz, égal à l'horizon d'entraînement du prédicteur) — random
     shooting sur l'espace des 4 actions, avec persistance d'action dans les
     séquences ; option CEM si nécessaire ;
  2. dérouler chaque séquence dans l'espace latent avec le prédicteur ;
  3. coût = Σ_t danger(ẑ_t) (V2 : + λ·(− score attendu)) ;
  4. exécuter la première action de la meilleure séquence, replanifier au pas suivant.
- **V2** : ajout des bumpers scoreurs dans `BoardConfig`, tête score (MLP sur (z, a) →
  points attendus), nouvelle collecte incluant les bumpers, coût combiné.

## 11. Risques et parades

| Risque | Parade |
|---|---|
| Collapse latent (z constant, perte nulle) | EMA + stop-gradient ; monitoring de variance ; régularisation VICReg en secours |
| La politique aléatoire ne frappe jamais la balle | Actions collantes (maintenues plusieurs pas) |
| Tunneling (balle traverse un mur) | Physique 120 Hz, murs épais, plafond de vitesse |
| Balle coincée | Détection de stagnation → fin d'épisode |
| Tête danger mal calibrée → MPC aveugle | AUC sur validation ; comparaison à l'heuristique hauteur de balle |
| Déconnexions Colab | Dataset et checkpoints sur Drive ; chaque notebook reprend où il en était |
| Prédicteur mauvais au-delà de quelques pas | Entraînement multi-pas aligné sur l'horizon de planification |

## 12. Tests

- **Physique (pytest, sans GPU)** : chute libre conforme à la gravité configurée ; un
  flipper actionné augmente la vitesse de la balle ; la perte de balle est détectée ;
  modifier `BoardConfig` change effectivement la géométrie.
- **Environnement** : shapes/dtypes des observations, déterminisme sous seed fixée.
- **Modèle** : overfit volontaire sur 100 transitions (la perte doit s'écraser) ;
  mini-run de quelques centaines de pas sans collapse (variance latente > seuil).
- **Planificateur** : sur un état où la balle fonce vers le drain côté gauche, la
  séquence choisie actionne le flipper gauche (test d'intégration avec modèle entraîné,
  exécuté en notebook plutôt qu'en CI).
- **Critère d'acceptation V1** : temps de survie moyen de l'agent > baselines aléatoire
  et « toujours appuyé » sur 50 épisodes.

## 13. Hors périmètre

- Apprentissage par renforcement (DQN, PPO...).
- Rendu réaliste / textures ; le rendu reste schématique.
- Multi-balles, tilt/nudge joueur, plateaux à étages.
- Transfert vers un vrai flipper ou un jeu vidéo existant.
