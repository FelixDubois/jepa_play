# JEPA Pinball

Projet pédagogique : un agent **JEPA** (Joint-Embedding Predictive Architecture)
apprend à jouer au flipper à partir des seules images 64×64 du plateau. Le monde
est un vrai simulateur physique 2D (pymunk), la table est paramétrable, et le
contrôle émerge de la **planification dans l'espace latent** (MPC) — aucun
apprentissage par renforcement.

```
image 64×64×2 ──▶ encodeur CNN ──▶ z ──▶ prédicteur(z, action) ──▶ ẑ futur
                                              │
                          planification : imaginer ~260 futurs, éviter le danger
```

## Mise en place (locale)

```bash
python -m venv .venv            # ou : uv venv --seed .venv
.venv/bin/pip install -e ".[dev,notebooks]"
.venv/bin/python -m pytest      # 57 tests, ~10 s
.venv/bin/jupyter lab           # depuis la RACINE du repo (chemins relatifs)
```

Tout tourne en CPU ; un GPU accélère seulement l'entraînement (notebook 03).

## Les notebooks, dans l'ordre

| Notebook | Contenu | Durée (CPU 6 cœurs) |
|---|---|---|
| `01_simulateur` | la table, la physique, jouer à la main | ~10 min de lecture |
| `02_collecte` | 100k transitions de jeu aléatoire → `data/hard_v1/` | < 1 min |
| `03_jepa` | entraînement du world model + diagnostics anti-collapse | ~1-2 h |
| `04_controle` | tête danger + agent MPC + évaluation vs baselines | ~20-40 min |
| `05_iteration` | l'agent collecte ses propres données, itération du modèle | ~2-3 h |

Données et checkpoints restent locaux (`data/`, `checkpoints_hard*/` — non
versionnés). Les entraînements écrivent un checkpoint à chaque epoch : on peut
interrompre et relancer, la reprise est automatique.

## La table d'expérience

Les expériences utilisent la « table dure » (`pinball.config.hard_board()`) :
drain central largement ouvert et flippers courts. Les stratégies aveugles y
meurent en ~2 s — il faut *voir* la balle pour survivre. C'est ce qui rend
l'évaluation honnête (voir le panneau des *nudges* du notebook 05).

## Structure

```
pinball/   simulateur pymunk, rendu, environnement Gym-like, collecte
jepa/      modèle JEPA, entraînement, tête danger, planificateur MPC, éval
notebooks/ 01 → 05 (sources jupytext .py + .ipynb)
tests/     57 tests pytest (physique, données, modèle, planificateur)
docs/      spécification et plans d'implémentation
```
