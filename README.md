# JEPA Pinball

Projet pédagogique : un agent **JEPA** (Joint-Embedding Predictive Architecture)
apprend à jouer au flipper à partir des seules images 64×64 du plateau. Le monde
est un vrai simulateur physique 2D (pymunk), la table est paramétrable, et le
contrôle émerge de la **planification dans l'espace latent** (MPC) — aucun
apprentissage par renforcement.

**V2 « Cibles »** : chaque partie place au hasard 1 à 3 cibles à toucher — la
partie est **gagnée** quand toutes le sont. La métrique reine est le taux de
victoire ; le coût de planification combine trois objectifs appris
(`danger − 0,5·hauteur − 2·cible`), et le danger est « honnête » : seules les
fins perdues ou immobilisées comptent comme dangereuses, jamais les victoires
— leçon du reward hacking observé pendant le développement.

```
image 64×64×2 ──▶ encodeur CNN ──▶ z ──▶ prédicteur(z, action) ──▶ ẑ futur
                                              │
     planification : imaginer ~260 futurs — éviter le danger, monter, viser
```

## Sur Colab (recommandé — GPU T4 gratuit)

Ouvre un notebook via son badge dans le tableau ci-dessous : la première
cellule clone le repo et installe le package ; données et checkpoints vivent
sur ton Google Drive (`MyDrive/jepa_pinball/`) — une déconnexion ne fait rien
perdre, tout reprend au dernier checkpoint.

## En local

```bash
python -m venv .venv            # ou : uv venv --seed .venv
.venv/bin/pip install -e ".[dev,notebooks]"
.venv/bin/python -m pytest      # 86 tests, ~7 s
.venv/bin/jupyter lab           # depuis la RACINE du repo (chemins relatifs)
```

Tout tourne en CPU ; un GPU n'accélère que les entraînements (03, 05, 06).

## Les notebooks, dans l'ordre

| Notebook | Contenu | Durée (T4) | Colab |
|---|---|---|---|
| `01_simulateur` | la table, la physique, les cibles, jouer à la main | ~10 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelixDubois/jepa_play/blob/main/notebooks/01_simulateur.ipynb) |
| `02_collecte` | 100k transitions de jeu aléatoire → `data/targets_v1` | ~15-30 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelixDubois/jepa_play/blob/main/notebooks/02_collecte.ipynb) |
| `03_jepa` | entraînement du world model + diagnostics anti-collapse | ~20-40 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelixDubois/jepa_play/blob/main/notebooks/03_jepa.ipynb) |
| `04_controle` | têtes danger/hauteur/cible/position, agent MPC multi-objectifs, éval en taux de victoire | ~15 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelixDubois/jepa_play/blob/main/notebooks/04_controle.ipynb) |
| `05_iteration` | l'agent collecte ses propres données, itération du modèle | ~1 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelixDubois/jepa_play/blob/main/notebooks/05_iteration.ipynb) |
| `06_visualisation` | superpositions prédit/réel + décodeur d'imagination | ~15 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelixDubois/jepa_play/blob/main/notebooks/06_visualisation.ipynb) |

En local, données et checkpoints restent à la racine (`data/`,
`checkpoints_targets*/` — non versionnés). Les entraînements écrivent un
checkpoint à chaque epoch : on peut interrompre et relancer, la reprise est
automatique.

## La table d'expérience

Les expériences utilisent la « table dure » (`pinball.config.hard_board()`) :
drain central largement ouvert et flippers courts — les stratégies aveugles y
meurent en ~2 s, il faut *voir* la balle pour survivre — **plus 1 à 3 cibles
aléatoires : les toucher toutes gagne la partie, et comme elles changent de
place à chaque épisode, impossible de jouer sans regarder l'image.** Le
panneau des *nudges* (notebook 05) garde l'évaluation honnête.

## Structure

```
pinball/   simulateur pymunk (cibles incluses), rendu, env Gym-like, collecte
jepa/      modèle JEPA, entraînement, têtes d'objectif, planificateur MPC
           multi-objectifs, décodeur d'imagination, superpositions, éval
notebooks/ 01 → 06 (sources jupytext .py + .ipynb)
tests/     86 tests pytest (physique, données, modèle, planificateur, viz)
docs/      spécifications et plans d'implémentation
```
