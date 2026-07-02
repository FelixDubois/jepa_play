# %% [markdown]
# # 05 — Itérer le world model : l'agent collecte ses propres données
#
# Le diagnostic de la V1 était clair : l'agent bat largement l'aléatoire mais
# son modèle du monde n'a vu QUE du jeu aléatoire — dès qu'il joue bien, il
# visite des états que son JEPA connaît mal. Il est hors distribution
# précisément quand il réussit.
#
# Depuis, le projet est passé sur la **table dure** (`hard_board()` : drain
# de 120 ouvert — la balle passive tombe au travers — et flippers de 90).
# Les stratégies aveugles s'y effondrent en ~2 s : il faut voir la balle.
# Prérequis : avoir relancé les notebooks 02 → 04 sur cette table.
#
# La réponse est la boucle classique des world models :
#
# 1. **l'agent V1 joue** et on enregistre tout (avec des rafales d'actions
#    aléatoires pour garder de la variété — `MixedPolicy`) ;
# 2. **le JEPA se réentraîne** sur le dataset mixte (aléatoire + agent), en
#    repartant de son checkpoint (warm start) ;
# 3. **la tête danger se réentraîne**, puis on ré-évalue : agent V2 contre
#    agent V1 contre les baselines.
#
# Bonus d'honnêteté : on compte les **nudges** consommés par épisode — la
# baseline « toujours appuyé » survit surtout grâce à eux.

# %%
from pathlib import Path
import shutil
import numpy as np
import torch

DATA_V1, DATA_V2 = Path("data/hard_v1"), Path("data/hard_v2")
CKPT_V1, CKPT_V2 = Path("checkpoints_hard"), Path("checkpoints_hard_v2")
print("device :", "cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Recharger l'agent V1

# %%
from pinball.collect import load_episodes
from jepa.train import load_jepa
from jepa.heads import DangerHead, train_danger_head
from jepa.planner import MPCPlanner

episodes_v1 = load_episodes(DATA_V1)
jepa_v1 = load_jepa(CKPT_V1 / "jepa.pt")
head_v1 = DangerHead()
head_v1.load_state_dict(torch.load(CKPT_V1 / "danger.pt", weights_only=True))
head_v1.eval()
agent_v1 = MPCPlanner(jepa_v1, head_v1, n_candidates=256)
print(f"{len(episodes_v1)} épisodes v1, agent V1 prêt (256 candidats)")

# %% [markdown]
# ## Étape 1 : l'agent collecte (~20-40 min sur CPU)
#
# `MixedPolicy` fait jouer l'agent avec ~20 % de rafales aléatoires collantes.
# Le simulateur n'attend personne : chaque pas coûte surtout la planification
# MPC (l'imagination de l'agent), d'où la durée.

# %%
from pinball.collect import MixedPolicy, collect_dataset
from pinball.config import hard_board
from pinball.env import PinballEnv

N_TRANSITIONS_V2 = 50_000

if DATA_V2.exists() and list(DATA_V2.glob("shard_*.npz")):
    print("Dataset v2 déjà présent — collecte sautée (supprimer data/hard_v2 "
          "ET checkpoints_hard_v2 pour refaire l'itération).")
else:
    env = PinballEnv(hard_board(), seed=123)
    explorer = MixedPolicy(agent_v1, np.random.default_rng(123))
    stats = collect_dataset(env, explorer, N_TRANSITIONS_V2, DATA_V2)
    print(stats)

# %%
episodes_v2 = load_episodes(DATA_V2)
lengths_v1 = np.array([len(ep["actions"]) for ep in episodes_v1])
lengths_v2 = np.array([len(ep["actions"]) for ep in episodes_v2])
print(f"durée moyenne des épisodes : aléatoire {lengths_v1.mean()/15:.1f} s "
      f"→ agent {lengths_v2.mean()/15:.1f} s")

# %% [markdown]
# ## Étape 2 : réentraîner le JEPA sur le dataset mixte (warm start)
#
# Astuce : `train_jepa` reprend automatiquement du checkpoint présent dans son
# dossier. On copie donc le checkpoint V1 dans un nouveau dossier et on demande
# 6 epochs de PLUS — il repart de l'epoch 10, sur les données mixtes.
#
# ⚠ Pour REFAIRE l'itération de zéro (p. ex. après avoir changé la part
# d'exploration) : supprimer `data/hard_v2` ET `checkpoints_hard_v2` — sinon
# la reprise repart de l'epoch 16 et n'entraîne RIEN de plus.

# %%
from jepa.train import train_jepa

CKPT_V2.mkdir(parents=True, exist_ok=True)
if not (CKPT_V2 / "jepa.pt").exists():
    shutil.copy(CKPT_V1 / "jepa.pt", CKPT_V2 / "jepa.pt")

episodes_mixed = episodes_v1 + episodes_v2
jepa_v2, history = train_jepa(episodes_mixed, CKPT_V2, epochs=16)

# %%
import matplotlib.pyplot as plt
epochs_ = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(epochs_, [h["pred_mse"] for h in history], label="prédicteur")
axes[0].plot(epochs_, [h["copy_mse"] for h in history], "--", label="baseline copie")
axes[0].axvline(10.5, color="gray", ls=":", label="← v1 | v2 →")
axes[0].legend(); axes[0].set_title("reprise sur données mixtes")
axes[1].plot(epochs_, [h["latent_std"] for h in history])
axes[1].axhline(0.05, color="r", ls=":"); axes[1].set_title("variance latente")
plt.show()

# %% [markdown]
# ## Étape 3 : tête danger V2, puis ré-évaluation

# %%
head_v2, val_auc = train_danger_head(jepa_v2, episodes_mixed)
torch.save(head_v2.state_dict(), CKPT_V2 / "danger.pt")
agent_v2 = MPCPlanner(jepa_v2, head_v2, n_candidates=256)

# %%
from pinball.collect import StickyRandomPolicy
from jepa.eval import AlwaysPressed, PeriodicFlapper, evaluate

env = PinballEnv(hard_board())
# instance fraîche pour l'éval : le RNG interne d'agent_v1 a été consommé par
# la collecte — une instance neuve garantit la reproductibilité entre exécutions
agent_v1_eval = MPCPlanner(jepa_v1, head_v1, n_candidates=256)
results = {}
for name, pol in [("agent V2", agent_v2),
                  ("agent V1", agent_v1_eval),
                  ("aléatoire", StickyRandomPolicy(np.random.default_rng(0))),
                  ("toujours appuyé", AlwaysPressed()),
                  ("flapper aveugle", PeriodicFlapper())]:
    results[name] = evaluate(env, pol, n_episodes=50)
    r = results[name]
    print(f"{name:16s}: {r['survival_s']:5.1f} s (méd. {r['median_steps']/15:.1f} s), "
          f"{r['mean_nudges']:.1f} nudges/épisode")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
names = list(results)
colors = ["tab:green", "tab:olive", "tab:gray", "tab:gray", "tab:orange"]
bars = ax.bar(names, [results[n]["survival_s"] for n in names], color=colors)
for bar, n in zip(bars, names):
    ax.annotate(f"{results[n]['mean_nudges']:.1f} nudges",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.set_ylabel("survie moyenne (s)")
ax.set_title("Itération 1 du world model (50 épisodes, seeds appariées)")
plt.show()

# %% [markdown]
# ## Lecture des résultats
#
# - **agent V2 vs agent V1** : le gain vient uniquement de meilleures données —
#   même architecture, même planificateur. C'est LA leçon de la boucle
#   world-model : la politique améliore les données, les données améliorent la
#   politique.
# - **les nudges** : regarde qui survit par lui-même et qui survit sous
#   perfusion de l'arbitre anti-stagnation. Une survie longue à ~6 nudges
#   n'a pas la même valeur qu'une survie active à ~0.
# - Sur la table dure, les baselines aveugles meurent en ~2-12 s : si l'agent
#   V2 dépasse nettement « toujours appuyé », **critère d'acceptation rempli**
#   — `git tag v1`.
# - La boucle peut se répéter (v3 : collecter avec agent_v2...) — les gains
#   diminuent à chaque tour, c'est normal et attendu.

# %% [markdown]
# ## Et après ?
#
# La V2 du projet (plan séparé) ajoutera les bumpers scoreurs et une tête
# score : le coût du planificateur deviendra « danger − λ·score » et l'agent
# devra VISER, pas seulement survivre.
