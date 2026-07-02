# %% [markdown]
# # 05 — Itérer le world model : l'agent collecte ses propres données
#
# Le diagnostic de la V1 était clair : l'agent bat largement l'aléatoire mais
# son modèle du monde n'a vu QUE du jeu aléatoire — dès qu'il joue bien, il
# visite des états que son JEPA connaît mal. Il est hors distribution
# précisément quand il réussit.
#
# La V2 a changé l'objectif : des CIBLES aléatoires à toucher (victoire
# quand toutes le sont) et un bonus de hauteur — le reward hacking du
# piégeage (itération 1 : ~3,4 nudges/épisode pour l'agent comme pour
# « toujours appuyé ») n'a plus de prise. L'itération reste utile pour la
# même raison qu'avant : le modèle n'a vu que du jeu aléatoire — et en V2
# elle enrichit surtout les exemples de CONTACTS de cibles (le hasard n'en
# touche que ~10 %). Prérequis : notebooks 02 → 04 relancés en V2.
#
# La réponse est la boucle classique des world models :
#
# 1. **l'agent V1 joue** et on enregistre tout (avec des rafales d'actions
#    aléatoires pour garder de la variété — `MixedPolicy`) ;
# 2. **le JEPA se réentraîne** sur le dataset mixte (aléatoire + agent), en
#    repartant de son checkpoint (warm start) ;
# 3. **les têtes d'objectif se réentraînent**, puis on ré-évalue : agent V2
#    contre agent V1 contre les baselines.
#
# Bonus d'honnêteté : on compte les **nudges** consommés par épisode — la
# baseline « toujours appuyé » survit surtout grâce à eux.

# %%
import importlib.util, subprocess, sys, os
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB and not os.path.exists("jepa_play"):
    subprocess.run(["git", "clone", "https://github.com/FelixDubois/jepa_play.git"], check=True)
    os.chdir("jepa_play")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# %%
from pathlib import Path
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/jepa_pinball")
else:
    ROOT = Path(".")

import shutil
import numpy as np
import torch
DATA_V1, DATA_V2 = ROOT / "data/targets_v1", ROOT / "data/targets_v2"
CKPT_V1, CKPT_V2 = ROOT / "checkpoints_targets", ROOT / "checkpoints_targets_v2"
print("device :", "cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Recharger l'agent V1

# %%
from pinball.collect import load_episodes
from jepa.train import load_jepa
from jepa.heads import DangerHead, HeightHead, PositionProbe, TargetHead
from jepa.planner import MPCPlanner

episodes_v1 = load_episodes(DATA_V1)
jepa_v1 = load_jepa(CKPT_V1 / "jepa.pt")
heads_v1 = {"danger": DangerHead(jepa_v1.z_dim), "height": HeightHead(jepa_v1.z_dim),
            "target": TargetHead(jepa_v1.z_dim), "pos": PositionProbe(jepa_v1.z_dim)}
for name, h in heads_v1.items():
    h.load_state_dict(torch.load(CKPT_V1 / f"{name}.pt", weights_only=True))
    h.eval()
agent_v1 = MPCPlanner(jepa_v1, heads_v1["danger"], n_candidates=256,
                      height_head=heads_v1["height"],
                      target_head=heads_v1["target"])
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
    print("Dataset v2 déjà présent — collecte sautée (supprimer data/targets_v2 "
          "ET checkpoints_targets_v2 pour refaire l'itération).")
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
# 6 epochs de PLUS que l'epoch du checkpoint — quel qu'il soit — sur les
# données mixtes.
#
# ⚠ Pour REFAIRE l'itération de zéro (p. ex. après avoir changé la part
# d'exploration) : supprimer `data/targets_v2` ET `checkpoints_targets_v2` —
# sinon la reprise repart de l'epoch 16 et n'entraîne RIEN de plus.

# %%
from jepa.train import train_jepa

CKPT_V2.mkdir(parents=True, exist_ok=True)
if not (CKPT_V2 / "jepa.pt").exists():
    shutil.copy(CKPT_V1 / "jepa.pt", CKPT_V2 / "jepa.pt")

episodes_mixed = episodes_v1 + episodes_v2
# reprise ROBUSTE : +6 epochs quel que soit l'entraînement du checkpoint
# (un epochs fixe inférieur à l'epoch du checkpoint n'entraînerait RIEN,
# en silence — p. ex. un notebook 03 poussé à 100 epochs)
ckpt_epoch = int(torch.load(CKPT_V2 / "jepa.pt", map_location="cpu",
                            weights_only=True)["epoch"])
jepa_v2, history = train_jepa(episodes_mixed, CKPT_V2, epochs=ckpt_epoch + 6)

# %%
import matplotlib.pyplot as plt
epochs_ = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(epochs_, [h["pred_mse"] for h in history], label="prédicteur")
axes[0].plot(epochs_, [h["copy_mse"] for h in history], "--", label="baseline copie")
axes[0].axvline(ckpt_epoch + 0.5, color="gray", ls=":", label="← v1 | v2 →")
axes[0].legend(); axes[0].set_title("reprise sur données mixtes")
axes[1].plot(epochs_, [h["latent_std"] for h in history])
axes[1].axhline(0.05, color="r", ls=":"); axes[1].set_title("variance latente")
plt.show()

# %% [markdown]
# ## Étape 3 : têtes d'objectif V2, puis ré-évaluation

# %%
from jepa.heads import train_objective_heads

heads_v2, metrics_v2 = train_objective_heads(jepa_v2, episodes_mixed)
for k, v in metrics_v2.items():
    print(f"{k}: {v:.3f}")
for name, h in heads_v2.items():
    torch.save(h.state_dict(), CKPT_V2 / f"{name}.pt")
agent_v2 = MPCPlanner(jepa_v2, heads_v2["danger"], n_candidates=256,
                      height_head=heads_v2["height"],
                      target_head=heads_v2["target"])

# %%
from pinball.collect import StickyRandomPolicy
from jepa.eval import AlwaysPressed, PeriodicFlapper, evaluate

env = PinballEnv(hard_board())
# instance fraîche pour l'éval : le RNG interne d'agent_v1 a été consommé par
# la collecte — une instance neuve garantit la reproductibilité entre exécutions
agent_v1_eval = MPCPlanner(jepa_v1, heads_v1["danger"], n_candidates=256,
                           height_head=heads_v1["height"],
                           target_head=heads_v1["target"])
results = {}
for name, pol in [("agent V2", agent_v2),
                  ("agent V1", agent_v1_eval),
                  ("aléatoire", StickyRandomPolicy(np.random.default_rng(0))),
                  ("toujours appuyé", AlwaysPressed()),
                  ("flapper aveugle", PeriodicFlapper())]:
    results[name] = evaluate(env, pol, n_episodes=50)
    r = results[name]
    print(f"{name:16s}: victoire {100*r['completion_rate']:3.0f} %  "
          f"cibles {100*r['targets_hit_rate']:3.0f} %  "
          f"hauteur {r['mean_height']:.2f}  survie {r['survival_s']:.1f} s  "
          f"nudges {r['mean_nudges']:.1f}")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
names = list(results)
colors = ["tab:green", "tab:olive", "tab:gray", "tab:gray", "tab:orange"]
bars = ax.bar(names, [100 * results[n]["completion_rate"] for n in names], color=colors)
for bar, n in zip(bars, names):
    ax.annotate(f"{results[n]['mean_nudges']:.1f} nudges",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.set_ylabel("taux de victoire (%)")
ax.set_title("Itération 1 — V2 cibles (50 épisodes, seeds appariées)")
plt.show()

# %% [markdown]
# ## Lecture des résultats
#
# - **agent V2 vs agent V1** : le gain vient uniquement de meilleures données —
#   même architecture, même planificateur. C'est LA leçon de la boucle
#   world-model : la politique améliore les données, les données améliorent la
#   politique.
# - **les nudges** : ce panneau a démasqué le piégeage de l'itération 1 —
#   l'agent ET « toujours appuyé » consommaient ~3,4 nudges/épisode chacun
#   pour survivre en bloquant la balle, sans jamais chercher à gagner. En V2,
#   la victoire ne se pirate plus ainsi : un nudge élevé reste un signal
#   d'alerte, mais ne suffit plus à faire gagner.
# - si l'agent V2 domine nettement toutes les baselines en taux de victoire,
#   le critère V2 est rempli — `git tag v2`.
# - La boucle peut se répéter (v3 : collecter avec agent_v2...) — les gains
#   diminuent à chaque tour, c'est normal et attendu.

# %% [markdown]
# ## Et après ?
#
# La suite : notebook 06, pour VOIR ce que le world model imagine —
# trajectoires prédites superposées au réel, puis décodeur d'imagination.
