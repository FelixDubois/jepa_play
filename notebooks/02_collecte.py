# %% [markdown]
# # 02 — Collecte d'expérience
#
# Personne ne joue : une politique aléatoire « collante » actionne les
# flippers au hasard, des milliers de parties, à vitesse machine. Le modèle
# n'a pas besoin de BON jeu — il a besoin de VARIÉTÉ : rebonds, frappes,
# et beaucoup de balles perdues (elles serviront de labels de danger).

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
    DATA_DIR = Path("/content/drive/MyDrive/jepa_pinball/data/v1")
else:
    DATA_DIR = Path("data/v1")
print("Dataset →", DATA_DIR)

# %%
import numpy as np
from pinball.collect import StickyRandomPolicy, collect_dataset, load_episodes
from pinball.env import PinballEnv

N_TRANSITIONS = 100_000

if DATA_DIR.exists() and list(DATA_DIR.glob("shard_*.npz")):
    print("Dataset déjà présent — collecte sautée (supprimer le dossier pour refaire).")
else:
    env = PinballEnv(seed=42)
    policy = StickyRandomPolicy(np.random.default_rng(42))
    stats = collect_dataset(env, policy, N_TRANSITIONS, DATA_DIR)
    print(stats)

# %% [markdown]
# ## Contrôle qualité du dataset
#
# Avant d'entraîner quoi que ce soit : vérifier que le dataset contient bien
# ce dont on aura besoin. Trois questions :
# 1. Les épisodes ont-ils des durées variées ?
# 2. Perd-on assez de balles (labels de danger) ?
# 3. Les 4 actions sont-elles toutes représentées ?

# %%
import matplotlib.pyplot as plt
episodes = load_episodes(DATA_DIR)
lengths = np.array([len(ep["actions"]) for ep in episodes])
losses = np.array([ep["ball_lost"] for ep in episodes])
print(f"{len(episodes)} épisodes, {lengths.sum()} transitions")
print(f"durée : moy {lengths.mean():.0f} pas ({lengths.mean()/15:.1f} s), "
      f"médiane {np.median(lengths):.0f}")
print(f"balles perdues : {losses.mean()*100:.0f} % des épisodes")
all_actions = np.concatenate([ep["actions"] for ep in episodes])
print("répartition des actions :", np.bincount(all_actions, minlength=4) / len(all_actions))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(lengths, bins=40); axes[0].set_title("Durées d'épisodes (pas)")
axes[1].hist(np.concatenate([ep["ball_pos"][:, 1] for ep in episodes]), bins=40)
axes[1].set_title("Hauteurs de balle visitées"); plt.show()

# %% [markdown]
# ## À quoi ressemble une transition ?

# %%
ep = episodes[0]
t = min(20, len(ep["actions"]) - 1)
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(ep["frames"][t - 1], cmap="gray"); axes[0].set_title("frame t-1")
axes[1].imshow(ep["frames"][t], cmap="gray"); axes[1].set_title(f"frame t (action={ep['actions'][t]})")
axes[2].imshow(ep["frames"][t + 1], cmap="gray"); axes[2].set_title("frame t+1")
for ax in axes: ax.axis("off")
plt.suptitle("Une transition : c'est TOUT ce que le JEPA verra"); plt.show()

# %% [markdown]
# Dataset prêt. Prochaine étape (notebook 03) : apprendre à PRÉDIRE —
# le cœur de JEPA.
