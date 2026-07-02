# %% [markdown]
# # 06 — Voir ce que l'IA prédit
#
# JEPA prédit le futur DANS SON ESPACE LATENT — il n'y a pas d'« image
# prédite » à regarder, et c'est voulu : prédire des pixels obligerait à
# modéliser des détails inutiles. Mais on peut TRADUIRE ses prédictions de
# deux façons, sans jamais toucher au world model :
#
# 1. **Sonde de position** : un mini-MLP lit (x, y) de la balle dans le latent.
#    On déroule le prédicteur 8 pas dans l'imagination et on superpose les
#    positions prédites à la trajectoire réelle. C'est le « prédit vs réel ».
# 2. **Décodeur d'imagination** : un petit déconv apprend latent → image
#    (encodeur gelé). On peut alors VOIR les états imaginés, et les superposer
#    au réel (rouge = imaginé, vert = réel, jaune = accord).

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
import torch
DATA_DIR, CKPT_DIR = ROOT / "data/targets_v1", ROOT / "checkpoints_targets"

# %%
from pinball.collect import load_episodes
from jepa.train import load_jepa
from jepa.heads import PositionProbe

episodes = load_episodes(DATA_DIR)
jepa = load_jepa(CKPT_DIR / "jepa.pt")
probe = PositionProbe(jepa.z_dim)
probe.load_state_dict(torch.load(CKPT_DIR / "pos.pt", weights_only=True))
probe.eval()
print(f"{len(episodes)} épisodes, modèle et sonde chargés")

# %% [markdown]
# ## 1. Trajectoires : prédit contre réel
#
# Cercles blancs = où la balle EST allée (8 pas). Croix colorées = où le
# modèle PENSAIT qu'elle irait, en ne partant que de l'image initiale et des
# actions jouées (jaune = 1 pas devant, violet = 8 pas). Plus les croix
# collent aux cercles, meilleur est le world model — regarde aussi comment
# l'erreur grandit avec l'horizon : c'est la difficulté de prédire loin.

# %%
import matplotlib.pyplot as plt
from jepa.viz import trajectory_overlay

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
shown = 0
for ep in episodes:
    if len(ep["actions"]) >= 20 and ep["hits"].sum() > 0:
        axes[shown].imshow(trajectory_overlay(jepa, probe, ep, t0=5, k=8))
        axes[shown].axis("off")
        axes[shown].set_title(f"épisode à contact ({shown + 1})")
        shown += 1
        if shown == 3:
            break
plt.suptitle("Blanc = réel, croix = imaginé (jaune → violet = horizon 1 → 8)")
plt.show()

# %% [markdown]
# ## 2. Entraîner le décodeur d'imagination (~10 min sur T4)
#
# Le décodeur N'EST PAS le modèle : c'est une loupe. Il apprend à redessiner
# l'image depuis le latent gelé — si le latent ne contient pas une
# information, le décodeur ne peut pas l'inventer. C'est d'ailleurs un test :
# si la balle décodée est floue, c'est que le latent code sa position avec
# incertitude.

# %%
from jepa.decoder import train_decoder
decoder = train_decoder(jepa, episodes, epochs=3)
torch.save(decoder.state_dict(), CKPT_DIR / "decoder.pt")

# %% [markdown]
# ### Contrôle : le plafond du décodeur
#
# Avant de décoder l'IMAGINATION, vérifions ce que le décodeur sait faire sur
# des latents RÉELS (encodés depuis de vraies images) : c'est sa performance
# maximale, l'imagination ne sera jamais plus nette. Si cette ligne est déjà
# noire ou floue, le problème est le décodeur — pas le prédicteur. (La perte
# est pondérée vers les pixels allumés : ~95 % du plateau est noir, une MSE
# nue apprendrait « tout noir ».)

# %%
import numpy as np
from jepa.data import MultiLabelDataset

ds_ctrl = MultiLabelDataset(episodes[:5])
idx = np.linspace(0, len(ds_ctrl) - 1, 8, dtype=int)
obs_ctrl = torch.stack([ds_ctrl[int(i)]["obs"] for i in idx])
dev = next(jepa.parameters()).device        # cuda sur Colab, cpu en local
with torch.no_grad():
    recon = decoder(jepa.encode_target(obs_ctrl.to(dev))).cpu().numpy()
fig, axes = plt.subplots(2, 8, figsize=(16, 4.2))
for j in range(8):
    axes[0, j].imshow(obs_ctrl[j, 1], cmap="gray", vmin=0, vmax=255)
    axes[1, j].imshow(recon[j], cmap="gray", vmin=0, vmax=1)
    axes[0, j].axis("off")
    axes[1, j].axis("off")
plt.suptitle("Plafond du décodeur : réel (haut) vs reconstruit (bas)")
plt.show()

# %% [markdown]
# ## 3. L'imagination en images
#
# Trois lignes : le RÉEL, l'IMAGINÉ (décodé des latents prédits ẑ), et la
# SUPERPOSITION — rouge = imaginé seul, vert = réel seul, JAUNE = accord.
# Une balle jaune = le modèle avait raison ; une paire rouge/verte disjointe =
# il s'est trompé (et de combien).

# %%
from jepa.viz import imagination_strip

ep = next(e for e in episodes if len(e["actions"]) >= 20)
strip = imagination_strip(jepa, decoder, ep, t0=5, k=8)
plt.figure(figsize=(16, 6))
plt.imshow(strip)
plt.axis("off")
plt.title("réel (haut) / imaginé (milieu) / superposition (bas) — horizon 1 → 8")
plt.show()

# %% [markdown]
# ## 4. Et quand l'agent joue ?
#
# Même exercice sur un épisode COLLECTÉ PAR L'AGENT (data/targets_v2, si le
# notebook 05 est passé) : l'imagination est-elle aussi bonne sur du bon jeu
# que sur du jeu aléatoire ? (C'est tout l'enjeu de l'itération.)

# %%
DATA_V2 = ROOT / "data/targets_v2"
if DATA_V2.exists() and list(DATA_V2.glob("shard_*.npz")):
    eps2 = load_episodes(DATA_V2)
    ep2 = next(e for e in eps2 if len(e["actions"]) >= 20)
    plt.figure(figsize=(16, 6))
    plt.imshow(imagination_strip(jepa, decoder, ep2, t0=5, k=8))
    plt.axis("off"); plt.title("imagination sur du jeu d'AGENT")
    plt.show()
else:
    print("data/targets_v2 absent — lancer le notebook 05 d'abord (optionnel).")

# %% [markdown]
# ## À retenir
#
# - JEPA ne prédit PAS des images : sonde et décodeur ne sont que des
#   traductions a posteriori, entraînées encodeur gelé.
# - La qualité de la superposition à 8 pas EST la qualité du world model —
#   c'est la version visuelle du diagnostic n°3 du notebook 03.
# - Le flou du décodeur est honnête : il montre l'incertitude du latent.
