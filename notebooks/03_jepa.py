# %% [markdown]
# # 03 — Le world model JEPA
#
# On apprend ici la pièce centrale : un modèle qui, donné l'état (en latent)
# et une action, prédit l'état SUIVANT (en latent). Trois idées à retenir :
#
# 1. **Prédire en latent, pas en pixels.** Redessiner l'image obligerait le
#    réseau à modéliser des détails inutiles. On prédit la représentation.
# 2. **Encodeur cible EMA + stop-gradient = anti-effondrement.** Sans cela,
#    `z = constante` annule la perte (collapse) : représentation morte.
# 3. **Rollout multi-pas.** Le prédicteur est entraîné à enchaîner 8 pas sur
#    ses PROPRES prédictions — car c'est ce que la planification lui demandera.

# %%
from pathlib import Path
import torch

DATA_DIR, CKPT_DIR = Path("data/hard_v1"), Path("checkpoints_hard")
print("device :", "cuda" if torch.cuda.is_available() else "cpu")

# %%
from pinball.collect import load_episodes
episodes = load_episodes(DATA_DIR)
print(f"{len(episodes)} épisodes, "
      f"{sum(len(e['actions']) for e in episodes)} transitions")

# %% [markdown]
# ## Entraînement
#
# C'est l'étape la plus longue du projet : ~1 à 2 h sur un CPU 6 cœurs
# (quelques dizaines de minutes avec un GPU). On peut interrompre sans
# risque : le checkpoint est écrit à CHAQUE epoch dans `checkpoints_hard/`
# et relancer cette cellule reprend automatiquement où on en était.

# %%
from jepa.train import train_jepa
model, history = train_jepa(episodes, CKPT_DIR, epochs=10)

# %% [markdown]
# ## Diagnostic n°1 : les courbes
#
# - `loss` doit décroître ;
# - `pred_mse` doit devenir NETTEMENT inférieur à `copy_mse` (la baseline
#   « le futur = le présent »). Sinon le modèle n'a rien appris de la dynamique ;
# - `latent_std` doit rester loin de 0 — c'est le détecteur de collapse.

# %%
import matplotlib.pyplot as plt
epochs_ = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].plot(epochs_, [h["loss"] for h in history]); axes[0].set_title("loss")
axes[1].plot(epochs_, [h["pred_mse"] for h in history], label="prédicteur")
axes[1].plot(epochs_, [h["copy_mse"] for h in history], "--", label="baseline copie")
axes[1].legend(); axes[1].set_title("le prédicteur bat-il la copie ?")
axes[2].plot(epochs_, [h["latent_std"] for h in history]); axes[2].axhline(0.05, color="r", ls=":")
axes[2].set_title("variance latente (collapse si → 0)")
plt.show()

# %% [markdown]
# ## Diagnostic n°2 : le latent « voit »-il la balle ?
#
# Le latent n'est pas fait pour être lu par un humain — mais on peut le sonder.
# PCA 2D des latents d'épisodes entiers, colorée par la position RÉELLE de la
# balle (qu'on connaît via `info`, jamais montrée au modèle) : si des dégradés
# apparaissent, le latent encode la position.

# %%
import numpy as np
from jepa.data import WindowDataset, stack_obs

model_cpu = model.to("cpu").eval()
lat, xs, ys = [], [], []
for ep in episodes[:30]:
    frames = torch.from_numpy(ep["frames"])
    for t in range(1, len(ep["actions"]) + 1, 3):
        obs = torch.stack([frames[t - 1], frames[t]]).unsqueeze(0)
        lat.append(model_cpu.encode_target(obs).squeeze(0).numpy())
        xs.append(ep["ball_pos"][t, 0]); ys.append(ep["ball_pos"][t, 1])
lat = np.stack(lat)
lat_c = lat - lat.mean(0)
_, _, Vt = np.linalg.svd(lat_c, full_matrices=False)
p2 = lat_c @ Vt[:2].T
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, c, name in ((axes[0], xs, "x balle"), (axes[1], ys, "y balle")):
    s = ax.scatter(p2[:, 0], p2[:, 1], c=c, s=4, cmap="viridis")
    plt.colorbar(s, ax=ax); ax.set_title(f"PCA des latents, couleur = {name}")
plt.show()

# %% [markdown]
# ## Diagnostic n°3 : la prédiction est-elle bonne à 8 pas ?
#
# On prend des fenêtres de validation, on déroule le prédicteur 8 pas, et on
# compare l'erreur à la baseline copie, PAS À PAS. L'erreur croît avec
# l'horizon (normal), mais doit rester sous la baseline.
#
# **Pourquoi deux encodeurs dans ce code ?** Le prédicteur consomme le latent
# de l'encodeur **online** (`encode`) — exactement comme à l'entraînement —
# tandis que toutes les cibles vivent dans l'espace de l'encodeur **cible
# EMA** (`encode_target`) : le prédicteur est donc une fonction
# « online → espace cible ». C'est aussi pour cela que la tête danger
# (notebook 04) sera entraînée sur des latents `encode_target` : en
# planification, elle ne verra que des ẑ prédits, qui approximent des z̄.

# %%
ds = WindowDataset(episodes[-50:], k=8)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
batch = next(iter(dl))
frames, actions = batch["frames"], batch["actions"]
with torch.no_grad():
    z = model_cpu.encode(stack_obs(frames, 0))
    z0_t = model_cpu.encode_target(stack_obs(frames, 0))
    pred_err, copy_err = [], []
    for i in range(8):
        z = model_cpu.predictor(z, actions[:, i])
        target = model_cpu.encode_target(stack_obs(frames, i + 1))
        pred_err.append(((z - target) ** 2).mean().item())
        copy_err.append(((z0_t - target) ** 2).mean().item())
plt.plot(range(1, 9), pred_err, "o-", label="prédicteur")
plt.plot(range(1, 9), copy_err, "s--", label="baseline copie")
plt.xlabel("horizon (pas)"); plt.ylabel("MSE latente"); plt.legend()
plt.title("Erreur de prédiction selon l'horizon"); plt.show()

# %% [markdown]
# Si les trois diagnostics sont bons, le modèle du monde est prêt.
# Prochaine étape (notebook 04) : s'en servir pour JOUER — tête danger,
# puis planification dans l'imagination.
