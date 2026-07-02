# %% [markdown]
# # 04 — L'agent joue : tête danger + planification
#
# Le world model sait prédire — il faut maintenant lui dire quoi ÉVITER,
# puis le laisser choisir ses actions en imaginant le futur.
#
# 1. **Tête danger** : P(balle perdue bientôt | latent). Supervisé, 5 min.
# 2. **Planificateur MPC** : à chaque pas, imaginer ~68 futurs possibles
#    sur 8 pas, exécuter la première action du futur le moins dangereux.
#    AUCUN apprentissage — le « bien jouer » émerge du modèle du monde.

# %%
import importlib.util, subprocess, sys, os
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB and not os.path.exists("jepa_play"):
    subprocess.run(["git", "clone", "https://github.com/VOTRE_COMPTE/jepa_play.git"], check=True)
    os.chdir("jepa_play")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# %%
from pathlib import Path
import numpy as np
import torch
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/jepa_pinball")
else:
    ROOT = Path(".")
DATA_DIR, CKPT_DIR = ROOT / "data/v1", ROOT / "checkpoints"

# %%
from pinball.collect import load_episodes
from jepa.train import load_jepa
episodes = load_episodes(DATA_DIR)
jepa = load_jepa(CKPT_DIR / "jepa.pt")
print(f"{len(episodes)} épisodes, modèle chargé")

# %% [markdown]
# ## Entraîner la tête danger
#
# Les labels sont gratuits : la politique aléatoire a perdu la balle des
# centaines de fois, on marque les 10 derniers pas de chaque perte.
# On vérifie la qualité par l'AUC sur des épisodes de validation, et on la
# compare à une heuristique évidente : « la balle est basse = danger »
# (calculée avec info privilégiée que le modèle, lui, n'a jamais vue).

# %%
from jepa.heads import auc, train_danger_head
from jepa.data import DangerDataset
head, val_auc = train_danger_head(jepa, episodes)
torch.save(head.state_dict(), CKPT_DIR / "danger.pt")

# heuristique de référence : hauteur de la balle (via info privilégiée)
n_val = max(1, int(len(episodes) * 0.1))
val_eps = episodes[-n_val:]
ds = DangerDataset(val_eps)
labels = np.array([ds[i]["label"].item() for i in range(len(ds))])
heights = np.concatenate([ep["ball_pos"][1:, 1] for ep in val_eps])
print(f"AUC tête danger (image seule) : {val_auc:.3f}")
print(f"AUC heuristique -hauteur (triche) : {auc(-heights, labels):.3f}")

# %% [markdown]
# ## L'agent joue
#
# D'abord un coup d'œil qualitatif : quatre GIF côte à côte.

# %%
from pinball.env import PinballEnv
from pinball.collect import StickyRandomPolicy
from jepa.eval import AlwaysPressed, PeriodicFlapper, evaluate, record_gif
from jepa.planner import MPCPlanner

agent = MPCPlanner(jepa, head)
env = PinballEnv()
for name, pol in [("agent", agent),
                  ("aleatoire", StickyRandomPolicy(np.random.default_rng(0))),
                  ("toujours", AlwaysPressed()),
                  ("flapper", PeriodicFlapper())]:
    r = record_gif(env, pol, f"{name}.gif", seed=2026)
    trunc = " (GIF tronqué à 30 s)" if r["truncated"] else ""
    print(f"{name:10s}: {r['steps']} pas ({r['steps']/15:.1f} s){trunc}")

# %%
from IPython.display import Image as IPImage, display
for name in ("agent", "aleatoire", "toujours", "flapper"):
    print(name); display(IPImage(f"{name}.gif"))

# %% [markdown]
# ## Le graphique de la V1 : 50 épisodes, seeds appariées
#
# **Critère d'acceptation** : la survie moyenne de l'agent doit dépasser
# NETTEMENT l'aléatoire et « toujours appuyé » (spec §2 et §12).
#
# On ajoute un quatrième concurrent, découvert pendant le développement :
# le **flapper aveugle**, qui bat des deux flippers en rythme sans jamais
# regarder l'écran — et qui survit étonnamment longtemps sur cette table.
# Il ne fait pas partie du critère d'acceptation, mais c'est le test
# d'honnêteté du projet : si l'agent ne fait pas mieux que lui, c'est qu'il
# a appris un rythme, pas à VOIR la balle.

# %%
import matplotlib.pyplot as plt
results = {}
for name, pol in [("agent JEPA", agent),
                  ("aléatoire", StickyRandomPolicy(np.random.default_rng(0))),
                  ("toujours appuyé", AlwaysPressed()),
                  ("flapper aveugle", PeriodicFlapper())]:
    results[name] = evaluate(env, pol, n_episodes=50)
    print(f"{name:16s}: {results[name]['survival_s']:5.1f} s en moyenne, "
          f"médiane {results[name]['median_steps']/15:.1f} s")

fig, ax = plt.subplots(figsize=(8, 4))
names = list(results)
ax.bar(names, [results[n]["survival_s"] for n in names],
       color=["tab:green", "tab:gray", "tab:gray", "tab:orange"])
ax.set_ylabel("survie moyenne (s)")
ax.set_title("V1 : l'agent JEPA contre les baselines (50 épisodes)")
plt.show()

# %% [markdown]
# ## Regarder l'agent « réfléchir » (bonus)
#
# Sur un état donné, affichons le coût imaginé des 4 actions constantes —
# on voit littéralement le danger prédit de chaque option.

# %%
obs = env.reset(seed=7)
for _ in range(10):
    obs, _ = env.step(0)
seqs = np.repeat(np.arange(4)[:, None], agent.horizon, axis=1)
with torch.no_grad():
    z = jepa.encode(torch.from_numpy(obs).unsqueeze(0).to(agent.device))
    z = z.expand(4, -1).contiguous()
    cost = torch.zeros(4, device=agent.device)
    for t in range(agent.horizon):
        z = jepa.predictor(z, torch.from_numpy(seqs[:, t]).to(agent.device))
        cost += torch.sigmoid(head.to(agent.device)(z))
for a, label in enumerate(["rien", "gauche", "droit", "les deux"]):
    print(f"action constante « {label:9s} » : danger imaginé = {cost[a]:.3f}")

# %% [markdown]
# ## Test de scénario (spec §12) : balle fonçant vers le drain, côté gauche
#
# On cherche un état critique réel — balle basse à gauche, en train de
# descendre — et on regarde ce que l'agent décide. Nuance : un MPC peut
# LÉGITIMEMENT retarder la frappe d'un pas ou deux (frapper trop tôt est
# souvent pire) ; l'important est de voir le flipper gauche actionné au
# moment critique. Si ce n'est pas le cas, inspecter les GIF et l'AUC.

# %%
state = None
for seed in range(300):
    env_s = PinballEnv(seed=seed)
    obs_s = env_s.reset()
    for _ in range(900):
        obs_s, info_s = env_s.step(0)
        if info_s["done"]:
            break
        (x_s, y_s), (vx_s, vy_s) = info_s["ball_pos"], info_s["ball_vel"]
        if x_s < 200 and y_s < 280 and vy_s < -100:
            state = (obs_s, x_s, y_s, vx_s, vy_s)
            break
    if state is not None:
        break
assert state is not None, "aucun état critique trouvé — élargir la plage de seeds"
obs_s, x_s, y_s, vx_s, vy_s = state
choice = agent.plan(obs_s)
print(f"état critique : balle à ({x_s:.0f}, {y_s:.0f}), vitesse ({vx_s:.0f}, {vy_s:.0f})")
print(f"action choisie = {choice} → flipper gauche actionné : {bool(choice & 1)}")

# %% [markdown]
# ## Si l'agent ne bat pas les baselines
#
# Diagnostiquer DANS L'ORDRE (ne pas toucher au planificateur d'abord) :
# 1. AUC de la tête danger < 0,8 → revoir la tête / le dataset ;
# 2. `pred_mse` ne bat pas `copy_mse` à 8 pas (notebook 03) → world model
#    à améliorer : plus d'epochs, plus de données ;
# 3. seulement ensuite : `horizon`, `n_candidates`, `switch_prob`.
#
# **V1 terminée** quand le graphique ci-dessus montre l'agent nettement
# au-dessus des deux baselines. La suite (plan V2) : bumpers scoreurs et
# tête score — le coût devient `danger − λ·score`.
