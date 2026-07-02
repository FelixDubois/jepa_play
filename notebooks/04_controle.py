# %% [markdown]
# # 04 — L'agent joue : têtes multiples, viser la victoire
#
# Le world model sait prédire — il faut maintenant lui dire quoi ÉVITER **et**
# quoi CHERCHER : le danger, la hauteur, les cibles.
#
# - **Tête danger** (honnête) : P(fin perdante bientôt | latent) — seules les
#   fins `ball_lost` / `stuck` marquent leur queue ; une victoire n'est JAMAIS
#   un danger (leçon du reward hacking de l'itération 1 : « éviter de perdre »
#   sans nuance récompensait le piégeage de la balle).
# - **Tête hauteur** : position verticale normalisée de la balle — un
#   objectif continu à faire progresser, pas seulement « éviter le pire ».
# - **Tête cible** : P(contact avec une cible bientôt | latent) — le nouvel
#   objectif offensif de la V2.
#
# Le planificateur MPC imagine ~260 futurs sur 8 pas et choisit celui qui
# minimise `danger − 0,5·hauteur − 2·cible` : éviter le drain, monter, et
# foncer sur les cibles. AUCUN apprentissage dans le planificateur — le
# « bien jouer » émerge entièrement du modèle du monde et des têtes.

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

import numpy as np
import torch
DATA_DIR, CKPT_DIR = ROOT / "data/targets_v1", ROOT / "checkpoints_targets"

# %%
from pinball.collect import load_episodes
from jepa.train import load_jepa
episodes = load_episodes(DATA_DIR)
jepa = load_jepa(CKPT_DIR / "jepa.pt")
print(f"{len(episodes)} épisodes, modèle chargé")

# %% [markdown]
# ## Entraîner les têtes d'objectif
#
# Les labels sont gratuits, fabriqués depuis `info` : la politique aléatoire a
# perdu la balle des centaines de fois (danger), touché quelques cibles par
# hasard (cible), et visité toute la hauteur du plateau (hauteur, position).
# Un seul passage d'encodage produit les latents partagés ; les 4 têtes
# s'entraînent côte à côte dessus.
#
# On vérifie la qualité de la tête danger par l'AUC sur des épisodes de
# validation, et on la compare à une heuristique évidente : « la balle est
# basse = danger » (calculée avec info privilégiée que le modèle, lui, n'a
# jamais vue).

# %%
from jepa.heads import auc, train_objective_heads
from jepa.data import MultiLabelDataset
heads, metrics = train_objective_heads(jepa, episodes)
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")
for name, h in heads.items():
    torch.save(h.state_dict(), CKPT_DIR / f"{name}.pt")

# heuristique de référence : hauteur de la balle (via info privilégiée)
n_val = max(1, int(len(episodes) * 0.1))
val_eps = episodes[-n_val:]
ds = MultiLabelDataset(val_eps)
labels = np.array([ds[i]["danger"].item() for i in range(len(ds))])
heights = np.concatenate([ep["ball_pos"][1:, 1] for ep in val_eps])
print(f"AUC tête danger (image seule) : {metrics['auc_danger']:.3f}")
print(f"AUC heuristique -hauteur (triche) : {auc(-heights, labels):.3f}")

# %% [markdown]
# ## L'agent joue
#
# D'abord un coup d'œil qualitatif : quatre GIF côte à côte.

# %%
from pinball.config import hard_board
from pinball.env import PinballEnv
from pinball.collect import StickyRandomPolicy
from jepa.eval import AlwaysPressed, PeriodicFlapper, evaluate, record_gif
from jepa.planner import MPCPlanner

agent = MPCPlanner(jepa, heads["danger"], n_candidates=256,
                   height_head=heads["height"], target_head=heads["target"])
env = PinballEnv(hard_board())
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
# ## Le graphique de la V2 : gagner la partie (50 épisodes, seeds appariées)
#
# **Critère d'acceptation** : l'agent domine NETTEMENT toutes les baselines en
# TAUX DE VICTOIRE — la MÉTRIQUE REINE de la V2 (les baselines aveugles
# gagnent ≤ ~2 % par chance ; toucher 1 à 3 cibles au hasard tout en évitant
# le drain reste rare).
#
# On ajoute un quatrième concurrent, découvert pendant le développement :
# le **flapper aveugle**, qui bat des deux flippers en rythme sans regarder
# l'écran. Sur la table par défaut il survivait 60 s ; sur la table dure,
# les stratégies aveugles s'effondrent — c'est tout l'intérêt.
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
    r = results[name]
    print(f"{name:16s}: victoire {100*r['completion_rate']:3.0f} %  "
          f"cibles {100*r['targets_hit_rate']:3.0f} %  "
          f"hauteur {r['mean_height']:.2f}  survie {r['survival_s']:.1f} s  "
          f"nudges {r['mean_nudges']:.1f}")

names = list(results)
colors = ["tab:green", "tab:gray", "tab:gray", "tab:orange"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
bars = ax1.bar(names, [100 * results[n]["completion_rate"] for n in names], color=colors)
for bar, n in zip(bars, names):
    ax1.annotate(f"{results[n]['mean_nudges']:.1f} nudges",
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha="center", va="bottom", fontsize=8)
ax1.set_ylabel("taux de victoire (%)")
ax1.set_title("victoire — MÉTRIQUE REINE")
ax2.bar(names, [100 * results[n]["targets_hit_rate"] for n in names], color=colors)
ax2.set_ylabel("cibles touchées (%)")
ax2.set_title("taux de cibles touchées")
fig.suptitle("V2 : gagner la partie (50 épisodes, seeds appariées)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Regarder l'agent « réfléchir » (bonus)
#
# Sur un état donné, affichons le coût imaginé des 4 actions constantes —
# les trois composantes (danger, hauteur, cible) séparément, puis leur somme
# pondérée (mêmes poids que l'agent : danger − 0,5·hauteur − 2·cible).

# %%
obs = env.reset(seed=7)
for _ in range(10):
    obs, _ = env.step(0)
seqs = np.repeat(np.arange(4)[:, None], agent.horizon, axis=1)
with torch.no_grad():
    z = jepa.encode(torch.from_numpy(obs).unsqueeze(0).to(agent.device))
    z = z.expand(4, -1).contiguous()
    danger_cost = torch.zeros(4, device=agent.device)
    height_cost = torch.zeros(4, device=agent.device)
    target_cost = torch.zeros(4, device=agent.device)
    for t in range(agent.horizon):
        z = jepa.predictor(z, torch.from_numpy(seqs[:, t]).to(agent.device))
        danger_cost += torch.sigmoid(agent.danger_head(z))
        height_cost += agent.height_head(z)
        target_cost += torch.sigmoid(agent.target_head(z))
    total_cost = (agent.w_danger * danger_cost - agent.w_height * height_cost
                 - agent.w_target * target_cost)
for a, label in enumerate(["rien", "gauche", "droit", "les deux"]):
    print(f"action constante « {label:9s} » : danger {danger_cost[a]:.3f}  "
          f"hauteur {height_cost[a]:.3f}  cible {target_cost[a]:.3f}  "
          f"→ coût total {total_cost[a]:.3f}")

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
    env_s = PinballEnv(hard_board(), seed=seed)
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
# 0. `auc_target` bas (< 0,75) ? Normal au premier tour — le hasard touche peu
#    de cibles ; c'est l'itération (notebook 05) qui l'améliore.
# 1. AUC de la tête danger < 0,8 → revoir la tête / le dataset ;
# 2. `pred_mse` ne bat pas `copy_mse` à 8 pas (notebook 03) → world model
#    à améliorer : plus d'epochs, plus de données ;
# 3. seulement ensuite : `horizon`, `n_candidates`, `switch_prob`.
#
# **V2 terminée** quand le graphique ci-dessus montre l'agent nettement
# au-dessus des baselines en taux de victoire. La suite : notebook 05
# (itération) puis 06 (visualisation des prédictions).
