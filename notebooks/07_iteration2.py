# %% [markdown]
# # 07 — Itération 2 : la boucle tient-elle un deuxième tour ?
#
# L'itération 1 (notebook 05) a rapporté +11 pts de victoires (~3σ à n=200) :
# 29 % → 40 %. La leçon des world models — la politique améliore les données,
# les données améliorent la politique — a fonctionné une fois. Ce notebook
# pose la question suivante : **encore une fois ?**
#
# Les gains diminuent normalement à chaque tour. Deux issues, toutes deux
# publiables :
#
# - **gain ≥ 8 pts** (≈2σ à n=200, seeds appariées) → `git tag v3`, nouveau
#   champion ;
# - **plateau ou régression** → la leçon attendue des rendements
#   décroissants, documentée telle quelle.
#
# Risque propre à ce tour : le champion part de l'epoch 22 et l'érosion de la
# dynamique rampe déjà (copy_h8 : 0.0128 → 0.0096 au dernier réentraînement).
# D'où un GARDE-FOU : le diagnostic n°4 (lisibilité de la balle, notebook 03)
# tourne AVANT l'évaluation — s'il alerte, le modèle warm-starté est
# disqualifié et l'ISSUE DE SECOURS prend le relais : réentraînement from
# scratch sur le même mélange (compteur d'érosion remis à zéro).
#
# *Vécu (runs 1-2 sur Colab)* : le garde-fou ABSOLU (seuil 0.12) a disqualifié
# le warm-start (0.135)… puis la contre-expertise a montré que le CHAMPION
# lui-même lit 0.127 sur cette distribution — le seuil, calibré sur du jeu
# aléatoire, mesurait la difficulté du jeu de champion, pas la casse du
# modèle. Le garde-fou est donc RELATIF : le candidat doit rester au niveau
# du champion (+0.02 max) sur les MÊMES épisodes. Le from scratch 10 epochs,
# lui, est réellement inférieur (0.172) — voir l'issue de secours.

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
DATA_V1, DATA_V2, DATA_V3 = (ROOT / "data/targets_v1", ROOT / "data/targets_v2",
                             ROOT / "data/targets_v3")
CKPT_V2, CKPT_V3 = ROOT / "checkpoints_targets_v2", ROOT / "checkpoints_targets_v3"
print("device :", "cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Recharger le champion (agent V2, 40 % de victoires à n=200)
#
# `checkpoints_targets_v2` est en LECTURE SEULE dans tout ce notebook :
# l'itération écrit exclusivement dans `data/targets_v3` et
# `checkpoints_targets_v3`. Refaire l'itération = supprimer ces deux
# derniers dossiers — jamais ceux du champion.

# %%
from pinball.collect import load_episodes
from jepa.train import load_jepa
from jepa.heads import DangerHead, HeightHead, PositionProbe, TargetHead
from jepa.planner import MPCPlanner

jepa_v2 = load_jepa(CKPT_V2 / "jepa.pt")
heads_v2 = {"danger": DangerHead(jepa_v2.z_dim), "height": HeightHead(jepa_v2.z_dim),
            "target": TargetHead(jepa_v2.z_dim), "pos": PositionProbe(jepa_v2.z_dim)}
for name, h in heads_v2.items():
    h.load_state_dict(torch.load(CKPT_V2 / f"{name}.pt", weights_only=True))
    h.eval()
agent_v2 = MPCPlanner(jepa_v2, heads_v2["danger"], n_candidates=256,
                      height_head=heads_v2["height"],
                      target_head=heads_v2["target"])
epoch_v2 = int(torch.load(CKPT_V2 / "jepa.pt", map_location="cpu",
                          weights_only=True)["epoch"])
print(f"champion : epoch {epoch_v2}, z_dim {jepa_v2.z_dim}, agent prêt (256 candidats)")

# %% [markdown]
# ## Étape 1 : le champion collecte (~20-40 min sur CPU)
#
# Même recette qu'au notebook 05 : `MixedPolicy` fait jouer l'agent,
# entrecoupé de rafales aléatoires collantes pour garder de la variété.
# Seed NEUF (456) : 123 a servi à l'itération 1 — on veut des dispositions
# de cibles fraîches, pas un replay.

# %%
from pinball.collect import MixedPolicy, collect_dataset
from pinball.config import hard_board
from pinball.env import PinballEnv

N_TRANSITIONS_V3 = 50_000

if DATA_V3.exists() and list(DATA_V3.glob("shard_*.npz")):
    print("Dataset v3 déjà présent — collecte sautée (supprimer data/targets_v3 "
          "ET checkpoints_targets_v3 pour refaire l'itération).")
else:
    env = PinballEnv(hard_board(), seed=456)
    explorer = MixedPolicy(agent_v2, np.random.default_rng(456))
    stats = collect_dataset(env, explorer, N_TRANSITIONS_V3, DATA_V3)
    print(stats)

# %%
episodes_v1 = load_episodes(DATA_V1)
episodes_v2 = load_episodes(DATA_V2)
episodes_v3 = load_episodes(DATA_V3)
for name, eps in (("aléatoire (v1)", episodes_v1), ("agent V1 (v2)", episodes_v2),
                  ("agent V2 (v3)", episodes_v3)):
    lengths = np.array([len(ep["actions"]) for ep in eps])
    print(f"{name:15s}: {len(eps):4d} épisodes, durée moyenne {lengths.mean()/15:.1f} s")

# %% [markdown]
# ## Étape 2 : warm-start du champion sur le mélange v1+v2+v3
#
# Même reprise robuste qu'au 05 : on copie le checkpoint du champion et on
# demande 6 epochs de PLUS que son epoch — un `epochs` fixe inférieur à
# l'epoch du checkpoint n'entraînerait RIEN, en silence.

# %%
from jepa.train import train_jepa

CKPT_V3.mkdir(parents=True, exist_ok=True)
if not (CKPT_V3 / "jepa.pt").exists():
    shutil.copy(CKPT_V2 / "jepa.pt", CKPT_V3 / "jepa.pt")

episodes_mixed = episodes_v1 + episodes_v2 + episodes_v3
# cible PINNÉE au champion (et non au checkpoint courant) : re-exécuter ce
# notebook ne doit PAS empiler 6 epochs de plus à chaque passage — la reprise
# s'arrête à epoch_v2 + 6 et ne réentraîne rien si on y est déjà
jepa_v3, history = train_jepa(episodes_mixed, CKPT_V3, epochs=epoch_v2 + 6)

# %%
import matplotlib.pyplot as plt
epochs_ = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(epochs_, [h["pred_mse"] for h in history], label="prédicteur")
axes[0].plot(epochs_, [h["copy_mse"] for h in history], "--", label="baseline copie")
axes[0].axvline(epoch_v2 + 0.5, color="gray", ls=":", label="← v2 | v3 →")
axes[0].legend(); axes[0].set_title("reprise sur données mixtes")
axes[1].plot(epochs_, [h["latent_std"] for h in history])
axes[1].axhline(0.05, color="r", ls=":"); axes[1].set_title("variance latente")
plt.show()

# %% [markdown]
# ## Garde-fou : le latent a-t-il encore la balle ? (diagnostic n°4)
#
# Chaque epoch de warm-start supplémentaire nourrit l'érosion connue de la
# dynamique. AVANT de payer des têtes et 400 épisodes d'évaluation, on
# vérifie que la balle est restée lisible : une sonde apprend (x, y) sur des
# latents gelés, validation = les 40 derniers épisodes du mélange (du jeu
# d'agent V2 — là où la lisibilité compte). Verdict :
#
# - MAE ≤ MAE_champion + 0.02 (mêmes épisodes, même protocole) → QUALIFIÉ ;
# - au-delà → le réentraînement a réellement dégradé la lecture : MODÈLE
#   DISQUALIFIÉ, l'issue de secours prend le relais.
#
# Pourquoi RELATIF et plus absolu ? Le run 1 avait disqualifié le warm-start
# au seuil 0.12 (calibré notebook 03 sur du jeu aléatoire : 0.108 en prod) ;
# le run 2 a montré que le champion lui-même lit 0.127 ici — le jeu de
# champion est plus dur à lire (balle rapide, haute, près des cibles). Un
# seuil absolu mesurait la distribution, pas le modèle. La référence honnête
# est le champion sur les MÊMES épisodes ; la marge 0.02 absorbe le bruit de
# sonde (~0.01) sans laisser passer une vraie casse (le from scratch est à
# +0.045).

# %%
from jepa.data import MultiLabelDataset, WindowDataset, stack_obs

def readability_mae(model, episodes_train, episodes_val):
    """Protocole du diagnostic n°4 : sonde gelée 400 pas, MAE de validation.

    Factorisé car il sert trois fois : warm-start, contre-expertise du
    champion, et candidat from scratch — toujours sur les MÊMES épisodes.
    """
    def encode_set(eps):
        ds = MultiLabelDataset(eps)
        idx = np.linspace(0, len(ds) - 1, min(3000, len(ds)), dtype=int)
        obs = torch.stack([ds[int(i)]["obs"] for i in idx])
        pos = torch.stack([ds[int(i)]["pos"] for i in idx])
        with torch.no_grad():
            z = torch.cat([model.encode_target(obs[j:j + 256])
                           for j in range(0, len(obs), 256)])
        return z, pos
    z_tr, p_tr = encode_set(episodes_train)
    z_va, p_va = encode_set(episodes_val)
    torch.manual_seed(0)
    probe = PositionProbe(model.z_dim)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for _ in range(400):
        perm = torch.randperm(len(z_tr))[:512]
        loss_p = ((probe(z_tr[perm]) - p_tr[perm]) ** 2).mean()
        popt.zero_grad(); loss_p.backward(); popt.step()
    probe.eval()
    with torch.no_grad():
        mae = (probe(z_va) - p_va).abs().mean().item()
    naif = (p_tr.mean(0) - p_va).abs().mean().item()
    return mae, naif

MARGE = 0.02  # tolérance vs champion : > bruit de sonde (~0.01), < vraie casse

model_cpu = jepa_v3.to("cpu").eval()
champion_cpu = jepa_v2.to("cpu").eval()
mae_warm, naif = readability_mae(model_cpu, episodes_mixed[:-40], episodes_mixed[-40:])
mae_champ, _ = readability_mae(champion_cpu, episodes_mixed[:-40], episodes_mixed[-40:])
print(f"champion   (epoch {epoch_v2})     : MAE = {mae_champ:.3f}  ← référence")
print(f"warm-start (epoch {epoch_v2 + 6}) : MAE = {mae_warm:.3f} "
      f"(devin naïf : {naif:.3f})")
LISIBLE = mae_warm <= mae_champ + MARGE
print("VERDICT :", "QUALIFIÉ — au niveau du champion" if LISIBLE
      else "DISQUALIFIÉ — dégradation réelle : issue de secours")

# %%
# la glissade connue : copy_h8 mesure combien le monde BOUGE dans le latent
# (0.0128 → 0.0096 au dernier réentraînement ; plus bas = latents plus
# statiques = érosion). À noter dans le journal de bord à chaque tour.
ds8 = WindowDataset(episodes_mixed[-50:], k=8)
dl8 = torch.utils.data.DataLoader(ds8, batch_size=128, shuffle=True)
batch = next(iter(dl8))
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
print(f"pred_h8 = {pred_err[-1]:.4f}  copy_h8 = {copy_err[-1]:.4f} "
      f"(pred sous copy ×{copy_err[-1]/pred_err[-1]:.1f})")

# %% [markdown]
# ## Issue de secours : réentraîner from scratch sur v1+v2+v3
#
# Si le warm-start a réellement dégradé la lecture, on remet le compteur à
# zéro : même recette que le notebook 03 (10 epochs), mêmes données mixtes,
# dossier NEUF (`checkpoints_targets_v3_scratch`). Si le warm-start est
# qualifié, cette cellule ne fait rien.
#
# *Vécu (run 2)* : le from scratch 10 epochs lit 0.172 (champion + 0.045) —
# NETTEMENT inférieur, avec une variance latente qui plafonne à 0.14 (vs
# 0.21 pour la lignée du champion). Hypothèse : effet CURRICULUM — la lignée
# du champion a appris la dynamique sur du jeu aléatoire lent avant de voir
# du jeu d'agent ; le from scratch encaisse le jeu de champion dès l'epoch 1,
# avec 10 epochs là où la lignée en cumule 22. La recette du champion ne se
# reproduit pas à bas coût.

# %%
CKPT_SCRATCH = ROOT / "checkpoints_targets_v3_scratch"

if LISIBLE:
    jepa_final, ckpt_final = jepa_v3, CKPT_V3
    mae_final, mode = mae_warm, f"warm-start (epoch {epoch_v2 + 6})"
    print("warm-start sain — issue de secours inutile")
else:
    jepa_scratch, history_s = train_jepa(episodes_mixed, CKPT_SCRATCH, epochs=10)
    scratch_cpu = jepa_scratch.to("cpu").eval()
    mae_scratch, naif_s = readability_mae(scratch_cpu, episodes_mixed[:-40],
                                          episodes_mixed[-40:])
    print(f"lisibilité (from scratch, epoch 10) : MAE = {mae_scratch:.3f} "
          f"(champion {mae_champ:.3f} ; devin naïf : {naif_s:.3f})")
    jepa_final, ckpt_final = jepa_scratch, CKPT_SCRATCH
    mae_final, mode = mae_scratch, "from scratch (epoch 10)"

print(f"candidat V3 = {mode}, lisibilité {mae_final:.3f} "
      f"(référence champion {mae_champ:.3f} + marge {MARGE})")

# %% [markdown]
# ## Étape 3 : têtes V3, puis évaluation n=200 (V3 vs V2)
#
# La cellule suivante REFUSE de tourner si AUCUN candidat — warm-start puis
# from scratch — n'a passé le garde-fou : évaluer un modèle qui a perdu la
# balle coûterait ~1 h pour un chiffre sans signification.

# %%
assert mae_final <= mae_champ + MARGE, (
    "Garde-fou : aucun candidat au niveau du champion sur les mêmes "
    "épisodes (warm-start ET from scratch dégradés). Le résultat de "
    "l'itération 2 est qu'aucun réentraînement ne préserve la lecture : "
    "à documenter tel quel.")

from jepa.heads import train_objective_heads

heads_v3, metrics_v3 = train_objective_heads(jepa_final, episodes_mixed)
for k, v in metrics_v3.items():
    print(f"{k}: {v:.3f}")
for name, h in heads_v3.items():
    torch.save(h.state_dict(), ckpt_final / f"{name}.pt")

# %% [markdown]
# Pourquoi seulement DEUX agents ? V1 (29 %) et les baselines (8-10 %) ont
# déjà leurs mesures n=200 du finisher v2.2 — les rejouer coûterait des
# heures pour confirmer des chiffres connus. `evaluate` seede chaque épisode
# par `seed0 + i` : à `seed0` égal, V3 et V2 jouent EXACTEMENT les mêmes
# parties (seeds appariées).

# %%
from jepa.eval import evaluate

agent_v3 = MPCPlanner(jepa_final, heads_v3["danger"], n_candidates=256,
                      height_head=heads_v3["height"],
                      target_head=heads_v3["target"])
# instance fraîche pour l'éval : le RNG interne d'agent_v2 a été consommé
# par la collecte — une instance neuve garantit la reproductibilité
agent_v2_eval = MPCPlanner(jepa_v2, heads_v2["danger"], n_candidates=256,
                           height_head=heads_v2["height"],
                           target_head=heads_v2["target"])

env = PinballEnv(hard_board())
results = {}
for name, pol in [("agent V3", agent_v3), ("agent V2", agent_v2_eval)]:
    results[name] = evaluate(env, pol, n_episodes=200)
    r = results[name]
    print(f"{name:9s}: victoire {100*r['completion_rate']:3.0f} %  "
          f"cibles {100*r['targets_hit_rate']:3.0f} %  "
          f"hauteur {r['mean_height']:.2f}  survie {r['survival_s']:.1f} s  "
          f"nudges {r['mean_nudges']:.1f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
names = list(results)
bars = ax.bar(names, [100 * results[n]["completion_rate"] for n in names],
              color=["tab:green", "tab:olive"])
for bar, n in zip(bars, names):
    ax.annotate(f"{results[n]['mean_nudges']:.1f} nudges",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.axhline(29, color="gray", ls="--", lw=1)
ax.text(1.45, 29.5, "agent V1 (29 %, n=200)", fontsize=8, color="gray", ha="right")
ax.axhspan(8, 10, color="gray", alpha=0.2)
ax.text(1.45, 10.5, "baselines (8-10 %, n=200)", fontsize=8, color="gray", ha="right")
ax.set_ylabel("taux de victoire (%)")
ax.set_title("Itération 2 — n=200, seeds appariées")
plt.show()

# %%
gain = 100 * (results["agent V3"]["completion_rate"]
              - results["agent V2"]["completion_rate"])
print(f"gain V3 − V2 : {gain:+.0f} pts (V3 = {mode} ; seuil : +8 pts ≈ 2σ à n=200)")
if gain >= 8:
    print("→ la boucle tient un 2e tour : nouveau champion — poser `git tag v3`.")
else:
    print("→ plateau (ou régression) : la boucle a rendu l'essentiel au 1er tour.")
    print("  C'est la leçon des rendements décroissants — résultat publié tel quel.")

# %% [markdown]
# ## Lecture des résultats
#
# - **Si V3 gagne (≥ +8 pts)** : champion remplacé — `git tag v3`, et le
#   notebook 06 se repointe sur `checkpoints_targets_v3` (un chemin à
#   changer) pour VOIR ce que la nouvelle imagination a appris.
# - **Si ça plafonne** : rien d'anormal. Le 1er tour corrigeait le vrai
#   problème — un world model qui n'avait JAMAIS vu de bon jeu. Au 2e tour,
#   l'agent V2 visite déjà les états utiles : le mélange n'apporte plus
#   grand-chose de neuf. Les rendements décroissants sont la règle des
#   boucles world-model, pas l'exception.
# - **Leçon des runs 1-2 (garde-fou)** : un seuil absolu de lisibilité
#   mesurait la DISTRIBUTION (le jeu de champion est plus dur à lire :
#   champion 0.127 vs 0.108 sur jeu aléatoire), pas la santé du modèle —
#   d'où le garde-fou relatif au champion. Et le from scratch 10 epochs
#   (0.172) montre que la lignée du champion ne se reproduit pas à bas
#   coût : la boucle raffine, elle ne repart pas de zéro.
# - Dans tous les cas, les nudges restent le détecteur de triche : un agent
#   qui gagne en nudgeant a trouvé une faille, pas une stratégie.
