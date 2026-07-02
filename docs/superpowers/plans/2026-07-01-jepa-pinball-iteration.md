# JEPA Pinball V1.5 — Itération du world model (plan d'implémentation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fermer la boucle world-model : l'agent V1 collecte lui-même de nouvelles
données (avec rafales d'exploration), le JEPA et la tête danger se réentraînent sur
le dataset mixte, et l'agent V2 est comparé à V1 et aux baselines — avec un panneau
d'honnêteté sur les nudges consommés.

**Contexte (résultats Colab V1, 2026-07-01) :** agent 25,2 s (méd. 26,1) ; aléatoire
3,9 s ; toujours appuyé 34,1 s (méd. 34,0 — horloge des 6 nudges) ; flapper 59,3 s
(plafond 60 s). AUC danger 0,930 vs heuristique 0,789. Diagnostic : world model et
tête danger sains ; le plafond vient (1) de l'horizon court et surtout (2) du fait
que le modèle n'a vu que du jeu aléatoire — l'agent est hors distribution quand il
joue bien. Réponse V1.5 : collecte agent-dans-la-boucle (design validé par
l'utilisateur en conversation).

**Architecture:** aucune nouvelle abstraction — une politique de mélange
(`MixedPolicy`), deux clés de plus dans l'éval, et un notebook 05 qui orchestre le
tout avec les APIs existantes (le warm-start réutilise la reprise par checkpoint de
`train_jepa` : on copie `jepa.pt` dans un nouveau dossier et on augmente `epochs`).

**Tech Stack:** inchangé (Python ≥3.10, pymunk, torch ≥2.3, jupytext).

## Global Constraints

- Mêmes conventions que le plan V1 : identifiants en anglais, docstrings et
  commentaires en français ; notebooks jupytext py:percent + .ipynb commités ;
  tests CPU rapides ; commits fréquents avec trailers Claude.
- Protocole politique inchangé : `reset()` + `__call__(obs) -> int` — `MixedPolicy`
  doit le respecter ET le déléguer (elle enveloppe l'agent MPC).
- Changements d'API STRICTEMENT additifs : `run_episode`/`evaluate` gagnent des clés
  (`nudges`, `mean_nudges`) sans en retirer ; `collect.py` gagne une classe sans
  modifier les existantes. Le notebook 01 n'est pas touché ; les notebooks 02-04
  ne changent QUE par la Task 20 (table dure + chemins `*_hard`).
- Suite de tests attendue : 53 existants + 3 (Task 17) + 1 (Task 20) = 57 PASS.
- **Table d'expérience (décision utilisateur)** : `hard_board()` =
  `BoardConfig(drain_gap=120.0, flipper_length=90.0)` partout dans les notebooks
  02, 04, 05. Les tests du package restent sur la table par défaut.

## File Structure

```
pinball/collect.py        # Task 17 — + MixedPolicy
tests/test_collect.py     # Task 17 — + 3 tests
jepa/eval.py              # Task 18 — run_episode/evaluate + nudges (additif)
pinball/config.py         # Task 20 — + hard_board()
tests/test_config.py      # Task 20 — + 1 test
notebooks/02..04_*.py     # Task 20 — table dure + chemins *_hard (+ .ipynb)
notebooks/05_iteration.py # Task 19 — la boucle d'itération (+ .ipynb)
```

**Ordre d'exécution : 17 → 18 → 20 → 19** (le notebook 05 consomme `hard_board`).

---

### Task 17: MixedPolicy — l'agent explore par rafales

**Files:**
- Modify: `pinball/collect.py` (ajout en fin de fichier)
- Test: `tests/test_collect.py` (ajout)

**Interfaces:**
- Consumes: le protocole politique (`reset()`, `__call__(obs) -> int`).
- Produces: `MixedPolicy(primary, rng, burst_prob=0.03, burst_range=(3, 15))` —
  joue `primary`, mais à chaque pas hors rafale a une probabilité `burst_prob`
  d'entamer une rafale d'actions aléatoires collantes (durée uniforme dans
  `burst_range`, bornes incluses). Part d'exploration résultante ≈
  E[L]/(1/p + E[L]) ≈ 21 % avec les défauts. `reset()` propage à `primary`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_collect.py` :

```python
class _ConstantPolicy:
    """Politique factice : action constante, trace les appels et les reset."""

    def __init__(self, action=3):
        self.action = action
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, obs):
        self.calls += 1
        return self.action


def test_mixed_policy_mostly_plays_primary():
    from pinball.collect import MixedPolicy
    primary = _ConstantPolicy(action=3)
    policy = MixedPolicy(primary, np.random.default_rng(0))
    actions = [policy(None) for _ in range(3000)]
    # les rafales tirent uniformément dans {0..3} : ~3/4 des pas de rafale
    # diffèrent de l'action du primaire -> part observable ≈ 21 % × 3/4
    frac_other = sum(a != 3 for a in actions) / len(actions)
    assert 0.05 < frac_other < 0.35
    # hors rafale, c'est bien le primaire qui décide
    assert primary.calls > len(actions) * 0.5


def test_mixed_policy_bursts_are_sticky():
    from pinball.collect import MixedPolicy
    policy = MixedPolicy(_ConstantPolicy(action=3), np.random.default_rng(1),
                         burst_prob=1.0, burst_range=(5, 5))
    # burst_prob=1 : rafale permanente, par blocs de 5 actions identiques
    actions = [policy(None) for _ in range(20)]
    for i in range(0, 20, 5):
        assert len(set(actions[i:i + 5])) == 1


def test_mixed_policy_reset_propagates():
    from pinball.collect import MixedPolicy
    primary = _ConstantPolicy()
    policy = MixedPolicy(primary, np.random.default_rng(0))
    policy.reset()
    policy.reset()
    assert primary.resets >= 2
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_collect.py -v`
Expected: FAIL — `ImportError: cannot import name 'MixedPolicy'` (les 4 tests
existants restent PASS).

- [ ] **Step 3: Implémenter (ajout en fin de `pinball/collect.py`)**

```python
class MixedPolicy:
    """Politique d'itération : l'agent joue, entrecoupé de RAFALES d'actions
    aléatoires collantes.

    Pourquoi ? Pour réentraîner le world model sur les états que visite le BON
    jeu (l'agent), tout en gardant assez de variété pour ne pas figer le
    modèle sur une seule trajectoire. Une action aléatoire isolée ne sert à
    rien (flipper qui vibre) : on explore par rafales maintenues, comme la
    politique de collecte initiale.
    """

    def __init__(self, primary, rng: np.random.Generator,
                 burst_prob: float = 0.03,
                 burst_range: tuple[int, int] = (3, 15)):
        self.primary = primary
        self._rng = rng
        self.burst_prob = burst_prob
        self.burst_range = burst_range
        self.reset()

    def reset(self) -> None:
        self.primary.reset()
        self._burst_left = 0
        self._burst_action = 0

    def __call__(self, obs) -> int:
        if self._burst_left <= 0 and self._rng.random() < self.burst_prob:
            self._burst_action = int(self._rng.integers(4))
            self._burst_left = int(self._rng.integers(self.burst_range[0],
                                                      self.burst_range[1] + 1))
        if self._burst_left > 0:
            self._burst_left -= 1
            return self._burst_action
        return self.primary(obs)
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_collect.py -v`
Expected: 7 PASS (4 existants + 3 nouveaux).

- [ ] **Step 5: Commit**

```bash
git add pinball/collect.py tests/test_collect.py
git commit -m "feat: MixedPolicy, exploration par rafales pour la collecte agent-dans-la-boucle"
```

---

### Task 18: L'éval compte les nudges

**Files:**
- Modify: `jepa/eval.py` (`run_episode` et `evaluate`, changement additif)

**Interfaces:**
- Consumes: `info["nudged"]` de `PinballEnv.step`.
- Produces: `run_episode` retourne en plus `"nudges": int` (nombre de nudges de
  l'épisode) ; `evaluate` retourne en plus `"mean_nudges": float`. Aucune clé
  existante ne change — le notebook 04 committé continue de fonctionner tel quel.

- [ ] **Step 1: Modifier `run_episode`**

```python
def run_episode(env, policy, seed: int | None = None) -> dict:
    obs = env.reset(seed=seed)
    policy.reset()
    nudges = 0
    while True:
        obs, info = env.step(policy(obs))
        nudges += int(info["nudged"])
        if info["done"]:
            return {"steps": info["steps"], "ball_lost": info["ball_lost"],
                    "stuck": info["stuck"], "nudges": nudges}
```

- [ ] **Step 2: Modifier `evaluate`** — ajouter dans le dict retourné :

```python
        "mean_nudges": float(np.mean([r["nudges"] for r in results])),
```

- [ ] **Step 3: Smoke test (les nudges révèlent le jeu passif)**

Run: `python -c "
from pinball.env import PinballEnv
from jepa.eval import AlwaysPressed, evaluate
from pinball.collect import StickyRandomPolicy
import numpy as np
env = PinballEnv(seed=0)
a = evaluate(env, AlwaysPressed(), n_episodes=8)
r = evaluate(env, StickyRandomPolicy(np.random.default_rng(0)), n_episodes=8)
print(f'toujours appuyé : {a[\"mean_nudges\"]:.1f} nudges/épisode')
print(f'aléatoire       : {r[\"mean_nudges\"]:.1f} nudges/épisode')
assert a['mean_nudges'] > r['mean_nudges']
"`
Expected: « toujours appuyé » consomme nettement plus de nudges (≈ 5-6) que
l'aléatoire (≈ 0-1) ; assertion OK.

- [ ] **Step 4: Suite complète**

Run: `pytest`
Expected: 56 PASS (rien de cassé — le changement est additif).

- [ ] **Step 5: Commit**

```bash
git add jepa/eval.py
git commit -m "feat: comptage des nudges dans run_episode/evaluate (additif)"
```

---

### Task 19: Notebook 05 — la boucle d'itération

**Files:**
- Create: `notebooks/05_iteration.py` + `notebooks/05_iteration.ipynb`

**Interfaces:**
- Consumes: tout l'existant + `MixedPolicy` (Task 17) + `mean_nudges` (Task 18).
- Produces: notebook exécutable sur Colab T4, qui produit l'agent V2 et le
  graphique comparatif final.

- [ ] **Step 1: Écrire `notebooks/05_iteration.py`**

```python
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
import importlib.util, subprocess, sys, os
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB and not os.path.exists("jepa_play"):
    subprocess.run(["git", "clone", "https://github.com/FelixDubois/jepa_play.git"], check=True)
    os.chdir("jepa_play")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# %%
from pathlib import Path
import shutil
import numpy as np
import torch
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/jepa_pinball")
else:
    ROOT = Path(".")
DATA_V1, DATA_V2 = ROOT / "data/hard_v1", ROOT / "data/hard_v2"
CKPT_V1, CKPT_V2 = ROOT / "checkpoints_hard", ROOT / "checkpoints_hard_v2"
print("device :", "cuda" if torch.cuda.is_available() else "cpu (lent !)")

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
# ## Étape 1 : l'agent collecte (~15-30 min sur T4)
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
    print("Dataset v2 déjà présent — collecte sautée.")
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
results = {}
for name, pol in [("agent V2", agent_v2),
                  ("agent V1", agent_v1),
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
```

- [ ] **Step 2: Générer le .ipynb et vérifier la structure**

Run: `jupytext --to ipynb notebooks/05_iteration.py && python -c "
import jupytext
nb = jupytext.read('notebooks/05_iteration.py')
print(len(nb.cells), 'cellules')"`
Expected: le .ipynb est créé, le nombre de cellules s'affiche.

- [ ] **Step 3: Test d'intégration mini (CPU) de la boucle complète**

Run: `python -c "
import tempfile, shutil, numpy as np, torch
from pathlib import Path
from pinball.collect import MixedPolicy, StickyRandomPolicy, collect_dataset, load_episodes
from pinball.config import hard_board
from pinball.env import PinballEnv
from jepa.train import train_jepa
from jepa.heads import train_danger_head
from jepa.planner import MPCPlanner
from jepa.eval import evaluate
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    env = PinballEnv(hard_board(), seed=0)
    collect_dataset(env, StickyRandomPolicy(np.random.default_rng(0)), 1500, d/'v1')
    eps1 = load_episodes(d/'v1')
    jepa, _ = train_jepa(eps1, d/'ck1', epochs=2, batch_size=64, device='cpu', num_workers=0)
    head, _ = train_danger_head(jepa, eps1, epochs=3, device='cpu')
    agent = MPCPlanner(jepa, head, n_candidates=16, device='cpu')
    collect_dataset(env, MixedPolicy(agent, np.random.default_rng(1)), 600, d/'v2')
    eps2 = load_episodes(d/'v2')
    (d/'ck2').mkdir(parents=True)
    shutil.copy(d/'ck1/jepa.pt', d/'ck2/jepa.pt')
    jepa2, hist = train_jepa(eps1 + eps2, d/'ck2', epochs=3, batch_size=64, device='cpu', num_workers=0)
    assert hist[-1]['epoch'] == 3 and len(hist) == 3
    head2, _ = train_danger_head(jepa2, eps1 + eps2, epochs=3, device='cpu')
    r = evaluate(env, MPCPlanner(jepa2, head2, n_candidates=16, device='cpu'), n_episodes=2)
    print('OK — boucle d itération complète, survie:', round(r['survival_s'], 1), 's,',
          r['mean_nudges'], 'nudges/ep')
"`
Expected: `OK — boucle d itération complète` (~3-6 min CPU).

- [ ] **Step 4: Commit**

```bash
git add notebooks/05_iteration.py notebooks/05_iteration.ipynb
git commit -m "docs: notebook 05, iteration du world model (collecte par l'agent, warm start)"
```

---

## Validation (manuelle, sur Colab)

1. Pousser la branche mergée. **D'abord relancer les notebooks 02 → 04 sur la
   table dure** (nouveaux dossiers Drive `data/hard_v1` / `checkpoints_hard` —
   les anciens résultats de la table par défaut restent intacts).
2. Puis notebook 05 (~45-60 min). Critère de réussite de l'itération :
   **agent V2 > agent V1** (sinon la boucle n'apporte rien — vérifier la part
   d'exploration et la durée des épisodes v2).
3. Si agent V2 dépasse nettement « toujours appuyé » (~12 s mesurés en local
   sur la table dure) : critère d'acceptation rempli → `git tag v1`.
4. Ensuite : plan V2 (bumpers + score).

---

### Task 20: La table dure devient la configuration officielle

**Décision utilisateur (2026-07-01) :** toutes les expériences passent sur
`BoardConfig(drain_gap=120.0, flipper_length=90.0)` (la « env2 » du notebook 01).
Équilibrage mesuré localement : aléatoire 2,1 s (100 % pertes), toujours appuyé
11,8 s (méd. 2,8 s, 2,6 nudges), flapper aveugle 1,6 s — les stratégies aveugles
s'effondrent, la vision devient indispensable. Épisodes aléatoires : moy 35 pas,
min 15 → 0 % trop courts pour k=8. Les datasets/checkpoints Drive existants
(table par défaut) sont invalidés : nouveaux dossiers `*_hard`, re-run 02→04.

**Files:**
- Modify: `pinball/config.py` (ajout de `hard_board()` en fin de fichier)
- Modify: `tests/test_config.py` (ajout d'un test)
- Modify: `notebooks/02_collecte.py`, `notebooks/03_jepa.py`,
  `notebooks/04_controle.py` (+ .ipynb régénérés)

**Interfaces:**
- Produces: `hard_board() -> BoardConfig` — l'unique source de vérité de la
  table d'expérience, consommée par les notebooks 02, 04, 05.
- Les tests du package restent sur la table PAR DÉFAUT (leurs hypothèses de
  géométrie — piège entre les pointes, etc. — sont propres au défaut).

- [ ] **Step 1: Test qui échoue** — ajouter à `tests/test_config.py` :

```python
def test_hard_board_preset():
    from pinball.config import hard_board
    cfg = hard_board()
    assert cfg.drain_gap == 120.0 and cfg.flipper_length == 90.0
    # l'ouverture au repos dépasse le diamètre de la balle : vrai trou central
    assert cfg.drain_gap - 2 * cfg.flipper_thickness > 2 * cfg.ball_radius
    # les défauts de BoardConfig ne bougent pas
    assert BoardConfig().drain_gap == 44.0
```

Run: `pytest tests/test_config.py -v` — Expected: FAIL (ImportError hard_board).

- [ ] **Step 2: Implémenter** — ajouter en fin de `pinball/config.py` :

```python
def hard_board() -> BoardConfig:
    """La table officielle des expériences (notebooks 02, 04, 05).

    Drain largement ouvert (120 − 2×12 = 96 > diamètre 28 : la balle passive
    draine) et flippers courts. Mesuré : toutes les politiques aveugles
    s'effondrent (~2 s) — il faut VOIR la balle pour survivre.
    """
    return BoardConfig(drain_gap=120.0, flipper_length=90.0)
```

Run: `pytest tests/test_config.py -v` — Expected: 5 PASS.

- [ ] **Step 3: Basculer les notebooks (éditions exactes)**

`notebooks/02_collecte.py` :
- ligne `DATA_DIR = Path("/content/drive/MyDrive/jepa_pinball/data/v1")`
  → `DATA_DIR = Path("/content/drive/MyDrive/jepa_pinball/data/hard_v1")`
- ligne `DATA_DIR = Path("data/v1")` → `DATA_DIR = Path("data/hard_v1")`
- ligne `from pinball.env import PinballEnv` → ajouter au-dessus :
  `from pinball.config import hard_board`
- ligne `env = PinballEnv(seed=42)` → `env = PinballEnv(hard_board(), seed=42)`
- dans le markdown d'intro, après la phrase sur les balles perdues, ajouter :
  `# Les expériences tournent sur la TABLE DURE (hard_board) : drain ouvert,`
  `# flippers courts — les stratégies aveugles ne survivent pas ici.`

`notebooks/03_jepa.py` :
- `DATA_DIR, CKPT_DIR = ROOT / "data/v1", ROOT / "checkpoints"`
  → `DATA_DIR, CKPT_DIR = ROOT / "data/hard_v1", ROOT / "checkpoints_hard"`

`notebooks/04_controle.py` :
- `DATA_DIR, CKPT_DIR = ROOT / "data/v1", ROOT / "checkpoints"`
  → `DATA_DIR, CKPT_DIR = ROOT / "data/hard_v1", ROOT / "checkpoints_hard"`
- ajouter `from pinball.config import hard_board` au-dessus de
  `from pinball.env import PinballEnv`
- `env = PinballEnv()` → `env = PinballEnv(hard_board())`
- `env_s = PinballEnv(seed=seed)` (test de scénario)
  → `env_s = PinballEnv(hard_board(), seed=seed)`
- dans le markdown « flapper aveugle », remplacer la phrase
  `# le **flapper aveugle**, qui bat des deux flippers en rythme sans jamais`
  `# regarder l'écran — et qui survit étonnamment longtemps sur cette table.`
  par
  `# le **flapper aveugle**, qui bat des deux flippers en rythme sans regarder`
  `# l'écran. Sur la table par défaut il survivait 60 s ; sur la table dure,`
  `# les stratégies aveugles s'effondrent — c'est tout l'intérêt.`

Régénérer : `jupytext --to ipynb notebooks/02_collecte.py notebooks/03_jepa.py notebooks/04_controle.py`

- [ ] **Step 4: Suite complète**

Run: `pytest` — Expected: 57 PASS (56 + 1).

- [ ] **Step 5: Commit**

```bash
git add pinball/config.py tests/test_config.py notebooks/
git commit -m "feat: hard_board() — la table dure devient la configuration officielle des expériences"
```
