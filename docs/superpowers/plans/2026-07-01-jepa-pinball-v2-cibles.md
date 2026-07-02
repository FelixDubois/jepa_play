# JEPA Pinball V2 « Cibles » — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cibles aléatoires (1-3/épisode, victoire quand toutes touchées), objectif
de hauteur continu, planification multi-objectifs, et visualisation des prédictions
(sonde de position superposée au réel + décodeur d'imagination).

**Spec:** `docs/superpowers/specs/2026-07-01-jepa-pinball-v2-cibles-design.md`
(elle-même adossée à la spec V1). Valeurs cibles **validées par prototype**
(2026-07-01) : rayon 26, gris 110, zone x∈[70,470] y∈[420,860], séparation 100,
détection par distance — 10 % de cibles touchées par le hasard, 0 nouveau blocage.

**Tech Stack:** inchangé.

## Global Constraints

- Rétro-compatibilité STRICTE du package : la table PAR DÉFAUT n'a pas de cibles
  (`n_targets_range=(0, 0)`) — les 57 tests existants ne doivent pas être modifiés
  (sauf ajouts). `hard_board()` active les cibles (1, 3). `MPCPlanner` garde sa
  signature actuelle valide (nouvelles têtes optionnelles).
- Détection de contact PAR SOUS-PAS (120 Hz) dans `step_control` — au pas de
  contrôle (15 Hz) une balle rapide peut traverser la zone de contact entre deux
  vérifications.
- Les fins en VICTOIRE (`completed`) ne sont JAMAIS étiquetées dangereuses ;
  les fins `stuck` LE SONT (correctif « danger honnête » intégré ici).
- Nouveaux dossiers de données : `data/targets_v1`, `checkpoints_targets` (+ `_v2`
  pour l'itération). Notebooks en DOUBLE-MODE Colab/local (en-tête standard §A).
- Conventions inchangées (identifiants anglais, prose française, TDD, commits
  avec trailers Claude, suite CPU rapide).

## File Structure

```
pinball/config.py      # T21 — champs cibles (+ retrait du champ bumpers inutilisé)
pinball/sim.py         # T21 — sample_target_positions, cibles dans PinballSim
pinball/render.py      # T22 — cibles au rendu (gris 110 / debug coloré)
pinball/env.py         # T23 — tirage au reset, fin "completed", info enrichi
pinball/collect.py     # T24 — shards + hits/targets_total/completed/stuck
jepa/data.py           # T25 — MultiLabelDataset (danger honnête, hauteur, cible, pos)
jepa/heads.py          # T26 — HeightHead, TargetHead, PositionProbe, train_objective_heads
jepa/planner.py        # T27 — coût multi-objectifs (rétro-compatible)
jepa/decoder.py        # T28 — Decoder + train_decoder
jepa/viz.py            # T28 — superpositions prédit/réel (trajectoire + imagination)
jepa/eval.py           # T29 — completion_rate, mean_height, targets_hit_rate (additif)
notebooks/…            # T30 (01-03), T31 (04), T32 (05), T33 (06, nouveau)
```

**Ordre : 21 → 22 → 23 → 24 → 25 → 26 → 27 → 28 → 29 → 30 → 31 → 32 → 33.**

## §A — En-tête double-mode standard des notebooks (référencé par T30-T33)

Cellule 1 (installation) puis cellule 2 (chemins), à adapter par notebook pour
les noms de dossiers. **Exception voulue : le notebook 01 n'utilise QUE la
cellule 1** — il ne lit ni n'écrit aucune donnée, monter Drive lui imposerait
une autorisation Colab inutile :

```python
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
```

---

### Task 21: Cibles dans la config et le simulateur

**Files:**
- Modify: `pinball/config.py`, `pinball/sim.py`
- Test: `tests/test_config.py`, `tests/test_sim.py` (ajouts uniquement)

**Interfaces:**
- `BoardConfig` gagne : `n_targets_range=(0, 0)` (bornes incluses ; défaut = pas
  de cibles), `target_radius=26.0`, `target_elasticity=1.2`,
  `target_zone_x=(70.0, 470.0)`, `target_zone_y=(420.0, 860.0)`,
  `target_min_sep=100.0`, `target_hit_margin=2.0`. Le champ `bumpers` (réservé,
  jamais utilisé) est SUPPRIMÉ. `hard_board()` passe `n_targets_range=(1, 3)`.
- `sample_target_positions(config, rng, n) -> list[tuple[float, float]]`
  (dans sim.py) — n positions dans la zone, séparation minimale respectée,
  RuntimeError après 500 tirages infructueux.
- `PinballSim(config, rng, targets: list[tuple[float, float]] | None = None)` ;
  attributs `targets` (positions), `target_alive` (list[bool]) ;
  `consume_hits() -> list[int]` — indices touchés depuis le dernier appel
  (détection PAR SOUS-PAS dans `step_control`, accumulée dans `_pending_hits`).
  **Retrait DIFFÉRÉ** : la cible détectée est marquée morte immédiatement mais
  reste solide jusqu'à la FIN du pas de contrôle courant (le rebond du contact
  a lieu — la détection à marge +2 précède le contact physique), puis sa forme
  est retirée de l'espace avant le rendu : plus jamais d'obstacle fantôme.

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_config.py` :

```python
def test_target_fields_and_hard_board():
    cfg = BoardConfig()
    assert cfg.n_targets_range == (0, 0)      # défaut : pas de cibles
    from pinball.config import hard_board
    hard = hard_board()
    assert hard.n_targets_range == (1, 3)
    assert hard.target_zone_y[0] > 350        # au-dessus flippers/slingshots
    assert not hasattr(cfg, "bumpers")
```

et à `tests/test_sim.py` :

```python
def test_sample_target_positions_respects_zone_and_separation():
    from pinball.config import hard_board
    from pinball.sim import sample_target_positions
    cfg = hard_board()
    rng = np.random.default_rng(0)
    for _ in range(20):
        pts = sample_target_positions(cfg, rng, 3)
        assert len(pts) == 3
        for x, y in pts:
            assert cfg.target_zone_x[0] <= x <= cfg.target_zone_x[1]
            assert cfg.target_zone_y[0] <= y <= cfg.target_zone_y[1]
        for i in range(3):
            for j in range(i + 1, 3):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                assert (dx * dx + dy * dy) ** 0.5 >= cfg.target_min_sep
    assert sample_target_positions(cfg, rng, 0) == []   # n=0 : liste vide


def test_target_hit_detected_and_removed():
    # balle lâchée pile au-dessus d'une cible : contact garanti, cible retirée
    from pinball.config import hard_board
    cfg = hard_board()
    sim = PinballSim(cfg, np.random.default_rng(0), targets=[(270.0, 600.0)])
    sim.ball.position = (270.0, 700.0)
    sim.ball.velocity = (0.0, -300.0)
    hits = []
    for _ in range(30):
        sim.step_control()
        hits += sim.consume_hits()
        if hits:
            break
    assert hits == [0]
    assert sim.target_alive == [False]
    # plus de contact possible ensuite
    sim.ball.position = (270.0, 700.0)
    sim.ball.velocity = (0.0, -300.0)
    for _ in range(30):
        sim.step_control()
    assert sim.consume_hits() == []


def test_ball_bounces_off_target():
    # la cible est SOLIDE : la balle qui tombe dessus repart vers le haut
    from pinball.config import hard_board
    cfg = hard_board()
    sim = PinballSim(cfg, np.random.default_rng(0), targets=[(270.0, 600.0)])
    sim.ball.position = (270.0, 720.0)
    sim.ball.velocity = (0.0, -400.0)
    max_vy = -1e9
    for _ in range(30):
        sim.step_control()
        max_vy = max(max_vy, sim.ball.velocity.y)
    assert max_vy > 100.0


def test_no_targets_by_default():
    sim, _ = make_sim()
    assert sim.targets == [] and sim.consume_hits() == []


def test_dead_target_stops_blocking_after_step():
    # une cible morte ne doit plus être un obstacle : sans retrait de la
    # forme, elle resterait un mur INVISIBLE (absent du rendu, présent en
    # physique) et le world model verrait la balle rebondir sur du vide
    from pinball.config import hard_board
    cfg = hard_board()
    sim = PinballSim(cfg, np.random.default_rng(0), targets=[(270.0, 600.0)])
    sim.ball.position = (270.0, 700.0)
    sim.ball.velocity = (0.0, -300.0)
    for _ in range(60):
        sim.step_control()
        if sim.consume_hits():
            break
    assert sim.target_alive == [False]
    # balle relancée au même endroit : elle TRAVERSE (aucun rebond fantôme)
    sim.ball.position = (270.0, 700.0)
    sim.ball.velocity = (0.0, -400.0)
    max_vy = -1e9
    for _ in range(6):          # assez pour traverser la zone, sans atteindre
        sim.step_control()      # les slingshots plus bas
        max_vy = max(max_vy, sim.ball.velocity.y)
    assert max_vy < 50.0
```

- [ ] **Step 2: Vérifier l'échec** — `pytest tests/test_config.py tests/test_sim.py -v`
  → les nouveaux tests FAIL (champs/fonctions absents), les anciens PASS.

- [ ] **Step 3: Implémenter**

`pinball/config.py` — remplacer le bloc `bumpers` par :

```python
    # --- cibles (V2) : plots à toucher, placés aléatoirement par épisode.
    # Défaut (0, 0) = pas de cibles : la table de base et les tests V1
    # gardent leur comportement. hard_board() active (1, 3). ---
    n_targets_range: tuple[int, int] = (0, 0)   # bornes incluses
    target_radius: float = 26.0
    target_elasticity: float = 1.2
    target_zone_x: tuple[float, float] = (70.0, 470.0)
    target_zone_y: tuple[float, float] = (420.0, 860.0)
    target_min_sep: float = 100.0
    target_hit_margin: float = 2.0
```

(supprimer aussi l'import `field` devenu inutile), et dans `hard_board()` :
`return BoardConfig(drain_gap=120.0, flipper_length=90.0, n_targets_range=(1, 3))`
(compléter la docstring : « + 1 à 3 cibles aléatoires à toucher »).

`pinball/sim.py` — ajouter en tête de module :

```python
def sample_target_positions(config: BoardConfig, rng: np.random.Generator,
                            n: int) -> list[tuple[float, float]]:
    """n positions de cibles dans la zone sûre, séparées d'au moins
    target_min_sep (rejet, 500 essais max)."""
    if n == 0:
        return []
    pts: list[tuple[float, float]] = []
    for _ in range(500):
        p = (float(rng.uniform(*config.target_zone_x)),
             float(rng.uniform(*config.target_zone_y)))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= config.target_min_sep
               for q in pts):
            pts.append(p)
            if len(pts) == n:
                return pts
    raise RuntimeError("placement des cibles impossible (zone trop petite ?)")
```

(+ `import math`). Dans `PinballSim` :

```python
    def __init__(self, config: BoardConfig, rng: np.random.Generator,
                 targets: list[tuple[float, float]] | None = None):
        ...  # inchangé jusqu'après _add_ball()
        self.targets = [tuple(t) for t in (targets or [])]
        self.target_alive = [True] * len(self.targets)
        self._target_shapes: list[pymunk.Shape] = []
        self._pending_hits: list[int] = []
        for (x, y) in self.targets:
            s = pymunk.Circle(self.space.static_body, config.target_radius,
                              offset=(x, y))
            s.elasticity = config.target_elasticity
            s.friction = config.wall_friction
            self.space.add(s)
            self._target_shapes.append(s)
```

Dans `step_control` : `self._check_target_hits()` après le plafond de vitesse
de CHAQUE sous-pas, et `self._flush_dead_targets()` en TOUTE FIN de méthode
(après la boucle des sous-pas). Initialiser aussi `self._dead_to_remove: list[int] = []`
dans `__init__`. Les trois méthodes :

```python
    def _check_target_hits(self) -> None:
        # par SOUS-PAS : au pas de contrôle, une balle rapide (≤147 u / pas)
        # traverserait la zone de contact (~84 u) entre deux vérifications.
        # La cible détectée reste SOLIDE jusqu'à la fin du pas de contrôle
        # (la détection à marge +2 précède le contact : le rebond doit avoir
        # lieu), puis _flush_dead_targets la retire — jamais de mur invisible.
        if not self.targets:
            return
        cfg = self.config
        bx, by = self.ball_pos
        reach = cfg.ball_radius + cfg.target_radius + cfg.target_hit_margin
        for i, (tx, ty) in enumerate(self.targets):
            if self.target_alive[i] and math.hypot(bx - tx, by - ty) <= reach:
                self.target_alive[i] = False
                self._pending_hits.append(i)
                self._dead_to_remove.append(i)

    def _flush_dead_targets(self) -> None:
        for i in self._dead_to_remove:
            self.space.remove(self._target_shapes[i])
        self._dead_to_remove = []

    def consume_hits(self) -> list[int]:
        """Indices des cibles touchées depuis le dernier appel."""
        hits, self._pending_hits = self._pending_hits, []
        return hits
```

- [ ] **Step 4: Suite complète** — `pytest` → 57 + 6 = 63 PASS.
- [ ] **Step 5: Commit** — `feat: cibles aléatoires dans le simulateur (placement sûr, contact par sous-pas)`

---

### Task 22: Les cibles au rendu

**Files:**
- Modify: `pinball/render.py`
- Test: `tests/test_render.py` (ajouts)

**Interfaces:** les cibles VIVANTES sont dessinées : disque plein gris 110
(`COLOR_TARGET = 110`) dans `render_frame` ; disque cyan (80, 200, 220) dans
`render_debug`. Une cible morte disparaît des deux rendus. Rayon écran =
`cfg.target_radius * sx` (pas de plancher : ~3 px, déjà visible — validé).

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_render.py` :

```python
def test_targets_rendered_and_disappear():
    from pinball.config import hard_board
    from pinball.sim import PinballSim
    cfg = hard_board()
    sim = PinballSim(cfg, np.random.default_rng(0), targets=[(270.0, 600.0)])
    sim.ball.position = (100.0, 850.0)   # balle loin de la cible
    f1 = render_frame(sim)
    assert (f1 == 110).sum() >= 12       # le disque gris est là (~3 px de rayon)
    sim.target_alive[0] = False          # cible éteinte
    f2 = render_frame(sim)
    assert (f2 == 110).sum() == 0


def test_dead_target_leaves_debug_render():
    from pinball.config import hard_board
    from pinball.sim import PinballSim
    cfg = hard_board()
    sim = PinballSim(cfg, np.random.default_rng(0), targets=[(270.0, 600.0)])
    px1 = np.asarray(render_debug(sim))
    sim.target_alive[0] = False
    px2 = np.asarray(render_debug(sim))
    assert not np.array_equal(px1, px2)
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** — dans `_draw_board`, après les
slingshots et AVANT les flippers (les flippers passent devant), ajouter le
paramètre `target_color` à la signature et :

```python
    for i, (tx, ty) in enumerate(sim.targets):
        if sim.target_alive[i]:
            cx, cy = pt((tx, ty))
            tr = sim.config.target_radius * (pt((1, 0))[0] - pt((0, 0))[0])
            d.ellipse([cx - tr, cy - tr, cx + tr, cy + tr], fill=target_color)
```

`COLOR_TARGET = 110` en constante de module ; `render_frame` passe
`COLOR_TARGET`, `render_debug` passe `(80, 200, 220)`.

- [ ] **Step 4:** `pytest` → 63 + 2 = 65 PASS. **Step 5: Commit** —
  `feat: rendu des cibles (gris 150 / cyan debug), disparition à l'extinction`

---

### Task 23: L'environnement — tirage, victoire, info

**Files:**
- Modify: `pinball/env.py`
- Test: `tests/test_env.py` (ajouts)

**Interfaces:**
- `reset()` : si `n_targets_range != (0, 0)`, tire `n` (bornes incluses, RNG de
  l'env) puis les positions via `sample_target_positions`, et construit
  `PinballSim(..., targets=positions)`.
- `step()` : `hits = self.sim.consume_hits()` ; cumul `self._targets_hit` ;
  `completed = targets_total > 0 and targets_hit == targets_total` ;
  `done = ball_lost or stuck or completed or steps >= max`.
- `info` gagne : `targets_total: int`, `targets_hit: int`, `hit_now: int`
  (contacts ce pas), `completed: bool`. Clés existantes inchangées.

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_env.py` :

```python
def test_default_table_has_no_targets():
    env = PinballEnv(seed=0)
    env.reset()
    _, info = env.step(0)
    assert info["targets_total"] == 0 and not info["completed"]


def test_targets_sampled_and_deterministic():
    from pinball.config import hard_board
    env = PinballEnv(hard_board(), seed=3)
    env.reset()
    a = list(env.sim.targets)
    assert 1 <= len(a) <= 3
    env2 = PinballEnv(hard_board(), seed=3)
    env2.reset()
    assert list(env2.sim.targets) == a       # même seed -> mêmes cibles
    env3 = PinballEnv(hard_board(), seed=4)
    env3.reset()
    assert list(env3.sim.targets) != a       # seed différente -> différentes


def test_completion_ends_episode():
    # une seule cible, balle téléportée dessus : l'épisode finit en victoire
    from pinball.config import hard_board
    env = PinballEnv(hard_board(), seed=0)
    env.reset()
    tx, ty = env.sim.targets[0]
    for i in range(1, len(env.sim.targets)):        # éteindre les autres
        env.sim.target_alive[i] = False
        env.sim.space.remove(env.sim._target_shapes[i])
        env._targets_hit += 1
    env.sim.ball.position = (tx, ty + 60)
    env.sim.ball.velocity = (0.0, -300.0)
    for _ in range(30):
        _, info = env.step(0)
        if info["done"]:
            break
    assert info["completed"] and info["done"]
    assert not info["ball_lost"]
    assert info["targets_hit"] == info["targets_total"]
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** — dans `reset()` :

```python
        lo, hi = self.config.n_targets_range
        targets = None
        if hi > 0:
            n = int(self._rng.integers(lo, hi + 1))
            targets = sample_target_positions(self.config, self._rng, n)
        self.sim = PinballSim(self.config, self._rng, targets=targets)
        self._targets_hit = 0
```

dans `step()` (après `step_control`) :

```python
        hits = self.sim.consume_hits()
        self._targets_hit += len(hits)
        targets_total = len(self.sim.targets)
        completed = targets_total > 0 and self._targets_hit == targets_total
```

`done` gagne `or completed` ; `info` gagne les 4 clés. Imports :
`from .sim import PinballSim, sample_target_positions`.

- [ ] **Step 4:** `pytest` → 65 + 3 = 68 PASS (les 8 tests env existants, sur la
  table par défaut sans cibles, ne changent pas). **Step 5: Commit** —
  `feat: cibles tirées au reset, fin d'épisode en victoire, info enrichi`

---

### Task 24: Le dataset enregistre contacts et fins d'épisode

**Files:**
- Modify: `pinball/collect.py`
- Test: `tests/test_collect.py` (ajouts)

**Interfaces (format additif) :** chaque épisode gagne
`hits: (T,) uint8` (nb de contacts au pas t+1), `targets_total: int`,
`completed: bool`, `stuck: bool`. Shards : + `hits` (ΣTᵢ,) uint8 (aligné sur
`actions` et `action_counts`), + `targets_total` (n,) int64, `completed` (n,)
bool, `stuck` (n,) bool. `load_episodes` restitue les nouvelles clés.

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_collect.py` :

```python
def test_episode_records_hits_and_endings(tmp_path):
    from pinball.config import hard_board
    env = PinballEnv(hard_board(), seed=0)
    policy = StickyRandomPolicy(np.random.default_rng(0))
    collect_dataset(env, policy, n_transitions=600, out_dir=tmp_path)
    episodes = load_episodes(tmp_path)
    for ep in episodes:
        T = len(ep["actions"])
        assert ep["hits"].shape == (T,) and ep["hits"].dtype == np.uint8
        assert 1 <= int(ep["targets_total"]) <= 3
        assert isinstance(bool(ep["completed"]), bool)
        assert isinstance(bool(ep["stuck"]), bool)
        # cohérence : un épisode complété a touché toutes ses cibles
        if ep["completed"]:
            assert ep["hits"].sum() >= ep["targets_total"]
    # le jeu aléatoire touche des cibles de temps en temps (mesuré ~21 %)
    assert any(ep["hits"].sum() > 0 for ep in episodes)


def test_default_table_records_zero_targets(tmp_path):
    env = PinballEnv(BoardConfig(max_episode_steps=40), seed=0)
    policy = StickyRandomPolicy(np.random.default_rng(0))
    collect_dataset(env, policy, n_transitions=100, out_dir=tmp_path)
    episodes = load_episodes(tmp_path)
    assert all(int(ep["targets_total"]) == 0 for ep in episodes)
    assert all(not ep["completed"] for ep in episodes)
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** — dans `_play_episode` :
collecter `hits_list.append(info["hit_now"])` à chaque pas ; au retour, ajouter

```python
        "hits": np.asarray(hits_list, dtype=np.uint8),
        "targets_total": int(info["targets_total"]),
        "completed": bool(info["completed"]),
        "stuck": bool(info["stuck"]),
```

Dans `_write_shard` : `hits=np.concatenate(...)` (offsets = ceux des actions),
`targets_total=np.asarray([...], dtype=np.int64)`, `completed=...`, `stuck=...`.
Dans `load_episodes` : lire les quatre tableaux UNE fois (même hoisting
anti-OOM), découper `hits` avec `a_ofs/action_counts`, restituer les scalaires.

- [ ] **Step 4:** `pytest` → 68 + 2 = 70 PASS. **Step 5: Commit** —
  `feat: shards v2 — contacts par pas, cibles totales, fins completed/stuck`

---

### Task 25: MultiLabelDataset — tous les labels d'un coup

**Files:**
- Modify: `jepa/data.py`
- Test: `tests/test_data.py` (ajouts)

**Interfaces:**
- `MultiLabelDataset(episodes, k_danger=10, k_target=10, board_size=(540.0, 960.0))`
  — Dataset ; item = dict :
  - `obs` uint8 (2, 64, 64) — stack(f_{t-1}, f_t), t ∈ [1, T] ;
  - `danger` float32 — 1.0 ssi l'épisode finit MAL (`ball_lost` OU `stuck`) ET
    t ≥ T − k_danger + 1. Fins en victoire ou timeout → 0 partout ;
  - `height` float32 — `ball_pos[t, 1] / board_size[1]` ;
  - `pos` float32 (2,) — `ball_pos[t] / board_size` ;
  - `target` float32 — 1.0 ssi `hits[t : t + k_target].sum() > 0` (un contact
    dans les k prochains pas ; épisodes sans clé `hits` → 0 partout, pour la
    rétro-compatibilité avec les vieux datasets).
- `DangerDataset` existant : INCHANGÉ (les anciens tests restent verts).

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_data.py` :

```python
def labeled_episode(T, ball_lost=False, stuck=False, completed=False,
                    hit_at=(), seed=0):
    ep = fake_episode(T, ball_lost, seed)
    ep["stuck"] = stuck
    ep["completed"] = completed
    ep["targets_total"] = 2
    hits = np.zeros(T, dtype=np.uint8)
    for h in hit_at:
        hits[h] = 1
    ep["hits"] = hits
    return ep


def test_multilabel_danger_honest():
    from jepa.data import MultiLabelDataset
    # perdu -> queue dangereuse ; stuck -> queue dangereuse ;
    # complété -> AUCUN danger ; timeout -> aucun danger
    for kw, expected_tail in [(dict(ball_lost=True), 10),
                              (dict(stuck=True), 10),
                              (dict(completed=True), 0),
                              (dict(), 0)]:
        ds = MultiLabelDataset([labeled_episode(30, **kw)], k_danger=10)
        labels = np.array([ds[i]["danger"].item() for i in range(len(ds))])
        assert labels.sum() == expected_tail, kw


def test_multilabel_height_and_pos():
    from jepa.data import MultiLabelDataset
    ep = labeled_episode(20)
    ds = MultiLabelDataset([ep], board_size=(540.0, 960.0))
    item = ds[0]                       # t = 1
    assert abs(item["height"].item() - ep["ball_pos"][1, 1] / 960.0) < 1e-6
    assert np.allclose(item["pos"].numpy(),
                       ep["ball_pos"][1] / np.array([540.0, 960.0]),
                       atol=1e-6)
    assert 0.0 <= item["height"].item() <= 1.0


def test_multilabel_target_window():
    from jepa.data import MultiLabelDataset
    # contact au pas 15 (hits[14]=1) : positifs pour t = 5..14 (k=10)
    ds = MultiLabelDataset([labeled_episode(30, hit_at=(14,))], k_target=10)
    labels = np.array([ds[i]["target"].item() for i in range(len(ds))])
    assert labels.sum() == 10
    assert labels[4] == 1.0 and labels[13] == 1.0   # t=5 et t=14 (index t-1)
    assert labels[3] == 0.0 and labels[14] == 0.0


def test_multilabel_backward_compatible_without_hits():
    from jepa.data import MultiLabelDataset
    ds = MultiLabelDataset([fake_episode(20, True)])   # pas de clés v2
    item = ds[0]
    assert item["target"].item() == 0.0
    assert item["danger"].item() in (0.0, 1.0)
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** dans `jepa/data.py` :

```python
class MultiLabelDataset(Dataset):
    """Observations + tous les labels d'objectif, fabriqués depuis `info`.

    - danger HONNÊTE : seules les fins PERDANTES (ball_lost ou stuck) marquent
      leur queue — une victoire n'est pas un danger (leçon du reward hacking
      de l'itération 1 : « éviter de perdre » sans nuance récompensait le
      piégeage de la balle) ;
    - hauteur et position normalisées (objectif continu + sonde de visu) ;
    - cible : un contact aura-t-il lieu dans les k prochains pas ?
    """

    def __init__(self, episodes: list[dict], k_danger: int = 10,
                 k_target: int = 10,
                 board_size: tuple[float, float] = (540.0, 960.0)):
        self.episodes = episodes
        self.k_danger = k_danger
        self.k_target = k_target
        self.board = np.asarray(board_size, dtype=np.float32)
        self.index: list[tuple[int, int]] = []
        for e, ep in enumerate(episodes):
            for t in range(1, len(ep["actions"]) + 1):
                self.index.append((e, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        T = len(ep["actions"])
        end_bad = bool(ep["ball_lost"]) or bool(ep.get("stuck", False))
        danger = end_bad and t >= T - self.k_danger + 1
        hits = ep.get("hits")
        target = bool(hits is not None
                      and hits[t:t + self.k_target].sum() > 0)
        pos = ep["ball_pos"][t] / self.board
        return {
            "obs": torch.from_numpy(
                np.ascontiguousarray(ep["frames"][t - 1:t + 1])),
            "danger": torch.tensor(1.0 if danger else 0.0),
            "height": torch.tensor(float(pos[1])),
            "pos": torch.from_numpy(pos.astype(np.float32)),
            "target": torch.tensor(1.0 if target else 0.0),
        }
```

- [ ] **Step 4:** `pytest` → 70 + 4 = 74 PASS. **Step 5: Commit** —
  `feat: MultiLabelDataset — danger honnête, hauteur, position, contact imminent`

---

### Task 26: Les têtes d'objectif

**Files:**
- Modify: `jepa/heads.py`
- Test: `tests/test_heads.py` (ajouts)

**Interfaces:**
- `HeightHead(z_dim=256)` — MLP 256→128→1, sortie **sigmoïde** ∈ (0, 1),
  `forward(z) -> (B,)`.
- `TargetHead(z_dim=256)` — MLP 256→128→1, LOGITS (B,) (comme DangerHead).
- `PositionProbe(z_dim=256)` — MLP 256→128→2, sortie sigmoïde, `forward(z) -> (B, 2)`.
- `train_objective_heads(jepa, episodes, k_danger=10, k_target=10, epochs=3,
  batch_size=512, lr=1e-3, val_fraction=0.1, device=None)
  -> tuple[dict[str, nn.Module], dict[str, float]]` — encode UNE fois tout le
  dataset (`encode_target`, split par épisode comme `train_danger_head`, même
  garde ≥2 épisodes, même clamp de batch), puis entraîne les 4 têtes sur les
  latents partagés : danger (BCE pos_weight), target (BCE pos_weight, saute
  l'entraînement si aucun positif — metrics `auc_target = nan`), height (MSE),
  pos (MSE). Retour : (`{"danger", "height", "target", "pos"}`,
  `{"auc_danger", "auc_target", "mae_height", "mae_pos"}`) — MAE sur la
  validation, en unités normalisées.
- `train_danger_head` existant : INCHANGÉ.

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_heads.py` :

```python
def test_new_heads_shapes():
    from jepa.heads import HeightHead, PositionProbe, TargetHead
    z = torch.randn(5, 256)
    h = HeightHead()(z)
    assert h.shape == (5,) and (h >= 0).all() and (h <= 1).all()
    assert TargetHead()(z).shape == (5,)
    p = PositionProbe()(z)
    assert p.shape == (5, 2) and (p >= 0).all() and (p <= 1).all()


def _v2_episode(T, ball_lost, seed):
    # bande lumineuse qui SUIT la hauteur réelle (hauteur/position apprenables)
    # + marqueur de coin distinct pendant la queue dangereuse (séparabilité
    # du danger garantie même avec un encodeur aléatoire — motif V1)
    rng = np.random.default_rng(seed)
    frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
    ys = (np.linspace(880.0, 120.0, T + 1) if ball_lost
          else np.linspace(120.0, 880.0, T + 1))
    ball_pos = np.stack([np.full(T + 1, 270.0), ys], axis=1).astype(np.float32)
    for t in range(T + 1):
        row = int(63 - ys[t] / 960.0 * 63)
        frames[t, max(0, row - 2):row + 3, 20:44] = 255
        if ball_lost and t > T - 10:
            frames[t, 55:63, 55:63] = 200
    hits = np.zeros(T, dtype=np.uint8)
    hits[T // 2] = 0 if ball_lost else 1
    return {"frames": frames, "actions": rng.integers(0, 4, (T,)).astype(np.int64),
            "ball_pos": ball_pos, "ball_lost": ball_lost, "stuck": False,
            "completed": not ball_lost, "targets_total": 1, "hits": hits}


def test_train_objective_heads_learns():
    from jepa.heads import train_objective_heads
    from jepa.model import JEPA
    torch.manual_seed(0)
    eps = [_v2_episode(30, s % 2 == 0, s) for s in range(20)]
    heads, metrics = train_objective_heads(JEPA(), eps, epochs=10,
                                           batch_size=64, device="cpu")
    assert set(heads) == {"danger", "height", "target", "pos"}
    assert metrics["auc_danger"] > 0.9          # signal séparable (cf. V1)
    assert metrics["mae_height"] < 0.2          # la hauteur se lit dans l'image
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** — `HeightHead`/`PositionProbe` :
même gabarit que `DangerHead` avec `torch.sigmoid` final (et `Linear(128, 2)`
pour la sonde) ; `TargetHead` = copie structurelle de `DangerHead`.
`train_objective_heads` : réutiliser le motif de `train_danger_head`
(split par épisode, garde ≥2 épisodes, `_encode_dataset` généralisé qui
retourne `(z, {clé: tenseur de labels})` depuis `MultiLabelDataset`, clamp du
batch) ; une boucle d'epochs unique qui optimise les 4 têtes côte à côte
(4 optimiseurs AdamW, 4 pertes indépendantes sur le même mini-batch de
latents) ; AUC par la fonction `auc` existante, MAE = moyenne des |écarts|
sur la validation ; imprimer un résumé en français.

- [ ] **Step 4:** `pytest` → 74 + 2 = 76 PASS. **Step 5: Commit** —
  `feat: têtes hauteur/cible/position + entraînement groupé sur latents partagés`

---

### Task 27: Planification multi-objectifs

**Files:**
- Modify: `jepa/planner.py`
- Test: `tests/test_planner.py` (ajouts)

**Interfaces:** `MPCPlanner.__init__` gagne les kwargs OPTIONNELS
`height_head=None, target_head=None, w_danger=1.0, w_height=0.5, w_target=2.0`
(après `seed`). Coût par pas imaginé :
`w_danger·σ(danger(z)) − w_height·height(z) − w_target·σ(target(z))`
(chaque terme seulement si sa tête est fournie ; les têtes fournies sont
déplacées sur le device et passées en eval, comme `danger_head`).
Rétro-compatibilité : tous les appels existants (danger seul) inchangés.

- [ ] **Step 1: Tests qui échouent** — ajouter à `tests/test_planner.py` :

```python
class ZeroDanger(nn.Module):
    def forward(self, z):
        return torch.zeros(z.shape[0])


class OracleHeight(nn.Module):
    """hauteur = sigmoïde de z[1] : l'action qui monte z[1] est la meilleure."""

    def forward(self, z):
        return torch.sigmoid(z[..., 1])


class AxisPredictor(nn.Module):
    """action 1 augmente z[1] ; action 3 augmente z[2] ; les autres neutres."""

    def forward(self, z, a):
        out = z.clone()
        out[..., 1] = out[..., 1] + torch.where(a == 1, 1.0, 0.0)
        out[..., 2] = out[..., 2] + torch.where(a == 3, 1.0, 0.0)
        return out


class OracleTarget(nn.Module):
    def forward(self, z):
        return z[..., 2]          # logits : hauts si z[2] grand


def make_multi_planner(**kwargs):
    torch.manual_seed(0)   # z0 déterministe : la course de marges entre les
    jepa = JEPA()          # deux sigmoïdes dépend du tirage de l'encodeur
    jepa.predictor = AxisPredictor()
    return MPCPlanner(jepa, ZeroDanger(), device="cpu", n_candidates=0, **kwargs)


def test_height_head_steers_planner():
    planner = make_multi_planner(height_head=OracleHeight(), w_height=1.0)
    obs = np.zeros((2, 64, 64), dtype=np.uint8)
    assert planner.plan(obs) == 1        # monter z[1] = monter la hauteur


def test_target_head_outweighs_height():
    planner = make_multi_planner(height_head=OracleHeight(), w_height=0.5,
                                 target_head=OracleTarget(), w_target=2.0)
    obs = np.zeros((2, 64, 64), dtype=np.uint8)
    assert planner.plan(obs) == 3        # w_target > w_height : viser la cible


def test_danger_only_still_works():
    # l'API historique (danger seul) reste intacte
    planner = make_planner(n_candidates=0)
    obs = np.zeros((2, 64, 64), dtype=np.uint8)
    assert planner.plan(obs) == 2
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** — stocker têtes et poids ;
dans la boucle de `plan` :

```python
            cost += self.w_danger * torch.sigmoid(self.danger_head(z))
            if self.height_head is not None:
                cost -= self.w_height * self.height_head(z)
            if self.target_head is not None:
                cost -= self.w_target * torch.sigmoid(self.target_head(z))
```

- [ ] **Step 4:** `pytest` → 76 + 3 = 79 PASS. **Step 5: Commit** —
  `feat: coût MPC multi-objectifs (danger, hauteur, cible), rétro-compatible`

---

### Task 28: Décodeur d'imagination et superpositions prédit/réel

**Files:**
- Create: `jepa/decoder.py`, `jepa/viz.py`
- Test: `tests/test_viz.py` (nouveau)

**Interfaces:**
- `Decoder(z_dim=256)` — `forward(z (B, 256)) -> (B, 64, 64)` ∈ (0, 1)
  (fc 256→4096, reshape (256, 4, 4), 4 ConvTranspose2d stride 2 : 256→128→64→32→1,
  SiLU entre, sigmoïde finale).
- `train_decoder(jepa, episodes, epochs=3, batch_size=256, lr=1e-3, device=None)
  -> Decoder` — cible = frame PRÉSENTE (`obs[:, 1]/255`), latents
  `encode_target` SANS gradient (le décodeur ne touche jamais au JEPA — il ne
  sert qu'à la visualisation), garde anti-DataLoader-vide, print par epoch.
- `rollout_latents(jepa, ep, t0, k=8, device="cpu") -> (k, 256)` — encode
  o_{t0} (online) puis déroule le prédicteur avec les actions RÉELLEMENT
  jouées `ep["actions"][t0:t0+k]`.
- `trajectory_overlay(jepa, pos_probe, ep, t0, k=8, upscale=6,
  board_size=(540.0, 960.0), device="cpu") -> PIL.Image` — frame réelle
  o_{t0+k} agrandie, annotée : positions RÉELLES t0+1..t0+k (cercles blancs)
  et positions PRÉDITES (sonde sur les ẑ ; croix colorées, dégradé jaune →
  violet selon l'horizon).
- `imagination_strip(jepa, decoder, ep, t0, k=8, upscale=4, device="cpu")
  -> PIL.Image` — planche 3 lignes × k colonnes : RÉEL / IMAGINÉ (décodé de
  ẑ) / SUPERPOSITION RGB (rouge = imaginé, vert = réel → jaune = accord).

- [ ] **Step 1: Tests qui échouent** — créer `tests/test_viz.py` :

```python
import numpy as np
import torch
from jepa.decoder import Decoder, train_decoder
from jepa.model import JEPA
from jepa.viz import imagination_strip, rollout_latents, trajectory_overlay
from jepa.heads import PositionProbe


def fake_ep(T=30, seed=0):
    rng = np.random.default_rng(seed)
    return {"frames": rng.integers(0, 255, (T + 1, 64, 64), dtype=np.uint8),
            "actions": rng.integers(0, 4, (T,)).astype(np.int64),
            "ball_pos": rng.uniform(50, 900, (T + 1, 2)).astype(np.float32),
            "ball_lost": True}


def test_decoder_shapes():
    out = Decoder()(torch.randn(3, 256))
    assert out.shape == (3, 64, 64)
    assert (out >= 0).all() and (out <= 1).all()


def test_train_decoder_reduces_loss(capsys):
    torch.manual_seed(0)
    eps = [fake_ep(T=20, seed=s) for s in range(2)]
    dec = train_decoder(JEPA(), eps, epochs=2, batch_size=16, device="cpu")
    assert not dec.training
    lines = [l for l in capsys.readouterr().out.splitlines() if "mse=" in l]
    first = float(lines[0].split("mse=")[1])
    last = float(lines[-1].split("mse=")[1])
    assert last <= first


def test_rollout_latents_shape():
    zs = rollout_latents(JEPA(), fake_ep(), t0=2, k=8)
    assert zs.shape == (8, 256)


def test_decoder_learns_bright_pixels():
    # ~95-97 % de pixels noirs : une MSE nue apprend « tout noir » (mesuré :
    # 0 pixel > 0.5 en sortie). La perte pondérée doit reproduire la bille.
    torch.manual_seed(0)

    def sparse_ep(T=25, seed=0):
        rng = np.random.default_rng(seed)
        frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
        frames[:, 60:63, :] = 90
        for t in range(T + 1):
            x = 8 + (t * 2) % 48
            frames[t, 20:25, x:x + 5] = 255
        return {"frames": frames,
                "actions": rng.integers(0, 4, (T,)).astype(np.int64),
                "ball_pos": np.zeros((T + 1, 2), dtype=np.float32),
                "ball_lost": True}

    from jepa.data import MultiLabelDataset
    eps = [sparse_ep(seed=s) for s in range(2)]
    jepa = JEPA()
    dec = train_decoder(jepa, eps, epochs=5, batch_size=16, device="cpu")
    ds = MultiLabelDataset(eps)
    obs = torch.stack([ds[i]["obs"] for i in range(8)])
    with torch.no_grad():
        out = dec(jepa.encode_target(obs))
    assert (out > 0.5).sum() > 0      # la bille brillante est reproduite


def test_overlays_return_images():
    ep = fake_ep()
    img1 = trajectory_overlay(JEPA(), PositionProbe(), ep, t0=2, k=8, upscale=4)
    assert img1.size == (256, 256) and img1.mode == "RGB"
    img2 = imagination_strip(JEPA(), Decoder(), ep, t0=2, k=8, upscale=2)
    assert img2.size == (8 * 128, 3 * 128) and img2.mode == "RGB"
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** `jepa/decoder.py` :

```python
"""Décodeur d'imagination : latent → image, POUR LA VISUALISATION UNIQUEMENT.

JEPA n'apprend jamais en pixels — c'est son principe. Mais pour VOIR ce que
le modèle imagine, on entraîne À PART un petit décodeur z̄ → image, encodeur
gelé : il ne modifie rien au world model, il le traduit en images.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import MultiLabelDataset


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(-1, 256, 4, 4)
        return torch.sigmoid(self.deconvs(h)).squeeze(1)


def train_decoder(jepa, episodes, epochs: int = 3, batch_size: int = 256,
                  lr: float = 1e-3, bright_weight: float = 10.0,
                  device: str | None = None) -> Decoder:
    """Entraîne le décodeur de visualisation (encodeur gelé).

    Piège des images de flipper : ~95 % des pixels sont noirs — une MSE nue
    apprend « tout noir » (mesuré : aucun pixel > 0,5 en sortie) et la
    sigmoïde sature. Chaque pixel est donc pondéré par 1 + bright_weight·cible :
    les pixels allumés (balle, cibles, murs) dominent la perte.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    jepa = jepa.to(dev).eval()
    ds = MultiLabelDataset(episodes)
    batch_size = min(batch_size, len(ds))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    if len(dl) == 0:
        raise ValueError("pas assez de données pour un batch de décodeur")
    dec = Decoder().to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr)
    for epoch in range(epochs):
        total, nb = 0.0, 0
        for batch in dl:
            obs = batch["obs"].to(dev)
            with torch.no_grad():
                z = jepa.encode_target(obs)
            target = obs[:, 1].float() / 255.0
            weights = 1.0 + bright_weight * target
            loss = (weights * (dec(z) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"décodeur epoch {epoch + 1}/{epochs}  wmse={total / max(nb, 1):.5f}")
    return dec.eval()
```

et `jepa/viz.py` :

```python
"""Visualiser les prédictions du world model, superposées au réel."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

# dégradé d'horizon : jaune (t+1) → violet (t+8)
HORIZON_COLORS = [(255, 220, 60), (255, 170, 60), (255, 120, 70), (255, 80, 90),
                  (230, 60, 120), (200, 50, 160), (160, 50, 200), (120, 60, 230)]


def _obs_at(ep: dict, t: int) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(ep["frames"][t - 1:t + 1]))


@torch.no_grad()
def rollout_latents(jepa, ep: dict, t0: int, k: int = 8,
                    device: str = "cpu") -> torch.Tensor:
    """Encode o_{t0} (online) puis déroule le prédicteur avec les actions
    réellement jouées — les ẑ retournés vivent dans l'espace cible."""
    jepa = jepa.to(device).eval()
    z = jepa.encode(_obs_at(ep, t0).unsqueeze(0).to(device))
    zs = []
    for j in range(k):
        a = torch.tensor([int(ep["actions"][t0 + j])], device=device)
        z = jepa.predictor(z, a)
        zs.append(z.squeeze(0).clone())
    return torch.stack(zs)


@torch.no_grad()
def trajectory_overlay(jepa, pos_probe, ep: dict, t0: int, k: int = 8,
                       upscale: int = 6,
                       board_size: tuple[float, float] = (540.0, 960.0),
                       device: str = "cpu") -> Image.Image:
    """Superposition « prédit vs réel » : la frame réelle finale, la vraie
    trajectoire (cercles blancs) et les positions imaginées (croix colorées)."""
    zs = rollout_latents(jepa, ep, t0, k, device)
    pred = pos_probe.to(device).eval()(zs).cpu().numpy()
    size = 64 * upscale
    img = Image.fromarray(ep["frames"][t0 + k]).convert("RGB").resize(
        (size, size), Image.NEAREST)
    d = ImageDraw.Draw(img)
    w, h = board_size
    for j in range(k):
        x, y = ep["ball_pos"][t0 + j + 1]
        px, py = x / w * size, (1 - y / h) * size
        d.ellipse([px - 3, py - 3, px + 3, py + 3], outline=(255, 255, 255))
    for j in range(k):
        px, py = pred[j, 0] * size, (1 - pred[j, 1]) * size
        c = HORIZON_COLORS[j % len(HORIZON_COLORS)]
        d.line([px - 5, py, px + 5, py], fill=c, width=2)
        d.line([px, py - 5, px, py + 5], fill=c, width=2)
    return img


@torch.no_grad()
def imagination_strip(jepa, decoder, ep: dict, t0: int, k: int = 8,
                      upscale: int = 4, device: str = "cpu") -> Image.Image:
    """Planche 3×k : RÉEL / IMAGINÉ (décodé des ẑ) / SUPERPOSITION
    (rouge = imaginé, vert = réel — jaune là où ils s'accordent)."""
    zs = rollout_latents(jepa, ep, t0, k, device)
    decoded = decoder.to(device).eval()(zs).cpu().numpy()
    cell = 64 * upscale
    strip = Image.new("RGB", (k * cell, 3 * cell), (15, 15, 25))
    for j in range(k):
        real = ep["frames"][t0 + j + 1].astype(np.float32) / 255.0
        imag = decoded[j]
        rows = [np.stack([real] * 3, axis=-1),
                np.stack([imag] * 3, axis=-1),
                np.stack([imag, real, np.zeros_like(real)], axis=-1)]
        for r, arr in enumerate(rows):
            im = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (cell, cell), Image.NEAREST)
            strip.paste(im, (j * cell, r * cell))
    return strip
```

- [ ] **Step 4:** `pytest` → 79 + 5 = 84 PASS. **Step 5: Commit** —
  `feat: décodeur d'imagination + superpositions trajectoire/imagination`

---

### Task 29: L'évaluation parle en victoires

**Files:**
- Modify: `jepa/eval.py`
- Test: `tests/test_eval.py` (nouveau — couvre au passage les clés nudges,
  suivi différé de la V1.5)

**Interfaces (additif) :**
- `run_episode` gagne : `completed: bool`, `targets_total: int`,
  `targets_hit: int`, `mean_height: float` (moyenne des `ball_pos.y /
  env.config.height` sur les pas de l'épisode).
- `evaluate` gagne : `completion_rate` (part d'épisodes gagnés),
  `mean_completion_s` (durée moyenne des seuls épisodes gagnés ; NaN si aucun),
  `targets_hit_rate` (cibles touchées / cibles disponibles ; NaN si aucune),
  `mean_height` (moyenne des moyennes par épisode).

- [ ] **Step 1: Tests qui échouent** — créer `tests/test_eval.py` :

```python
import numpy as np
from pinball.collect import StickyRandomPolicy
from pinball.config import BoardConfig, hard_board
from pinball.env import PinballEnv
from jepa.eval import AlwaysPressed, evaluate, run_episode


def test_run_episode_new_keys():
    env = PinballEnv(hard_board(), seed=0)
    r = run_episode(env, StickyRandomPolicy(np.random.default_rng(0)), seed=0)
    assert {"steps", "ball_lost", "stuck", "nudges", "completed",
            "targets_total", "targets_hit", "mean_height"} <= r.keys()
    assert 0.0 < r["mean_height"] < 1.0
    assert 0 <= r["targets_hit"] <= r["targets_total"] <= 3


def test_evaluate_new_keys_and_ranges():
    env = PinballEnv(hard_board(), seed=0)
    r = evaluate(env, StickyRandomPolicy(np.random.default_rng(0)), n_episodes=6)
    assert 0.0 <= r["completion_rate"] <= 1.0
    assert 0.0 <= r["targets_hit_rate"] <= 1.0
    assert 0.0 < r["mean_height"] < 1.0
    assert r["mean_nudges"] >= 0.0


def test_evaluate_no_targets_table_gives_nan_rate():
    env = PinballEnv(BoardConfig(max_episode_steps=30), seed=0)
    r = evaluate(env, AlwaysPressed(), n_episodes=3)
    assert r["completion_rate"] == 0.0
    assert np.isnan(r["targets_hit_rate"])
```

- [ ] **Step 2: échec** ; **Step 3: Implémenter** — `run_episode` accumule
`heights.append(info["ball_pos"][1] / env.config.height)` et retourne les
nouvelles clés depuis le dernier `info` ; `evaluate` calcule :

```python
        "completion_rate": float(np.mean([r["completed"] for r in results])),
        "mean_completion_s": float(np.mean([r["steps"] for r in results
                                            if r["completed"]]) / hz)
                             if any(r["completed"] for r in results) else float("nan"),
        "targets_hit_rate": (float(sum(r["targets_hit"] for r in results))
                             / total_targets) if (total_targets := sum(
                                 r["targets_total"] for r in results)) > 0
                            else float("nan"),
        "mean_height": float(np.mean([r["mean_height"] for r in results])),
```

- [ ] **Step 4:** `pytest` → 83 + 3 = 86 PASS. **Step 5: Commit** —
  `feat: éval V2 — taux de victoire, temps de complétion, hauteur moyenne`

---

### Task 30: Notebooks 01-03 — double-mode et table à cibles

**Files:**
- Modify: `notebooks/01_simulateur.py`, `notebooks/02_collecte.py`,
  `notebooks/03_jepa.py` (+ .ipynb régénérés)

Éditions exactes :

**01_simulateur.py**
- Après le markdown de titre, insérer la cellule 1 de §A (installation
  double-mode) — le notebook 01 n'a pas besoin de chemins de données.
- Retirer du markdown de titre les deux lignes « Exécution locale : ... voir le
  README. » (remplacées par le double-mode).
- La cellule `large = BoardConfig(drain_gap=120.0, flipper_length=90.0)` devient :

```python
from pinball.config import hard_board
hard = hard_board()
env2 = PinballEnv(hard, seed=0)
env2.reset()
plt.figure(figsize=(5, 5)); plt.imshow(render_debug(env2.sim)); plt.axis("off")
plt.title("hard_board : la table OFFICIELLE (drain ouvert, flippers courts, cibles)")
plt.show()
print(f"cette partie a {len(env2.sim.targets)} cible(s), placées au hasard")
```

et son markdown au-dessus devient : « Regardons la table par défaut, puis LA
table officielle des expériences : `hard_board()` — drain ouvert, flippers
courts, et 1 à 3 cibles placées au hasard à chaque partie (les plots ronds). »
- Dans « À retenir », ajouter : « - Les CIBLES (plots gris) changent de place à
  chaque partie : impossible de jouer sans regarder l'image. »

**02_collecte.py**
- Remplacer les 2 premières cellules de code par §A puis
  `DATA_DIR = ROOT / "data/targets_v1"` + `print("Dataset →", DATA_DIR)`.
- `env = PinballEnv(hard_board(), seed=42)` : inchangé (déjà correct).
- Dans l'intro, après la ligne « ... les stratégies aveugles ne survivent pas
  ici. », ajouter : « Nouveauté V2 : chaque partie a 1 à 3 CIBLES aléatoires —
  les toucher toutes gagne la partie. Le hasard en touche ~10 % : assez pour
  apprendre ce qu'est un contact. »
- Dans la cellule de contrôle qualité, ajouter à la fin :

```python
hits_eps = np.array([ep["hits"].sum() for ep in episodes])
wins = np.array([ep["completed"] for ep in episodes])
print(f"épisodes avec ≥1 contact de cible : {100*(hits_eps>0).mean():.0f} %")
print(f"victoires chanceuses : {100*wins.mean():.1f} %")
```

**03_jepa.py**
- Remplacer les 2 premières cellules de code par §A puis :

```python
import torch
DATA_DIR, CKPT_DIR = ROOT / "data/targets_v1", ROOT / "checkpoints_targets"
print("device :", "cuda" if torch.cuda.is_available() else
      "cpu (sur Colab : Exécution → Modifier le type d'exécution → T4)")
```

- Le markdown « Entraînement » redevient : « ~20-40 min sur T4 (1-2 h sur CPU).
  Le checkpoint est écrit à CHAQUE epoch : une déconnexion Colab ou une
  interruption locale ne coûte que quelques minutes — relancer la cellule
  reprend automatiquement. »

Vérifications : `jupytext --to ipynb notebooks/01_simulateur.py notebooks/02_collecte.py notebooks/03_jepa.py` ;
`grep -L "IN_COLAB" notebooks/0[123]*.py` vide ; suite pytest inchangée (86 PASS).

Commit : `docs: notebooks 01-03 — double-mode Colab/local, table à cibles, chemins targets_v1`

---

### Task 31: Notebook 04 — têtes multiples, victoire pour métrique reine

**Files:**
- Modify: `notebooks/04_controle.py` (+ .ipynb)

Restructuration (les cellules non mentionnées restent) :

1. **En-tête** : §A + `DATA_DIR, CKPT_DIR = ROOT / "data/targets_v1", ROOT / "checkpoints_targets"`.
2. **Titre/intro** : « ...il faut maintenant lui dire quoi éviter ET quoi
   chercher : le danger, la hauteur, les cibles. » (3 puces : tête danger
   honnête ; tête hauteur ; tête cible — et le coût MPC
   `danger − 0,5·hauteur − 2·cible`).
3. **Cellule têtes** (remplace `train_danger_head`) :

```python
from jepa.heads import auc, train_objective_heads
from jepa.data import MultiLabelDataset
heads, metrics = train_objective_heads(jepa, episodes)
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")
for name, h in heads.items():
    torch.save(h.state_dict(), CKPT_DIR / f"{name}.pt")
```

   suivi de la comparaison AUC/heuristique hauteur EXISTANTE adaptée :
   `ds = MultiLabelDataset(val_eps)` et `labels` lus depuis `ds[i]["danger"]`.
4. **Agent** :

```python
agent = MPCPlanner(jepa, heads["danger"], n_candidates=256,
                   height_head=heads["height"], target_head=heads["target"])
```

5. **Éval** — même boucle 4 politiques mais l'affichage devient :

```python
    r = results[name]
    print(f"{name:16s}: victoire {100*r['completion_rate']:3.0f} %  "
          f"cibles {100*r['targets_hit_rate']:3.0f} %  "
          f"hauteur {r['mean_height']:.2f}  survie {r['survival_s']:.1f} s  "
          f"nudges {r['mean_nudges']:.1f}")
```

   et le graphique : 2 sous-graphiques — à gauche `completion_rate` (barres,
   la MÉTRIQUE REINE), à droite `targets_hit_rate` ; annotation nudges sur le
   premier. Titre : « V2 : gagner la partie (50 épisodes, seeds appariées) ».
   Markdown : critère = l'agent domine NETTEMENT toutes les baselines en taux
   de victoire (les baselines aveugles gagnent ≤ ~2 % par chance).
6. **Bonus « coût imaginé »** : le calcul ajoute les termes hauteur/cible du
   coût (mêmes poids que l'agent) — imprimer les trois composantes par action.
7. **Scénario spec §12** : inchangé.
8. **Checklist finale** : ajouter « 0. `auc_target` bas (< 0,75) ? normal au
   premier tour — le hasard touche peu ; c'est l'itération (notebook 05) qui
   l'améliore » et remplacer la mention V2/bumpers par « la suite : notebook 05
   (itération) puis 06 (visualisation des prédictions) ».

Vérifications : jupytext ; `grep -c "train_danger_head" notebooks/04_controle.py`
= 0 ; suite pytest inchangée.

Commit : `docs: notebook 04 v2 — têtes multiples, taux de victoire en métrique reine`

---

### Task 32: Notebook 05 — itération sur la table à cibles

**Files:**
- Modify: `notebooks/05_iteration.py` (+ .ipynb)

Éditions exactes :
- **En-tête** : §A + :

```python
import shutil
import numpy as np
import torch
DATA_V1, DATA_V2 = ROOT / "data/targets_v1", ROOT / "data/targets_v2"
CKPT_V1, CKPT_V2 = ROOT / "checkpoints_targets", ROOT / "checkpoints_targets_v2"
print("device :", "cuda" if torch.cuda.is_available() else "cpu")
```

- **Intro** : remplacer le paragraphe « Depuis, le projet est passé sur la
  table dure... » par : « La V2 a changé l'objectif : des CIBLES aléatoires à
  toucher (victoire quand toutes le sont) et un bonus de hauteur — le
  reward hacking du piégeage (itération 1 : ~3,4 nudges/épisode pour l'agent
  comme pour « toujours appuyé ») n'a plus de prise. L'itération reste utile
  pour la même raison qu'avant : le modèle n'a vu que du jeu aléatoire — et en
  V2 elle enrichit surtout les exemples de CONTACTS de cibles (le hasard n'en
  touche que ~10 %). Prérequis : notebooks 02 → 04 relancés en V2. »
- **Rechargement agent V1** : charger les 4 têtes :

```python
from jepa.heads import DangerHead, HeightHead, PositionProbe, TargetHead
heads_v1 = {"danger": DangerHead(), "height": HeightHead(),
            "target": TargetHead(), "pos": PositionProbe()}
for name, h in heads_v1.items():
    h.load_state_dict(torch.load(CKPT_V1 / f"{name}.pt", weights_only=True))
    h.eval()
agent_v1 = MPCPlanner(jepa_v1, heads_v1["danger"], n_candidates=256,
                      height_head=heads_v1["height"],
                      target_head=heads_v1["target"])
```

- **Étape 3** : `train_objective_heads(jepa_v2, episodes_mixed)` (sauvegarde
  des 4 têtes dans CKPT_V2) ; `agent_v2` construit comme `agent_v1` (têtes v2) ;
  `agent_v1_eval` idem têtes v1 (instance fraîche, commentaire conservé).
- **Éval/graphique** : impression 5 politiques au format V2 du notebook 04
  (victoire/cibles/hauteur/survie/nudges) ; graphique = barres de
  `completion_rate` annotées nudges (titre « Itération 1 — V2 cibles »).
- **Lecture des résultats** : remplacer la puce « Sur la table dure... » par :
  « si l'agent V2 domine nettement toutes les baselines en taux de victoire,
  le critère V2 est rempli — `git tag v2` » ; et la puce nudges rappelle
  l'histoire de l'itération 1 (le panneau a démasqué le piégeage).
- **Avertissement reprise** : mettre à jour les noms de dossiers
  (`data/targets_v2` ET `checkpoints_targets_v2`).
- **Amendement post-validation (2026-07-02)** : le warm-start calcule désormais
  `epochs = epoch_du_checkpoint + 6` dynamiquement (un `epochs=16` fixe
  n'entraînait RIEN en silence quand l'utilisateur avait poussé le notebook 03
  à 100 epochs — cas réel rencontré) ; l'axvline du graphique suit.

Vérifications : jupytext ; grep : plus aucune occurrence de `hard_v1/hard_v2/
checkpoints_hard` dans notebooks/ ; suite pytest inchangée.

Commit : `docs: notebook 05 v2 — itération avec têtes multiples et métrique victoire`

---

### Task 33: Notebook 06 — voir ce que l'IA prédit

**Files:**
- Create: `notebooks/06_visualisation.py` + `.ipynb`

Source complète :

```python
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
probe = PositionProbe()
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
```

Vérifications : `jupytext --to ipynb notebooks/06_visualisation.py` ; structure
(nombre de cellules affiché) ; suite pytest inchangée (86 PASS).

Commit : `docs: notebook 06 — superpositions prédit/réel et décodeur d'imagination`

---

## Validation (manuelle, Colab T4 recommandé)

1. Merger, pousser. Ouvrir les notebooks depuis GitHub (badges Colab du README).
2. 02 (collecte targets_v1, ~15-30 min T4) → 03 (JEPA, ~20-40 min) →
   04 (têtes + agent multi-objectifs + graphique VICTOIRES) →
   05 (itération, ~1 h) → 06 (visualisation, ~15 min).
3. Critère V2 : taux de victoire de l'agent NETTEMENT au-dessus de toutes les
   baselines (elles plafonnent à ~2 % par chance) → `git tag v2`.
4. Suivi non bloquant : mettre à jour le README (badges + section V2).

---

### Task 34: Architecture BIG+GAMMA et checkpoints auto-descriptifs

**Décision (2026-07-02, expérience contrôlée `_exp_arch`, 5 variantes) :**
grossir seul ne change rien (BIG ≈ BASE) ; la grille spatiale 8×8 échoue
partout (FINE) ; la **pondération des horizons courts** (γ=0,7) améliore tout
— lisibilité balle 0,054→0,041, bras 0,114→0,059, trajectoire déroulée h=1
0,064→0,046, variance dynamique ×2,8 (antidote à l'érosion). BIG+GAMMA
retenu (gain marginal supplémentaire, coût GPU nul) : canaux (48, 96, 192,
384), z_dim 384, prédicteur 768, a_dim 64, γ=0,7. L'observation ne change
pas : les datasets restent valides, seuls les checkpoints sont obsolètes —
d'où des checkpoints AUTO-DESCRIPTIFS (hparams embarqués).

**Files:**
- Modify: `jepa/model.py`, `jepa/train.py`, `jepa/heads.py` (si nécessaire),
  `jepa/decoder.py`
- Test: `tests/test_model.py`, `tests/test_train.py`, `tests/test_viz.py` (mises à jour + ajouts)

**Interfaces:**
- `Encoder(in_ch=2, z_dim=384, channels=(48, 96, 192, 384))` — boucle conv
  inchangée, `side = 64 // 2**len(channels)`, fc adapté. GroupNorm(8, c)
  reste valide (canaux divisibles par 8).
- `Predictor(z_dim=384, n_actions=4, a_dim=64, hidden=768)`.
- `JEPA(z_dim=384, enc_channels=(48, 96, 192, 384), pred_hidden=768,
  a_dim=64)` ; attribut `self.hparams = {"z_dim", "enc_channels" (tuple),
  "pred_hidden", "a_dim"}` ; propriété `z_dim`.
- `JEPA.loss(frames, actions, gamma: float | None = 0.7)` — chaque pas i
  pondéré par `gamma**i`, perte = somme pondérée / somme des poids ;
  `gamma=None` = uniforme (rétro-compatible). MÉTRIQUES INCHANGÉES
  (pred_mse/copy_mse restent des moyennes non pondérées, comparables).
  Docstring pédagogique : sans pondération, l'erreur irréductible des
  horizons lointains domine le gradient et pousse à moyenner ; γ concentre
  l'apprentissage sur la dynamique fine (bras, balle proche) — chiffres de
  l'expérience cités.
- `train_jepa(..., gamma: float | None = 0.7)` — transmis à `loss` ; le
  checkpoint gagne `"hparams": model.hparams` ; la REPRISE reconstruit
  `JEPA(**ckpt["hparams"])`.
- `load_jepa` : reconstruit depuis `ckpt["hparams"]` ; si la clé manque
  (checkpoint antérieur) → `ValueError` explicite « checkpoint d'une version
  antérieure (architecture inconnue) — réentraîner via le notebook 03 ».
- `train_objective_heads` / `train_danger_head` : les têtes DOIVENT être
  construites avec la dimension des latents (`z_train.shape[1]`) — vérifier
  que c'est déjà le cas (motif de `train_danger_head`), corriger sinon.
- `train_decoder` : `Decoder(z_dim=jepa.z_dim)` (fallback
  `getattr(jepa, "z_dim", 256)` inutile : JEPA a toujours la propriété).
  La classe `Decoder` garde son défaut 256 (tests unitaires explicites).

Mises à jour de tests (exhaustives) :
- `test_model.py` : shapes 256→384 dans `test_encoder_shapes`,
  `test_predictor_shapes_and_action_sensitivity` (out shape),
  `test_encode_routes_online_and_target` ((3, 384)) ; la fourchette de
  paramètres devient `1_000_000 < n < 8_000_000` (encodeur ≈ 3,2 M).
- `test_viz.py` : `rollout_latents` → (8, 384) ; `test_overlays_return_images`
  construit `PositionProbe(384)` (et `Decoder(384)` pour `imagination_strip`).
- Nouveaux tests :

```python
def test_loss_gamma_weighting_differs():
    torch.manual_seed(0)
    jepa = JEPA()
    frames = torch.randint(0, 255, (3, 10, 64, 64), dtype=torch.uint8)
    actions = torch.randint(0, 4, (3, 8))
    l_g, _ = jepa.loss(frames, actions, gamma=0.7)
    l_u, _ = jepa.loss(frames, actions, gamma=None)
    assert l_g.isfinite() and l_u.isfinite()
    assert not torch.isclose(l_g, l_u)   # la pondération change bien la perte


def test_checkpoint_self_describing(tmp_path):
    # un checkpoint embarque son architecture : load_jepa reconstruit
    # à l'identique même pour des dimensions non standard
    eps = [moving_dot_episode(T=15, seed=0)]
    import jepa.train as jt
    torch.manual_seed(0)
    small = JEPA(z_dim=128, enc_channels=(16, 32, 64, 128),
                 pred_hidden=256, a_dim=16)
    opt = torch.optim.AdamW(
        [p for p in small.parameters() if p.requires_grad], lr=1e-3)
    torch.save({"model": small.state_dict(), "optimizer": opt.state_dict(),
                "epoch": 1, "history": [], "hparams": small.hparams},
               tmp_path / "jepa.pt")
    loaded = jt.load_jepa(tmp_path / "jepa.pt", device="cpu")
    assert loaded.z_dim == 128
    obs = torch.randint(0, 255, (2, 2, 64, 64), dtype=torch.uint8)
    assert loaded.encode(obs).shape == (2, 128)


def test_load_jepa_rejects_legacy_checkpoint(tmp_path):
    torch.save({"model": {}, "optimizer": {}, "epoch": 1, "history": []},
               tmp_path / "jepa.pt")
    with pytest.raises(ValueError):
        jt_load = __import__("jepa.train", fromlist=["load_jepa"]).load_jepa
        jt_load(tmp_path / "jepa.pt", device="cpu")
```

(`import pytest` si absent dans test_train.py ; GroupNorm(8, 16) valide.)

Suite attendue : 87 − 0 + 3 = 90 PASS (les tests modifiés restent au même
nombre). Commit :
`feat: architecture BIG+GAMMA (z=384, perte pondérée γ=0.7) + checkpoints auto-descriptifs`

---

### Task 35: Notebooks — dimensions dynamiques et diagnostic n°4

**Files:**
- Modify: `notebooks/03_jepa.py`, `notebooks/05_iteration.py`,
  `notebooks/06_visualisation.py` (+ .ipynb régénérés)

Éditions exactes :

**03** — après le diagnostic n°3, AVANT le markdown final, insérer :

```python
# %% [markdown]
# ## Diagnostic n°4 : la balle est-elle LISIBLE dans le latent ?
#
# Les diagnostics 1-3 peuvent être bons alors que la position précise de la
# balle s'est érodée du latent (vécu : cibles trop saillantes, puis
# sur-entraînement). Test direct : une petite sonde apprend à lire (x, y)
# depuis les latents gelés — son erreur sur des épisodes de validation dit
# ce que le latent contient VRAIMENT. Référence : un devin qui répond
# toujours la position moyenne.

# %%
from jepa.data import MultiLabelDataset
from jepa.heads import PositionProbe

def readability(episodes_subset):
    ds = MultiLabelDataset(episodes_subset)
    idx = np.linspace(0, len(ds) - 1, min(3000, len(ds)), dtype=int)
    obs = torch.stack([ds[int(i)]["obs"] for i in idx])
    pos = torch.stack([ds[int(i)]["pos"] for i in idx])
    with torch.no_grad():
        z = torch.cat([model_cpu.encode_target(obs[j:j + 256])
                       for j in range(0, len(obs), 256)])
    return z, pos

z_tr, p_tr = readability(episodes[:-40])
z_va, p_va = readability(episodes[-40:])
torch.manual_seed(0)
probe = PositionProbe(model_cpu.z_dim)
popt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
for _ in range(400):
    perm = torch.randperm(len(z_tr))[:512]
    loss_p = ((probe(z_tr[perm]) - p_tr[perm]) ** 2).mean()
    popt.zero_grad(); loss_p.backward(); popt.step()
probe.eval()
with torch.no_grad():
    mae = (probe(z_va) - p_va).abs().mean().item()
naif = (p_tr.mean(0) - p_va).abs().mean().item()
print(f"lisibilité de la balle : MAE = {mae:.3f} (devin naïf : {naif:.3f})")
print("bon signe si < 0.08 ; au-delà de 0.12, le latent a perdu la balle.")
```

**05** — le rechargement des têtes v1 devient dimension-conscient :
`DangerHead(jepa_v1.z_dim)`, `HeightHead(jepa_v1.z_dim)`,
`TargetHead(jepa_v1.z_dim)`, `PositionProbe(jepa_v1.z_dim)` (dict inchangé
par ailleurs).

**06** — `probe = PositionProbe(jepa.z_dim)` (chargement pos.pt inchangé).

Vérifications : jupytext ×3 ; ast ; suite inchangée (90 PASS). Commit :
`docs: notebooks — dimensions dynamiques (z_dim du checkpoint) + diagnostic n°4 lisibilité`

## Validation V2.2 (Colab)

Les datasets restent VALIDES (observation inchangée). Supprimer uniquement
`checkpoints_targets/` et `checkpoints_targets_v2/` du Drive, puis 03 (10
epochs, ~30-60 min — modèle plus gros) → vérifier diagnostics n°3 ET n°4
(< 0.08 attendu) → 04 → 05 → 06. Attendu : croix de trajectoire sur les
cercles à h=1-3, bras animés dans l'imagination.
