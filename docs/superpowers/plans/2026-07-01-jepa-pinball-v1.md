# JEPA Pinball V1 — Plan d'implémentation (Phases 0 à 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un agent qui apprend à garder une balle de flipper en jeu : simulateur physique pymunk paramétrable, world model JEPA conditionné par l'action, tête danger, planification MPC — le tout piloté par des notebooks Colab pédagogiques en français.

**Architecture:** Package Python `pinball` (simulation, rendu, env, collecte) + package `jepa` (modèle, entraînement, têtes, planificateur, éval). Notebooks minces en jupytext py:percent convertis en .ipynb. Entraînement offline sur dataset collecté par politique aléatoire.

**Tech Stack:** Python ≥3.10, pymunk ≥7.0, numpy, Pillow, PyTorch, pytest, jupytext.

**Spec:** `docs/superpowers/specs/2026-07-01-jepa-pinball-design.md`. La Phase 4 (V2 score) fera l'objet d'un plan séparé après validation de la V1.

## Global Constraints

- Python ≥ 3.10 ; pymunk ≥ 7.0 (API 7.x — testé avec 7.3.0).
- Observations : 2 frames empilées 64×64 niveaux de gris, uint8, shape `(2, 64, 64)`.
- Actions : 4 actions discrètes ; bit 0 = flipper gauche, bit 1 = flipper droit (`a & 1`, `a >> 1`).
- Physique 120 Hz, contrôle 15 Hz (frame_skip = 8).
- Latent `z ∈ R^256`. Horizon d'entraînement du prédicteur = horizon de planification = 8 pas.
- Le modèle ne voit JAMAIS `info` (positions/vitesses exactes) — uniquement les images.
- Tous les tests tournent sur CPU, sans GPU, en < 2 min au total.
- Code : identifiants en anglais, docstrings/commentaires en français. Notebooks : pédagogie en français.
- Commits fréquents, un par tâche minimum. Chaque message de commit se termine par les trailers Claude d'usage.
- Valeurs physiques par défaut : **validées par prototype** (voir tableau en Annexe A) — ne pas les « corriger » sans re-tester l'équilibrage.

## File Structure

```
jepa_play/
├── pyproject.toml               # Task 1
├── .gitignore                   # Task 1
├── pinball/
│   ├── __init__.py              # Task 1
│   ├── config.py                # Task 2 — BoardConfig + géométrie dérivée
│   ├── sim.py                   # Task 3 — PinballSim (pymunk)
│   ├── render.py                # Task 4 — rendu 64×64 + rendu debug
│   ├── env.py                   # Task 5 — PinballEnv (reset/step/info)
│   └── collect.py               # Task 7 — StickyRandomPolicy + shards npz
├── jepa/
│   ├── __init__.py              # Task 1
│   ├── data.py                  # Task 9 — WindowDataset, DangerDataset
│   ├── model.py                 # Task 10 — Encoder, Predictor, JEPA
│   ├── train.py                 # Task 11 — boucle d'entraînement + checkpoints
│   ├── heads.py                 # Task 13 — DangerHead + entraînement + AUC
│   ├── planner.py               # Task 14 — MPCPlanner
│   └── eval.py                  # Task 15 — baselines, évaluation, GIF
├── notebooks/
│   ├── 01_simulateur.py         # Task 6  (jupytext py:percent → .ipynb)
│   ├── 02_collecte.py           # Task 8
│   ├── 03_jepa.py               # Task 12
│   └── 04_controle.py           # Task 16
└── tests/
    ├── test_config.py           # Task 2
    ├── test_sim.py              # Task 3
    ├── test_render.py           # Task 4
    ├── test_env.py              # Task 5
    ├── test_collect.py          # Task 7
    ├── test_data.py             # Task 9
    ├── test_model.py            # Task 10
    ├── test_train.py            # Task 11
    ├── test_heads.py            # Task 13
    └── test_planner.py          # Task 14
```

## Annexe A — Valeurs physiques validées par prototype (2026-07-01)

Prototype pymunk 7.3.0 exécuté localement ; résultats mesurés :

| Paramètre | Valeur | Justification mesurée |
|---|---|---|
| Plateau | 540 × 960 unités | ratio flipper réaliste |
| Gravité | 1800 u/s² | chute libre vérifiée exacte |
| Rayon balle | 14 | ~2 px au rendu 64×64 (plancher 2 px) |
| `drain_gap` (écart pointes au repos) | **44** | équilibrage mesuré : aléatoire ≈ 5,7 s ; heuristique grossière ≈ 42 s → large marge de progression |
| Flipper : longueur 110, épaisseur (rayon) 12, masse 8 | | frappe mesurée : balle arrivant à −810 u/s repart à +1728 u/s |
| Angles flipper : repos −0,52 rad, appuyé +0,55 rad (gauche ; miroir à droite) | | |
| Moteur : vitesse 25 rad/s, max_force 1e7 | | |
| **Convention SimpleMotor** : `SimpleMotor(static_body, flipper_body, rate)` avec **rate > 0 ⇒ l'angle du flipper DIMINUE** | | mesuré : rate=+10 fait passer l'angle de −0,520 à −0,525 |
| Guides : du mur (y=320) vers un point **20 unités au-dessus du pivot** | | supprime la poche de coincement pivot/guide (mesuré : 0 stuck sur 60 épisodes) |
| Slingshots : triangles dont la **base est posée sur la ligne du guide** (t=95→205 le long du guide, apex à 48 vers l'intérieur), élasticité 1,4 | | supprime la poche guide/slingshot |
| Vitesse max balle | 2200 u/s | anti-tunneling : 0 évasion sur 60+ épisodes |
| Murs/guides : rayon 6, élasticité 0,65 ; balle : élasticité 0,5, friction 0,4 | | |
| Stagnation : vitesse < 5 pendant 45 pas de contrôle → **nudge** (impulsion aléatoire), max 6 nudges/épisode puis fin `stuck` | | « toujours appuyé » piège la balle sur les flippers levés : le nudge évite la stratégie dégénérée du piégeage |
| Débit mesuré | ~11 500 transitions/s (local, physique+rendu) | 100k transitions en quelques minutes, même si Colab est 3-5× plus lent |

---

### Task 1: Scaffolding du projet

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `pinball/__init__.py`, `jepa/__init__.py`
- Create: `tests/__init__.py` (vide)

**Interfaces:**
- Consumes: rien.
- Produces: packages importables `pinball` et `jepa` ; `pip install -e ".[dev]"` fonctionne ; `pytest` collecte 0 test sans erreur.

- [ ] **Step 1: Écrire `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "jepa-pinball"
version = "0.1.0"
description = "Un agent JEPA qui apprend le flipper (projet pédagogique)"
requires-python = ">=3.10"
dependencies = [
    "pymunk>=7.0",
    "numpy",
    "pillow",
    "torch",
]

[project.optional-dependencies]
dev = ["pytest", "jupytext"]

[tool.setuptools]
packages = ["pinball", "jepa"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Écrire `.gitignore`**

```gitignore
__pycache__/
*.egg-info/
.venv/
venv/
data/
checkpoints/
.ipynb_checkpoints/
*.gif
```

- [ ] **Step 3: Créer les `__init__.py`**

`pinball/__init__.py` :
```python
"""Simulateur de flipper paramétrable (pymunk) et environnement type Gym."""
```

`jepa/__init__.py` :
```python
"""World model JEPA conditionné par l'action + planification MPC."""
```

`tests/__init__.py` : fichier vide.

- [ ] **Step 4: Installer et vérifier**

Run: `pip install -e ".[dev]" && pytest`
Expected: installation OK ; pytest affiche `no tests ran`.

Note Colab : torch y est préinstallé ; en local l'installation de torch (CPU) peut être longue — c'est attendu.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore pinball/__init__.py jepa/__init__.py tests/__init__.py
git commit -m "feat: scaffolding du projet jepa-pinball"
```

---

### Task 2: BoardConfig — la table paramétrable

**Files:**
- Create: `pinball/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: rien.
- Produces: `BoardConfig` (dataclass) avec tous les champs de l'Annexe A, plus :
  - `pivot_offset -> float` (propriété) : distance pivot↔centre, dérivée de `drain_gap`.
  - `pivot_pos(side: int) -> tuple[float, float]` : position du pivot (side=+1 gauche, −1 droit).
  - `guide_points(side: int) -> tuple[tuple[float, float], tuple[float, float]]` : segment guide (du mur vers le pivot).
  - `sling_verts(side: int) -> list[tuple[float, float]]` : 3 sommets du slingshot, base sur le guide.

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_config.py
import math
from pinball.config import BoardConfig


def test_defaults_and_derived_geometry():
    cfg = BoardConfig()
    assert cfg.width == 540.0 and cfg.height == 960.0
    # pivot_offset = drain_gap/2 + longueur*cos(angle repos)
    expected = cfg.drain_gap / 2 + cfg.flipper_length * math.cos(cfg.flipper_rest_angle)
    assert abs(cfg.pivot_offset - expected) < 1e-9
    lx, ly = cfg.pivot_pos(+1)
    rx, ry = cfg.pivot_pos(-1)
    assert ly == ry == cfg.flipper_y
    assert abs((rx - lx) - 2 * cfg.pivot_offset) < 1e-9


def test_guide_ends_above_pivot():
    cfg = BoardConfig()
    (x0, y0), (x1, y1) = cfg.guide_points(+1)
    assert x0 == 0.0 and y0 == cfg.guide_top_y
    px, py = cfg.pivot_pos(+1)
    assert abs(x1 - px) < 1e-9
    assert abs(y1 - (py + cfg.guide_end_lift)) < 1e-9


def test_sling_base_lies_on_guide_line():
    cfg = BoardConfig()
    (ax, ay), (bx, by) = cfg.guide_points(+1)
    p1, p2, apex = cfg.sling_verts(+1)
    # p1 et p2 sont sur la droite du guide (produit vectoriel ~ 0)
    for px, py in (p1, p2):
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        assert abs(cross) < 1e-6
    # l'apex est du côté intérieur du plateau (x plus grand pour le côté gauche)
    assert apex[0] > min(p1[0], p2[0])


def test_config_is_customizable():
    cfg = BoardConfig(drain_gap=100.0, flipper_length=80.0)
    assert cfg.drain_gap == 100.0
    small = BoardConfig(flipper_length=80.0).pivot_offset
    assert small < BoardConfig().pivot_offset
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pinball.config'`.

- [ ] **Step 3: Implémenter `pinball/config.py`**

```python
"""Configuration paramétrable du plateau de flipper.

Toutes les valeurs par défaut ont été équilibrées par prototype (voir le plan,
Annexe A) : politique aléatoire ~6 s de survie, heuristique simple ~42 s.
Repère : origine en bas à gauche, y vers le haut (convention pymunk).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BoardConfig:
    # --- dimensions du plateau (unités arbitraires) ---
    width: float = 540.0
    height: float = 960.0

    # --- balle ---
    ball_radius: float = 14.0
    ball_mass: float = 1.0
    ball_elasticity: float = 0.5
    ball_friction: float = 0.4
    max_ball_speed: float = 2200.0   # plafond anti-tunneling

    # --- gravité (inclinaison de la table) ---
    gravity: float = 1800.0

    # --- murs et guides ---
    wall_radius: float = 6.0
    wall_elasticity: float = 0.65
    wall_friction: float = 0.4
    guide_top_y: float = 320.0       # hauteur d'accroche des guides sur les murs
    guide_end_lift: float = 20.0     # le guide s'arrête au-dessus du pivot
                                     # (au ras du dessus du flipper : pas de poche)

    # --- flippers ---
    flipper_length: float = 110.0
    flipper_thickness: float = 12.0  # rayon du segment
    flipper_mass: float = 8.0
    flipper_y: float = 120.0         # hauteur des pivots
    flipper_rest_angle: float = 0.52   # rad sous l'horizontale, au repos
    flipper_press_angle: float = 0.55  # rad au-dessus, appuyé
    flipper_speed: float = 25.0        # rad/s du moteur
    flipper_max_force: float = 1e7
    flipper_elasticity: float = 0.4
    flipper_friction: float = 0.6
    drain_gap: float = 44.0          # écart horizontal entre les POINTES au repos

    # --- slingshots (triangles rebondissants posés sur les guides) ---
    sling_start: float = 95.0        # abscisses curvilignes le long du guide
    sling_end: float = 205.0
    sling_height: float = 48.0       # hauteur de l'apex vers l'intérieur
    sling_elasticity: float = 1.4    # > 1 : redonne de l'énergie à la balle

    # --- temps ---
    physics_hz: int = 120
    frame_skip: int = 8              # contrôle à 120/8 = 15 Hz

    # --- épisode ---
    max_episode_steps: int = 900     # 60 s à 15 Hz
    drain_y: float = 60.0            # sous cette hauteur, balle perdue
    launch_margin: float = 80.0      # zone x de lancement : [margin, width-margin]
    launch_y_offset: float = 90.0    # lancement à height - offset
    launch_vx_max: float = 150.0     # |vx| initial max
    stuck_speed: float = 5.0         # vitesse sous laquelle la balle "stagne"
    stuck_steps: int = 45            # pas de contrôle stagnants avant nudge (3 s)
    nudge_impulse: float = 300.0     # impulsion du nudge (quantité de mouvement)
    max_nudges: int = 6              # au-delà, fin d'épisode "stuck"

    # --- V2 (bumpers scoreurs) : rempli dans le plan V2 ---
    bumpers: list = field(default_factory=list)

    # ---------- géométrie dérivée ----------
    @property
    def pivot_offset(self) -> float:
        """Distance horizontale pivot ↔ centre, telle que l'écart entre les
        pointes des flippers AU REPOS vaille exactement drain_gap."""
        return self.drain_gap / 2 + self.flipper_length * math.cos(self.flipper_rest_angle)

    def pivot_pos(self, side: int) -> tuple[float, float]:
        """Pivot du flipper. side=+1 : gauche, side=-1 : droit."""
        return (self.width / 2 - side * self.pivot_offset, self.flipper_y)

    def guide_points(self, side: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Guide d'entonnoir : du mur latéral vers un point juste au-dessus
        du pivot (au ras du dessus du flipper, pour ne pas créer de poche)."""
        x0 = 0.0 if side > 0 else self.width
        px, py = self.pivot_pos(side)
        return (x0, self.guide_top_y), (px, py + self.guide_end_lift)

    def sling_verts(self, side: int) -> list[tuple[float, float]]:
        """Slingshot : triangle dont la base est SUR la ligne du guide
        (aucun interstice où coincer la balle), apex tourné vers le plateau."""
        (ax, ay), (bx, by) = self.guide_points(side)
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        # normale pointant vers l'intérieur du plateau
        nx, ny = (-dy, dx) if side > 0 else (dy, -dx)
        p1 = (ax + dx * self.sling_start, ay + dy * self.sling_start)
        p2 = (ax + dx * self.sling_end, ay + dy * self.sling_end)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        apex = (mx + nx * self.sling_height, my + ny * self.sling_height)
        return [p1, p2, apex]
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_config.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add pinball/config.py tests/test_config.py
git commit -m "feat: BoardConfig, table de flipper paramétrable avec géométrie dérivée"
```

---

### Task 3: PinballSim — le simulateur pymunk

**Files:**
- Create: `pinball/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `BoardConfig` (Task 2).
- Produces: classe `PinballSim` :
  - `PinballSim(config: BoardConfig, rng: numpy.random.Generator)` — construit l'espace, lance la balle.
  - `set_flippers(left: bool, right: bool) -> None`
  - `step_control() -> None` — `frame_skip` sous-pas physiques + plafond de vitesse.
  - `nudge() -> None` — impulsion aléatoire (biais vers le haut) sur la balle.
  - Propriétés : `ball_pos -> tuple[float, float]`, `ball_speed -> float`,
    `flipper_angles -> tuple[float, float]`.
  - Attributs publics pour le rendu : `config`, `flipper_bodies` (liste `[gauche, droit]`).

**Convention moteur validée par prototype (NE PAS inverser) :** avec
`SimpleMotor(static_body, flipper_body, rate)`, un `rate > 0` fait DIMINUER
l'angle du flipper. Donc : flipper gauche appuyé ⇒ `rate = -flipper_speed` ;
flipper droit appuyé ⇒ `rate = +flipper_speed`.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_sim.py
import numpy as np
from pinball.config import BoardConfig
from pinball.sim import PinballSim


def make_sim(seed=0, **cfg_kwargs):
    cfg = BoardConfig(**cfg_kwargs)
    return PinballSim(cfg, np.random.default_rng(seed)), cfg


def test_free_fall_matches_gravity():
    # balle lancée sans vitesse : après 0,5 s, vy ≈ -g * t
    sim, cfg = make_sim()
    sim.ball.velocity = (0, 0)
    sim.ball.position = (cfg.width / 2, 800)
    for _ in range(int(0.5 * cfg.physics_hz / cfg.frame_skip) + 1):
        sim.step_control()
    t = (int(0.5 * cfg.physics_hz / cfg.frame_skip) + 1) * cfg.frame_skip / cfg.physics_hz
    assert abs(sim.ball.velocity.y + cfg.gravity * t) < 30.0


def test_flipper_press_raises_angle():
    sim, cfg = make_sim()
    left0, right0 = sim.flipper_angles
    sim.set_flippers(True, True)
    for _ in range(6):
        sim.step_control()
    left1, right1 = sim.flipper_angles
    assert left1 > left0    # gauche : appuyé = angle qui augmente
    assert right1 < right0  # droit : miroir


def test_flipper_strike_speeds_ball_up():
    # balle lâchée sur le flipper gauche au repos, puis frappe
    sim, cfg = make_sim()
    px, py = cfg.pivot_pos(+1)
    sim.ball.position = (px + cfg.flipper_length * 0.6, 400)
    sim.ball.velocity = (0, 0)
    for _ in range(9):          # ~0,6 s de chute sur le flipper
        sim.step_control()
    sim.set_flippers(True, False)
    max_vy = -1e9
    for _ in range(5):
        sim.step_control()
        max_vy = max(max_vy, sim.ball.velocity.y)
    assert max_vy > 400.0       # la balle est projetée vers le haut


def test_speed_is_capped():
    sim, cfg = make_sim()
    sim.ball.velocity = (0, -10 * cfg.max_ball_speed)
    sim.step_control()
    assert sim.ball_speed <= cfg.max_ball_speed * 1.01


def test_nudge_moves_stationary_ball():
    sim, cfg = make_sim()
    sim.ball.velocity = (0, 0)
    v0 = sim.ball_speed
    sim.nudge()
    assert sim.ball_speed > v0


def test_config_changes_geometry():
    sim_a, _ = make_sim(drain_gap=44.0)
    sim_b, _ = make_sim(drain_gap=120.0)
    ax = sim_a.flipper_bodies[0].position.x
    bx = sim_b.flipper_bodies[0].position.x
    assert bx < ax   # drain plus large -> pivot gauche plus loin du centre


def test_ball_stays_in_bounds_long_episode():
    # anti-tunneling : 900 pas de jeu aléatoire, la balle reste dans le plateau
    sim, cfg = make_sim(seed=3)
    rng = np.random.default_rng(3)
    for i in range(900):
        if i % 7 == 0:
            a = int(rng.integers(4))
        sim.set_flippers(bool(a & 1), bool(a >> 1))
        sim.step_control()
        x, y = sim.ball_pos
        assert -5 <= x <= cfg.width + 5 and y <= cfg.height + 5
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_sim.py -v`
Expected: FAIL — `No module named 'pinball.sim'`.

- [ ] **Step 3: Implémenter `pinball/sim.py`**

```python
"""Simulateur physique du flipper (pymunk / Chipmunk2D).

La balle est un vrai corps rigide : gravité, restitution, friction.
Les flippers sont des corps dynamiques sur pivot, actionnés par moteur
angulaire avec butées — la balle est réellement frappée (transfert de
moment), rien n'est scripté.
"""
from __future__ import annotations

import numpy as np
import pymunk

from .config import BoardConfig


class PinballSim:
    def __init__(self, config: BoardConfig, rng: np.random.Generator):
        self.config = config
        self._rng = rng
        self.space = pymunk.Space()
        self.space.gravity = (0, -config.gravity)
        self._build_static()
        self.flipper_bodies: list[pymunk.Body] = []
        self._motors: list[pymunk.SimpleMotor] = []
        for side in (+1, -1):
            self._add_flipper(side)
        self._add_ball()

    # ---------- construction ----------
    def _build_static(self) -> None:
        cfg = self.config
        sb = self.space.static_body
        segments = [
            ((0, 0), (0, cfg.height)),                    # mur gauche
            ((cfg.width, 0), (cfg.width, cfg.height)),    # mur droit
            ((0, cfg.height), (cfg.width, cfg.height)),   # plafond
            cfg.guide_points(+1),                         # guides d'entonnoir
            cfg.guide_points(-1),
        ]
        shapes = []
        for a, b in segments:
            s = pymunk.Segment(sb, a, b, cfg.wall_radius)
            s.elasticity = cfg.wall_elasticity
            s.friction = cfg.wall_friction
            shapes.append(s)
        for side in (+1, -1):
            p = pymunk.Poly(sb, cfg.sling_verts(side))
            p.elasticity = cfg.sling_elasticity
            p.friction = cfg.wall_friction
            shapes.append(p)
        self.space.add(*shapes)

    def _add_flipper(self, side: int) -> None:
        cfg = self.config
        a, b = (0, 0), (side * cfg.flipper_length, 0)
        moment = pymunk.moment_for_segment(cfg.flipper_mass, a, b, cfg.wall_radius)
        body = pymunk.Body(cfg.flipper_mass, moment)
        body.position = cfg.pivot_pos(side)
        shape = pymunk.Segment(body, a, b, cfg.flipper_thickness)
        shape.elasticity = cfg.flipper_elasticity
        shape.friction = cfg.flipper_friction
        if side > 0:
            lo, hi = -cfg.flipper_rest_angle, cfg.flipper_press_angle
        else:
            lo, hi = -cfg.flipper_press_angle, cfg.flipper_rest_angle
        sb = self.space.static_body
        pivot = pymunk.PivotJoint(sb, body, body.position)
        limit = pymunk.RotaryLimitJoint(sb, body, lo, hi)
        motor = pymunk.SimpleMotor(sb, body, 0.0)
        motor.max_force = cfg.flipper_max_force
        body.angle = lo if side > 0 else hi   # position de repos
        self.space.add(body, shape, pivot, limit, motor)
        self.flipper_bodies.append(body)
        self._motors.append(motor)

    def _add_ball(self) -> None:
        cfg = self.config
        moment = pymunk.moment_for_circle(cfg.ball_mass, 0, cfg.ball_radius)
        self.ball = pymunk.Body(cfg.ball_mass, moment)
        x = self._rng.uniform(cfg.launch_margin, cfg.width - cfg.launch_margin)
        self.ball.position = (x, cfg.height - cfg.launch_y_offset)
        self.ball.velocity = (self._rng.uniform(-cfg.launch_vx_max, cfg.launch_vx_max), 0)
        shape = pymunk.Circle(self.ball, cfg.ball_radius)
        shape.elasticity = cfg.ball_elasticity
        shape.friction = cfg.ball_friction
        self.space.add(self.ball, shape)

    # ---------- contrôle ----------
    def set_flippers(self, left: bool, right: bool) -> None:
        # Convention pymunk mesurée : rate > 0 fait DIMINUER l'angle.
        speed = self.config.flipper_speed
        self._motors[0].rate = -speed if left else speed
        self._motors[1].rate = speed if right else -speed

    def step_control(self) -> None:
        """Un pas de contrôle = frame_skip sous-pas physiques + plafond vitesse."""
        cfg = self.config
        dt = 1.0 / cfg.physics_hz
        for _ in range(cfg.frame_skip):
            self.space.step(dt)
            v = self.ball.velocity
            if v.length > cfg.max_ball_speed:
                self.ball.velocity = v * (cfg.max_ball_speed / v.length)

    def nudge(self) -> None:
        """Petite impulsion aléatoire, biaisée vers le haut (anti-stagnation)."""
        cfg = self.config
        angle = self._rng.uniform(np.pi * 0.25, np.pi * 0.75)  # vers le haut
        impulse = cfg.nudge_impulse * np.array([np.cos(angle), np.sin(angle)])
        self.ball.apply_impulse_at_local_point(tuple(impulse))

    # ---------- lecture d'état ----------
    @property
    def ball_pos(self) -> tuple[float, float]:
        p = self.ball.position
        return (p.x, p.y)

    @property
    def ball_speed(self) -> float:
        return self.ball.velocity.length

    @property
    def flipper_angles(self) -> tuple[float, float]:
        return (self.flipper_bodies[0].angle, self.flipper_bodies[1].angle)
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_sim.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add pinball/sim.py tests/test_sim.py
git commit -m "feat: simulateur pymunk (flippers à pivot moteur, nudge, anti-tunneling)"
```

---

### Task 4: Rendu image 64×64

**Files:**
- Create: `pinball/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `PinballSim` (Task 3), `BoardConfig` (Task 2).
- Produces:
  - `render_frame(sim: PinballSim, size: int = 64) -> np.ndarray` — uint8 `(size, size)`,
    fond 0, murs/guides/slings gris 90, flippers gris 180, balle blanc 255.
  - `render_debug(sim: PinballSim, scale: int = 5) -> PIL.Image.Image` — grande image RGB
    annotée pour les vidéos des notebooks.

Le rendu est anisotrope (largeur et hauteur du plateau étirées chacune sur
`size` pixels) — assumé et documenté. Le rayon de la balle à l'écran a un
plancher de 2 px pour rester visible.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_render.py
import numpy as np
from pinball.config import BoardConfig
from pinball.render import render_frame, render_debug
from pinball.sim import PinballSim


def make_sim(seed=0):
    cfg = BoardConfig()
    return PinballSim(cfg, np.random.default_rng(seed)), cfg


def test_frame_shape_and_dtype():
    sim, _ = make_sim()
    f = render_frame(sim)
    assert f.shape == (64, 64) and f.dtype == np.uint8


def test_ball_is_brightest_and_visible():
    sim, cfg = make_sim()
    sim.ball.position = (cfg.width / 2, cfg.height / 2)
    f = render_frame(sim)
    assert f.max() == 255
    # la balle occupe au moins un disque de rayon ~2 px
    assert (f == 255).sum() >= 8
    # et elle est bien au centre de l'image
    ys, xs = np.where(f == 255)
    assert abs(xs.mean() - 32) < 3 and abs(ys.mean() - 32) < 3


def test_ball_moves_in_image():
    sim, cfg = make_sim()
    sim.ball.position = (100, 700)
    f1 = render_frame(sim)
    sim.ball.position = (400, 300)
    f2 = render_frame(sim)
    assert not np.array_equal(f1, f2)


def test_flippers_change_pixels_when_pressed():
    sim, _ = make_sim()
    f_rest = render_frame(sim)
    sim.set_flippers(True, True)
    for _ in range(6):
        sim.step_control()
    f_up = render_frame(sim)
    assert not np.array_equal(f_rest, f_up)


def test_render_debug_is_rgb_and_scaled():
    sim, _ = make_sim()
    img = render_debug(sim, scale=5)
    assert img.mode == "RGB"
    assert img.size == (64 * 5, 64 * 5)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_render.py -v`
Expected: FAIL — `No module named 'pinball.render'`.

- [ ] **Step 3: Implémenter `pinball/render.py`**

```python
"""Rendu du plateau en image, sans fenêtre graphique (PIL uniquement).

C'est l'observation du modèle : 64×64 niveaux de gris. Le monde physique
(origine en bas à gauche, y vers le haut) est projeté sur l'image
(origine en haut à gauche, y vers le bas), de façon anisotrope.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .sim import PinballSim

COLOR_WALL = 90
COLOR_FLIPPER = 180
COLOR_BALL = 255
BALL_MIN_PX = 2.0   # plancher de visibilité de la balle


def _project(cfg, size: int):
    sx, sy = size / cfg.width, size / cfg.height

    def pt(p):
        return (p[0] * sx, (cfg.height - p[1]) * sy)

    return pt, sx


def _draw_board(d: ImageDraw.ImageDraw, sim: PinballSim, pt, wall, flip, ball,
                width: int = 1) -> None:
    cfg = sim.config
    d.line([pt((0, 0)), pt((0, cfg.height)), pt((cfg.width, cfg.height)),
            pt((cfg.width, 0))], fill=wall, width=width)
    for side in (+1, -1):
        a, b = cfg.guide_points(side)
        d.line([pt(a), pt(b)], fill=wall, width=width)
        d.polygon([pt(v) for v in cfg.sling_verts(side)], outline=wall)
    for i, body in enumerate(sim.flipper_bodies):
        side = +1 if i == 0 else -1
        tip = body.local_to_world((side * cfg.flipper_length, 0))
        d.line([pt(tuple(body.position)), pt(tuple(tip))], fill=flip,
               width=max(2, width * 2))
    bx, by = pt(sim.ball_pos)
    r = ball
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=COLOR_BALL)


def render_frame(sim: PinballSim, size: int = 64) -> np.ndarray:
    """Observation du modèle : uint8 (size, size)."""
    cfg = sim.config
    pt, sx = _project(cfg, size)
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    ball_px = max(BALL_MIN_PX, cfg.ball_radius * sx)
    _draw_board(d, sim, pt, COLOR_WALL, COLOR_FLIPPER, ball_px)
    return np.asarray(img)


def render_debug(sim: PinballSim, scale: int = 5) -> Image.Image:
    """Grande image RGB pour les vidéos des notebooks (pas pour le modèle)."""
    cfg = sim.config
    size = 64 * scale
    pt, sx = _project(cfg, size)
    img = Image.new("RGB", (size, size), (10, 10, 30))
    d = ImageDraw.Draw(img)
    ball_px = max(BALL_MIN_PX * scale, cfg.ball_radius * sx)
    _draw_board(d, sim, pt, (200, 200, 200), (255, 160, 40), ball_px,
                width=max(1, scale // 2))
    x, y = sim.ball_pos
    d.text((6, 6), f"balle: ({x:.0f}, {y:.0f})  v={sim.ball_speed:.0f}",
           fill=(150, 220, 150))
    return img
```

Note : `_draw_board` dessine la balle avec `fill=COLOR_BALL` (blanc) dans les
deux rendus ; le paramètre `ball` est le rayon en pixels.

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_render.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add pinball/render.py tests/test_render.py
git commit -m "feat: rendu PIL 64x64 (observation) et rendu debug annoté"
```

---

### Task 5: PinballEnv — l'environnement type Gym

**Files:**
- Create: `pinball/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `BoardConfig`, `PinballSim`, `render_frame`.
- Produces: classe `PinballEnv` :
  - `PinballEnv(config: BoardConfig | None = None, seed: int | None = None, obs_size: int = 64)`
  - `N_ACTIONS = 4` (attribut de classe)
  - `reset(seed: int | None = None) -> np.ndarray` — obs `(2, obs_size, obs_size)` uint8,
    les deux frames identiques au premier pas.
  - `step(action: int) -> tuple[np.ndarray, dict]` — `info` contient :
    `ball_pos: tuple`, `ball_vel: tuple`, `ball_lost: bool`, `stuck: bool`,
    `nudged: bool`, `steps: int`, `done: bool`.
  - Attribut `sim` (le `PinballSim` courant) pour `render_debug`.
  - `done` = `ball_lost` OU `stuck` OU `steps >= config.max_episode_steps`.

Sémantique de fin d'épisode : `ball_lost` quand `ball_pos.y < config.drain_y` ;
stagnation (vitesse < `stuck_speed` pendant `stuck_steps` pas) → `nudge()` du sim
et `nudged=True` ; au-delà de `max_nudges` nudges dans l'épisode, `stuck=True`.
Appeler `step` après `done` doit lever `RuntimeError` (on doit `reset`).

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_env.py
import numpy as np
import pytest
from pinball.config import BoardConfig
from pinball.env import PinballEnv


def test_obs_shape_dtype_and_first_stack():
    env = PinballEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (2, 64, 64) and obs.dtype == np.uint8
    assert np.array_equal(obs[0], obs[1])  # premier pas : frames identiques


def test_step_advances_and_stacks():
    env = PinballEnv(seed=0)
    obs0 = env.reset()
    obs1, info = env.step(0)
    assert obs1.shape == (2, 64, 64)
    # la frame "présent" de obs0 devient la frame "passé" de obs1
    assert np.array_equal(obs1[0], obs0[1])
    assert info["steps"] == 1 and not info["done"]


def test_determinism_with_same_seed():
    def rollout(seed):
        env = PinballEnv(seed=seed)
        env.reset()
        frames = []
        for i in range(40):
            obs, info = env.step(i % 4)
            frames.append(obs.copy())
            if info["done"]:
                break
        return np.stack(frames)

    a, b = rollout(7), rollout(7)
    assert np.array_equal(a, b)


def test_episode_ends_by_drain():
    # sans jamais actionner les flippers, la balle finit par drainer
    env = PinballEnv(seed=1)
    env.reset()
    for _ in range(900):
        obs, info = env.step(0)
        if info["done"]:
            break
    assert info["done"]
    assert info["ball_lost"] or info["stuck"] or info["steps"] == 900
    # cas le plus courant avec flippers baissés : le drain
    assert info["ball_lost"]


def test_step_after_done_raises():
    env = PinballEnv(seed=1)
    env.reset()
    for _ in range(900):
        _, info = env.step(0)
        if info["done"]:
            break
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_action_raises():
    env = PinballEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)


def test_custom_config_is_used():
    env = PinballEnv(config=BoardConfig(max_episode_steps=5), seed=0)
    env.reset()
    for _ in range(5):
        _, info = env.step(0)
    assert info["done"] and info["steps"] == 5
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_env.py -v`
Expected: FAIL — `No module named 'pinball.env'`.

- [ ] **Step 3: Implémenter `pinball/env.py`**

```python
"""Environnement type Gymnasium au-dessus du simulateur.

L'observation est UNIQUEMENT l'image (2 frames empilées : le mouvement est
visible, l'état devient ~Markovien). Les grandeurs exactes (position,
vitesse...) sortent dans `info` pour le debug et les labels — jamais pour
le modèle.
"""
from __future__ import annotations

import numpy as np

from .config import BoardConfig
from .render import render_frame
from .sim import PinballSim


class PinballEnv:
    N_ACTIONS = 4  # bit 0 = flipper gauche, bit 1 = flipper droit

    def __init__(self, config: BoardConfig | None = None,
                 seed: int | None = None, obs_size: int = 64):
        self.config = config or BoardConfig()
        self.obs_size = obs_size
        self._rng = np.random.default_rng(seed)
        self.sim: PinballSim | None = None
        self._done = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.sim = PinballSim(self.config, self._rng)
        self._steps = 0
        self._stuck_count = 0
        self._nudges = 0
        self._done = False
        frame = render_frame(self.sim, self.obs_size)
        self._prev_frame = frame
        return np.stack([frame, frame])

    def step(self, action: int) -> tuple[np.ndarray, dict]:
        if self._done:
            raise RuntimeError("épisode terminé : appeler reset()")
        if not 0 <= int(action) < self.N_ACTIONS:
            raise ValueError(f"action invalide : {action}")
        action = int(action)

        self.sim.set_flippers(bool(action & 1), bool(action >> 1))
        self.sim.step_control()
        self._steps += 1

        x, y = self.sim.ball_pos
        ball_lost = y < self.config.drain_y
        nudged = False
        stuck = False
        if not ball_lost and self.sim.ball_speed < self.config.stuck_speed:
            self._stuck_count += 1
            if self._stuck_count >= self.config.stuck_steps:
                if self._nudges < self.config.max_nudges:
                    self.sim.nudge()
                    self._nudges += 1
                    nudged = True
                    self._stuck_count = 0
                else:
                    stuck = True
        else:
            self._stuck_count = 0

        self._done = ball_lost or stuck or self._steps >= self.config.max_episode_steps
        frame = render_frame(self.sim, self.obs_size)
        obs = np.stack([self._prev_frame, frame])
        self._prev_frame = frame

        vx, vy = self.sim.ball.velocity
        info = {
            "ball_pos": (x, y),
            "ball_vel": (vx, vy),
            "ball_lost": ball_lost,
            "stuck": stuck,
            "nudged": nudged,
            "steps": self._steps,
            "done": self._done,
        }
        return obs, info
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_env.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Vérifier toute la suite**

Run: `pytest`
Expected: tous les tests des Tasks 2-5 PASS.

- [ ] **Step 6: Commit**

```bash
git add pinball/env.py tests/test_env.py
git commit -m "feat: PinballEnv, observations image empilees et info privilegie"
```

---

### Task 6: Notebook 01 — le simulateur en action

**Files:**
- Create: `notebooks/01_simulateur.py` (source jupytext, format py:percent)
- Create: `notebooks/01_simulateur.ipynb` (généré, commité pour ouverture directe dans Colab)

**Interfaces:**
- Consumes: `BoardConfig`, `PinballEnv`, `render_frame`, `render_debug`.
- Produces: notebook exécutable de bout en bout dans Colab ET localement.

Convention pour TOUS les notebooks : première cellule = installation
(`%pip install -q -e .` si le repo est cloné, sinon instructions de clone),
détection Colab via `google.colab` importable ; montage de Drive uniquement sur
Colab. Les notebooks appellent le package — aucune logique métier dans les cellules.

- [ ] **Step 1: Écrire `notebooks/01_simulateur.py`**

```python
# %% [markdown]
# # 01 — Le simulateur de flipper
#
# Dans ce notebook : construire la table, comprendre ses paramètres, regarder
# la physique tourner, et jouer manuellement pour vérifier que le flipper est
# agréable. **Aucun apprentissage ici** — d'abord, un monde qui fonctionne.

# %%
# Installation (Colab : cloner le repo ; local : repo déjà présent)
import importlib.util, subprocess, sys, os
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB and not os.path.exists("jepa_play"):
    subprocess.run(["git", "clone", "https://github.com/VOTRE_COMPTE/jepa_play.git"], check=True)
    os.chdir("jepa_play")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from pinball.config import BoardConfig
from pinball.env import PinballEnv
from pinball.render import render_debug

# %% [markdown]
# ## La table est une configuration
#
# Tout le plateau est décrit par une dataclass. Changer la table = changer des
# nombres. Regardons la table par défaut, puis une variante.

# %%
cfg = BoardConfig()
env = PinballEnv(cfg, seed=0)
env.reset()
plt.figure(figsize=(5, 5)); plt.imshow(render_debug(env.sim)); plt.axis("off")
plt.title("Table par défaut"); plt.show()

# %%
large = BoardConfig(drain_gap=120.0, flipper_length=90.0)
env2 = PinballEnv(large, seed=0)
env2.reset()
plt.figure(figsize=(5, 5)); plt.imshow(render_debug(env2.sim)); plt.axis("off")
plt.title("Variante : drain large, flippers courts (plus difficile)"); plt.show()

# %% [markdown]
# ## Ce que voit le modèle : 64×64, 2 frames
#
# L'observation est volontairement pauvre : une petite image en niveaux de
# gris. Pourquoi DEUX frames ? Une image seule ne contient pas la vitesse de
# la balle — deux images identiques peuvent cacher une balle montante ou
# descendante. Avec deux frames, le mouvement est visible : l'état redevient
# (approximativement) Markovien.

# %%
obs = env.reset(seed=3)
for _ in range(12):
    obs, info = env.step(0)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i, ax in enumerate(axes):
    ax.imshow(obs[i], cmap="gray"); ax.set_title(f"frame t-{1-i}"); ax.axis("off")
plt.suptitle("L'observation du modèle : le mouvement est dans la paire"); plt.show()

# %% [markdown]
# ## Une partie en vidéo (politique : ne rien faire)

# %%
from PIL import Image

def episode_gif(env, policy, path, seed=None, max_steps=450):
    """Joue un épisode et l'enregistre en GIF (rendu debug)."""
    env.reset(seed=seed)
    frames = [render_debug(env.sim)]
    for _ in range(max_steps):
        _, info = env.step(policy())
        frames.append(render_debug(env.sim))
        if info["done"]:
            break
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=66, loop=0)
    return info

info = episode_gif(PinballEnv(seed=5), lambda: 0, "episode_passif.gif", seed=5)
print(f"Épisode terminé en {info['steps']} pas ({info['steps']/15:.1f} s), "
      f"balle perdue : {info['ball_lost']}")

# %%
from IPython.display import Image as IPImage, display
display(IPImage("episode_passif.gif"))

# %% [markdown]
# ## Jouer à la main (widgets)
#
# Deux cases à cocher = les deux boutons du flipper. Le bouton « Avancer 1 s »
# fait 15 pas de contrôle. Vérifie que : la balle rebondit, les flippers
# frappent fort, les slingshots relancent, et que la balle finit au drain.

# %%
import ipywidgets as widgets

env_manual = PinballEnv(seed=None)
env_manual.reset()
left_box = widgets.Checkbox(description="Flipper gauche")
right_box = widgets.Checkbox(description="Flipper droit")
out = widgets.Output()

def advance(_):
    with out:
        out.clear_output(wait=True)
        action = int(left_box.value) | (int(right_box.value) << 1)
        for _ in range(15):
            _, info = env_manual.step(action)
            if info["done"]:
                break
        plt.figure(figsize=(5, 5)); plt.imshow(render_debug(env_manual.sim))
        plt.axis("off"); plt.show()
        if info["done"]:
            print("Balle perdue ! Relance...")
            env_manual.reset()

button = widgets.Button(description="Avancer 1 s")
button.on_click(advance)
display(widgets.HBox([left_box, right_box, button]), out)
advance(None)

# %% [markdown]
# ## À retenir
#
# - La table est **paramétrable** (`BoardConfig`) : gravité, flippers, drain...
# - La physique est **réelle** (pymunk) : la balle est frappée, pas téléportée.
# - Le modèle ne verra QUE l'image 64×64×2 — jamais les positions exactes.
#
# Prochaine étape (notebook 02) : générer le dataset d'expérience en laissant
# une politique aléatoire jouer des milliers de parties, toute seule.
```

- [ ] **Step 2: Générer le .ipynb et exécuter localement**

Run: `jupytext --to ipynb notebooks/01_simulateur.py`
Expected: crée `notebooks/01_simulateur.ipynb`.

Vérification de structure (l'exécution complète, widgets compris, se fait à
la main dans Colab lors de la validation de phase) :
Run: `python -c "
import jupytext
nb = jupytext.read('notebooks/01_simulateur.py')
print(len(nb.cells), 'cellules')"`
Expected: le nombre de cellules s'affiche sans erreur.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_simulateur.py notebooks/01_simulateur.ipynb
git commit -m "docs: notebook 01, decouverte du simulateur (table, obs, jeu manuel)"
```

---

### Task 7: Collecte — politique collante et shards npz

**Files:**
- Create: `pinball/collect.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Consumes: `PinballEnv`.
- Produces:
  - `StickyRandomPolicy(rng: np.random.Generator, hold_range: tuple[int, int] = (3, 15))`
    — callable : `policy(obs) -> int` ; méthode `reset()`. Tire une action et la
    maintient un nombre aléatoire de pas.
  - `collect_dataset(env, policy, n_transitions: int, out_dir: str | Path, shard_episodes: int = 64) -> dict`
    — joue des épisodes complets jusqu'à atteindre `n_transitions`, écrit des
    shards `shard_00000.npz`, ... Retourne `{"episodes": int, "transitions": int, "shards": int}`.
  - `load_episodes(data_dir: str | Path) -> list[dict]` — recharge tous les shards ;
    chaque épisode est un dict :
    `frames: (T+1, 64, 64) uint8` (frames BRUTES, non empilées),
    `actions: (T,) int64`, `ball_pos: (T+1, 2) float32`, `ball_lost: bool`.

Format d'un shard npz (épisodes concaténés — les npz ne gèrent pas le ragged) :
`frames` (ΣTᵢ+n, 64, 64) uint8, `actions` (ΣTᵢ,) int64, `ball_pos` (ΣTᵢ+n, 2)
float32, `frame_counts` (n,) int64, `action_counts` (n,) int64,
`ball_lost` (n,) bool. La frame t d'un épisode est l'image APRÈS t pas ;
`frames[0]` est l'image du reset. L'observation empilée se reconstruit à la
lecture : `obs_t = stack(frames[max(t-1, 0)], frames[t])`.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_collect.py
import numpy as np
from pinball.collect import StickyRandomPolicy, collect_dataset, load_episodes
from pinball.config import BoardConfig
from pinball.env import PinballEnv


def test_sticky_policy_holds_actions():
    policy = StickyRandomPolicy(np.random.default_rng(0), hold_range=(4, 6))
    actions = [policy(None) for _ in range(60)]
    # les actions changent, mais pas à chaque pas : il existe des paliers >= 4
    runs, current, length = [], actions[0], 1
    for a in actions[1:]:
        if a == current:
            length += 1
        else:
            runs.append(length); current, length = a, 1
    runs.append(length)
    assert max(runs) >= 4
    assert len(set(actions)) > 1


def test_collect_and_reload_roundtrip(tmp_path):
    env = PinballEnv(BoardConfig(max_episode_steps=40), seed=0)
    policy = StickyRandomPolicy(np.random.default_rng(0))
    stats = collect_dataset(env, policy, n_transitions=150, out_dir=tmp_path,
                            shard_episodes=2)
    assert stats["transitions"] >= 150
    episodes = load_episodes(tmp_path)
    assert len(episodes) == stats["episodes"]
    total = sum(len(ep["actions"]) for ep in episodes)
    assert total == stats["transitions"]
    for ep in episodes:
        T = len(ep["actions"])
        assert ep["frames"].shape == (T + 1, 64, 64)
        assert ep["frames"].dtype == np.uint8
        assert ep["ball_pos"].shape == (T + 1, 2)
        assert isinstance(bool(ep["ball_lost"]), bool)
        assert ep["actions"].min() >= 0 and ep["actions"].max() < 4


def test_random_play_produces_ball_losses(tmp_path):
    # crucial pour la tête danger : le jeu aléatoire doit perdre des balles
    env = PinballEnv(seed=1)
    policy = StickyRandomPolicy(np.random.default_rng(1))
    collect_dataset(env, policy, n_transitions=800, out_dir=tmp_path)
    episodes = load_episodes(tmp_path)
    losses = sum(bool(ep["ball_lost"]) for ep in episodes)
    assert losses >= 1
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_collect.py -v`
Expected: FAIL — `No module named 'pinball.collect'`.

- [ ] **Step 3: Implémenter `pinball/collect.py`**

```python
"""Collecte d'expérience : une politique aléatoire joue seule dans le
simulateur et on enregistre tout.

Pourquoi des actions « collantes » ? Une action retirée à chaque pas fait
vibrer les flippers sans jamais frapper : le dataset ne contiendrait aucun
exemple de frappe réussie et le modèle ne pourrait pas apprendre l'effet des
actions. On tire donc une action et on la MAINTIENT plusieurs pas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import PinballEnv


class StickyRandomPolicy:
    def __init__(self, rng: np.random.Generator,
                 hold_range: tuple[int, int] = (3, 15)):
        self._rng = rng
        self._hold_range = hold_range
        self.reset()

    def reset(self) -> None:
        self._action = 0
        self._hold = 0

    def __call__(self, obs) -> int:
        if self._hold <= 0:
            self._action = int(self._rng.integers(4))
            self._hold = int(self._rng.integers(self._hold_range[0],
                                                 self._hold_range[1] + 1))
        self._hold -= 1
        return self._action


def _play_episode(env: PinballEnv, policy) -> dict:
    obs = env.reset()
    policy.reset()
    frames = [obs[1]]           # frame brute du reset
    actions, positions = [], []
    info = {"ball_pos": None}
    while True:
        a = policy(obs)
        obs, info = env.step(a)
        actions.append(a)
        frames.append(obs[1])   # la frame "présent" du stack
        positions.append(info["ball_pos"])
        if info["done"]:
            break
    pos0 = positions[0]         # position au reset non exposée : duplique t=1
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "actions": np.asarray(actions, dtype=np.int64),
        "ball_pos": np.asarray([pos0] + positions, dtype=np.float32),
        "ball_lost": bool(info["ball_lost"]),
    }


def _write_shard(path: Path, episodes: list[dict]) -> None:
    np.savez_compressed(
        path,
        frames=np.concatenate([ep["frames"] for ep in episodes]),
        actions=np.concatenate([ep["actions"] for ep in episodes]),
        ball_pos=np.concatenate([ep["ball_pos"] for ep in episodes]),
        frame_counts=np.asarray([len(ep["frames"]) for ep in episodes]),
        action_counts=np.asarray([len(ep["actions"]) for ep in episodes]),
        ball_lost=np.asarray([ep["ball_lost"] for ep in episodes]),
    )


def collect_dataset(env: PinballEnv, policy, n_transitions: int,
                    out_dir, shard_episodes: int = 64) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer, n_done, n_eps, n_shards = [], 0, 0, 0
    while n_done < n_transitions:
        ep = _play_episode(env, policy)
        buffer.append(ep)
        n_done += len(ep["actions"])
        n_eps += 1
        if len(buffer) >= shard_episodes:
            _write_shard(out / f"shard_{n_shards:05d}.npz", buffer)
            n_shards += 1
            buffer = []
    if buffer:
        _write_shard(out / f"shard_{n_shards:05d}.npz", buffer)
        n_shards += 1
    return {"episodes": n_eps, "transitions": n_done, "shards": n_shards}


def load_episodes(data_dir) -> list[dict]:
    episodes = []
    for path in sorted(Path(data_dir).glob("shard_*.npz")):
        with np.load(path) as z:
            f_ofs = a_ofs = 0
            for fc, ac, lost in zip(z["frame_counts"], z["action_counts"],
                                    z["ball_lost"]):
                episodes.append({
                    "frames": z["frames"][f_ofs:f_ofs + fc],
                    "actions": z["actions"][a_ofs:a_ofs + ac],
                    "ball_pos": z["ball_pos"][f_ofs:f_ofs + fc],
                    "ball_lost": bool(lost),
                })
                f_ofs += fc
                a_ofs += ac
    return episodes
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_collect.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add pinball/collect.py tests/test_collect.py
git commit -m "feat: collecte d'experience (politique collante, shards npz par episodes)"
```

---

### Task 8: Notebook 02 — la collecte

**Files:**
- Create: `notebooks/02_collecte.py` + `notebooks/02_collecte.ipynb`

**Interfaces:**
- Consumes: `collect_dataset`, `load_episodes`, `StickyRandomPolicy`, `PinballEnv`.
- Produces: notebook qui écrit le dataset dans `data/v1/` (local) ou
  `/content/drive/MyDrive/jepa_pinball/data/v1/` (Colab).

- [ ] **Step 1: Écrire `notebooks/02_collecte.py`**

```python
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
    subprocess.run(["git", "clone", "https://github.com/VOTRE_COMPTE/jepa_play.git"], check=True)
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
```

- [ ] **Step 2: Générer le .ipynb**

Run: `jupytext --to ipynb notebooks/02_collecte.py`
Expected: crée `notebooks/02_collecte.ipynb`.

- [ ] **Step 3: Test d'intégration local rapide**

Run: `python -c "
from pathlib import Path
import numpy as np
from pinball.collect import StickyRandomPolicy, collect_dataset, load_episodes
from pinball.env import PinballEnv
import tempfile
with tempfile.TemporaryDirectory() as d:
    env = PinballEnv(seed=42)
    stats = collect_dataset(env, StickyRandomPolicy(np.random.default_rng(42)), 3000, d)
    eps = load_episodes(d)
    lengths = [len(e['actions']) for e in eps]
    losses = sum(e['ball_lost'] for e in eps)
    print(f'{stats} | duree moy {np.mean(lengths):.0f} pas | {losses}/{len(eps)} pertes')
    assert losses / len(eps) > 0.5, 'le jeu aleatoire doit perdre la majorite des balles'
"`
Expected: stats affichées, assertion OK (durée moyenne attendue ≈ 60-120 pas,
majorité d'épisodes perdus).

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_collecte.py notebooks/02_collecte.ipynb
git commit -m "docs: notebook 02, collecte du dataset et controle qualite"
```

---

### Task 9: Datasets PyTorch — fenêtres et labels danger

**Files:**
- Create: `jepa/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: la liste d'épisodes de `load_episodes` (Task 7).
- Produces:
  - `WindowDataset(episodes: list[dict], k: int = 8)` — `torch.utils.data.Dataset`.
    Item : `frames: uint8 (k+2, 64, 64)` (frames brutes f_{t-1}..f_{t+k}),
    `actions: int64 (k,)` (a_t..a_{t+k-1}). Indexe tous les `(ep, t)` avec
    `1 <= t` et `t + k <= T` (T = nb d'actions de l'épisode).
  - `DangerDataset(episodes: list[dict], k_danger: int = 10)` — Dataset.
    Item : `obs: uint8 (2, 64, 64)` (stack f_{t-1}, f_t), `label: float32`
    (1.0 si l'épisode se termine par `ball_lost` ET `t >= T - k_danger`, sinon 0.0),
    pour `1 <= t <= T`.
  - `stack_obs(frames: Tensor (B, k+2, H, W), i: int) -> Tensor (B, 2, H, W)` —
    utilitaire : l'observation empilée au pas i de la fenêtre (paires de frames
    consécutives ; i=0 → frames 0 et 1, qui correspondent à obs_t).

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_data.py
import numpy as np
import torch
from jepa.data import DangerDataset, WindowDataset, stack_obs


def fake_episode(T, ball_lost, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "frames": rng.integers(0, 255, (T + 1, 64, 64), dtype=np.uint8),
        "actions": rng.integers(0, 4, (T,)).astype(np.int64),
        "ball_pos": rng.uniform(0, 500, (T + 1, 2)).astype(np.float32),
        "ball_lost": ball_lost,
    }


def test_window_dataset_shapes_and_count():
    eps = [fake_episode(20, True), fake_episode(9, False)]
    ds = WindowDataset(eps, k=8)
    # ep1 : t dans [1, 12] -> 12 fenêtres ; ep2 : t dans [1, 1] -> 1 fenêtre
    assert len(ds) == 13
    item = ds[0]
    assert item["frames"].shape == (10, 64, 64) and item["frames"].dtype == torch.uint8
    assert item["actions"].shape == (8,) and item["actions"].dtype == torch.int64


def test_window_content_matches_episode():
    ep = fake_episode(20, True)
    ds = WindowDataset([ep], k=8)
    item = ds[0]  # premier index -> t=1
    assert np.array_equal(item["frames"].numpy(), ep["frames"][0:10])
    assert np.array_equal(item["actions"].numpy(), ep["actions"][1:9])


def test_stack_obs_pairs_consecutive_frames():
    frames = torch.arange(2 * 10 * 4 * 4, dtype=torch.uint8).reshape(2, 10, 4, 4)
    s0 = stack_obs(frames, 0)
    assert s0.shape == (2, 2, 4, 4)
    assert torch.equal(s0[:, 0], frames[:, 0]) and torch.equal(s0[:, 1], frames[:, 1])
    s3 = stack_obs(frames, 3)
    assert torch.equal(s3[:, 0], frames[:, 3]) and torch.equal(s3[:, 1], frames[:, 4])


def test_danger_labels():
    T = 30
    ds = DangerDataset([fake_episode(T, True)], k_danger=10)
    labels = np.array([ds[i]["label"].item() for i in range(len(ds))])
    assert len(ds) == T          # t de 1 à T
    assert labels.sum() == 10    # les 10 derniers pas sont dangereux
    assert labels[-1] == 1.0 and labels[0] == 0.0
    # épisode SANS perte de balle : aucun label positif
    ds2 = DangerDataset([fake_episode(T, False)], k_danger=10)
    labels2 = np.array([ds2[i]["label"].item() for i in range(len(ds2))])
    assert labels2.sum() == 0
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `No module named 'jepa.data'`.

- [ ] **Step 3: Implémenter `jepa/data.py`**

```python
"""Datasets PyTorch au-dessus des épisodes collectés.

Les shards stockent des frames BRUTES (une par pas). Les observations
empilées (2 frames) sont reconstruites ici, à la volée — le dataset sur
disque reste deux fois plus petit.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def stack_obs(frames: torch.Tensor, i: int) -> torch.Tensor:
    """Observation au pas i d'une fenêtre : paire (frame_i, frame_i+1).

    frames : (B, k+2, H, W). Retour : (B, 2, H, W).
    La fenêtre commence à f_{t-1}, donc i=0 donne obs_t, i=1 donne obs_{t+1}...
    """
    return frames[:, i:i + 2]


class WindowDataset(Dataset):
    """Fenêtres de trajectoires pour l'entraînement JEPA multi-pas."""

    def __init__(self, episodes: list[dict], k: int = 8):
        self.episodes = episodes
        self.k = k
        self.index: list[tuple[int, int]] = []
        for e, ep in enumerate(episodes):
            T = len(ep["actions"])
            for t in range(1, T - k + 1):
                self.index.append((e, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        return {
            "frames": torch.from_numpy(
                np.ascontiguousarray(ep["frames"][t - 1:t + self.k + 1])),
            "actions": torch.from_numpy(
                np.ascontiguousarray(ep["actions"][t:t + self.k])),
        }


class DangerDataset(Dataset):
    """Paires (observation, danger) pour la tête danger.

    label = 1 si la balle sera perdue dans les k_danger prochains pas.
    Les labels sont fabriqués automatiquement depuis la fin des épisodes —
    aucune annotation humaine.
    """

    def __init__(self, episodes: list[dict], k_danger: int = 10):
        self.episodes = episodes
        self.k_danger = k_danger
        self.index: list[tuple[int, int]] = []
        for e, ep in enumerate(episodes):
            T = len(ep["actions"])
            for t in range(1, T + 1):
                self.index.append((e, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        T = len(ep["actions"])
        dangerous = ep["ball_lost"] and t >= T - self.k_danger + 1
        return {
            "obs": torch.from_numpy(
                np.ascontiguousarray(ep["frames"][t - 1:t + 1])),
            "label": torch.tensor(1.0 if dangerous else 0.0),
        }
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_data.py -v`
Expected: 4 PASS.

Attention au compte de `test_danger_labels` : avec `t ∈ [1, T]` et la condition
`t >= T - k_danger + 1`, un épisode perdu de T=30 pas donne exactement 10
labels positifs (t = 21..30). Si le test échoue sur ce compte, corriger la
borne dans l'implémentation, pas dans le test.

- [ ] **Step 5: Commit**

```bash
git add jepa/data.py tests/test_data.py
git commit -m "feat: datasets fenetres JEPA et labels danger auto-generes"
```

---

### Task 10: Le modèle JEPA (encodeur, prédicteur, EMA)

**Files:**
- Create: `jepa/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `stack_obs` (Task 9).
- Produces:
  - `Encoder(in_ch: int = 2, z_dim: int = 256)` — `(B, 2, 64, 64)` uint8 ou float → `(B, 256)`.
    Convertit lui-même uint8 → float [0,1] si nécessaire. Sortie LayerNorm-ée.
  - `Predictor(z_dim: int = 256, n_actions: int = 4)` — `(z (B, 256), a (B,))` → `(B, 256)`.
    Résiduel + LayerNorm final.
  - `JEPA(z_dim: int = 256)` — attributs `encoder`, `predictor`, `target_encoder` ;
    - `loss(frames (B, k+2, H, W), actions (B, k)) -> tuple[Tensor, dict]` — perte
      scalaire + métriques `{"pred_mse", "copy_mse", "latent_std"}`.
    - `update_target(tau: float = 0.996) -> None` — EMA.
    - `encode(obs) -> Tensor` — encodeur ONLINE (l'entrée du prédicteur, comme
      à l'entraînement). `@torch.no_grad()`.
    - `encode_target(obs) -> Tensor` — encodeur cible EMA. `@torch.no_grad()`.

**Choix d'espace latent (à documenter dans le code et le notebook 03) :** le
prédicteur consomme un latent de l'encodeur ONLINE (z_t) et produit des
prédictions dans l'espace de l'encodeur CIBLE (z̄). La tête danger (Task 13)
est donc entraînée sur des latents CIBLE — car en planification elle ne verra
que des ẑ, qui approximent des z̄. Le planificateur encode l'observation
courante avec `encode()` (online), exactement comme à l'entraînement.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_model.py
import torch
from jepa.model import Encoder, JEPA, Predictor


def test_encoder_shapes_and_uint8():
    enc = Encoder()
    x8 = torch.randint(0, 255, (4, 2, 64, 64), dtype=torch.uint8)
    z = enc(x8)
    assert z.shape == (4, 256)
    xf = x8.float() / 255.0
    assert torch.allclose(enc(xf), z, atol=1e-5)


def test_encoder_param_count():
    n = sum(p.numel() for p in Encoder().parameters())
    assert 500_000 < n < 4_000_000  # "petit CNN" de la spec (~1-2M)


def test_predictor_shapes_and_action_sensitivity():
    torch.manual_seed(0)
    pred = Predictor()
    z = torch.randn(4, 256)
    a0 = torch.zeros(4, dtype=torch.long)
    a3 = torch.full((4,), 3, dtype=torch.long)
    out0, out3 = pred(z, a0), pred(z, a3)
    assert out0.shape == (4, 256)
    assert not torch.allclose(out0, out3)  # l'action doit influencer le futur


def test_jepa_loss_backward_and_metrics():
    torch.manual_seed(0)
    jepa = JEPA()
    frames = torch.randint(0, 255, (3, 10, 64, 64), dtype=torch.uint8)  # k=8
    actions = torch.randint(0, 4, (3, 8))
    loss, metrics = jepa.loss(frames, actions)
    assert loss.requires_grad
    loss.backward()
    assert {"pred_mse", "copy_mse", "latent_std"} <= metrics.keys()
    # l'encodeur cible ne reçoit JAMAIS de gradient
    assert all(p.grad is None for p in jepa.target_encoder.parameters())
    assert any(p.grad is not None for p in jepa.encoder.parameters())
    assert any(p.grad is not None for p in jepa.predictor.parameters())


def test_ema_update_moves_target():
    torch.manual_seed(0)
    jepa = JEPA()
    with torch.no_grad():
        for p in jepa.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in jepa.target_encoder.parameters()]
    jepa.update_target(tau=0.5)
    after = list(jepa.target_encoder.parameters())
    assert all(not torch.allclose(b, a) for b, a in zip(before, after))
    # tau=1 : la cible ne bouge plus
    frozen = [p.clone() for p in jepa.target_encoder.parameters()]
    jepa.update_target(tau=1.0)
    assert all(torch.allclose(f, a) for f, a in
               zip(frozen, jepa.target_encoder.parameters()))
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `No module named 'jepa.model'`.

- [ ] **Step 3: Implémenter `jepa/model.py`**

```python
"""Le cœur du projet : JEPA conditionné par l'action.

JEPA ne prédit pas les pixels du futur — il prédit sa REPRÉSENTATION.
Trois pièces :
  - Encoder (online)      : image -> z, entraîné par gradient ;
  - Predictor             : (z_t, action) -> ẑ_{t+1}, déroulé en chaîne ;
  - target_encoder (EMA)  : copie lente de l'encodeur, SANS gradient,
                            qui fabrique les cibles z̄.
L'EMA + stop-gradient est LE mécanisme anti-effondrement : sans lui, la
solution triviale z = constante annule la perte et tue la représentation.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .data import stack_obs


def _to_float(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 2, z_dim: int = 256):
        super().__init__()
        chans = [32, 64, 128, 256]
        layers, prev = [], in_ch
        for c in chans:
            layers += [nn.Conv2d(prev, c, 3, stride=2, padding=1),
                       nn.GroupNorm(8, c), nn.SiLU()]
            prev = c
        self.convs = nn.Sequential(*layers)          # 64 -> 4x4
        self.fc = nn.Linear(256 * 4 * 4, z_dim)
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(_to_float(x))
        return self.norm(self.fc(h.flatten(1)))


class Predictor(nn.Module):
    """(z_t, a_t) -> ẑ_{t+1}. Résiduel : il prédit le CHANGEMENT d'état."""

    def __init__(self, z_dim: int = 256, n_actions: int = 4, a_dim: int = 32):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, a_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + a_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, z_dim),
        )
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(torch.cat([z, self.action_emb(a)], dim=-1))
        return self.norm(z + delta)


class JEPA(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.predictor = Predictor(z_dim=z_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def loss(self, frames: torch.Tensor, actions: torch.Tensor):
        """Rollout multi-pas dans le latent, perte à chaque pas.

        frames : (B, k+2, H, W) — f_{t-1} .. f_{t+k} ; actions : (B, k).
        """
        k = actions.shape[1]
        z = self.encoder(stack_obs(frames, 0))          # z_t (online)
        total, pred_mse = 0.0, []
        with torch.no_grad():
            z_prev_target = self.target_encoder(stack_obs(frames, 0))
        copy_mse, latent_std = [], None
        for i in range(k):
            z = self.predictor(z, actions[:, i])        # ẑ_{t+i+1}
            with torch.no_grad():
                target = self.target_encoder(stack_obs(frames, i + 1))
            step_loss = nn.functional.mse_loss(z, target)
            total = total + step_loss
            pred_mse.append(step_loss.detach())
            # baseline naïve « le futur = le présent » : erreur si on avait
            # simplement recopié z̄_t — le prédicteur doit faire mieux
            copy_mse.append(nn.functional.mse_loss(z_prev_target, target))
            if i == 0:
                latent_std = target.std(dim=0).mean()
        metrics = {
            "pred_mse": torch.stack(pred_mse).mean().item(),
            "copy_mse": torch.stack(copy_mse).mean().item(),
            "latent_std": latent_std.item(),
        }
        return total / k, metrics

    @torch.no_grad()
    def update_target(self, tau: float = 0.996) -> None:
        for p, tp in zip(self.encoder.parameters(),
                         self.target_encoder.parameters()):
            tp.lerp_(p, 1.0 - tau)

    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Latent online — l'entrée du prédicteur (planification)."""
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Latent cible EMA — l'espace des prédictions (tête danger)."""
        return self.target_encoder(obs)
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_model.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add jepa/model.py tests/test_model.py
git commit -m "feat: modele JEPA (encodeur CNN, predicteur residuel, cible EMA)"
```

---

### Task 11: Boucle d'entraînement, checkpoints, anti-collapse

**Files:**
- Create: `jepa/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `JEPA` (Task 10), `WindowDataset` (Task 9), épisodes (Task 7).
- Produces:
  - `train_jepa(episodes, out_dir, epochs: int = 10, k: int = 8, batch_size: int = 256,
    lr: float = 3e-4, tau: float = 0.996, device: str | None = None,
    resume: bool = True, num_workers: int = 2) -> tuple[JEPA, list[dict]]`
    — entraîne, checkpointe à CHAQUE epoch dans `out_dir/jepa.pt`, reprend
    automatiquement si le checkpoint existe. Retourne (modèle, historique) ;
    historique = un dict par epoch : `{"epoch", "loss", "pred_mse", "copy_mse", "latent_std"}`.
  - `load_jepa(ckpt_path, device: str | None = None) -> JEPA` — recharge un
    checkpoint en mode eval.

Détails d'implémentation imposés :
- device auto : `"cuda" if torch.cuda.is_available() else "cpu"`.
- AMP (`torch.autocast` + `GradScaler`) uniquement si cuda.
- `update_target(tau)` après CHAQUE pas d'optimiseur.
- Avertissement explicite (print) si `latent_std < 0.05` en fin d'epoch :
  « ⚠ variance latente faible — collapse possible ».
- Le checkpoint contient : `model` (state_dict complet, cible EMA incluse),
  `optimizer`, `epoch`, `history`.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_train.py
import numpy as np
import torch
from jepa.train import load_jepa, train_jepa


def moving_dot_episode(T=20, seed=0):
    """Épisode synthétique APPRENABLE : un point qui descend, action = décalage x.

    Dynamique simple et déterministe pour que l'overfit soit possible sur CPU.
    """
    rng = np.random.default_rng(seed)
    frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
    actions = rng.integers(0, 4, (T,)).astype(np.int64)
    x, y = 32, 4
    frames[0, y, x] = 255
    for t in range(T):
        y = min(60, y + 2)
        x = int(np.clip(x + (actions[t] - 1) * 3, 2, 61))
        frames[t + 1, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 255
    return {"frames": frames, "actions": actions,
            "ball_pos": np.zeros((T + 1, 2), dtype=np.float32),
            "ball_lost": True}


def test_train_checkpoints_and_resumes(tmp_path):
    eps = [moving_dot_episode(T=15, seed=s) for s in range(3)]
    model, hist = train_jepa(eps, tmp_path, epochs=2, k=4, batch_size=8,
                             device="cpu", num_workers=0)
    assert len(hist) == 2
    assert (tmp_path / "jepa.pt").exists()
    # reprise : 1 epoch de plus seulement
    model2, hist2 = train_jepa(eps, tmp_path, epochs=3, k=4, batch_size=8,
                               device="cpu", num_workers=0)
    assert len(hist2) == 3 and hist2[:2] == hist


def test_overfit_small_dataset(tmp_path):
    # Critère de la spec : sur un petit dataset, la perte doit s'écraser
    # et la variance latente rester saine (pas de collapse).
    torch.manual_seed(0)
    eps = [moving_dot_episode(T=15, seed=s) for s in range(2)]
    _, hist = train_jepa(eps, tmp_path, epochs=40, k=4, batch_size=16,
                         lr=1e-3, device="cpu", num_workers=0)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    assert hist[-1]["latent_std"] > 0.01


def test_load_jepa_roundtrip(tmp_path):
    eps = [moving_dot_episode(T=15, seed=0)]
    model, _ = train_jepa(eps, tmp_path, epochs=1, k=4, batch_size=8,
                          device="cpu", num_workers=0)
    loaded = load_jepa(tmp_path / "jepa.pt", device="cpu")
    obs = torch.randint(0, 255, (2, 2, 64, 64), dtype=torch.uint8)
    assert torch.allclose(model.encode(obs), loaded.encode(obs), atol=1e-6)
    assert not loaded.training
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `No module named 'jepa.train'`.

- [ ] **Step 3: Implémenter `jepa/train.py`**

```python
"""Boucle d'entraînement JEPA : supervisée, stable, reprenable.

Conçue pour le Colab gratuit : AMP sur GPU, checkpoint à chaque epoch sur
Drive, reprise automatique après déconnexion.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import WindowDataset
from .model import JEPA


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_jepa(episodes, out_dir, epochs: int = 10, k: int = 8,
               batch_size: int = 256, lr: float = 3e-4, tau: float = 0.996,
               device: str | None = None, resume: bool = True,
               num_workers: int = 2):
    dev = _device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "jepa.pt"

    model = JEPA().to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    history: list[dict] = []
    start_epoch = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        print(f"reprise du checkpoint : epoch {start_epoch}")

    ds = WindowDataset(episodes, k=k)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, drop_last=True)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(start_epoch, epochs):
        agg = {"loss": 0.0, "pred_mse": 0.0, "copy_mse": 0.0, "latent_std": 0.0}
        n_batches = 0
        for batch in dl:
            frames = batch["frames"].to(dev, non_blocking=True)
            actions = batch["actions"].to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, metrics = model.loss(frames, actions)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.update_target(tau)
            agg["loss"] += loss.item()
            for key in ("pred_mse", "copy_mse", "latent_std"):
                agg[key] += metrics[key]
            n_batches += 1
        row = {"epoch": epoch + 1,
               **{key: val / max(n_batches, 1) for key, val in agg.items()}}
        history.append(row)
        if row["latent_std"] < 0.05:
            print("⚠ variance latente faible — collapse possible "
                  f"(latent_std={row['latent_std']:.4f})")
        print(f"epoch {row['epoch']}/{epochs}  loss={row['loss']:.4f}  "
              f"pred={row['pred_mse']:.4f}  copy={row['copy_mse']:.4f}  "
              f"std={row['latent_std']:.3f}")
        torch.save({"model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch + 1,
                    "history": history}, ckpt_path)
    return model, history


def load_jepa(ckpt_path, device: str | None = None) -> JEPA:
    dev = _device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model = JEPA().to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_train.py -v`
Expected: 3 PASS (l'overfit prend ~30-60 s sur CPU — acceptable).

Si `weights_only=True` pose problème (history contient des dicts Python
simples : OK normalement), passer `weights_only=False` — le checkpoint est
produit localement, pas de risque de désérialisation.

- [ ] **Step 5: Vérifier toute la suite**

Run: `pytest`
Expected: tout PASS.

- [ ] **Step 6: Commit**

```bash
git add jepa/train.py tests/test_train.py
git commit -m "feat: entrainement JEPA (AMP, EMA, checkpoints reprenables, alerte collapse)"
```

---

### Task 12: Notebook 03 — entraîner le JEPA

**Files:**
- Create: `notebooks/03_jepa.py` + `notebooks/03_jepa.ipynb`

**Interfaces:**
- Consumes: `load_episodes`, `train_jepa`, `load_jepa`, `WindowDataset`, `stack_obs`.
- Produces: notebook d'entraînement avec diagnostics anti-collapse et
  visualisation de la qualité de prédiction. Checkpoint dans
  `checkpoints/` (local) ou `MyDrive/jepa_pinball/checkpoints/` (Colab).

- [ ] **Step 1: Écrire `notebooks/03_jepa.py`**

```python
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
import importlib.util, subprocess, sys, os
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB and not os.path.exists("jepa_play"):
    subprocess.run(["git", "clone", "https://github.com/VOTRE_COMPTE/jepa_play.git"], check=True)
    os.chdir("jepa_play")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# %%
from pathlib import Path
import torch
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/jepa_pinball")
else:
    ROOT = Path(".")
DATA_DIR, CKPT_DIR = ROOT / "data/v1", ROOT / "checkpoints"
print("device :", "cuda" if torch.cuda.is_available() else "cpu (lent ! activer le GPU)")

# %%
from pinball.collect import load_episodes
episodes = load_episodes(DATA_DIR)
print(f"{len(episodes)} épisodes, "
      f"{sum(len(e['actions']) for e in episodes)} transitions")

# %% [markdown]
# ## Entraînement
#
# ~20-40 min sur T4. Le checkpoint est écrit sur Drive à CHAQUE epoch :
# si Colab déconnecte, relancer cette cellule reprend où on en était.

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
```

- [ ] **Step 2: Générer le .ipynb**

Run: `jupytext --to ipynb notebooks/03_jepa.py`
Expected: crée `notebooks/03_jepa.ipynb`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_jepa.py notebooks/03_jepa.ipynb
git commit -m "docs: notebook 03, entrainement JEPA et diagnostics anti-collapse"
```

---

### Task 13: La tête danger

**Files:**
- Create: `jepa/heads.py`
- Test: `tests/test_heads.py`

**Interfaces:**
- Consumes: `JEPA.encode_target` (Task 10), `DangerDataset` (Task 9).
- Produces:
  - `auc(scores: np.ndarray, labels: np.ndarray) -> float` — AUC ROC par
    statistique de rangs (pas de dépendance sklearn).
  - `DangerHead(z_dim: int = 256)` — MLP 256→128→1, `forward(z) -> logits (B,)`.
  - `train_danger_head(jepa, episodes, k_danger: int = 10, epochs: int = 3,
    batch_size: int = 512, lr: float = 1e-3, val_fraction: float = 0.1,
    device: str | None = None) -> tuple[DangerHead, float]` — encode les
    observations avec l'encodeur CIBLE (gelé), entraîne la tête en BCE avec
    `pos_weight` (classes déséquilibrées), retourne `(head, val_auc)`.
    Le split validation se fait PAR ÉPISODE (pas par transition — sinon fuite
    entre train et val, les pas voisins étant quasi identiques).

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_heads.py
import numpy as np
import torch
from jepa.heads import DangerHead, auc, train_danger_head
from jepa.model import JEPA


def test_auc_known_values():
    labels = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0   # parfait
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0   # inversé
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(auc(scores, labels) - 0.5) < 1e-9                # hasard


def test_danger_head_shapes():
    head = DangerHead()
    logits = head(torch.randn(7, 256))
    assert logits.shape == (7,)


def separable_episode(T, ball_lost, seed):
    """La frame encode grossièrement le danger : bande basse allumée en fin
    d'épisode perdu — un signal qu'un encodeur même non entraîné transmet."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
    for t in range(T + 1):
        y = 10 if not (ball_lost and t > T - 10) else 58
        frames[t, y - 2:y + 2, 20:44] = 255
    return {"frames": frames,
            "actions": rng.integers(0, 4, (T,)).astype(np.int64),
            "ball_pos": np.zeros((T + 1, 2), dtype=np.float32),
            "ball_lost": ball_lost}


def test_train_danger_head_learns_separable_signal():
    torch.manual_seed(0)
    eps = [separable_episode(30, s % 2 == 0, s) for s in range(20)]
    jepa = JEPA()  # encodeur NON entraîné : projection aléatoire, suffisant ici
    head, val_auc = train_danger_head(jepa, eps, epochs=10, batch_size=64,
                                      device="cpu")
    assert val_auc > 0.9
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_heads.py -v`
Expected: FAIL — `No module named 'jepa.heads'`.

- [ ] **Step 3: Implémenter `jepa/heads.py`**

```python
"""Tête danger : P(balle perdue dans les k prochains pas | latent).

C'est elle qui donne un SENS au futur imaginé : le prédicteur déroule des
latents, la tête danger dit lesquels mènent au drain. Entraînement supervisé
classique — les labels sortent gratuitement du dataset (fins d'épisodes).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DangerDataset


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC ROC par statistique de Mann-Whitney (gère les ex aequo)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # rang moyen des ex aequo
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


class DangerHead(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)


def _encode_dataset(jepa, dataset, batch_size, device):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs, ys = [], []
    for batch in dl:
        obs = batch["obs"].to(device)
        # espace CIBLE : c'est là que vivront les prédictions ẑ du planificateur
        zs.append(jepa.encode_target(obs).cpu())
        ys.append(batch["label"])
    return torch.cat(zs), torch.cat(ys)


def train_danger_head(jepa, episodes, k_danger: int = 10, epochs: int = 3,
                      batch_size: int = 512, lr: float = 1e-3,
                      val_fraction: float = 0.1, device: str | None = None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    jepa = jepa.to(dev).eval()
    # split PAR ÉPISODE : deux pas voisins sont quasi identiques, un split
    # par transition ferait fuir le train dans la validation
    n_val = max(1, int(len(episodes) * val_fraction))
    train_eps, val_eps = episodes[:-n_val], episodes[-n_val:]
    z_train, y_train = _encode_dataset(jepa, DangerDataset(train_eps, k_danger),
                                       batch_size, dev)
    z_val, y_val = _encode_dataset(jepa, DangerDataset(val_eps, k_danger),
                                   batch_size, dev)

    head = DangerHead(z_train.shape[1]).to(dev)
    n_pos = float(y_train.sum().item())
    pos_weight = torch.tensor((len(y_train) - n_pos) / max(n_pos, 1.0),
                              device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    z_train, y_train = z_train.to(dev), y_train.to(dev)
    for _ in range(epochs):
        perm = torch.randperm(len(z_train), device=dev)
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(head(z_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        scores = torch.sigmoid(head(z_val.to(dev))).cpu().numpy()
    val_auc = auc(scores, y_val.numpy())
    print(f"tête danger : AUC validation = {val_auc:.3f} "
          f"({int(y_val.sum())} positifs / {len(y_val)})")
    return head, val_auc
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_heads.py -v`
Expected: 3 PASS.

Note : `test_train_danger_head_learns_separable_signal` utilise un encodeur
aléatoire — c'est voulu (une projection aléatoire préserve un signal aussi
grossier). S'il est flaky, augmenter `epochs` à 20 dans le test, pas ailleurs.

- [ ] **Step 5: Commit**

```bash
git add jepa/heads.py tests/test_heads.py
git commit -m "feat: tete danger (BCE pos_weight, split par episode, AUC maison)"
```

---

### Task 14: Le planificateur MPC

**Files:**
- Create: `jepa/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `JEPA` (`encode`, `predictor`), `DangerHead`.
- Produces: `MPCPlanner(jepa, danger_head, horizon: int = 8, n_candidates: int = 64,
  switch_prob: float = 0.2, device: str | None = None, seed: int = 0)` :
  - `plan(obs: np.ndarray (2, 64, 64) uint8) -> int` — l'action à exécuter.
  - `reset() -> None` — no-op (uniformité avec les autres politiques).
  - `__call__(obs) -> int` — alias de `plan` (interface politique commune).

Fonctionnement (aucun apprentissage) :
1. générer `n_candidates` séquences d'actions `(N, horizon)` : première action
   uniforme, puis à chaque pas, probabilité `switch_prob` de changer d'action
   (persistance, comme la politique de collecte) ; les 4 séquences CONSTANTES
   sont toujours incluses en tête (garantit que « tenir le flipper » est considéré) ;
2. encoder l'observation (`encode`, online) et répliquer z sur N ;
3. dérouler le prédicteur `horizon` pas, en accumulant
   `cost += sigmoid(danger(ẑ))` à chaque pas ;
4. retourner la première action de la séquence de coût minimal.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# tests/test_planner.py
import numpy as np
import torch
import torch.nn as nn
from jepa.model import JEPA
from jepa.planner import MPCPlanner


class OracleDanger(nn.Module):
    """Tête factice : danger = -z[0]. Pilote le planificateur de façon connue."""

    def forward(self, z):
        return -z[..., 0]


class CountingPredictor(nn.Module):
    """Prédicteur factice : l'action 2 augmente z[0] (donc baisse le danger),
    les autres le diminuent. La meilleure séquence est donc 'toujours 2'."""

    def forward(self, z, a):
        delta = torch.where(a == 2, 1.0, -1.0)
        out = z.clone()
        out[..., 0] = out[..., 0] + delta
        return out


def make_planner(**kwargs):
    jepa = JEPA()
    jepa.predictor = CountingPredictor()
    return MPCPlanner(jepa, OracleDanger(), device="cpu", **kwargs)


def test_planner_returns_valid_action():
    planner = make_planner()
    obs = np.random.default_rng(0).integers(0, 255, (2, 64, 64)).astype(np.uint8)
    a = planner.plan(obs)
    assert a in (0, 1, 2, 3)


def test_planner_picks_the_least_dangerous_action():
    planner = make_planner(n_candidates=64)
    obs = np.zeros((2, 64, 64), dtype=np.uint8)
    # avec le prédicteur factice, la séquence constante '2' minimise le coût
    assert planner.plan(obs) == 2


def test_constant_sequences_always_candidates():
    # même avec 0 candidat aléatoire, les 4 constantes suffisent
    planner = make_planner(n_candidates=0)
    obs = np.zeros((2, 64, 64), dtype=np.uint8)
    assert planner.plan(obs) == 2


def test_planner_is_deterministic_given_seed():
    obs = np.random.default_rng(1).integers(0, 255, (2, 64, 64)).astype(np.uint8)
    a1 = make_planner(seed=5).plan(obs)
    a2 = make_planner(seed=5).plan(obs)
    assert a1 == a2
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL — `No module named 'jepa.planner'`.

- [ ] **Step 3: Implémenter `jepa/planner.py`**

```python
"""Planification MPC dans l'espace latent : l'agent « imagine et choisit ».

À chaque pas : générer des séquences d'actions candidates, les dérouler
DANS LE LATENT avec le prédicteur (aucune physique !), sommer le danger
prédit, exécuter la première action de la meilleure séquence, replanifier.
Aucun apprentissage ici — tout le savoir est dans le world model et la tête.
"""
from __future__ import annotations

import numpy as np
import torch


class MPCPlanner:
    def __init__(self, jepa, danger_head, horizon: int = 8,
                 n_candidates: int = 64, switch_prob: float = 0.2,
                 device: str | None = None, seed: int = 0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.jepa = jepa.to(self.device).eval()
        self.danger_head = danger_head.to(self.device).eval()
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.switch_prob = switch_prob
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def _candidate_sequences(self) -> np.ndarray:
        """(4 + n_candidates, horizon) : constantes d'abord, puis persistantes."""
        constants = np.repeat(np.arange(4)[:, None], self.horizon, axis=1)
        if self.n_candidates == 0:
            return constants
        seqs = np.empty((self.n_candidates, self.horizon), dtype=np.int64)
        seqs[:, 0] = self._rng.integers(4, size=self.n_candidates)
        for t in range(1, self.horizon):
            switch = self._rng.random(self.n_candidates) < self.switch_prob
            seqs[:, t] = np.where(switch, self._rng.integers(4, size=self.n_candidates),
                                  seqs[:, t - 1])
        return np.concatenate([constants, seqs])

    @torch.no_grad()
    def plan(self, obs: np.ndarray) -> int:
        seqs = self._candidate_sequences()
        actions = torch.from_numpy(seqs).to(self.device)
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0)
        z0 = self.jepa.encode(obs_t.to(self.device))
        z = z0.expand(len(seqs), -1).contiguous()
        cost = torch.zeros(len(seqs), device=self.device)
        for t in range(self.horizon):
            z = self.jepa.predictor(z, actions[:, t])
            cost += torch.sigmoid(self.danger_head(z))
        best = int(torch.argmin(cost).item())
        return int(seqs[best, 0])

    __call__ = plan
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `pytest tests/test_planner.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add jepa/planner.py tests/test_planner.py
git commit -m "feat: planificateur MPC (random shooting persistant + constantes)"
```

---

### Task 15: Évaluation, baselines et vidéos

**Files:**
- Create: `jepa/eval.py`
- Test: intégré (smoke test dans le step 4 — les vrais chiffres sortent au notebook 04)

**Interfaces:**
- Consumes: `PinballEnv`, `StickyRandomPolicy`, `MPCPlanner`, `render_debug`.
- Produces:
  - `AlwaysPressed()` — politique baseline : `__call__(obs) -> 3`, `reset()`.
  - `run_episode(env, policy, seed: int | None = None) -> dict` —
    `{"steps": int, "ball_lost": bool, "stuck": bool}`.
  - `evaluate(env, policy, n_episodes: int = 50, seed0: int = 1000) -> dict` —
    épisodes avec seeds `seed0..seed0+n-1` (mêmes seeds pour toutes les
    politiques → comparaison appariée) ;
    `{"mean_steps", "median_steps", "survival_s", "loss_rate", "lengths"}`.
  - `record_gif(env, policy, path, seed: int | None = None, max_steps: int = 450) -> dict` —
    GIF au rendu debug, retourne le dict de `run_episode`.

- [ ] **Step 1: Implémenter `jepa/eval.py`**

```python
"""Évaluation : l'agent JEPA contre les baselines, à seeds appariées.

Le graphique de victoire de la V1 : temps de survie moyen de l'agent
NETTEMENT au-dessus de la politique aléatoire et de « toujours appuyé ».
"""
from __future__ import annotations

import numpy as np

from pinball.render import render_debug


class AlwaysPressed:
    """Baseline : les deux flippers levés en permanence."""

    def reset(self) -> None:
        pass

    def __call__(self, obs) -> int:
        return 3


def run_episode(env, policy, seed: int | None = None) -> dict:
    obs = env.reset(seed=seed)
    policy.reset()
    while True:
        obs, info = env.step(policy(obs))
        if info["done"]:
            return {"steps": info["steps"], "ball_lost": info["ball_lost"],
                    "stuck": info["stuck"]}


def evaluate(env, policy, n_episodes: int = 50, seed0: int = 1000) -> dict:
    results = [run_episode(env, policy, seed=seed0 + i)
               for i in range(n_episodes)]
    lengths = np.array([r["steps"] for r in results])
    hz = env.config.physics_hz / env.config.frame_skip
    return {
        "mean_steps": float(lengths.mean()),
        "median_steps": float(np.median(lengths)),
        "survival_s": float(lengths.mean() / hz),
        "loss_rate": float(np.mean([r["ball_lost"] for r in results])),
        "lengths": lengths,
    }


def record_gif(env, policy, path, seed: int | None = None,
               max_steps: int = 450) -> dict:
    obs = env.reset(seed=seed)
    policy.reset()
    frames = [render_debug(env.sim)]
    info = {"steps": 0, "ball_lost": False, "stuck": False}
    for _ in range(max_steps):
        obs, info = env.step(policy(obs))
        frames.append(render_debug(env.sim))
        if info["done"]:
            break
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=66, loop=0)
    return {"steps": info["steps"], "ball_lost": info["ball_lost"],
            "stuck": info["stuck"]}
```

- [ ] **Step 2: Smoke test (baselines uniquement, sans modèle entraîné)**

Run: `python -c "
from pinball.collect import StickyRandomPolicy
from pinball.env import PinballEnv
from jepa.eval import AlwaysPressed, evaluate
import numpy as np
env = PinballEnv(seed=0)
r = evaluate(env, StickyRandomPolicy(np.random.default_rng(0)), n_episodes=8)
print('aléatoire :', round(r['survival_s'], 1), 's')
a = evaluate(env, AlwaysPressed(), n_episodes=8)
print('toujours appuyé :', round(a['survival_s'], 1), 's')
"`
Expected: deux durées de survie s'affichent (ordre de grandeur : 4-10 s
chacune), sans erreur.

- [ ] **Step 3: Commit**

```bash
git add jepa/eval.py
git commit -m "feat: evaluation a seeds appariees, baselines et export GIF"
```

---

### Task 16: Notebook 04 — l'agent joue (V1)

**Files:**
- Create: `notebooks/04_controle.py` + `notebooks/04_controle.ipynb`

**Interfaces:**
- Consumes: tout ce qui précède.
- Produces: le notebook de la V1 : entraîne la tête danger, fait jouer l'agent,
  produit le graphique d'acceptation et les vidéos.

- [ ] **Step 1: Écrire `notebooks/04_controle.py`**

```python
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
# D'abord un coup d'œil qualitatif : trois GIF côte à côte.

# %%
from pinball.env import PinballEnv
from pinball.collect import StickyRandomPolicy
from jepa.eval import AlwaysPressed, evaluate, record_gif
from jepa.planner import MPCPlanner

agent = MPCPlanner(jepa, head)
env = PinballEnv()
for name, pol in [("agent", agent),
                  ("aleatoire", StickyRandomPolicy(np.random.default_rng(0))),
                  ("toujours", AlwaysPressed())]:
    r = record_gif(env, pol, f"{name}.gif", seed=2026)
    print(f"{name:10s}: {r['steps']} pas ({r['steps']/15:.1f} s)")

# %%
from IPython.display import Image as IPImage, display
for name in ("agent", "aleatoire", "toujours"):
    print(name); display(IPImage(f"{name}.gif"))

# %% [markdown]
# ## Le graphique de la V1 : 50 épisodes, seeds appariées
#
# **Critère d'acceptation** : la survie moyenne de l'agent doit dépasser
# NETTEMENT les deux baselines (spec §2 et §12).

# %%
import matplotlib.pyplot as plt
results = {}
for name, pol in [("agent JEPA", agent),
                  ("aléatoire", StickyRandomPolicy(np.random.default_rng(0))),
                  ("toujours appuyé", AlwaysPressed())]:
    results[name] = evaluate(env, pol, n_episodes=50)
    print(f"{name:16s}: {results[name]['survival_s']:5.1f} s en moyenne, "
          f"médiane {results[name]['median_steps']/15:.1f} s")

fig, ax = plt.subplots(figsize=(7, 4))
names = list(results)
ax.bar(names, [results[n]["survival_s"] for n in names],
       color=["tab:green", "tab:gray", "tab:gray"])
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
```

- [ ] **Step 2: Générer le .ipynb**

Run: `jupytext --to ipynb notebooks/04_controle.py`
Expected: crée `notebooks/04_controle.ipynb`.

- [ ] **Step 3: Test d'intégration bout-en-bout (mini, CPU)**

Vérifie que TOUTE la chaîne tient debout sur un mini-dataset local —
sans prétendre à la performance (ça, c'est le job du notebook sur T4) :

Run: `python -c "
import tempfile, numpy as np, torch
from pathlib import Path
from pinball.collect import StickyRandomPolicy, collect_dataset, load_episodes
from pinball.env import PinballEnv
from jepa.train import train_jepa
from jepa.heads import train_danger_head
from jepa.planner import MPCPlanner
from jepa.eval import evaluate
with tempfile.TemporaryDirectory() as d:
    env = PinballEnv(seed=0)
    collect_dataset(env, StickyRandomPolicy(np.random.default_rng(0)), 2000, d)
    eps = load_episodes(d)
    jepa, hist = train_jepa(eps, Path(d)/'ckpt', epochs=2, batch_size=64, device='cpu', num_workers=0)
    head, auc_val = train_danger_head(jepa, eps, epochs=3, device='cpu')
    agent = MPCPlanner(jepa, head, n_candidates=16, device='cpu')
    r = evaluate(env, agent, n_episodes=3)
    print('OK — chaîne complète, survie mini-agent :', round(r['survival_s'],1), 's')
"`
Expected: `OK — chaîne complète` s'affiche (2-5 min CPU). Aucune assertion de
performance — uniquement l'absence d'erreur d'intégration.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_controle.py notebooks/04_controle.ipynb
git commit -m "docs: notebook 04, tete danger + agent MPC + graphique V1"
```

---

## Validation de la V1 (manuelle, sur Colab)

Après la Task 16, la validation finale se fait sur Colab T4 (pas en CI) :

1. Pousser le repo sur GitHub ; remplacer `VOTRE_COMPTE` dans les 4 notebooks.
2. Notebook 02 : collecter les 100k transitions sur Drive (~10-30 min).
3. Notebook 03 : entraîner 10 epochs (~20-40 min), vérifier les 3 diagnostics.
4. Notebook 04 : **critère d'acceptation** — survie moyenne de l'agent
   au-dessus des deux baselines sur 50 épisodes à seeds appariées.
   Ordres de grandeur attendus : aléatoire ≈ 5-8 s ; l'agent doit faire
   au moins 2× mieux, sinon suivre la checklist de diagnostic du notebook.
5. Marquer la V1 : `git tag v1 && git push --tags`.

## Hors périmètre de ce plan

Phase 4 (V2 score) : bumpers dans `BoardConfig` (champ `bumpers` déjà
réservé), collision handler pymunk pour compter les points, tête score,
coût MPC combiné `danger − λ·score`, notebook 05. Fera l'objet d'un plan
séparé après validation de la V1 — les interfaces ci-dessus (info dict,
BoardConfig, coût du planner) ont été conçues pour l'accueillir sans rupture.
