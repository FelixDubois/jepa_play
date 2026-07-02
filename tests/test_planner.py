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
