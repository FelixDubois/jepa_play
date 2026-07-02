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


def test_planner_returns_first_action_of_best_sequence():
    # Le gagnant est une séquence NON constante : première action 0, puis
    # que des 2. Épingle l'extraction seqs[best, 0] — un bug seqs[best, -1]
    # retournerait 2. (Avec les mocks, la séquence B = [0,2,2,...] domine
    # A = [2,0,0,...] point à point à partir du 3e pas.)
    planner = make_planner(n_candidates=0)
    seq_a = np.full((1, planner.horizon), 0, dtype=np.int64)
    seq_a[0, 0] = 2
    seq_b = np.full((1, planner.horizon), 2, dtype=np.int64)
    seq_b[0, 0] = 0
    planner._candidate_sequences = lambda: np.concatenate([seq_a, seq_b])
    obs = np.zeros((2, 64, 64), dtype=np.uint8)
    assert planner.plan(obs) == 0


def test_candidate_sequences_use_instance_seed():
    # Épingle le câblage du RNG par instance (le test de déterminisme
    # ci-dessus passerait même sans seed : le gagnant y est insensible
    # aux candidats aléatoires).
    a = make_planner(seed=5)._candidate_sequences()
    b = make_planner(seed=5)._candidate_sequences()
    c = make_planner(seed=6)._candidate_sequences()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# Task 27: Multi-objective planner tests


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
