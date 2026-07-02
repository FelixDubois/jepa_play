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
                 device: str | None = None, seed: int = 0,
                 height_head=None, target_head=None,
                 w_danger: float = 1.0, w_height: float = 0.5,
                 w_target: float = 2.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.jepa = jepa.to(self.device).eval()
        self.danger_head = danger_head.to(self.device).eval()
        self.height_head = height_head.to(self.device).eval() if height_head is not None else None
        self.target_head = target_head.to(self.device).eval() if target_head is not None else None
        self.w_danger = w_danger
        self.w_height = w_height
        self.w_target = w_target
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
            cost += self.w_danger * torch.sigmoid(self.danger_head(z))
            if self.height_head is not None:
                cost -= self.w_height * self.height_head(z)
            if self.target_head is not None:
                cost -= self.w_target * torch.sigmoid(self.target_head(z))
        best = int(torch.argmin(cost).item())
        return int(seqs[best, 0])

    __call__ = plan
