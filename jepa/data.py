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
