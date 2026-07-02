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

    Version V1 historique — préférer MultiLabelDataset (danger honnête
    incluant les fins stuck).
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
