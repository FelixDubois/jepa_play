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
    if len(episodes) - n_val < 1:
        raise ValueError("il faut au moins 2 épisodes pour un split "
                         "train/validation par épisode")
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
    # clamp : sinon batch_size > données => zéro pas d'optimiseur, en silence
    batch_size = min(batch_size, len(z_train))
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
