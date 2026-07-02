"""Têtes d'objectif (danger, hauteur, cible, position) + entraînement partagé.

Ce sont elles qui donnent un SENS au futur imaginé : le prédicteur déroule des
latents, et chaque tête dit ce qui s'y joue — danger (drain), hauteur, contact
avec une cible, position sur le plateau. `train_objective_heads` les entraîne
toutes les quatre sur un seul passage d'encodage. Entraînement supervisé
classique — les labels sortent gratuitement du dataset (fins d'épisodes,
positions, contacts).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DangerDataset, MultiLabelDataset


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


class HeightHead(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(z).squeeze(-1))


class TargetHead(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)


class PositionProbe(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(z))


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


def _encode_multi_dataset(jepa, dataset, batch_size, device):
    """Comme `_encode_dataset`, mais pour `MultiLabelDataset` : un seul passage
    d'encodage produit les latents ET les 4 jeux de labels partagés par les
    têtes d'objectif."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    keys = ("danger", "height", "pos", "target")
    zs = []
    ys: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    for batch in dl:
        obs = batch["obs"].to(device)
        zs.append(jepa.encode_target(obs).cpu())
        for k in keys:
            ys[k].append(batch[k])
    z = torch.cat(zs)
    labels = {k: torch.cat(v) for k, v in ys.items()}
    return z, labels


def train_objective_heads(jepa, episodes, k_danger: int = 10, k_target: int = 10,
                          epochs: int = 3, batch_size: int = 512, lr: float = 1e-3,
                          val_fraction: float = 0.1, device: str | None = None):
    """Entraîne les 4 têtes d'objectif (danger, hauteur, cible, position) sur
    des latents partagés : un seul encodage du dataset, 4 optimiseurs
    indépendants qui avancent côte à côte sur le même mini-batch de latents.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    jepa = jepa.to(dev).eval()
    # split PAR ÉPISODE : deux pas voisins sont quasi identiques, un split
    # par transition ferait fuir le train dans la validation
    n_val = max(1, int(len(episodes) * val_fraction))
    if len(episodes) - n_val < 1:
        raise ValueError("il faut au moins 2 épisodes pour un split "
                         "train/validation par épisode")
    train_eps, val_eps = episodes[:-n_val], episodes[-n_val:]
    z_train, y_train = _encode_multi_dataset(
        jepa, MultiLabelDataset(train_eps, k_danger, k_target), batch_size, dev)
    z_val, y_val = _encode_multi_dataset(
        jepa, MultiLabelDataset(val_eps, k_danger, k_target), batch_size, dev)

    z_dim = z_train.shape[1]
    heads = {
        "danger": DangerHead(z_dim).to(dev),
        "height": HeightHead(z_dim).to(dev),
        "target": TargetHead(z_dim).to(dev),
        "pos": PositionProbe(z_dim).to(dev),
    }
    opts = {name: torch.optim.AdamW(head.parameters(), lr=lr)
            for name, head in heads.items()}

    n_pos_danger = float(y_train["danger"].sum().item())
    pos_weight_danger = torch.tensor(
        (len(y_train["danger"]) - n_pos_danger) / max(n_pos_danger, 1.0),
        device=dev)
    loss_danger = nn.BCEWithLogitsLoss(pos_weight=pos_weight_danger)
    loss_height = nn.MSELoss()
    loss_pos = nn.MSELoss()

    # aucun positif cible dans le train : rien à apprendre, on saute
    n_pos_target = float(y_train["target"].sum().item())
    train_target = n_pos_target > 0
    if train_target:
        pos_weight_target = torch.tensor(
            (len(y_train["target"]) - n_pos_target) / max(n_pos_target, 1.0),
            device=dev)
        loss_target = nn.BCEWithLogitsLoss(pos_weight=pos_weight_target)

    z_train = z_train.to(dev)
    y_train = {k: v.to(dev) for k, v in y_train.items()}
    # clamp : sinon batch_size > données => zéro pas d'optimiseur, en silence
    batch_size = min(batch_size, len(z_train))
    for _ in range(epochs):
        perm = torch.randperm(len(z_train), device=dev)
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            zb = z_train[idx]

            opts["danger"].zero_grad(set_to_none=True)
            loss_danger(heads["danger"](zb), y_train["danger"][idx]).backward()
            opts["danger"].step()

            opts["height"].zero_grad(set_to_none=True)
            loss_height(heads["height"](zb), y_train["height"][idx]).backward()
            opts["height"].step()

            opts["pos"].zero_grad(set_to_none=True)
            loss_pos(heads["pos"](zb), y_train["pos"][idx]).backward()
            opts["pos"].step()

            if train_target:
                opts["target"].zero_grad(set_to_none=True)
                loss_target(heads["target"](zb),
                           y_train["target"][idx]).backward()
                opts["target"].step()

    for head in heads.values():
        head.eval()
    z_val_dev = z_val.to(dev)
    with torch.no_grad():
        scores_danger = torch.sigmoid(heads["danger"](z_val_dev)).cpu().numpy()
        pred_height = heads["height"](z_val_dev).cpu().numpy()
        pred_pos = heads["pos"](z_val_dev).cpu().numpy()
        scores_target = (torch.sigmoid(heads["target"](z_val_dev)).cpu().numpy()
                         if train_target else None)

    auc_danger = auc(scores_danger, y_val["danger"].numpy())
    auc_target = (auc(scores_target, y_val["target"].numpy()) if train_target
                 else float("nan"))
    mae_height = float(np.abs(pred_height - y_val["height"].numpy()).mean())
    mae_pos = float(np.abs(pred_pos - y_val["pos"].numpy()).mean())

    metrics = {
        "auc_danger": auc_danger,
        "auc_target": auc_target,
        "mae_height": mae_height,
        "mae_pos": mae_pos,
    }
    print(f"têtes objectif : AUC danger = {auc_danger:.3f}, "
          f"AUC cible = {auc_target:.3f}, "
          f"MAE hauteur = {mae_height:.3f}, MAE position = {mae_pos:.3f}")
    return heads, metrics
