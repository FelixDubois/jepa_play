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
                  lr: float = 1e-3, device: str | None = None) -> Decoder:
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
            loss = nn.functional.mse_loss(dec(z), obs[:, 1].float() / 255.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"décodeur epoch {epoch + 1}/{epochs}  mse={total / max(nb, 1):.5f}")
    return dec.eval()
