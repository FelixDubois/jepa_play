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
