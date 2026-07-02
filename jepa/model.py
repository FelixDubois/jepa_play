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
    def __init__(self, in_ch: int = 2, z_dim: int = 384,
                 channels: tuple[int, ...] = (48, 96, 192, 384)):
        super().__init__()
        layers, prev = [], in_ch
        for c in channels:
            layers += [nn.Conv2d(prev, c, 3, stride=2, padding=1),
                       nn.GroupNorm(8, c), nn.SiLU()]
            prev = c
        self.convs = nn.Sequential(*layers)          # 64 -> side x side
        side = 64 // 2 ** len(channels)
        self.fc = nn.Linear(channels[-1] * side * side, z_dim)
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(_to_float(x))
        return self.norm(self.fc(h.flatten(1)))


class Predictor(nn.Module):
    """(z_t, a_t) -> ẑ_{t+1}. Résiduel : il prédit le CHANGEMENT d'état."""

    def __init__(self, z_dim: int = 384, n_actions: int = 4, a_dim: int = 64,
                 hidden: int = 768):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, a_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + a_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(torch.cat([z, self.action_emb(a)], dim=-1))
        return self.norm(z + delta)


class JEPA(nn.Module):
    def __init__(self, z_dim: int = 384,
                 enc_channels: tuple[int, ...] = (48, 96, 192, 384),
                 pred_hidden: int = 768, a_dim: int = 64):
        super().__init__()
        # hparams embarqués dans le checkpoint : un checkpoint AUTO-DESCRIPTIF
        # se reconstruit à l'identique via JEPA(**hparams), sans dépendre des
        # défauts du code au moment du chargement (cf. jepa.train.load_jepa).
        self.hparams = {"z_dim": z_dim, "enc_channels": tuple(enc_channels),
                        "pred_hidden": pred_hidden, "a_dim": a_dim}
        self.encoder = Encoder(z_dim=z_dim, channels=enc_channels)
        self.predictor = Predictor(z_dim=z_dim, a_dim=a_dim, hidden=pred_hidden)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @property
    def z_dim(self) -> int:
        return self.hparams["z_dim"]

    def loss(self, frames: torch.Tensor, actions: torch.Tensor,
             gamma: float | None = 0.7):
        """Rollout multi-pas dans le latent, perte à chaque pas.

        frames : (B, k+2, H, W) — f_{t-1} .. f_{t+k} ; actions : (B, k).

        gamma : pondération des horizons courts. Le pas i (prédiction à
        t+i+1) est pondéré par gamma**i, et la perte totale est la somme
        pondérée divisée par la somme des poids — gamma=None revient à la
        moyenne uniforme d'origine (rétro-compatible).

        Pourquoi : sans pondération, tous les horizons pèsent pareil dans le
        gradient. Or l'erreur des horizons lointains (h=8) est en grande
        partie IRRÉDUCTIBLE (la balle a divergé, l'information n'est plus
        dans z_t) — elle domine le gradient moyen et pousse le prédicteur
        vers la solution qui la minimise localement : moyenner, c'est-à-dire
        éroder les détails fins (bras des flippers, position précise de la
        balle) au profit d'un flou "moyen" qui gagne un peu partout. Avec
        gamma=0,7, les premiers pas (dynamique fine, largement prévisible)
        dominent le gradient et les horizons lointains ne font que
        l'affiner. Mesuré (expérience contrôlée `_exp_arch`, 5 variantes) :
        lisibilité de la balle 0,054→0,041, des bras 0,114→0,059, erreur de
        trajectoire déroulée à h=1 0,064→0,046, variance latente dynamique
        ×2,8 (antidote à l'érosion/collapse). Les MÉTRIQUES (pred_mse,
        copy_mse) restent volontairement des moyennes NON pondérées : elles
        doivent rester comparables entre configurations gamma différentes.
        """
        k = actions.shape[1]
        z = self.encoder(stack_obs(frames, 0))          # z_t (online)
        total, weight_sum, pred_mse = 0.0, 0.0, []
        with torch.no_grad():
            z_prev_target = self.target_encoder(stack_obs(frames, 0))
        copy_mse, latent_std = [], None
        for i in range(k):
            z = self.predictor(z, actions[:, i])        # ẑ_{t+i+1}
            with torch.no_grad():
                target = self.target_encoder(stack_obs(frames, i + 1))
            step_loss = nn.functional.mse_loss(z, target)
            weight = 1.0 if gamma is None else gamma ** i
            total = total + weight * step_loss
            weight_sum += weight
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
        return total / weight_sum, metrics

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
