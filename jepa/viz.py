"""Visualiser les prédictions du world model, superposées au réel."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

# dégradé d'horizon : jaune (t+1) → violet (t+8)
HORIZON_COLORS = [(255, 220, 60), (255, 170, 60), (255, 120, 70), (255, 80, 90),
                  (230, 60, 120), (200, 50, 160), (160, 50, 200), (120, 60, 230)]


def _obs_at(ep: dict, t: int) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(ep["frames"][t - 1:t + 1]))


@torch.no_grad()
def rollout_latents(jepa, ep: dict, t0: int, k: int = 8,
                    device: str = "cpu") -> torch.Tensor:
    """Encode o_{t0} (online) puis déroule le prédicteur avec les actions
    réellement jouées — les ẑ retournés vivent dans l'espace cible."""
    jepa = jepa.to(device).eval()
    z = jepa.encode(_obs_at(ep, t0).unsqueeze(0).to(device))
    zs = []
    for j in range(k):
        a = torch.tensor([int(ep["actions"][t0 + j])], device=device)
        z = jepa.predictor(z, a)
        zs.append(z.squeeze(0).clone())
    return torch.stack(zs)


@torch.no_grad()
def trajectory_overlay(jepa, pos_probe, ep: dict, t0: int, k: int = 8,
                       upscale: int = 6,
                       board_size: tuple[float, float] = (540.0, 960.0),
                       device: str = "cpu") -> Image.Image:
    """Superposition « prédit vs réel » : la frame réelle finale, la vraie
    trajectoire (cercles blancs) et les positions imaginées (croix colorées)."""
    zs = rollout_latents(jepa, ep, t0, k, device)
    pred = pos_probe.to(device).eval()(zs).cpu().numpy()
    size = 64 * upscale
    img = Image.fromarray(ep["frames"][t0 + k]).convert("RGB").resize(
        (size, size), Image.NEAREST)
    d = ImageDraw.Draw(img)
    w, h = board_size
    for j in range(k):
        x, y = ep["ball_pos"][t0 + j + 1]
        px, py = x / w * size, (1 - y / h) * size
        d.ellipse([px - 3, py - 3, px + 3, py + 3], outline=(255, 255, 255))
    for j in range(k):
        px, py = pred[j, 0] * size, (1 - pred[j, 1]) * size
        c = HORIZON_COLORS[j % len(HORIZON_COLORS)]
        d.line([px - 5, py, px + 5, py], fill=c, width=2)
        d.line([px, py - 5, px, py + 5], fill=c, width=2)
    return img


@torch.no_grad()
def imagination_strip(jepa, decoder, ep: dict, t0: int, k: int = 8,
                      upscale: int = 4, device: str = "cpu") -> Image.Image:
    """Planche 3×k : RÉEL / IMAGINÉ (décodé des ẑ) / SUPERPOSITION
    (rouge = imaginé, vert = réel — jaune là où ils s'accordent)."""
    zs = rollout_latents(jepa, ep, t0, k, device)
    decoded = decoder.to(device).eval()(zs).cpu().numpy()
    cell = 64 * upscale
    strip = Image.new("RGB", (k * cell, 3 * cell), (15, 15, 25))
    for j in range(k):
        real = ep["frames"][t0 + j + 1].astype(np.float32) / 255.0
        imag = decoded[j]
        rows = [np.stack([real] * 3, axis=-1),
                np.stack([imag] * 3, axis=-1),
                np.stack([imag, real, np.zeros_like(real)], axis=-1)]
        for r, arr in enumerate(rows):
            im = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (cell, cell), Image.NEAREST)
            strip.paste(im, (j * cell, r * cell))
    return strip
