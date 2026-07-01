"""Rendu du plateau en image, sans fenêtre graphique (PIL uniquement).

C'est l'observation du modèle : 64×64 niveaux de gris. Le monde physique
(origine en bas à gauche, y vers le haut) est projeté sur l'image
(origine en haut à gauche, y vers le bas), de façon anisotrope.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .sim import PinballSim

COLOR_WALL = 90
COLOR_FLIPPER = 180
COLOR_BALL = 255
BALL_MIN_PX = 2.0   # plancher de visibilité de la balle


def _project(cfg, size: int):
    sx, sy = size / cfg.width, size / cfg.height

    def pt(p):
        return (p[0] * sx, (cfg.height - p[1]) * sy)

    return pt, sx


def _draw_board(d: ImageDraw.ImageDraw, sim: PinballSim, pt, wall, flip,
                ball_px, ball_color, width: int = 1) -> None:
    cfg = sim.config
    d.line([pt((0, 0)), pt((0, cfg.height)), pt((cfg.width, cfg.height)),
            pt((cfg.width, 0))], fill=wall, width=width)
    for side in (+1, -1):
        a, b = cfg.guide_points(side)
        d.line([pt(a), pt(b)], fill=wall, width=width)
        d.polygon([pt(v) for v in cfg.sling_verts(side)], outline=wall)
    for i, body in enumerate(sim.flipper_bodies):
        side = +1 if i == 0 else -1
        tip = body.local_to_world((side * cfg.flipper_length, 0))
        d.line([pt(tuple(body.position)), pt(tuple(tip))], fill=flip,
               width=max(2, width * 2))
    bx, by = pt(sim.ball_pos)
    r = ball_px
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=ball_color)


def render_frame(sim: PinballSim, size: int = 64) -> np.ndarray:
    """Observation du modèle : uint8 (size, size)."""
    cfg = sim.config
    pt, sx = _project(cfg, size)
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    ball_px = max(BALL_MIN_PX, cfg.ball_radius * sx)
    _draw_board(d, sim, pt, COLOR_WALL, COLOR_FLIPPER, ball_px, COLOR_BALL)
    return np.asarray(img)


def render_debug(sim: PinballSim, scale: int = 5) -> Image.Image:
    """Grande image RGB pour les vidéos des notebooks (pas pour le modèle)."""
    cfg = sim.config
    size = 64 * scale
    pt, sx = _project(cfg, size)
    img = Image.new("RGB", (size, size), (10, 10, 30))
    d = ImageDraw.Draw(img)
    ball_px = max(BALL_MIN_PX * scale, cfg.ball_radius * sx)
    _draw_board(d, sim, pt, (200, 200, 200), (255, 160, 40), ball_px,
                (255, 255, 255), width=max(1, scale // 2))
    x, y = sim.ball_pos
    d.text((6, 6), f"balle: ({x:.0f}, {y:.0f})  v={sim.ball_speed:.0f}",
           fill=(150, 220, 150))
    return img
