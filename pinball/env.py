"""Environnement type Gymnasium au-dessus du simulateur.

L'observation est UNIQUEMENT l'image (2 frames empilées : le mouvement est
visible, l'état devient ~Markovien). Les grandeurs exactes (position,
vitesse...) sortent dans `info` pour le debug et les labels — jamais pour
le modèle.
"""
from __future__ import annotations

import numpy as np

from .config import BoardConfig
from .render import render_frame
from .sim import PinballSim


class PinballEnv:
    N_ACTIONS = 4  # bit 0 = flipper gauche, bit 1 = flipper droit

    def __init__(self, config: BoardConfig | None = None,
                 seed: int | None = None, obs_size: int = 64):
        self.config = config or BoardConfig()
        self.obs_size = obs_size
        self._rng = np.random.default_rng(seed)
        self.sim: PinballSim | None = None
        self._done = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.sim = PinballSim(self.config, self._rng)
        self._steps = 0
        self._stuck_count = 0
        self._nudges = 0
        self._done = False
        frame = render_frame(self.sim, self.obs_size)
        self._prev_frame = frame
        # premier pas : frames dupliquées — état jamais vu à l'entraînement
        # (les datasets commencent à t=1) ; une seule décision par épisode
        return np.stack([frame, frame])

    def step(self, action: int) -> tuple[np.ndarray, dict]:
        if self._done:
            raise RuntimeError("épisode terminé : appeler reset()")
        if not 0 <= int(action) < self.N_ACTIONS:
            raise ValueError(f"action invalide : {action}")
        action = int(action)

        self.sim.set_flippers(bool(action & 1), bool(action >> 1))
        self.sim.step_control()
        self._steps += 1

        x, y = self.sim.ball_pos
        ball_lost = y < self.config.drain_y
        nudged = False
        stuck = False
        if not ball_lost and self.sim.ball_speed < self.config.stuck_speed:
            self._stuck_count += 1
            if self._stuck_count >= self.config.stuck_steps:
                if self._nudges < self.config.max_nudges:
                    self.sim.nudge()
                    self._nudges += 1
                    nudged = True
                    self._stuck_count = 0
                else:
                    stuck = True
        else:
            self._stuck_count = 0

        self._done = ball_lost or stuck or self._steps >= self.config.max_episode_steps
        frame = render_frame(self.sim, self.obs_size)
        obs = np.stack([self._prev_frame, frame])
        self._prev_frame = frame

        vx, vy = self.sim.ball.velocity
        info = {
            "ball_pos": (x, y),
            "ball_vel": (vx, vy),
            "ball_lost": ball_lost,
            "stuck": stuck,
            "nudged": nudged,
            "steps": self._steps,
            "done": self._done,
        }
        return obs, info
