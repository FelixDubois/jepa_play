"""Évaluation : l'agent JEPA contre les baselines, à seeds appariées.

Le graphique de victoire de la V1 : temps de survie moyen de l'agent
NETTEMENT au-dessus de la politique aléatoire et de « toujours appuyé ».
"""
from __future__ import annotations

import numpy as np

from pinball.render import render_debug


class AlwaysPressed:
    """Baseline : les deux flippers levés en permanence."""

    def reset(self) -> None:
        pass

    def __call__(self, obs) -> int:
        return 3


class PeriodicFlapper:
    """Baseline « jongleur aveugle » : battement des deux flippers à période
    fixe, sans jamais regarder l'écran. Sur cette table, ce rythme garde
    souvent la balle en jeu très longtemps : si l'agent ne fait pas mieux,
    c'est qu'il n'exploite pas la vision."""

    def __init__(self, half_period: int = 15):
        self.half_period = half_period
        self.reset()

    def reset(self) -> None:
        self._step = 0

    def __call__(self, obs) -> int:
        action = 3 if (self._step // self.half_period) % 2 == 0 else 0
        self._step += 1
        return action


def run_episode(env, policy, seed: int | None = None) -> dict:
    obs = env.reset(seed=seed)
    policy.reset()
    while True:
        obs, info = env.step(policy(obs))
        if info["done"]:
            return {"steps": info["steps"], "ball_lost": info["ball_lost"],
                    "stuck": info["stuck"]}


def evaluate(env, policy, n_episodes: int = 50, seed0: int = 1000) -> dict:
    results = [run_episode(env, policy, seed=seed0 + i)
               for i in range(n_episodes)]
    lengths = np.array([r["steps"] for r in results])
    hz = env.config.physics_hz / env.config.frame_skip
    return {
        "mean_steps": float(lengths.mean()),
        "median_steps": float(np.median(lengths)),
        "survival_s": float(lengths.mean() / hz),
        "loss_rate": float(np.mean([r["ball_lost"] for r in results])),
        "lengths": lengths,
    }


def record_gif(env, policy, path, seed: int | None = None,
               max_steps: int = 450) -> dict:
    obs = env.reset(seed=seed)
    policy.reset()
    frames = [render_debug(env.sim)]
    info = {"steps": 0, "ball_lost": False, "stuck": False}
    for _ in range(max_steps):
        obs, info = env.step(policy(obs))
        frames.append(render_debug(env.sim))
        if info["done"]:
            break
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=66, loop=0)
    return {"steps": info["steps"], "ball_lost": info["ball_lost"],
            "stuck": info["stuck"]}
