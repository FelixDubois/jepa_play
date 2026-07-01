"""Collecte d'expérience : une politique aléatoire joue seule dans le
simulateur et on enregistre tout.

Pourquoi des actions « collantes » ? Une action retirée à chaque pas fait
vibrer les flippers sans jamais frapper : le dataset ne contiendrait aucun
exemple de frappe réussie et le modèle ne pourrait pas apprendre l'effet des
actions. On tire donc une action et on la MAINTIENT plusieurs pas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import PinballEnv


class StickyRandomPolicy:
    def __init__(self, rng: np.random.Generator,
                 hold_range: tuple[int, int] = (3, 15)):
        self._rng = rng
        self._hold_range = hold_range
        self.reset()

    def reset(self) -> None:
        self._action = 0
        self._hold = 0

    def __call__(self, obs) -> int:
        if self._hold <= 0:
            self._action = int(self._rng.integers(4))
            self._hold = int(self._rng.integers(self._hold_range[0],
                                                 self._hold_range[1] + 1))
        self._hold -= 1
        return self._action


def _play_episode(env: PinballEnv, policy) -> dict:
    obs = env.reset()
    policy.reset()
    frames = [obs[1]]           # frame brute du reset
    actions, positions = [], []
    info = {"ball_pos": None}
    while True:
        a = policy(obs)
        obs, info = env.step(a)
        actions.append(a)
        frames.append(obs[1])   # la frame "présent" du stack
        positions.append(info["ball_pos"])
        if info["done"]:
            break
    pos0 = positions[0]         # position au reset non exposée : duplique t=1
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "actions": np.asarray(actions, dtype=np.int64),
        "ball_pos": np.asarray([pos0] + positions, dtype=np.float32),
        "ball_lost": bool(info["ball_lost"]),
    }


def _write_shard(path: Path, episodes: list[dict]) -> None:
    np.savez_compressed(
        path,
        frames=np.concatenate([ep["frames"] for ep in episodes]),
        actions=np.concatenate([ep["actions"] for ep in episodes]),
        ball_pos=np.concatenate([ep["ball_pos"] for ep in episodes]),
        frame_counts=np.asarray([len(ep["frames"]) for ep in episodes]),
        action_counts=np.asarray([len(ep["actions"]) for ep in episodes]),
        ball_lost=np.asarray([ep["ball_lost"] for ep in episodes]),
    )


def collect_dataset(env: PinballEnv, policy, n_transitions: int,
                    out_dir, shard_episodes: int = 64) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer, n_done, n_eps, n_shards = [], 0, 0, 0
    while n_done < n_transitions:
        ep = _play_episode(env, policy)
        buffer.append(ep)
        n_done += len(ep["actions"])
        n_eps += 1
        if len(buffer) >= shard_episodes:
            _write_shard(out / f"shard_{n_shards:05d}.npz", buffer)
            n_shards += 1
            buffer = []
    if buffer:
        _write_shard(out / f"shard_{n_shards:05d}.npz", buffer)
        n_shards += 1
    return {"episodes": n_eps, "transitions": n_done, "shards": n_shards}


def load_episodes(data_dir) -> list[dict]:
    episodes = []
    for path in sorted(Path(data_dir).glob("shard_*.npz")):
        with np.load(path) as z:
            f_ofs = a_ofs = 0
            for fc, ac, lost in zip(z["frame_counts"], z["action_counts"],
                                    z["ball_lost"]):
                episodes.append({
                    "frames": z["frames"][f_ofs:f_ofs + fc],
                    "actions": z["actions"][a_ofs:a_ofs + ac],
                    "ball_pos": z["ball_pos"][f_ofs:f_ofs + fc],
                    "ball_lost": bool(lost),
                })
                f_ofs += fc
                a_ofs += ac
    return episodes
