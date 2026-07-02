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
    actions, positions, hits_list = [], [], []
    info = {"ball_pos": None}
    while True:
        a = policy(obs)
        obs, info = env.step(a)
        actions.append(a)
        frames.append(obs[1])   # la frame "présent" du stack
        positions.append(info["ball_pos"])
        hits_list.append(info["hit_now"])
        if info["done"]:
            break
    pos0 = positions[0]         # position au reset non exposée : duplique t=1
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "actions": np.asarray(actions, dtype=np.int64),
        "ball_pos": np.asarray([pos0] + positions, dtype=np.float32),
        "ball_lost": bool(info["ball_lost"]),
        "hits": np.asarray(hits_list, dtype=np.uint8),
        "targets_total": int(info["targets_total"]),
        "completed": bool(info["completed"]),
        "stuck": bool(info["stuck"]),
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
        hits=np.concatenate([ep["hits"] for ep in episodes]),
        targets_total=np.asarray([ep["targets_total"] for ep in episodes], dtype=np.int64),
        completed=np.asarray([ep["completed"] for ep in episodes], dtype=bool),
        stuck=np.asarray([ep["stuck"] for ep in episodes], dtype=bool),
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
            # Lire chaque tableau UNE seule fois : NpzFile décompresse à
            # CHAQUE accès, et chaque tranche retiendrait alors sa propre
            # copie du shard entier (base du view) — ~64× le dataset en RAM,
            # OOM garanti sur Colab à l'échelle 100k transitions.
            frames = z["frames"]
            actions = z["actions"]
            ball_pos = z["ball_pos"]
            frame_counts = z["frame_counts"]
            action_counts = z["action_counts"]
            ball_lost = z["ball_lost"]
            hits = z["hits"]
            targets_total = z["targets_total"]
            completed = z["completed"]
            stuck = z["stuck"]
        f_ofs = a_ofs = 0
        for fc, ac, lost, tt, comp, stk in zip(frame_counts, action_counts, ball_lost, targets_total, completed, stuck):
            episodes.append({
                "frames": frames[f_ofs:f_ofs + fc],
                "actions": actions[a_ofs:a_ofs + ac],
                "ball_pos": ball_pos[f_ofs:f_ofs + fc],
                "ball_lost": bool(lost),
                "hits": hits[a_ofs:a_ofs + ac],
                "targets_total": int(tt),
                "completed": bool(comp),
                "stuck": bool(stk),
            })
            f_ofs += fc
            a_ofs += ac
    return episodes


class MixedPolicy:
    """Politique d'itération : l'agent joue, entrecoupé de RAFALES d'actions
    aléatoires collantes.

    Pourquoi ? Pour réentraîner le world model sur les états que visite le BON
    jeu (l'agent), tout en gardant assez de variété pour ne pas figer le
    modèle sur une seule trajectoire. Une action aléatoire isolée ne sert à
    rien (flipper qui vibre) : on explore par rafales maintenues, comme la
    politique de collecte initiale.

    Note : le primaire n'est PAS consulté pendant une rafale — il doit donc
    replanifier de zéro à chaque pas (c'est le cas du MPCPlanner) plutôt que
    dépendre d'un état interne suivant le flux d'observations.
    """

    def __init__(self, primary, rng: np.random.Generator,
                 burst_prob: float = 0.03,
                 burst_range: tuple[int, int] = (3, 15)):
        self.primary = primary
        self._rng = rng
        self.burst_prob = burst_prob
        self.burst_range = burst_range
        self.reset()

    def reset(self) -> None:
        self.primary.reset()
        self._burst_left = 0
        self._burst_action = 0

    def __call__(self, obs) -> int:
        if self._burst_left <= 0 and self._rng.random() < self.burst_prob:
            self._burst_action = int(self._rng.integers(4))
            self._burst_left = int(self._rng.integers(self.burst_range[0],
                                                      self.burst_range[1] + 1))
        if self._burst_left > 0:
            self._burst_left -= 1
            return self._burst_action
        return self.primary(obs)
