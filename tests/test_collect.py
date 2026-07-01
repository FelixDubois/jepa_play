import numpy as np
from pinball.collect import StickyRandomPolicy, collect_dataset, load_episodes
from pinball.config import BoardConfig
from pinball.env import PinballEnv


def test_sticky_policy_holds_actions():
    policy = StickyRandomPolicy(np.random.default_rng(0), hold_range=(4, 6))
    actions = [policy(None) for _ in range(60)]
    # les actions changent, mais pas à chaque pas : il existe des paliers >= 4
    runs, current, length = [], actions[0], 1
    for a in actions[1:]:
        if a == current:
            length += 1
        else:
            runs.append(length); current, length = a, 1
    runs.append(length)
    assert max(runs) >= 4
    assert len(set(actions)) > 1


def test_collect_and_reload_roundtrip(tmp_path):
    env = PinballEnv(BoardConfig(max_episode_steps=40), seed=0)
    policy = StickyRandomPolicy(np.random.default_rng(0))
    stats = collect_dataset(env, policy, n_transitions=150, out_dir=tmp_path,
                            shard_episodes=2)
    assert stats["transitions"] >= 150
    episodes = load_episodes(tmp_path)
    assert len(episodes) == stats["episodes"]
    total = sum(len(ep["actions"]) for ep in episodes)
    assert total == stats["transitions"]
    for ep in episodes:
        T = len(ep["actions"])
        assert ep["frames"].shape == (T + 1, 64, 64)
        assert ep["frames"].dtype == np.uint8
        assert ep["ball_pos"].shape == (T + 1, 2)
        assert isinstance(bool(ep["ball_lost"]), bool)
        assert ep["actions"].min() >= 0 and ep["actions"].max() < 4


def test_random_play_produces_ball_losses(tmp_path):
    # crucial pour la tête danger : le jeu aléatoire doit perdre des balles
    env = PinballEnv(seed=1)
    policy = StickyRandomPolicy(np.random.default_rng(1))
    collect_dataset(env, policy, n_transitions=800, out_dir=tmp_path)
    episodes = load_episodes(tmp_path)
    losses = sum(bool(ep["ball_lost"]) for ep in episodes)
    assert losses >= 1
