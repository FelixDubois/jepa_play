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


def test_load_episodes_shares_shard_memory(tmp_path):
    # NpzFile décompresse à CHAQUE accès : sans hoisting des lectures, chaque
    # épisode retiendrait sa propre copie du shard entier (~64× le dataset en
    # RAM → OOM Colab). Les épisodes d'un même shard partagent une base.
    env = PinballEnv(BoardConfig(max_episode_steps=30), seed=0)
    policy = StickyRandomPolicy(np.random.default_rng(0))
    collect_dataset(env, policy, n_transitions=120, out_dir=tmp_path,
                    shard_episodes=64)
    episodes = load_episodes(tmp_path)
    assert len(episodes) >= 2
    assert episodes[0]["frames"].base is episodes[1]["frames"].base


class _ConstantPolicy:
    """Politique factice : action constante, trace les appels et les reset."""

    def __init__(self, action=3):
        self.action = action
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, obs):
        self.calls += 1
        return self.action


def test_mixed_policy_mostly_plays_primary():
    from pinball.collect import MixedPolicy
    primary = _ConstantPolicy(action=3)
    policy = MixedPolicy(primary, np.random.default_rng(0))
    actions = [policy(None) for _ in range(3000)]
    # les rafales tirent uniformément dans {0..3} : ~3/4 des pas de rafale
    # diffèrent de l'action du primaire -> part observable ≈ 21 % × 3/4
    frac_other = sum(a != 3 for a in actions) / len(actions)
    assert 0.05 < frac_other < 0.35
    # hors rafale, c'est bien le primaire qui décide
    assert primary.calls > len(actions) * 0.5


def test_mixed_policy_bursts_are_sticky():
    from pinball.collect import MixedPolicy
    policy = MixedPolicy(_ConstantPolicy(action=3), np.random.default_rng(1),
                         burst_prob=1.0, burst_range=(5, 5))
    # burst_prob=1 : rafale permanente, par blocs de 5 actions identiques
    actions = [policy(None) for _ in range(20)]
    for i in range(0, 20, 5):
        assert len(set(actions[i:i + 5])) == 1


def test_mixed_policy_reset_propagates():
    from pinball.collect import MixedPolicy
    primary = _ConstantPolicy()
    policy = MixedPolicy(primary, np.random.default_rng(0))
    policy.reset()
    policy.reset()
    assert primary.resets >= 2
