import numpy as np
import pytest
from pinball.config import BoardConfig
from pinball.env import PinballEnv


def test_obs_shape_dtype_and_first_stack():
    env = PinballEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (2, 64, 64) and obs.dtype == np.uint8
    assert np.array_equal(obs[0], obs[1])  # premier pas : frames identiques


def test_step_advances_and_stacks():
    env = PinballEnv(seed=0)
    obs0 = env.reset()
    obs1, info = env.step(0)
    assert obs1.shape == (2, 64, 64)
    # la frame "présent" de obs0 devient la frame "passé" de obs1
    assert np.array_equal(obs1[0], obs0[1])
    assert info["steps"] == 1 and not info["done"]


def test_determinism_with_same_seed():
    def rollout(seed):
        env = PinballEnv(seed=seed)
        env.reset()
        frames = []
        for i in range(40):
            obs, info = env.step(i % 4)
            frames.append(obs.copy())
            if info["done"]:
                break
        return np.stack(frames)

    a, b = rollout(7), rollout(7)
    assert np.array_equal(a, b)


def test_episode_ends_passive():
    # flippers au repos, l'ouverture effective du drain (drain_gap - 2×épaisseur
    # de flipper = 20) est plus étroite que la balle (28) : elle se coince dans
    # le V entre les pointes, est nudgée max_nudges fois, puis fin "stuck".
    # Jamais de timeout.
    env = PinballEnv(seed=1)
    env.reset()
    for _ in range(900):
        obs, info = env.step(0)
        if info["done"]:
            break
    assert info["done"]
    assert info["ball_lost"] or info["stuck"]
    assert info["steps"] < 900


def test_episode_ends_by_drain():
    # le drain ne s'ouvre que quand les flippers bougent : un battement
    # périodique finit par perdre la balle sur au moins une seed (mesuré :
    # seeds 0 et 2 drainent ; boucle multi-seeds = robustesse inter-builds).
    drained = False
    for seed in (0, 1, 2, 3):
        env = PinballEnv(seed=seed)
        env.reset()
        for i in range(900):
            action = 3 if (i // 15) % 2 == 0 else 0
            obs, info = env.step(action)
            if info["done"]:
                break
        if info["ball_lost"]:
            drained = True
            break
    assert drained


def test_step_after_done_raises():
    env = PinballEnv(seed=1)
    env.reset()
    for _ in range(900):
        _, info = env.step(0)
        if info["done"]:
            break
    assert info["done"]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_action_raises():
    env = PinballEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)


def test_custom_config_is_used():
    env = PinballEnv(config=BoardConfig(max_episode_steps=5), seed=0)
    env.reset()
    for _ in range(5):
        _, info = env.step(0)
    assert info["done"] and info["steps"] == 5


def test_default_table_has_no_targets():
    env = PinballEnv(seed=0)
    env.reset()
    _, info = env.step(0)
    assert info["targets_total"] == 0 and not info["completed"]


def test_targets_sampled_and_deterministic():
    from pinball.config import hard_board
    env = PinballEnv(hard_board(), seed=3)
    env.reset()
    a = list(env.sim.targets)
    assert 1 <= len(a) <= 3
    env2 = PinballEnv(hard_board(), seed=3)
    env2.reset()
    assert list(env2.sim.targets) == a       # même seed -> mêmes cibles
    env3 = PinballEnv(hard_board(), seed=4)
    env3.reset()
    assert list(env3.sim.targets) != a       # seed différente -> différentes


def test_completion_ends_episode():
    # une seule cible, balle téléportée dessus : l'épisode finit en victoire
    from pinball.config import hard_board
    env = PinballEnv(hard_board(), seed=0)
    env.reset()
    tx, ty = env.sim.targets[0]
    for i in range(1, len(env.sim.targets)):        # éteindre les autres
        env.sim.target_alive[i] = False
        env.sim.space.remove(env.sim._target_shapes[i])
        env._targets_hit += 1
    env.sim.ball.position = (tx, ty + 60)
    env.sim.ball.velocity = (0.0, -300.0)
    for _ in range(30):
        _, info = env.step(0)
        if info["done"]:
            break
    assert info["completed"] and info["done"]
    assert not info["ball_lost"]
    assert info["targets_hit"] == info["targets_total"]
