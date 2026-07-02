import numpy as np
from pinball.collect import StickyRandomPolicy
from pinball.config import BoardConfig, hard_board
from pinball.env import PinballEnv
from jepa.eval import AlwaysPressed, evaluate, run_episode


def test_run_episode_new_keys():
    env = PinballEnv(hard_board(), seed=0)
    r = run_episode(env, StickyRandomPolicy(np.random.default_rng(0)), seed=0)
    assert {"steps", "ball_lost", "stuck", "nudges", "completed",
            "targets_total", "targets_hit", "mean_height"} <= r.keys()
    assert 0.0 < r["mean_height"] < 1.0
    assert 0 <= r["targets_hit"] <= r["targets_total"] <= 3


def test_evaluate_new_keys_and_ranges():
    env = PinballEnv(hard_board(), seed=0)
    r = evaluate(env, StickyRandomPolicy(np.random.default_rng(0)), n_episodes=6)
    assert 0.0 <= r["completion_rate"] <= 1.0
    assert 0.0 <= r["targets_hit_rate"] <= 1.0
    assert 0.0 < r["mean_height"] < 1.0
    assert r["mean_nudges"] >= 0.0


def test_evaluate_no_targets_table_gives_nan_rate():
    env = PinballEnv(BoardConfig(max_episode_steps=30), seed=0)
    r = evaluate(env, AlwaysPressed(), n_episodes=3)
    assert r["completion_rate"] == 0.0
    assert np.isnan(r["targets_hit_rate"])
