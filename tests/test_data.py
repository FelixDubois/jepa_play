import numpy as np
import torch
from jepa.data import DangerDataset, WindowDataset, stack_obs, MultiLabelDataset


def fake_episode(T, ball_lost, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "frames": rng.integers(0, 255, (T + 1, 64, 64), dtype=np.uint8),
        "actions": rng.integers(0, 4, (T,)).astype(np.int64),
        "ball_pos": rng.uniform(0, 500, (T + 1, 2)).astype(np.float32),
        "ball_lost": ball_lost,
    }


def labeled_episode(T, ball_lost=False, stuck=False, completed=False,
                    hit_at=(), seed=0):
    ep = fake_episode(T, ball_lost, seed)
    ep["stuck"] = stuck
    ep["completed"] = completed
    ep["targets_total"] = 2
    hits = np.zeros(T, dtype=np.uint8)
    for h in hit_at:
        hits[h] = 1
    ep["hits"] = hits
    return ep


def test_window_dataset_shapes_and_count():
    eps = [fake_episode(20, True), fake_episode(9, False)]
    ds = WindowDataset(eps, k=8)
    # ep1 : t dans [1, 12] -> 12 fenêtres ; ep2 : t dans [1, 1] -> 1 fenêtre
    assert len(ds) == 13
    item = ds[0]
    assert item["frames"].shape == (10, 64, 64) and item["frames"].dtype == torch.uint8
    assert item["actions"].shape == (8,) and item["actions"].dtype == torch.int64


def test_window_content_matches_episode():
    ep = fake_episode(20, True)
    ds = WindowDataset([ep], k=8)
    item = ds[0]  # premier index -> t=1
    assert np.array_equal(item["frames"].numpy(), ep["frames"][0:10])
    assert np.array_equal(item["actions"].numpy(), ep["actions"][1:9])


def test_stack_obs_pairs_consecutive_frames():
    frames = torch.arange(2 * 10 * 4 * 4, dtype=torch.uint8).reshape(2, 10, 4, 4)
    s0 = stack_obs(frames, 0)
    assert s0.shape == (2, 2, 4, 4)
    assert torch.equal(s0[:, 0], frames[:, 0]) and torch.equal(s0[:, 1], frames[:, 1])
    s3 = stack_obs(frames, 3)
    assert torch.equal(s3[:, 0], frames[:, 3]) and torch.equal(s3[:, 1], frames[:, 4])


def test_danger_labels():
    T = 30
    ds = DangerDataset([fake_episode(T, True)], k_danger=10)
    labels = np.array([ds[i]["label"].item() for i in range(len(ds))])
    assert len(ds) == T          # t de 1 à T
    assert labels.sum() == 10    # les 10 derniers pas sont dangereux
    assert labels[-1] == 1.0 and labels[0] == 0.0
    # épisode SANS perte de balle : aucun label positif
    ds2 = DangerDataset([fake_episode(T, False)], k_danger=10)
    labels2 = np.array([ds2[i]["label"].item() for i in range(len(ds2))])
    assert labels2.sum() == 0


def test_multilabel_danger_honest():
    # perdu -> queue dangereuse ; stuck -> queue dangereuse ;
    # complété -> AUCUN danger ; timeout -> aucun danger
    for kw, expected_tail in [(dict(ball_lost=True), 10),
                              (dict(stuck=True), 10),
                              (dict(completed=True), 0),
                              (dict(), 0)]:
        ds = MultiLabelDataset([labeled_episode(30, **kw)], k_danger=10)
        labels = np.array([ds[i]["danger"].item() for i in range(len(ds))])
        assert labels.sum() == expected_tail, kw


def test_multilabel_height_and_pos():
    ep = labeled_episode(20)
    ds = MultiLabelDataset([ep], board_size=(540.0, 960.0))
    item = ds[0]                       # t = 1
    assert abs(item["height"].item() - ep["ball_pos"][1, 1] / 960.0) < 1e-6
    assert np.allclose(item["pos"].numpy(),
                       ep["ball_pos"][1] / np.array([540.0, 960.0]),
                       atol=1e-6)
    assert 0.0 <= item["height"].item() <= 1.0


def test_multilabel_target_window():
    # contact au pas 15 (hits[14]=1) : positifs pour t = 5..14 (k=10)
    ds = MultiLabelDataset([labeled_episode(30, hit_at=(14,))], k_target=10)
    labels = np.array([ds[i]["target"].item() for i in range(len(ds))])
    assert labels.sum() == 10
    assert labels[4] == 1.0 and labels[13] == 1.0   # t=5 et t=14 (index t-1)
    assert labels[3] == 0.0 and labels[14] == 0.0


def test_multilabel_backward_compatible_without_hits():
    ds = MultiLabelDataset([fake_episode(20, True)])   # pas de clés v2
    item = ds[0]
    assert item["target"].item() == 0.0
    assert item["danger"].item() in (0.0, 1.0)
