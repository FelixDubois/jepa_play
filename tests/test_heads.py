import numpy as np
import torch
from jepa.heads import DangerHead, auc, train_danger_head
from jepa.model import JEPA


def test_auc_known_values():
    labels = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0   # parfait
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0   # inversé
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(auc(scores, labels) - 0.5) < 1e-9                # hasard


def test_danger_head_shapes():
    head = DangerHead()
    logits = head(torch.randn(7, 256))
    assert logits.shape == (7,)


def separable_episode(T, ball_lost, seed):
    """La frame encode grossièrement le danger : bande basse allumée en fin
    d'épisode perdu — un signal qu'un encodeur même non entraîné transmet."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
    for t in range(T + 1):
        y = 10 if not (ball_lost and t > T - 10) else 58
        frames[t, y - 2:y + 2, 20:44] = 255
    return {"frames": frames,
            "actions": rng.integers(0, 4, (T,)).astype(np.int64),
            "ball_pos": np.zeros((T + 1, 2), dtype=np.float32),
            "ball_lost": ball_lost}


def test_train_danger_head_learns_separable_signal():
    torch.manual_seed(0)
    eps = [separable_episode(30, s % 2 == 0, s) for s in range(20)]
    jepa = JEPA()  # encodeur NON entraîné : projection aléatoire, suffisant ici
    head, val_auc = train_danger_head(jepa, eps, epochs=10, batch_size=64,
                                      device="cpu")
    assert val_auc > 0.9


def test_train_danger_head_clamps_large_batch():
    # batch_size > nb d'échantillons : sans clamp, la boucle interne ne ferait
    # AUCUN pas d'optimiseur et retournerait une tête aléatoire, en silence
    # (mesuré : AUC 0.10 sans clamp, 1.00 avec, seed 0).
    torch.manual_seed(0)
    eps = [separable_episode(30, s % 2 == 0, s) for s in range(20)]
    head, val_auc = train_danger_head(JEPA(), eps, epochs=10,
                                      batch_size=10_000, device="cpu")
    assert val_auc > 0.9


def test_new_heads_shapes():
    from jepa.heads import HeightHead, PositionProbe, TargetHead
    z = torch.randn(5, 256)
    h = HeightHead()(z)
    assert h.shape == (5,) and (h >= 0).all() and (h <= 1).all()
    assert TargetHead()(z).shape == (5,)
    p = PositionProbe()(z)
    assert p.shape == (5, 2) and (p >= 0).all() and (p <= 1).all()


def _v2_episode(T, ball_lost, seed):
    # bande lumineuse qui SUIT la hauteur réelle (hauteur/position apprenables)
    # + marqueur de coin distinct pendant la queue dangereuse (séparabilité
    # du danger garantie même avec un encodeur aléatoire — motif V1)
    rng = np.random.default_rng(seed)
    frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
    ys = (np.linspace(880.0, 120.0, T + 1) if ball_lost
          else np.linspace(120.0, 880.0, T + 1))
    ball_pos = np.stack([np.full(T + 1, 270.0), ys], axis=1).astype(np.float32)
    for t in range(T + 1):
        row = int(63 - ys[t] / 960.0 * 63)
        frames[t, max(0, row - 2):row + 3, 20:44] = 255
        if ball_lost and t > T - 10:
            frames[t, 55:63, 55:63] = 200
    hits = np.zeros(T, dtype=np.uint8)
    hits[T // 2] = 0 if ball_lost else 1
    return {"frames": frames, "actions": rng.integers(0, 4, (T,)).astype(np.int64),
            "ball_pos": ball_pos, "ball_lost": ball_lost, "stuck": False,
            "completed": not ball_lost, "targets_total": 1, "hits": hits}


def test_train_objective_heads_learns():
    from jepa.heads import train_objective_heads
    from jepa.model import JEPA
    torch.manual_seed(0)
    eps = [_v2_episode(30, s % 2 == 0, s) for s in range(20)]
    heads, metrics = train_objective_heads(JEPA(), eps, epochs=10,
                                           batch_size=64, device="cpu")
    assert set(heads) == {"danger", "height", "target", "pos"}
    assert metrics["auc_danger"] > 0.9          # signal séparable (cf. V1)
    assert metrics["mae_height"] < 0.2          # la hauteur se lit dans l'image
