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
