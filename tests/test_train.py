import numpy as np
import pytest
import torch
from jepa.train import load_jepa, train_jepa


def moving_dot_episode(T=20, seed=0):
    """Épisode synthétique APPRENABLE : un point qui descend, action = décalage x.

    Dynamique simple et déterministe pour que l'overfit soit possible sur CPU.
    """
    rng = np.random.default_rng(seed)
    frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
    actions = rng.integers(0, 4, (T,)).astype(np.int64)
    x, y = 32, 4
    frames[0, y, x] = 255
    for t in range(T):
        y = min(60, y + 2)
        x = int(np.clip(x + (actions[t] - 1) * 3, 2, 61))
        frames[t + 1, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 255
    return {"frames": frames, "actions": actions,
            "ball_pos": np.zeros((T + 1, 2), dtype=np.float32),
            "ball_lost": True}


def test_train_checkpoints_and_resumes(tmp_path):
    eps = [moving_dot_episode(T=15, seed=s) for s in range(3)]
    model, hist = train_jepa(eps, tmp_path, epochs=2, k=4, batch_size=8,
                             device="cpu", num_workers=0)
    assert len(hist) == 2
    assert (tmp_path / "jepa.pt").exists()
    # reprise : 1 epoch de plus seulement
    model2, hist2 = train_jepa(eps, tmp_path, epochs=3, k=4, batch_size=8,
                               device="cpu", num_workers=0)
    assert len(hist2) == 3 and hist2[:2] == hist


def test_overfit_small_dataset(tmp_path):
    # Critère de la spec : sur un petit dataset, la perte doit s'écraser
    # et la variance latente rester saine (pas de collapse).
    torch.manual_seed(0)
    eps = [moving_dot_episode(T=15, seed=s) for s in range(2)]
    _, hist = train_jepa(eps, tmp_path, epochs=40, k=4, batch_size=16,
                         lr=1e-3, device="cpu", num_workers=0)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    assert hist[-1]["latent_std"] > 0.01


def test_load_jepa_roundtrip(tmp_path):
    eps = [moving_dot_episode(T=15, seed=0)]
    model, _ = train_jepa(eps, tmp_path, epochs=1, k=4, batch_size=8,
                          device="cpu", num_workers=0)
    loaded = load_jepa(tmp_path / "jepa.pt", device="cpu")
    obs = torch.randint(0, 255, (2, 2, 64, 64), dtype=torch.uint8)
    assert torch.allclose(model.encode(obs), loaded.encode(obs), atol=1e-6)
    assert not loaded.training


def test_train_raises_on_empty_loader(tmp_path):
    # batch_size > nb de fenêtres + drop_last : sans garde, l'epoch "réussit"
    # à vide (loss 0, fausse alerte collapse). On exige un échec explicite.
    eps = [moving_dot_episode(T=15, seed=0)]  # 11 fenêtres à k=4
    with pytest.raises(ValueError):
        train_jepa(eps, tmp_path, epochs=1, k=4, batch_size=64,
                   device="cpu", num_workers=0)
