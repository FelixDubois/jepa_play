import numpy as np
import torch
from jepa.decoder import Decoder, train_decoder
from jepa.model import JEPA
from jepa.viz import imagination_strip, rollout_latents, trajectory_overlay
from jepa.heads import PositionProbe


def fake_ep(T=30, seed=0):
    rng = np.random.default_rng(seed)
    return {"frames": rng.integers(0, 255, (T + 1, 64, 64), dtype=np.uint8),
            "actions": rng.integers(0, 4, (T,)).astype(np.int64),
            "ball_pos": rng.uniform(50, 900, (T + 1, 2)).astype(np.float32),
            "ball_lost": True}


def test_decoder_shapes():
    out = Decoder()(torch.randn(3, 256))
    assert out.shape == (3, 64, 64)
    assert (out >= 0).all() and (out <= 1).all()


def test_train_decoder_reduces_loss(capsys):
    torch.manual_seed(0)
    eps = [fake_ep(T=20, seed=s) for s in range(2)]
    dec = train_decoder(JEPA(), eps, epochs=2, batch_size=16, device="cpu")
    assert not dec.training
    lines = [l for l in capsys.readouterr().out.splitlines() if "wmse=" in l]
    first = float(lines[0].split("wmse=")[1])
    last = float(lines[-1].split("wmse=")[1])
    assert last <= first


def test_rollout_latents_shape():
    zs = rollout_latents(JEPA(), fake_ep(), t0=2, k=8)
    assert zs.shape == (8, 256)


def test_decoder_learns_bright_pixels():
    # ~95-97 % de pixels noirs : une MSE nue apprend « tout noir » (mesuré :
    # 0 pixel > 0.5 en sortie). La perte pondérée doit reproduire la bille.
    torch.manual_seed(0)

    def sparse_ep(T=25, seed=0):
        rng = np.random.default_rng(seed)
        frames = np.zeros((T + 1, 64, 64), dtype=np.uint8)
        frames[:, 60:63, :] = 90
        for t in range(T + 1):
            x = 8 + (t * 2) % 48
            frames[t, 20:25, x:x + 5] = 255
        return {"frames": frames,
                "actions": rng.integers(0, 4, (T,)).astype(np.int64),
                "ball_pos": np.zeros((T + 1, 2), dtype=np.float32),
                "ball_lost": True}

    from jepa.data import MultiLabelDataset
    eps = [sparse_ep(seed=s) for s in range(2)]
    jepa = JEPA()
    dec = train_decoder(jepa, eps, epochs=5, batch_size=16, device="cpu")
    ds = MultiLabelDataset(eps)
    obs = torch.stack([ds[i]["obs"] for i in range(8)])
    with torch.no_grad():
        out = dec(jepa.encode_target(obs))
    assert (out > 0.5).sum() > 0      # la bille brillante est reproduite


def test_overlays_return_images():
    ep = fake_ep()
    img1 = trajectory_overlay(JEPA(), PositionProbe(), ep, t0=2, k=8, upscale=4)
    assert img1.size == (256, 256) and img1.mode == "RGB"
    img2 = imagination_strip(JEPA(), Decoder(), ep, t0=2, k=8, upscale=2)
    assert img2.size == (8 * 128, 3 * 128) and img2.mode == "RGB"
