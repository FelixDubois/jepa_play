import torch
from jepa.model import Encoder, JEPA, Predictor


def test_encoder_shapes_and_uint8():
    enc = Encoder()
    x8 = torch.randint(0, 255, (4, 2, 64, 64), dtype=torch.uint8)
    z = enc(x8)
    assert z.shape == (4, 256)
    xf = x8.float() / 255.0
    assert torch.allclose(enc(xf), z, atol=1e-5)


def test_encoder_param_count():
    n = sum(p.numel() for p in Encoder().parameters())
    assert 500_000 < n < 4_000_000  # "petit CNN" de la spec (~1-2M)


def test_predictor_shapes_and_action_sensitivity():
    torch.manual_seed(0)
    pred = Predictor()
    z = torch.randn(4, 256)
    a0 = torch.zeros(4, dtype=torch.long)
    a3 = torch.full((4,), 3, dtype=torch.long)
    out0, out3 = pred(z, a0), pred(z, a3)
    assert out0.shape == (4, 256)
    assert not torch.allclose(out0, out3)  # l'action doit influencer le futur


def test_jepa_loss_backward_and_metrics():
    torch.manual_seed(0)
    jepa = JEPA()
    frames = torch.randint(0, 255, (3, 10, 64, 64), dtype=torch.uint8)  # k=8
    actions = torch.randint(0, 4, (3, 8))
    loss, metrics = jepa.loss(frames, actions)
    assert loss.requires_grad
    loss.backward()
    assert {"pred_mse", "copy_mse", "latent_std"} <= metrics.keys()
    # l'encodeur cible ne reçoit JAMAIS de gradient
    assert all(p.grad is None for p in jepa.target_encoder.parameters())
    assert any(p.grad is not None for p in jepa.encoder.parameters())
    assert any(p.grad is not None for p in jepa.predictor.parameters())


def test_ema_update_moves_target():
    torch.manual_seed(0)
    jepa = JEPA()
    with torch.no_grad():
        for p in jepa.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in jepa.target_encoder.parameters()]
    jepa.update_target(tau=0.5)
    after = list(jepa.target_encoder.parameters())
    assert all(not torch.allclose(b, a) for b, a in zip(before, after))
    # tau=1 : la cible ne bouge plus
    frozen = [p.clone() for p in jepa.target_encoder.parameters()]
    jepa.update_target(tau=1.0)
    assert all(torch.allclose(f, a) for f, a in
               zip(frozen, jepa.target_encoder.parameters()))


def test_encode_routes_online_and_target():
    # Pin le routage : encode() = encodeur ONLINE, encode_target() = cible EMA.
    # Une inversion silencieuse casserait les Tasks 13-14 sans faire échouer
    # aucun autre test — d'où la perturbation asymétrique.
    torch.manual_seed(0)
    jepa = JEPA()
    obs = torch.randint(0, 255, (3, 2, 64, 64), dtype=torch.uint8)
    z_on, z_tg = jepa.encode(obs), jepa.encode_target(obs)
    assert z_on.shape == z_tg.shape == (3, 256)
    assert not z_on.requires_grad and not z_tg.requires_grad
    # perturber l'encodeur ONLINE seul : encode() change, encode_target() non
    with torch.no_grad():
        for p in jepa.encoder.parameters():
            p.add_(0.1)
    assert not torch.allclose(jepa.encode(obs), z_on)
    assert torch.allclose(jepa.encode_target(obs), z_tg)
