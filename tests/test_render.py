import numpy as np
from pinball.config import BoardConfig
from pinball.render import render_frame, render_debug
from pinball.sim import PinballSim


def make_sim(seed=0):
    cfg = BoardConfig()
    return PinballSim(cfg, np.random.default_rng(seed)), cfg


def test_frame_shape_and_dtype():
    sim, _ = make_sim()
    f = render_frame(sim)
    assert f.shape == (64, 64) and f.dtype == np.uint8


def test_ball_is_brightest_and_visible():
    sim, cfg = make_sim()
    sim.ball.position = (cfg.width / 2, cfg.height / 2)
    f = render_frame(sim)
    assert f.max() == 255
    # la balle occupe au moins un disque de rayon ~2 px
    assert (f == 255).sum() >= 8
    # et elle est bien au centre de l'image
    ys, xs = np.where(f == 255)
    assert abs(xs.mean() - 32) < 3 and abs(ys.mean() - 32) < 3


def test_ball_moves_in_image():
    sim, cfg = make_sim()
    sim.ball.position = (100, 700)
    f1 = render_frame(sim)
    sim.ball.position = (400, 300)
    f2 = render_frame(sim)
    assert not np.array_equal(f1, f2)


def test_flippers_change_pixels_when_pressed():
    sim, _ = make_sim()
    f_rest = render_frame(sim)
    sim.set_flippers(True, True)
    for _ in range(6):
        sim.step_control()
    f_up = render_frame(sim)
    assert not np.array_equal(f_rest, f_up)


def test_render_debug_is_rgb_and_scaled():
    sim, _ = make_sim()
    img = render_debug(sim, scale=5)
    assert img.mode == "RGB"
    assert img.size == (64 * 5, 64 * 5)


def test_render_debug_ball_is_white():
    # piège PIL : fill=255 sur une image RGB ne colore QUE le canal rouge.
    # La balle doit être blanc pur (255, 255, 255) dans le rendu debug aussi.
    sim, cfg = make_sim()
    sim.ball.position = (cfg.width / 2, cfg.height / 2)
    px = np.asarray(render_debug(sim, scale=5))
    center = px[140:180, 140:180]
    assert (center == 255).all(axis=-1).any()
