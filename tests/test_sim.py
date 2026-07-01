import numpy as np
from pinball.config import BoardConfig
from pinball.sim import PinballSim


def make_sim(seed=0, **cfg_kwargs):
    cfg = BoardConfig(**cfg_kwargs)
    return PinballSim(cfg, np.random.default_rng(seed)), cfg


def test_free_fall_matches_gravity():
    # balle lancée sans vitesse : après 0,5 s, vy ≈ -g * t
    sim, cfg = make_sim()
    sim.ball.velocity = (0, 0)
    sim.ball.position = (cfg.width / 2, 800)
    for _ in range(int(0.5 * cfg.physics_hz / cfg.frame_skip) + 1):
        sim.step_control()
    t = (int(0.5 * cfg.physics_hz / cfg.frame_skip) + 1) * cfg.frame_skip / cfg.physics_hz
    assert abs(sim.ball.velocity.y + cfg.gravity * t) < 30.0


def test_flipper_press_raises_angle():
    sim, cfg = make_sim()
    left0, right0 = sim.flipper_angles
    sim.set_flippers(True, True)
    for _ in range(6):
        sim.step_control()
    left1, right1 = sim.flipper_angles
    assert left1 > left0    # gauche : appuyé = angle qui augmente
    assert right1 < right0  # droit : miroir


def test_flipper_strike_speeds_ball_up():
    # balle lâchée sur le flipper gauche au repos, puis frappe
    sim, cfg = make_sim()
    px, py = cfg.pivot_pos(+1)
    sim.ball.position = (px + cfg.flipper_length * 0.6, 400)
    sim.ball.velocity = (0, 0)
    for _ in range(9):          # ~0,6 s de chute sur le flipper
        sim.step_control()
    sim.set_flippers(True, False)
    max_vy = -1e9
    for _ in range(5):
        sim.step_control()
        max_vy = max(max_vy, sim.ball.velocity.y)
    assert max_vy > 400.0       # la balle est projetée vers le haut


def test_speed_is_capped():
    sim, cfg = make_sim()
    sim.ball.velocity = (0, -10 * cfg.max_ball_speed)
    sim.step_control()
    assert sim.ball_speed <= cfg.max_ball_speed * 1.01


def test_nudge_moves_stationary_ball():
    sim, cfg = make_sim()
    sim.ball.velocity = (0, 0)
    v0 = sim.ball_speed
    sim.nudge()
    assert sim.ball_speed > v0


def test_config_changes_geometry():
    sim_a, _ = make_sim(drain_gap=44.0)
    sim_b, _ = make_sim(drain_gap=120.0)
    ax = sim_a.flipper_bodies[0].position.x
    bx = sim_b.flipper_bodies[0].position.x
    assert bx < ax   # drain plus large -> pivot gauche plus loin du centre


def test_ball_stays_in_bounds_long_episode():
    # anti-tunneling : la balle ne traverse JAMAIS un mur. La sortie par le
    # bas (drain, ouvert par conception — c'est l'env qui la détecte) est une
    # fin légitime : on passe alors à la seed suivante.
    for seed in range(5):
        sim, cfg = make_sim(seed=seed)
        rng = np.random.default_rng(seed)
        a = 0
        for i in range(900):
            if i % 7 == 0:
                a = int(rng.integers(4))
            sim.set_flippers(bool(a & 1), bool(a >> 1))
            sim.step_control()
            x, y = sim.ball_pos
            if y < 0:          # sortie par le drain : fin légitime de la seed
                break
            assert -5 <= x <= cfg.width + 5 and y <= cfg.height + 5
