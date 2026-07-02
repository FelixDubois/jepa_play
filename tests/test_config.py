import math
from pinball.config import BoardConfig


def test_defaults_and_derived_geometry():
    cfg = BoardConfig()
    assert cfg.width == 540.0 and cfg.height == 960.0
    # pivot_offset = drain_gap/2 + longueur*cos(angle repos)
    expected = cfg.drain_gap / 2 + cfg.flipper_length * math.cos(cfg.flipper_rest_angle)
    assert abs(cfg.pivot_offset - expected) < 1e-9
    lx, ly = cfg.pivot_pos(+1)
    rx, ry = cfg.pivot_pos(-1)
    assert ly == ry == cfg.flipper_y
    assert abs((rx - lx) - 2 * cfg.pivot_offset) < 1e-9


def test_guide_ends_above_pivot():
    cfg = BoardConfig()
    (x0, y0), (x1, y1) = cfg.guide_points(+1)
    assert x0 == 0.0 and y0 == cfg.guide_top_y
    px, py = cfg.pivot_pos(+1)
    assert abs(x1 - px) < 1e-9
    assert abs(y1 - (py + cfg.guide_end_lift)) < 1e-9


def test_sling_base_lies_on_guide_line():
    cfg = BoardConfig()
    (ax, ay), (bx, by) = cfg.guide_points(+1)
    p1, p2, apex = cfg.sling_verts(+1)
    # p1 et p2 sont sur la droite du guide (produit vectoriel ~ 0)
    for px, py in (p1, p2):
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        assert abs(cross) < 1e-6
    # l'apex est du côté intérieur du plateau (x plus grand pour le côté gauche)
    assert apex[0] > min(p1[0], p2[0])


def test_config_is_customizable():
    cfg = BoardConfig(drain_gap=100.0, flipper_length=80.0)
    assert cfg.drain_gap == 100.0
    small = BoardConfig(flipper_length=80.0).pivot_offset
    assert small < BoardConfig().pivot_offset


def test_hard_board_preset():
    from pinball.config import hard_board
    cfg = hard_board()
    assert cfg.drain_gap == 120.0 and cfg.flipper_length == 90.0
    # l'ouverture au repos dépasse le diamètre de la balle : vrai trou central
    assert cfg.drain_gap - 2 * cfg.flipper_thickness > 2 * cfg.ball_radius
    # les défauts de BoardConfig ne bougent pas
    assert BoardConfig().drain_gap == 44.0
