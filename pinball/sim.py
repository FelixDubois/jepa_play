"""Simulateur physique du flipper (pymunk / Chipmunk2D).

La balle est un vrai corps rigide : gravité, restitution, friction.
Les flippers sont des corps dynamiques sur pivot, actionnés par moteur
angulaire avec butées — la balle est réellement frappée (transfert de
moment), rien n'est scripté.
"""
from __future__ import annotations

import numpy as np
import pymunk

from .config import BoardConfig


class PinballSim:
    def __init__(self, config: BoardConfig, rng: np.random.Generator):
        self.config = config
        self._rng = rng
        self.space = pymunk.Space()
        self.space.gravity = (0, -config.gravity)
        self._build_static()
        self.flipper_bodies: list[pymunk.Body] = []
        self._motors: list[pymunk.SimpleMotor] = []
        for side in (+1, -1):
            self._add_flipper(side)
        self._add_ball()

    # ---------- construction ----------
    def _build_static(self) -> None:
        cfg = self.config
        sb = self.space.static_body
        segments = [
            ((0, 0), (0, cfg.height)),                    # mur gauche
            ((cfg.width, 0), (cfg.width, cfg.height)),    # mur droit
            ((0, cfg.height), (cfg.width, cfg.height)),   # plafond
            cfg.guide_points(+1),                         # guides d'entonnoir
            cfg.guide_points(-1),
        ]
        shapes = []
        for a, b in segments:
            s = pymunk.Segment(sb, a, b, cfg.wall_radius)
            s.elasticity = cfg.wall_elasticity
            s.friction = cfg.wall_friction
            shapes.append(s)
        for side in (+1, -1):
            p = pymunk.Poly(sb, cfg.sling_verts(side))
            p.elasticity = cfg.sling_elasticity
            p.friction = cfg.wall_friction
            shapes.append(p)
        self.space.add(*shapes)

    def _add_flipper(self, side: int) -> None:
        cfg = self.config
        a, b = (0, 0), (side * cfg.flipper_length, 0)
        moment = pymunk.moment_for_segment(cfg.flipper_mass, a, b, cfg.wall_radius)
        body = pymunk.Body(cfg.flipper_mass, moment)
        body.position = cfg.pivot_pos(side)
        shape = pymunk.Segment(body, a, b, cfg.flipper_thickness)
        shape.elasticity = cfg.flipper_elasticity
        shape.friction = cfg.flipper_friction
        if side > 0:
            lo, hi = -cfg.flipper_rest_angle, cfg.flipper_press_angle
        else:
            lo, hi = -cfg.flipper_press_angle, cfg.flipper_rest_angle
        sb = self.space.static_body
        pivot = pymunk.PivotJoint(sb, body, body.position)
        limit = pymunk.RotaryLimitJoint(sb, body, lo, hi)
        motor = pymunk.SimpleMotor(sb, body, 0.0)
        motor.max_force = cfg.flipper_max_force
        body.angle = lo if side > 0 else hi   # position de repos
        self.space.add(body, shape, pivot, limit, motor)
        self.flipper_bodies.append(body)
        self._motors.append(motor)

    def _add_ball(self) -> None:
        cfg = self.config
        moment = pymunk.moment_for_circle(cfg.ball_mass, 0, cfg.ball_radius)
        self.ball = pymunk.Body(cfg.ball_mass, moment)
        x = self._rng.uniform(cfg.launch_margin, cfg.width - cfg.launch_margin)
        self.ball.position = (x, cfg.height - cfg.launch_y_offset)
        self.ball.velocity = (self._rng.uniform(-cfg.launch_vx_max, cfg.launch_vx_max), 0)
        shape = pymunk.Circle(self.ball, cfg.ball_radius)
        shape.elasticity = cfg.ball_elasticity
        shape.friction = cfg.ball_friction
        self.space.add(self.ball, shape)

    # ---------- contrôle ----------
    def set_flippers(self, left: bool, right: bool) -> None:
        # Convention pymunk mesurée : rate > 0 fait DIMINUER l'angle.
        speed = self.config.flipper_speed
        self._motors[0].rate = -speed if left else speed
        self._motors[1].rate = speed if right else -speed

    def step_control(self) -> None:
        """Un pas de contrôle = frame_skip sous-pas physiques + plafond vitesse."""
        cfg = self.config
        dt = 1.0 / cfg.physics_hz
        for _ in range(cfg.frame_skip):
            self.space.step(dt)
            v = self.ball.velocity
            if v.length > cfg.max_ball_speed:
                self.ball.velocity = v * (cfg.max_ball_speed / v.length)

    def nudge(self) -> None:
        """Petite impulsion aléatoire, biaisée vers le haut (anti-stagnation)."""
        cfg = self.config
        angle = self._rng.uniform(np.pi * 0.25, np.pi * 0.75)  # vers le haut
        impulse = cfg.nudge_impulse * np.array([np.cos(angle), np.sin(angle)])
        self.ball.apply_impulse_at_local_point(tuple(impulse))

    # ---------- lecture d'état ----------
    @property
    def ball_pos(self) -> tuple[float, float]:
        p = self.ball.position
        return (p.x, p.y)

    @property
    def ball_speed(self) -> float:
        return self.ball.velocity.length

    @property
    def flipper_angles(self) -> tuple[float, float]:
        return (self.flipper_bodies[0].angle, self.flipper_bodies[1].angle)
