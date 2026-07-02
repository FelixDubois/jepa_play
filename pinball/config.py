"""Configuration paramétrable du plateau de flipper.

Toutes les valeurs par défaut ont été équilibrées par prototype (voir le plan,
Annexe A) : politique aléatoire ~6 s de survie, heuristique simple ~42 s.
Repère : origine en bas à gauche, y vers le haut (convention pymunk).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BoardConfig:
    # --- dimensions du plateau (unités arbitraires) ---
    width: float = 540.0
    height: float = 960.0

    # --- balle ---
    ball_radius: float = 14.0
    ball_mass: float = 1.0
    ball_elasticity: float = 0.5
    ball_friction: float = 0.4
    max_ball_speed: float = 2200.0   # plafond anti-tunneling

    # --- gravité (inclinaison de la table) ---
    gravity: float = 1800.0

    # --- murs et guides ---
    wall_radius: float = 6.0
    wall_elasticity: float = 0.65
    wall_friction: float = 0.4
    guide_top_y: float = 320.0       # hauteur d'accroche des guides sur les murs
    guide_end_lift: float = 20.0     # le guide s'arrête au-dessus du pivot
                                     # (au ras du dessus du flipper : pas de poche)

    # --- flippers ---
    flipper_length: float = 110.0
    flipper_thickness: float = 12.0  # rayon du segment
    flipper_mass: float = 8.0
    flipper_y: float = 120.0         # hauteur des pivots
    flipper_rest_angle: float = 0.52   # rad sous l'horizontale, au repos
    flipper_press_angle: float = 0.55  # rad au-dessus, appuyé
    flipper_speed: float = 25.0        # rad/s du moteur
    flipper_max_force: float = 1e7
    flipper_elasticity: float = 0.4
    flipper_friction: float = 0.6
    drain_gap: float = 44.0          # écart horizontal entre les POINTES au repos

    # --- slingshots (triangles rebondissants posés sur les guides) ---
    sling_start: float = 95.0        # abscisses curvilignes le long du guide
    sling_end: float = 205.0
    sling_height: float = 48.0       # hauteur de l'apex vers l'intérieur
    sling_elasticity: float = 1.4    # > 1 : redonne de l'énergie à la balle

    # --- temps ---
    physics_hz: int = 120
    frame_skip: int = 8              # contrôle à 120/8 = 15 Hz

    # --- épisode ---
    max_episode_steps: int = 900     # 60 s à 15 Hz
    drain_y: float = 60.0            # sous cette hauteur, balle perdue
    launch_margin: float = 80.0      # zone x de lancement : [margin, width-margin]
    launch_y_offset: float = 90.0    # lancement à height - offset
    launch_vx_max: float = 150.0     # |vx| initial max
    stuck_speed: float = 5.0         # vitesse sous laquelle la balle "stagne"
    stuck_steps: int = 45            # pas de contrôle stagnants avant nudge (3 s)
    nudge_impulse: float = 300.0     # impulsion du nudge (quantité de mouvement)
    max_nudges: int = 6              # au-delà, fin d'épisode "stuck"

    # --- cibles (V2) : plots à toucher, placés aléatoirement par épisode.
    # Défaut (0, 0) = pas de cibles : la table de base et les tests V1
    # gardent leur comportement. hard_board() active (1, 3). ---
    n_targets_range: tuple[int, int] = (0, 0)   # bornes incluses
    target_radius: float = 26.0
    target_elasticity: float = 1.2
    target_zone_x: tuple[float, float] = (70.0, 470.0)
    target_zone_y: tuple[float, float] = (420.0, 860.0)
    target_min_sep: float = 100.0    # > 2×(r_balle+r_cible+marge) : un seul contact par sous-pas
    target_hit_margin: float = 2.0

    # ---------- géométrie dérivée ----------
    @property
    def pivot_offset(self) -> float:
        """Distance horizontale pivot ↔ centre, telle que l'écart entre les
        pointes des flippers AU REPOS vaille exactement drain_gap."""
        return self.drain_gap / 2 + self.flipper_length * math.cos(self.flipper_rest_angle)

    def pivot_pos(self, side: int) -> tuple[float, float]:
        """Pivot du flipper. side=+1 : gauche, side=-1 : droit."""
        return (self.width / 2 - side * self.pivot_offset, self.flipper_y)

    def guide_points(self, side: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Guide d'entonnoir : du mur latéral vers un point juste au-dessus
        du pivot (au ras du dessus du flipper, pour ne pas créer de poche)."""
        x0 = 0.0 if side > 0 else self.width
        px, py = self.pivot_pos(side)
        return (x0, self.guide_top_y), (px, py + self.guide_end_lift)

    def sling_verts(self, side: int) -> list[tuple[float, float]]:
        """Slingshot : triangle dont la base est SUR la ligne du guide
        (aucun interstice où coincer la balle), apex tourné vers le plateau."""
        (ax, ay), (bx, by) = self.guide_points(side)
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        # normale pointant vers l'intérieur du plateau
        nx, ny = (-dy, dx) if side > 0 else (dy, -dx)
        p1 = (ax + dx * self.sling_start, ay + dy * self.sling_start)
        p2 = (ax + dx * self.sling_end, ay + dy * self.sling_end)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        apex = (mx + nx * self.sling_height, my + ny * self.sling_height)
        return [p1, p2, apex]


def hard_board() -> BoardConfig:
    """La table officielle des expériences (notebooks 02, 04, 05).

    Drain largement ouvert (120 − 2×12 = 96 > diamètre 28 : la balle passive
    draine) et flippers courts. Mesuré : toutes les politiques aveugles
    s'effondrent (~2 s) — il faut VOIR la balle pour survivre.
    + 1 à 3 cibles aléatoires à toucher.
    """
    return BoardConfig(drain_gap=120.0, flipper_length=90.0, n_targets_range=(1, 3))
