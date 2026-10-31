"""Geometry adapters for the two comparison notebooks."""

import ast
import numpy as np
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket.aero_surface import TrapezoidalFins

MACH = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

ALPHA = [
    -20,
    -10,
    -5,
    -2,
    -1,
    -0.5,
    -0.25,
    -0.1,
    -0.05,
    0,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
    20,
]

BETA = [-20, -5, 0, 5, 20]


class LocalPaths(ast.NodeTransformer):
    def __init__(self, root):
        self.root = root

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("../../data/"):
            return ast.copy_location(
                ast.Constant(str(self.root / node.value[6:])), node
            )
        return node


def attach_datcom(rocket, executable, schedules):
    """Replace fin definitions by identical planforms, assuming 2% HEX sections."""
    options = dict(
        mach=MACH,
        alpha=ALPHA,
        beta=BETA,
        altitude=schedules,
        powered_base_drag="nozzle",
    )
    originals = [
        (s, p)
        for s, p, _ in rocket.aerodynamic_surfaces
        if isinstance(s, TrapezoidalFins)
    ]
    if not originals:
        raise ValueError("Source rocket has no supported trapezoidal fin set")
    geometry = []
    for fin, position in originals:
        rocket.aerodynamic_surfaces.remove(fin)
        replacement = rocket.add_datcom_hex_fins(
            n=fin.n,
            root_chord=fin.root_chord,
            tip_chord=fin.tip_chord,
            span=fin.span,
            sweep_length=fin.sweep_length,
            cant_angle=fin.cant_angle,
            position=position,
            path=executable,
            thickness_to_chord=0.02,
            max_thickness_position=0.05,
            flat_length=0.9,
            datcom_options=options,
        )
        geometry.append(
            dict(
                n=fin.n,
                root_chord_m=fin.root_chord,
                tip_chord_m=fin.tip_chord,
                span_m=fin.span,
                sweep_m=fin.sweep_length,
                cant_deg=fin.cant_angle,
                position_m=list(position),
                assumed_section="symmetric HEX, 2% thickness, 5% edge bevels",
            )
        )
        assert replacement.root_chord == fin.root_chord
    return geometry


def envelope_diagnostics(flight, model, mach_grid=None, alpha_grid=None):
    """Count sampled envelope excursions at meaningful pressure (q >= 100 Pa).

    Also report solver/probe flags separately; these include pad/near-apogee
    flow. Sampling is uniform in time and is diagnostic, not an accuracy bound.
    """
    solutions = np.asarray(flight.solution)
    mach_grid = MACH if mach_grid is None else mach_grid
    alpha_grid = ALPHA if alpha_grid is None else alpha_grid
    samples = np.linspace(0, flight.apogee_time, 501)
    counts = dict(samples=0, outside_mach=0, outside_alpha=0, outside_beta=0)
    max_angles = np.zeros(2)
    for time in samples:
        state = [
            np.interp(time, solutions[:, 0], solutions[:, j]) for j in range(1, 14)
        ]
        rotation = Matrix.transformation(state[6:10])
        offset = model.reference_offset(time)
        z = state[2] + (rotation @ offset).z
        wind = Vector([flight.env.wind_velocity_x(z), flight.env.wind_velocity_y(z), 0])
        velocity = rotation.transpose @ (Vector(state[3:6]) - wind) + (
            Vector(state[10:13]) ^ offset
        )
        speed = abs(velocity)
        if 0.5 * flight.env.density(z) * speed**2 < 100:
            continue
        alpha = np.degrees(np.arctan2(velocity.y, velocity.z))
        beta = np.degrees(np.arctan2(velocity.x, velocity.z))
        mach = speed / flight.env.speed_of_sound(z)
        counts["samples"] += 1
        counts["outside_mach"] += int(not mach_grid[0] <= mach <= mach_grid[-1])
        counts["outside_alpha"] += int(not alpha_grid[0] <= alpha <= alpha_grid[-1])
        counts["outside_beta"] += int(not BETA[0] <= beta <= BETA[-1])
        max_angles = np.maximum(max_angles, np.abs([alpha, beta]))
    return dict(
        counts,
        max_abs_alpha_deg=float(max_angles[0]),
        max_abs_beta_deg=float(max_angles[1]),
        solver_mach_excursion=bool(model.outside_mach_envelope),
        solver_alpha_excursion=bool(model.outside_alpha_envelope),
        solver_beta_excursion=bool(model.outside_beta_envelope),
    )
