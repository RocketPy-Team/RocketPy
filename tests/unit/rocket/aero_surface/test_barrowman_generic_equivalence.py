"""Regression tests for the GenericSurface-rooted aerodynamic hierarchy.

After the refactor, every aerodynamic surface (Barrowman or generic) exposes
the coefficient derivatives ``cN_alpha``/``cY_beta`` and the
``aerodynamic_center`` accessor used by the rocket's center-of-pressure /
stability-margin computation. Barrowman surfaces compute their flight forces
with the classic geometric method, and their public ``cN``/``cY``/``cA``/``cl``
coefficients are that same method written as coefficients. These tests pin the
properties the refactor guarantees.
"""

import warnings

import numpy as np
import pytest

from rocketpy import (
    LinearGenericSurface,
    NoseCone,
    Tail,
    TrapezoidalFin,
    TrapezoidalFins,
)
from rocketpy.mathutils import Vector


def test_barrowman_derived_cp_matches_geometric_cp():
    """The derived ``aerodynamic_center`` diagnostic must reproduce the
    geometric cp of each Barrowman surface. It is given along the body z-axis
    (positive toward the nose), while ``cpz`` is measured from the nose toward
    the tail, hence the sign."""
    nose = NoseCone(
        length=0.55829, kind="vonkarman", base_radius=0.0635, rocket_radius=0.0635
    )
    tail = Tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, rocket_radius=0.0635
    )
    fins = TrapezoidalFins(
        n=4, span=0.100, root_chord=0.120, tip_chord=0.040, rocket_radius=0.0635
    )

    for surface in (nose, tail, fins):
        for mach in (0.0, 0.5, 0.9):
            assert (
                pytest.approx(
                    surface.aerodynamic_center.get_value_opt(mach), rel=1e-6, abs=1e-9
                )
                == -surface.cpz
            )
        # The normal-force slope derivative must equal the Barrowman clalpha.
        assert pytest.approx(
            nose.cN_alpha.get_value_opt(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ) == nose.clalpha.get_value_opt(0.0)


def test_generic_surface_contributes_to_static_margin(calisto_motorless):
    """A generic surface must now contribute to the rocket center of pressure
    (previously generic surfaces were skipped, breaking stability margin)."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)

    cp_without_generic = rocket.aerodynamic_center.get_value_opt(0.2)

    # A lifting generic surface placed aft should move the cp aft (more stable).
    generic = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={
            "cN_alpha": lambda a, b, m, re, p, q, r: 2.0,
            "cm_alpha": lambda a, b, m, re, p, q, r: -1.0,
        },
        name="generic_fins",
    )
    rocket.add_surfaces(generic, positions=-1.0)

    cp_with_generic = rocket.aerodynamic_center.get_value_opt(0.2)
    assert cp_with_generic != pytest.approx(cp_without_generic)
    assert np.isfinite(cp_with_generic)


def test_zero_lift_surface_does_not_break_cp(calisto_motorless):
    """A surface with no normal-force slope must drop out of the lift-weighted
    cp average without producing NaNs."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    cp_reference = rocket.aerodynamic_center.get_value_opt(0.2)

    drag_only = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={"cA_0": lambda a, b, m, re, p, q, r: 0.5},
        name="drag_only",
    )
    rocket.add_surfaces(drag_only, positions=-1.0)

    cp_after = rocket.aerodynamic_center.get_value_opt(0.2)
    assert np.isfinite(cp_after)
    assert cp_after == pytest.approx(cp_reference)


def test_axisymmetric_rocket_pitch_equals_yaw_margin(calisto_motorless):
    """An axisymmetric rocket must have identical pitch and yaw margins and
    must not raise the asymmetry warning."""
    rocket = calisto_motorless
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # asymmetry warning would fail the test
        rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
        rocket.add_trapezoidal_fins(
            n=4, span=0.100, root_chord=0.120, tip_chord=0.040, position=-1.04
        )
        rocket.add_tail(
            top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194
        )

    for mach in (0.0, 0.5, 0.9):
        assert rocket.aerodynamic_center.get_value_opt(mach) == pytest.approx(
            rocket.aerodynamic_center_yaw.get_value_opt(mach), abs=1e-9
        )
    assert rocket.static_margin.get_value_opt(0) == pytest.approx(
        rocket.static_margin_yaw.get_value_opt(0), abs=1e-9
    )


def test_non_axisymmetric_rocket_splits_margins_and_warns(calisto_motorless):
    """A non-axisymmetric generic surface must yield distinct pitch/yaw margins
    and raise a warning that the scalar margin describes the pitch plane only.

    The advisory is emitted lazily -- on the first evaluation of the aerodynamic
    center, not eagerly at add time -- so adding the surface itself is silent."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    asymmetric = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={
            "cN_alpha": lambda a, b, m, re, p, q, r: 2.0,
            "cm_alpha": lambda a, b, m, re, p, q, r: -1.0,
            "cY_beta": lambda a, b, m, re, p, q, r: -2.0,
            "cn_beta": lambda a, b, m, re, p, q, r: 2.0,
        },
        name="asym",
    )

    rocket.add_surfaces(asymmetric, positions=-1.0)

    # Warning fires once, on the first aerodynamic-center evaluation.
    with pytest.warns(UserWarning, match="not\\s+axisymmetric"):
        ac_pitch = rocket.aerodynamic_center.get_value_opt(0.2)

    assert ac_pitch != pytest.approx(rocket.aerodynamic_center_yaw.get_value_opt(0.2))
    assert rocket.static_margin.get_value_opt(0) != pytest.approx(
        rocket.static_margin_yaw.get_value_opt(0)
    )


def test_barrowman_surface_uses_geometric_compute_path():
    """Barrowman surfaces compute their normal force and moment with the classic
    Barrowman method (their own ``compute_forces_and_moments``): the resultant
    force is reported at the geometric center of pressure and its moment is
    transported geometrically from there."""
    from rocketpy.rocket.aero_surface._barrowman_surface import _BarrowmanSurface
    from rocketpy.rocket.aero_surface.generic_surface import GenericSurface

    nose = NoseCone(
        length=0.55829, kind="vonkarman", base_radius=0.0635, rocket_radius=0.0635
    )
    assert isinstance(nose, GenericSurface)
    # Force is reported at the surface's geometric center of pressure.
    assert tuple(nose.force_application_point) == (nose.cpx, nose.cpy, nose.cpz)
    # Uses the Barrowman geometric compute, not the generic coefficient path.
    assert (
        nose.compute_forces_and_moments.__func__
        is _BarrowmanSurface.compute_forces_and_moments
    )


# Airflow relative to the surface, in the body frame (m/s): small and large
# angles, both planes at once, no crossflow and tail-first flow.
_STREAM_VELOCITIES = [
    (3.0, -2.0, -100.0),
    (20.0, 35.0, -80.0),
    (-50.0, 10.0, -60.0),
    (5.0, 0.0, -100.0),
    (0.0, 0.0, -100.0),
    (30.0, 30.0, 40.0),
]


def _barrowman_surfaces():
    return [
        NoseCone(
            length=0.55829, kind="vonkarman", base_radius=0.0635, rocket_radius=0.0635
        ),
        Tail(
            top_radius=0.0635, bottom_radius=0.0435, length=0.060, rocket_radius=0.0635
        ),
        TrapezoidalFins(
            n=4,
            span=0.100,
            root_chord=0.120,
            tip_chord=0.040,
            rocket_radius=0.0635,
            cant_angle=1.0,
        ),
        TrapezoidalFin(
            angular_position=30,
            span=0.100,
            root_chord=0.120,
            tip_chord=0.040,
            rocket_radius=0.0635,
            cant_angle=2.0,
        ),
    ]


@pytest.mark.parametrize("stream_velocity", _STREAM_VELOCITIES)
def test_public_coefficients_are_what_flies(stream_velocity):
    """The forces used in the simulation (the fast classic Barrowman computation)
    must equal the surface's public ``cN``, ``cY`` and ``cA`` coefficients times
    the dynamic pressure and the reference area, so that what a user reads or
    plots is what flies."""
    rho, mach = 1.2, 0.3
    stream = Vector(stream_velocity)
    speed = abs(stream)
    # The coefficients take the angles of the rocket's velocity relative to the air
    alpha = np.arctan2(-stream[1], -stream[2])
    beta = np.arctan2(-stream[0], -stream[2])
    args = (alpha, beta, mach, 0.0, 0.0, 0.0, 0.0)

    for surface in _barrowman_surfaces():
        r1, r2, r3, *_ = surface.compute_forces_and_moments(
            stream, speed, mach, rho, Vector([0, 0, 0]), (0, 0, 0)
        )
        scale = 0.5 * rho * speed**2 * surface.reference_area
        assert r1 == pytest.approx(scale * surface.cY(*args), rel=1e-12, abs=1e-9)
        assert r2 == pytest.approx(-scale * surface.cN(*args), rel=1e-12, abs=1e-9)
        assert r3 == pytest.approx(-scale * surface.cA(*args), rel=1e-12, abs=1e-9)


def test_public_roll_coefficient_is_what_flies():
    """A fin set's roll moment in the simulation must equal its public ``cl``
    (cant forcing plus roll damping) times the dynamic pressure, the reference
    area and the reference length; an individual fin's ``cl`` is its roll
    damping."""
    rho, mach, speed, roll = 1.2, 0.3, 100.0, 6.0
    stream = Vector([0.0, 0.0, -speed])  # no crossflow: only the roll moment is left
    nose, _, fins, fin = _barrowman_surfaces()

    for surface in (nose, fins, fin):
        moment = surface.compute_forces_and_moments(
            stream, speed, mach, rho, Vector([0, 0, 0]), (0, 0, roll)
        )[5]
        reduced_roll = roll * surface.reference_length / (2 * speed)
        scale = 0.5 * rho * speed**2 * surface.reference_area * surface.reference_length
        expected = scale * surface.cl(0.0, 0.0, mach, 0.0, 0.0, 0.0, reduced_roll)
        assert moment == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert nose.cl.is_zero
    assert fins.cl_0(0, 0, mach, 0, 0, 0, 0) != 0
