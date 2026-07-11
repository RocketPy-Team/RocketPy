"""Unit tests for the optional alpha_dot/beta_dot unsteady coefficient axes of
GenericSurface."""

import pytest

from rocketpy import Function, GenericSurface
from rocketpy.mathutils.vector_matrix import Vector

DENSITY = Function(lambda z: 1.16)
VISCOSITY = Function(lambda z: 1.8e-5)


def _pitch_moment(surface, alpha_dot):
    return surface.compute_forces_and_moments(
        Vector([0, 0, -100]),
        100,
        0.29,
        1.16,
        Vector([0, 0, 0]),
        Vector([0, 0, 0]),
        DENSITY,
        VISCOSITY,
        100.0,
        alpha_dot=alpha_dot,
        beta_dot=0.0,
    )[3]


def test_unsteady_axes_extend_independent_vars():
    surface = GenericSurface(
        reference_area=1, reference_length=0.2, coefficients={}, unsteady_aero=True
    )
    assert surface.independent_vars[7:] == ["alpha_dot", "beta_dot"]


def test_prescribed_alpha_dot_produces_unsteady_pitch_moment():
    surface = GenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={
            "cm": lambda a, b, m, re, p, q, r, alpha_dot, beta_dot: 0.7 * alpha_dot
        },
        unsteady_aero=True,
    )
    m0 = _pitch_moment(surface, 0.0)
    m1 = _pitch_moment(surface, 0.5)
    m2 = _pitch_moment(surface, 1.0)
    assert m0 == pytest.approx(0.0)
    assert m1 != pytest.approx(0.0)
    assert m2 == pytest.approx(2 * m1)


def test_default_surface_ignores_alpha_dot_and_stays_seven_var():
    """Existing 7-variable surfaces must be unaffected: independent vars
    unchanged and alpha_dot/beta_dot ignored at evaluation."""
    surface = GenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={"cm": lambda a, b, m, re, p, q, r: 0.1},
    )
    assert surface.independent_vars == [
        "alpha",
        "beta",
        "mach",
        "reynolds",
        "pitch_rate",
        "yaw_rate",
        "roll_rate",
    ]
    # passing nonzero alpha_dot must not change the result
    assert _pitch_moment(surface, 0.0) == pytest.approx(_pitch_moment(surface, 99.0))
