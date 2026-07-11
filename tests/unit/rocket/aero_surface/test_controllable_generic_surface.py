"""Unit tests for ControllableGenericSurface and the controllable-surface
controller linkage."""

import pytest

from rocketpy import ControllableGenericSurface, Function, GenericSurface
from rocketpy.mathutils.vector_matrix import Vector

DENSITY = Function(lambda z: 1.16)
VISCOSITY = Function(lambda z: 1.8e-5)


def _moment_at_deflection(surface, deflection, comp="pitch"):
    surface.set_control("deflection", deflection)
    r1, r2, r3, m1, m2, m3 = surface.compute_forces_and_moments(
        Vector([0, 0, -100]),
        100,
        0.29,
        1.16,
        Vector([0, 0, 0]),
        Vector([0, 0, 0]),
        DENSITY,
        VISCOSITY,
        100.0,
    )
    return {"pitch": m1, "yaw": m2, "roll": m3}[comp]


def test_control_variable_extends_independent_vars():
    surface = ControllableGenericSurface(
        reference_area=1, reference_length=0.2, coefficients={}
    )
    assert surface.independent_vars[:7] == [
        "alpha",
        "beta",
        "mach",
        "reynolds",
        "pitch_rate",
        "yaw_rate",
        "roll_rate",
    ]
    assert surface.independent_vars[7:] == ["deflection"]
    assert surface.control_state == {"deflection": 0.0}


def test_deflection_produces_proportional_control_moment():
    surface = ControllableGenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={"cm": lambda a, b, m, re, p, q, r, deflection: 0.5 * deflection},
    )
    m0 = _moment_at_deflection(surface, 0.0)
    m1 = _moment_at_deflection(surface, 0.1)
    m2 = _moment_at_deflection(surface, 0.2)
    assert m0 == pytest.approx(0.0)
    assert m2 == pytest.approx(2 * m1)
    assert m1 != pytest.approx(0.0)


def test_multiple_named_controls():
    surface = ControllableGenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={"cn": lambda a, b, m, re, p, q, r, dp, dy: 0.3 * dy},
        controls=("delta_pitch", "delta_yaw"),
    )
    assert surface.independent_vars[7:] == ["delta_pitch", "delta_yaw"]
    surface.set_control("delta_yaw", 0.5)
    yaw = surface.compute_forces_and_moments(
        Vector([0, 0, -100]),
        100,
        0.29,
        1.16,
        Vector([0, 0, 0]),
        Vector([0, 0, 0]),
        DENSITY,
        VISCOSITY,
        100.0,
    )[4]
    assert yaw != pytest.approx(0.0)


def test_set_control_unknown_name_raises():
    surface = ControllableGenericSurface(
        reference_area=1, reference_length=0.2, coefficients={}
    )
    with pytest.raises(KeyError):
        surface.set_control("not_a_control", 0.1)


def test_plain_generic_surface_default_independent_vars_unchanged():
    surface = GenericSurface(reference_area=1, reference_length=0.2, coefficients={})
    assert surface.independent_vars == [
        "alpha",
        "beta",
        "mach",
        "reynolds",
        "pitch_rate",
        "yaw_rate",
        "roll_rate",
    ]
