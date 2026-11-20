"""Unit tests for ControllableGenericSurface and the controllable-surface
controller linkage."""

import pytest

from rocketpy import ControllableGenericSurface, Function, GenericSurface
from rocketpy.mathutils.vector_matrix import Vector

DENSITY = Function(lambda z: 1.16)
VISCOSITY = Function(lambda z: 1.8e-5)


def _moment_at_deflection(surface, deflection, comp="pitch"):
    surface.set_control("deflection", deflection)
    _, _, _, m1, m2, m3 = surface.compute_forces_and_moments(
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


def test_active_during_preset_round_trips_through_dict():
    """A preset activation policy survives to_dict/from_dict (jet-vane case)."""
    surface = ControllableGenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={},
        active_during="power_on",
    )
    restored = ControllableGenericSurface.from_dict(surface.to_dict())
    assert restored.active_during == "power_on"


def test_starting_switched_off_round_trips_through_dict():
    """Whether the surface starts each flight switched on is saved with it."""
    surface = ControllableGenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={},
        active=False,
    )
    restored = ControllableGenericSurface.from_dict(surface.to_dict())
    assert restored.active is False
    assert (
        ControllableGenericSurface.from_dict(
            ControllableGenericSurface(1, 0.2, {}).to_dict()
        ).active
        is True
    )


def test_controls_and_coefficients_round_trip_through_dict():
    """Saving and loading keeps the control names and the coefficient values,
    including for coefficients given in the wind frame."""
    surface = ControllableGenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={
            "cL": lambda alpha, canard: 2 * alpha + 0.5 * canard,
            "cD": 0.3,
        },
        controls=("canard", "elevon"),
        force_convention="wind",
    )
    surface.set_control("canard", 0.2)

    restored = ControllableGenericSurface.from_dict(surface.to_dict())
    restored.set_control("canard", 0.2)

    assert restored.control_variables == ["canard", "elevon"]
    args = (0.05, 0.02, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0)
    for name in ("cN", "cY", "cA"):
        assert getattr(restored, name)(*args) == pytest.approx(
            getattr(surface, name)(*args)
        )


def test_controllable_surface_from_csv_reads_the_control_column(tmp_path):
    csv_file = tmp_path / "canard.csv"
    rows = [f"{d},{m},{1.5 * d * (1 + m)}" for d in (-0.2, 0.0, 0.2) for m in (0, 1, 2)]
    csv_file.write_text("canard,mach,cN\n" + "\n".join(rows) + "\n")

    surface = ControllableGenericSurface.from_csv(
        str(csv_file), 1.0, 1.0, controls=("canard",)
    )
    surface.set_control("canard", 0.1)

    assert surface.cN.depends_on == ("canard", "mach")
    args = surface._coefficient_arguments(0.0, 0.0, 1.0, 0, 0, 0, 0)
    assert surface.cN(*args) == pytest.approx(0.3)
