"""Unit tests for the controls of an aerodynamic surface: the extra inputs its
coefficients read, and how they are saved."""

import json
import warnings

import numpy as np
import pytest

from rocketpy import (
    AirBrakes,
    Function,
    GenericSurface,
    LinearGenericSurface,
    StochasticAirBrakes,
)
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control.control_signal import initial_controls
from rocketpy.mathutils.vector_matrix import Vector

DENSITY = Function(lambda z: 1.16)
VISCOSITY = Function(lambda z: 1.8e-5)
FLOW = (
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


def _canard(**kwargs):
    return GenericSurface(
        reference_area=1,
        reference_length=0.2,
        coefficients={"cm": lambda mach, deflection: 0.5 * deflection},
        name="Canards",
        controls=("deflection",),
        **kwargs,
    )


def _pitch_moment(surface, controls=None):
    return surface.compute_forces_and_moments(*FLOW, controls)[3]


def test_a_surface_has_no_controls_unless_given():
    surface = GenericSurface(1, 0.2, {})
    assert surface.controls == ()
    assert surface.control_defaults == {}
    assert surface.control_limits == {}
    assert surface.active is True


def test_a_control_extends_the_variables_of_the_coefficients():
    surface = _canard()
    assert surface.controls == ("deflection",)
    assert surface.independent_vars == [
        "alpha",
        "beta",
        "mach",
        "reynolds",
        "pitch_rate",
        "yaw_rate",
        "roll_rate",
        "deflection",
    ]
    assert surface.control_defaults == {"deflection": 0.0}
    assert surface.control_limits == {"deflection": None}


def test_the_controls_given_produce_a_proportional_moment():
    surface = _canard()
    m0 = _pitch_moment(surface, [0.0])
    m1 = _pitch_moment(surface, [0.1])
    m2 = _pitch_moment(surface, [0.2])
    assert m0 == pytest.approx(0.0)
    assert m1 != pytest.approx(0.0)
    assert m2 == pytest.approx(2 * m1)


def test_without_controls_the_defaults_are_used():
    """Outside a flight a surface is evaluated at its defaults, the same as
    with those values given explicitly."""
    surface = _canard(control_defaults={"deflection": 0.1})
    assert _pitch_moment(surface) == _pitch_moment(surface, [0.1])
    assert surface._coefficient_arguments(0, 0, 0.5, 0, 0, 0, 0) == (
        surface._coefficient_arguments(0, 0, 0.5, 0, 0, 0, 0, controls=[0.1])
    )
    # a surface without controls is called with the seven flow variables
    plain = GenericSurface(1, 0.2, {"cA": 0.4})
    assert plain._coefficient_arguments(1, 2, 3, 4, 5, 6, 7) == (1, 2, 3, 4, 5, 6, 7)
    assert plain._coefficient_arguments(1, 2, 3, 4, 5, 6, 7, controls=[]) == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )


def test_several_controls_keep_their_order():
    surface = GenericSurface(
        1,
        0.2,
        {"cn": lambda delta_pitch, delta_yaw: 0.3 * delta_yaw},
        controls=("delta_pitch", "delta_yaw"),
    )
    assert surface.independent_vars[7:] == ["delta_pitch", "delta_yaw"]
    yaw = surface.compute_forces_and_moments(*FLOW, [0.0, 0.5])[4]
    assert yaw != pytest.approx(0.0)
    assert surface.compute_forces_and_moments(*FLOW, [0.5, 0.0])[4] == 0.0


def test_defaults_outside_the_limits_are_clamped_or_warned():
    clamped = _canard(
        control_defaults={"deflection": 0.5}, control_limits={"deflection": (-0.2, 0.2)}
    )
    assert clamped.control_defaults["deflection"] == 0.2
    with pytest.warns(UserWarning, match="outside its limits"):
        kept = _canard(
            control_defaults={"deflection": 0.5},
            control_limits={"deflection": (-0.2, 0.2)},
            clamp=False,
        )
    assert kept.control_defaults["deflection"] == 0.5


@pytest.mark.parametrize(
    "controls, match",
    [
        (("mach",), "one of the flow variables"),
        (("alpha_deg",), "one of the flow variables"),
        (("deflection", "deflection"), "lists a control twice"),
        (("2nd",), "must be made of letters"),
        (("_hidden",), "must be made of letters"),
    ],
)
def test_a_bad_control_name_is_refused(controls, match):
    with pytest.raises(ValueError, match=match):
        GenericSurface(1, 0.2, {}, controls=controls)


def test_defaults_and_limits_must_name_a_control():
    with pytest.raises(ValueError, match="control_defaults for \\['flap'\\]"):
        _canard(control_defaults={"flap": 0.1})
    with pytest.raises(ValueError, match="control_limits for \\['flap'\\]"):
        _canard(control_limits={"flap": (0, 1)})
    with pytest.raises(ValueError, match="must not be larger"):
        _canard(control_limits={"deflection": (0.2, -0.2)})


def test_a_flight_never_changes_the_surface():
    """A controller writes the flight's control inputs: the surface keeps no
    state."""
    surface = _canard()
    before = dict(surface.control_defaults)
    controls, _, _ = initial_controls([surface])
    controls["canards.deflection"] = 0.15
    assert surface.control_defaults == before
    assert not hasattr(surface, "control_state")
    assert _pitch_moment(surface) == pytest.approx(0.0)


def test_controls_round_trip_through_dict():
    surface = GenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={
            "cL": lambda alpha, canard: 2 * alpha + 0.5 * canard,
            "cD": 0.3,
        },
        controls=("canard", "elevon"),
        control_defaults={"canard": 0.05},
        control_limits={"canard": (-0.3, 0.3)},
        clamp=False,
        force_convention="wind",
        active_during="power_on",
        active=False,
    )
    restored = GenericSurface.from_dict(surface.to_dict())

    assert restored.controls == ("canard", "elevon")
    assert restored.control_defaults == surface.control_defaults
    assert restored.control_limits == surface.control_limits
    assert restored.clamp is False
    assert restored.active_during == "power_on"
    assert restored.active is False
    args = (0.05, 0.02, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0)
    for name in ("cN", "cY", "cA"):
        assert getattr(restored, name)(*args) == pytest.approx(
            getattr(surface, name)(*args)
        )


def test_a_surface_without_controls_saves_as_before():
    data = GenericSurface(1, 0.2, {"cA": 0.4}).to_dict()
    assert "controls" not in data
    assert data["active"] is True


def test_a_saved_controllable_generic_surface_loads_as_a_generic_surface():
    """Files saved while ControllableGenericSurface existed keep loading: it
    saved the same arguments, ``controls`` included."""
    surface = GenericSurface(
        1,
        0.2,
        {"cm": ([[-0.2, -0.1], [0.0, 0.0], [0.2, 0.1]], ["deflection"])},
        controls=("deflection",),
    )
    saved = json.loads(json.dumps(surface, cls=RocketPyEncoder))
    saved["signature"].update(
        module="rocketpy.rocket.aero_surface.controllable_generic_surface",
        name="ControllableGenericSurface",
    )
    loaded = json.loads(json.dumps(saved), cls=RocketPyDecoder)

    assert type(loaded) is GenericSurface  # pylint: disable=unidiomatic-typecheck
    assert loaded.controls == ("deflection",)
    assert _pitch_moment(loaded, [0.2]) == pytest.approx(_pitch_moment(surface, [0.2]))


def test_from_csv_reads_the_control_column(tmp_path):
    csv_file = tmp_path / "canard.csv"
    rows = [f"{d},{m},{1.5 * d * (1 + m)}" for d in (-0.2, 0.0, 0.2) for m in (0, 1, 2)]
    csv_file.write_text("canard,mach,cN\n" + "\n".join(rows) + "\n")

    surface = GenericSurface.from_csv(
        str(csv_file), 1.0, 1.0, controls=("canard",), control_defaults={"canard": 0.1}
    )

    assert surface.cN.depends_on == ("canard", "mach")
    args = surface._coefficient_arguments(0.0, 0.0, 1.0, 0, 0, 0, 0)
    assert surface.cN(*args) == pytest.approx(0.3)


def test_the_stability_slopes_are_taken_at_the_control_defaults():
    """A moment coupling the angle of attack and a control gives a slope that
    depends on the control; the stability analysis reads it at the default."""
    coefficients = {
        "cN": lambda alpha: 2.0 * alpha,
        "cm": lambda alpha, deflection: (1.0 + deflection) * alpha,
    }
    at_zero = GenericSurface(1, 0.2, coefficients, controls=("deflection",))
    at_half = GenericSurface(
        1,
        0.2,
        coefficients,
        controls=("deflection",),
        control_defaults={"deflection": 0.5},
    )
    assert at_zero.cm_alpha(0, 0, 0.3, 0, 0, 0, 0, 0) == pytest.approx(1.0)
    assert at_half.cm_alpha(0, 0, 0.3, 0, 0, 0, 0, 0) == pytest.approx(1.5)
    assert at_half.aerodynamic_center(0.3) != at_zero.aerodynamic_center(0.3)


def test_a_linear_surface_can_have_controls():
    surface = LinearGenericSurface(
        0.01,
        0.1,
        {"cl_0": lambda mach, deflection: 0.5 * deflection},
        controls=("deflection",),
        control_defaults={"deflection": 0.2},
    )
    assert surface.controls == ("deflection",)
    assert surface.cl.get_value_opt(0, 0, 0.5, 0, 0, 0, 0, 0.1) == pytest.approx(0.05)
    restored = LinearGenericSurface.from_dict(surface.to_dict())
    assert restored.control_defaults == {"deflection": 0.2}


# ---------------------------------------------------------------------------
# Air brakes
# ---------------------------------------------------------------------------


def _air_brakes(**kwargs):
    return AirBrakes(
        drag_coefficient_curve=lambda deployment_level, mach: deployment_level,
        reference_area=0.01,
        **kwargs,
    )


def test_air_brakes_have_one_control_with_limits():
    air_brakes = _air_brakes(deployment_level=0.3)
    assert air_brakes.controls == ("deployment_level",)
    assert air_brakes.control_defaults == {"deployment_level": 0.3}
    assert air_brakes.control_limits["deployment_level"] == (0, 1)
    assert air_brakes.clamp is True


def test_the_deployment_level_of_the_object_is_the_default():
    """It is the level every flight starts at, clamped like a write."""
    air_brakes = _air_brakes()
    air_brakes.deployment_level = 0.4
    assert air_brakes.control_defaults["deployment_level"] == 0.4
    assert air_brakes.initial_deployment_level == 0.4
    air_brakes.deployment_level = 1.5
    assert air_brakes.deployment_level == 1
    loose = _air_brakes(clamp=False)
    with pytest.warns(UserWarning, match="outside its limits"):
        loose.deployment_level = 1.5
    assert loose.deployment_level == 1.5


def test_air_brakes_drag_follows_the_deployment_given():
    air_brakes = _air_brakes(deployment_level=0.3)
    args = (0, 0, 0.3, 0, 0, 0, 0)
    assert air_brakes.cA(*air_brakes._coefficient_arguments(*args)) == 0.3
    assert air_brakes.cA(
        *air_brakes._coefficient_arguments(*args, controls=[0.8])
    ) == pytest.approx(0.8)
    # retracted brakes add no drag, unless they replace the rocket's drag
    assert air_brakes.cA(*air_brakes._coefficient_arguments(*args, controls=[0])) == 0


def test_air_brakes_round_trip_keeps_their_deployment_level():
    air_brakes = _air_brakes(deployment_level=0.25, clamp=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = AirBrakes.from_dict(air_brakes.to_dict())
    assert restored.deployment_level == 0.25
    assert restored.clamp is False


def test_dispersed_air_brakes_keep_their_name():
    """The name names the control channels, so a Monte Carlo copy keeps it."""
    air_brakes = _air_brakes(name="Upper Brakes")
    copy = StochasticAirBrakes(air_brakes=air_brakes).create_object()
    assert copy.name == "Upper Brakes"
    assert initial_controls([copy])[0] == {"upper_brakes.deployment_level": 0.0}


def test_prints_and_plots_read_the_surface_at_its_defaults(capsys):
    surface = GenericSurface(
        1,
        0.2,
        {"cN": lambda alpha, deflection: 2 * alpha + deflection},
        controls=("deflection",),
        control_defaults={"deflection": 0.1},
    )
    assert surface._reference_arguments() == [0.0] * 7 + [0.1]
    surface.prints.coefficients()
    expected = 2 * np.deg2rad(5) + 0.1
    assert f"cN = {expected:.4f}" in capsys.readouterr().out
