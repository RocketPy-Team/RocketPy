"""Air brakes as an aerodynamic surface: their drag comes from the surface
loop like every other surface's, with the rocket's own drag left out while
brakes built with ``override_rocket_drag`` are deployed."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import AirBrakes, Rocket
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control.control_signal import initial_controls
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.simulation.helpers.flight_derivatives import (
    _compute_drag_area,
    u_dot_generalized,
)

CURVE = "data/rockets/calisto/air_brakes_cd.csv"


def _brakes(override=False, name="AirBrakes"):
    return AirBrakes(CURVE, 0.01, override_rocket_drag=override, name=name)


def _stand_in(rocket, environment):
    """A stand-in flight of ``rocket``, holding what the derivatives read."""
    controls, channels, _ = initial_controls(rocket.actuators)
    air_brakes = rocket._air_brakes
    return controls, SimpleNamespace(
        rocket=rocket,
        env=environment,
        _surface_switches={},
        _control_channels=channels,
        _rail_air_brakes=air_brakes,
        _drag_overrides=tuple(
            channels[brakes][0] for brakes in air_brakes if brakes.override_rocket_drag
        ),
    )


def _forces_and_moments(flight, controls, speed=200.0):
    """R1, R2, R3, M1, M2, M3 of the rocket coasting nose-first at ``speed``,
    straight up, no wind."""
    upright = [1.0, 0.0, 0.0, 0.0]
    state = [0.0, 0.0, 1500.0, 0.0, 0.0, speed, *upright, 0.0, 0.0, 0.0]
    out = u_dot_generalized(flight, 10.0, state, controls, post_processing=True)
    return np.array(out[6:12])


@pytest.mark.parametrize("override", [False, True])
def test_retracted_air_brakes_make_no_drag(override):
    brakes = _brakes(override)
    assert brakes.cA(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0
    forces = brakes.compute_forces_and_moments(
        Vector([0, 0, -200.0]),
        200.0,
        0.6,
        1.0,
        Vector([0, 0, 0]),
        Vector([0, 0, 0]),
        None,
        None,
        0.0,
        [0.0],
    )
    assert all(value == 0 for value in forces)


def test_override_drops_the_body_drag_while_deployed(calisto_robust, example_plain_env):
    calisto_robust.add_surfaces(_brakes(override=True), 0.0)
    controls, flight = _stand_in(calisto_robust, example_plain_env)
    arguments = (Vector([0, 0, -200.0]), 200.0, 0.6, 1.0, 1.8e-5, (0, 0, 0))

    body = _compute_drag_area(flight, 10.0, *arguments, controls)
    controls["air_brakes.deployment_level"] = 0.5
    assert body > 0
    assert _compute_drag_area(flight, 10.0, *arguments, controls) == 0.0


def test_the_drag_is_the_same_as_before_at_zero_angle_of_attack(
    calisto_robust, example_plain_env
):
    """Nose-first, the brakes' drag is what the separate air-brakes code gave:
    0.5 rho V^2 times the area and the curve at the deployment level."""
    calisto_robust.add_air_brakes(CURVE, lambda context: None, 10)
    controls, flight = _stand_in(calisto_robust, example_plain_env)
    retracted = _forces_and_moments(flight, controls)
    controls["air_brakes.deployment_level"] = 0.7
    deployed = _forces_and_moments(flight, controls)

    (brakes,) = calisto_robust._air_brakes
    rho = example_plain_env.density(1500.0)
    mach = 200.0 / example_plain_env.speed_of_sound(1500.0)
    expected = -0.5 * rho * 200.0**2 * brakes.reference_area
    expected *= brakes.drag_coefficient(0.7, mach)
    assert deployed[2] - retracted[2] == pytest.approx(expected, rel=1e-12)
    # pinned at the center of dry mass: no moment, and no side force
    assert list(deployed[[0, 1, 3, 4, 5]]) == list(retracted[[0, 1, 3, 4, 5]])


def test_air_brakes_without_a_position_follow_the_center_of_dry_mass(
    calisto_motorless, cesaroni_m1670
):
    rocket = calisto_motorless
    rocket.add_air_brakes(CURVE, lambda context: None, 10)
    rocket.add_motor(cesaroni_m1670, position=-1.373)
    rocket._refresh_aerodynamics()
    (brakes,) = rocket._air_brakes
    assert list(rocket.surfaces_cp_to_cdm[brakes]) == [0, 0, 0]


def test_air_brakes_off_the_axis_make_a_moment(calisto_robust, example_plain_env):
    calisto_robust.add_air_brakes(
        CURVE, lambda context: None, 10, position=(0.05, 0.0, 0.0)
    )
    controls, flight = _stand_in(calisto_robust, example_plain_env)
    retracted = _forces_and_moments(flight, controls)
    controls["air_brakes.deployment_level"] = 0.7
    deployed = _forces_and_moments(flight, controls)

    (brakes,) = calisto_robust._air_brakes
    lever = calisto_robust.surfaces_cp_to_cdm[brakes]
    force = Vector(list(deployed[:3] - retracted[:3]))
    moment = deployed[3:] - retracted[3:]
    assert lever.x != 0
    assert list(moment) == pytest.approx(list(lever ^ force), rel=1e-12)
    assert moment[1] != 0


def test_air_brakes_are_saved_once_inside_the_surfaces(calisto_robust):
    calisto_robust.add_air_brakes(CURVE, lambda context: None, 10)
    data = calisto_robust.to_dict()
    assert "air_brakes" not in data
    assert any(isinstance(s, AirBrakes) for s, _ in data["aerodynamic_surfaces"])

    loaded = json.loads(
        json.dumps(calisto_robust, cls=RocketPyEncoder), cls=RocketPyDecoder
    )
    (brakes,) = loaded._air_brakes
    assert brakes._at_center_of_dry_mass
    assert loaded.control_channels == ("air_brakes.deployment_level",)


def test_air_brakes_saved_before_they_were_surfaces_load(calisto_robust):
    """Older files keep air brakes in their own list, apart from the surfaces;
    they load as surfaces at the center of dry mass."""
    saved = json.loads(json.dumps(calisto_robust, cls=RocketPyEncoder))
    brakes = json.loads(json.dumps(_brakes(), cls=RocketPyEncoder))
    del brakes["at_center_of_dry_mass"]
    saved["air_brakes"] = [brakes]

    loaded = json.loads(json.dumps(saved), cls=RocketPyDecoder)
    assert isinstance(loaded, Rocket)
    (loaded_brakes,) = loaded._air_brakes
    assert loaded_brakes._at_center_of_dry_mass
    loaded._refresh_aerodynamics()
    assert list(loaded.surfaces_cp_to_cdm[loaded_brakes]) == [0, 0, 0]


def test_rocket_air_brakes_is_a_deprecated_view(calisto_robust):
    brakes = calisto_robust.add_air_brakes(CURVE, lambda context: None, 10)
    with pytest.warns(DeprecationWarning, match="aerodynamic_surfaces"):
        assert calisto_robust.air_brakes == [brakes]


def test_memory_and_return_controller_are_deprecated(calisto_robust):
    with pytest.warns(DeprecationWarning, match="memory"):
        calisto_robust.add_air_brakes(
            CURVE, lambda context: None, 10, memory={"a": 1}, name="First"
        )
    with pytest.warns(DeprecationWarning, match="return_controller"):
        brakes, controller = calisto_robust.add_air_brakes(
            CURVE, lambda context: None, 10, return_controller=True, name="Second"
        )
    assert isinstance(brakes, AirBrakes)
    assert controller in calisto_robust._controllers
