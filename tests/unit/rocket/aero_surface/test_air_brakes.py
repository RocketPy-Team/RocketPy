"""Air brakes as an aerodynamic surface: their drag comes from the surface
loop like every other surface's and is added to the rocket's own drag. Brakes
built with ``override_rocket_drag`` add what their whole-rocket curve gains
over its retracted values."""

import json
import warnings
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import AirBrakes, Function, Rocket
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control._helpers import initial_controls
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.simulation.helpers.flight_derivatives import (
    u_dot_generalized,
)

CURVE = "data/rockets/calisto/air_brakes_cd.csv"


def _brakes(override=False, name="AirBrakes"):
    return AirBrakes(CURVE, 0.01, override_rocket_drag=override, name=name)


def _stand_in(rocket, environment):
    """A stand-in flight of ``rocket``, holding what the derivatives read."""
    controls, channels, _ = initial_controls(rocket.controllables)
    flight = SimpleNamespace(
        rocket=rocket,
        env=environment,
        _surface_switches={},
        _control_names=channels,
    )
    return controls, flight


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


# A curve of the whole rocket: 0.5 retracted, rising to 0.9 fully open, and
# 0.1 more at Mach 1
WHOLE_ROCKET_CURVE = [
    [0.0, 0.0, 0.5],
    [0.0, 1.0, 0.6],
    [1.0, 0.0, 0.9],
    [1.0, 1.0, 1.0],
]


def test_a_whole_rocket_curve_adds_its_gain_over_the_retracted_values():
    brakes = AirBrakes(WHOLE_ROCKET_CURVE, 0.01, override_rocket_drag=True)
    at = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0)
    assert brakes.cA(*at, 1.0) == pytest.approx(0.4)
    assert brakes.cA(*at, 0.5) == pytest.approx(0.2)
    # continuous through deployment level 0
    assert brakes.cA(*at, 1e-9) == pytest.approx(0.0, abs=1e-8)
    assert brakes.cA(*at, 0.0) == 0.0


def test_the_total_drag_is_the_whole_rocket_curve(calisto_robust, example_plain_env):
    """With the rocket's drag set to the curve's retracted values, the drag of
    the rocket with its brakes open is the curve at that deployment level."""
    rocket = calisto_robust
    curve = AirBrakes(WHOLE_ROCKET_CURVE, rocket.area).drag_coefficient
    rocket.power_off_drag = lambda mach: curve(0.0, mach)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rocket.add_air_brakes(
            WHOLE_ROCKET_CURVE, lambda context: None, 10, override_rocket_drag=True
        )
    controls, flight = _stand_in(rocket, example_plain_env)
    without_brakes = _forces_and_moments(flight, controls)
    rho = example_plain_env.density(1500.0)
    mach = 200.0 / example_plain_env.speed_of_sound(1500.0)
    pressure_area = 0.5 * rho * 200.0**2 * rocket.area

    surfaces = without_brakes[2] + pressure_area * curve(0.0, mach)
    for level in (0.0, 0.3, 1.0):
        controls["air_brakes.deployment_level"] = level
        axial = _forces_and_moments(flight, controls)[2]
        assert axial - surfaces == pytest.approx(
            -pressure_area * curve(level, mach), rel=1e-12
        )


def test_the_exported_coefficients_follow_a_whole_rocket_curve(calisto_robust):
    """The export adds the rocket's drag and the air brakes' gain, so its axial
    coefficient rises with the deployment level by what the curve does."""
    rocket = calisto_robust
    rocket.add_surfaces(
        AirBrakes(WHOLE_ROCKET_CURVE, rocket.area, override_rocket_drag=True), 0.0
    )
    axial = rocket.to_coefficients(
        machs=[0.0, 0.5, 1.0], controls={"deployment_level": [0.0, 0.5, 1.0]}
    )["power_off"]["cA_0"]
    assert axial(0.5, 1.0) - axial(0.5, 0.0) == pytest.approx(0.4)
    assert axial(0.5, 0.5) - axial(0.5, 0.0) == pytest.approx(0.2)


def test_a_whole_rocket_curve_that_disagrees_with_the_rocket_drag_warns(
    calisto_robust,
):
    with pytest.warns(UserWarning, match="deployment level 0"):
        calisto_robust.add_air_brakes(
            CURVE, lambda context: None, 10, override_rocket_drag=True
        )


def test_a_whole_rocket_curve_is_checked_whichever_way_it_is_added(calisto_robust):
    """Added as a surface, the air brakes get the same warning as through
    ``add_air_brakes``, once each way, and none for a curve of the air brakes
    alone."""
    rocket = calisto_robust

    def retracted_drag_warnings(record):
        return [w for w in record if "deployment level 0" in str(w.message)]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rocket.add_surfaces(
            AirBrakes(CURVE, rocket.area, override_rocket_drag=True, name="First"),
            0.0,
        )
    assert len(retracted_drag_warnings(record)) == 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rocket.add_air_brakes(
            CURVE, lambda context: None, 10, override_rocket_drag=True, name="Second"
        )
    assert len(retracted_drag_warnings(record)) == 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rocket.add_surfaces(AirBrakes(CURVE, rocket.area, name="Third"), 0.0)
    assert not retracted_drag_warnings(record)


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

    (brakes,) = calisto_robust.air_brakes
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
    """Without a position the brakes act at the center of dry mass, wherever a
    motor added before or after puts it, and no warning is raised."""
    rocket = calisto_motorless
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rocket.add_air_brakes(CURVE, lambda context: None, 10)
    rocket.add_motor(cesaroni_m1670, position=-1.373)
    rocket._refresh_aerodynamics()
    (brakes,) = rocket.air_brakes
    assert rocket._is_at_center_of_dry_mass(brakes)
    assert list(rocket.surfaces_cp_to_cdm[brakes]) == [0, 0, 0]
    # Saved and loaded, they still follow it
    loaded = json.loads(json.dumps(rocket, cls=RocketPyEncoder), cls=RocketPyDecoder)
    (loaded_brakes,) = loaded.air_brakes
    assert loaded._is_at_center_of_dry_mass(loaded_brakes)
    assert list(loaded.surfaces_cp_to_cdm[loaded_brakes]) == [0, 0, 0]


def test_air_brakes_off_the_axis_make_a_moment(calisto_robust, example_plain_env):
    calisto_robust.add_air_brakes(
        CURVE, lambda context: None, 10, position=(0.05, 0.0, 0.0)
    )
    controls, flight = _stand_in(calisto_robust, example_plain_env)
    retracted = _forces_and_moments(flight, controls)
    controls["air_brakes.deployment_level"] = 0.7
    deployed = _forces_and_moments(flight, controls)

    (brakes,) = calisto_robust.air_brakes
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
    assert len(loaded.air_brakes) == 1
    assert loaded.control_names == ("air_brakes.deployment_level",)


def test_air_brakes_saved_before_they_were_surfaces_load(calisto_robust):
    """Older files keep air brakes in their own list, apart from the surfaces;
    they load as surfaces at the center of dry mass."""
    saved = json.loads(json.dumps(calisto_robust, cls=RocketPyEncoder))
    saved["air_brakes"] = [json.loads(json.dumps(_brakes(), cls=RocketPyEncoder))]

    loaded = json.loads(json.dumps(saved), cls=RocketPyDecoder)
    assert isinstance(loaded, Rocket)
    (loaded_brakes,) = loaded.air_brakes
    assert loaded._is_at_center_of_dry_mass(loaded_brakes)
    assert list(loaded.surfaces_cp_to_cdm[loaded_brakes]) == [0, 0, 0]


def test_rocket_air_brakes_lists_the_air_brakes_among_the_surfaces(calisto_robust):
    brakes = calisto_robust.add_air_brakes(CURVE, lambda context: None, 10)
    assert calisto_robust.air_brakes == [brakes]
    assert any(surface is brakes for surface, _ in calisto_robust.aerodynamic_surfaces)


def test_return_controller_is_deprecated_and_still_returns_the_controller(
    calisto_robust,
):
    with pytest.warns(DeprecationWarning, match="`return_controller` is deprecated"):
        brakes, controller = calisto_robust.add_air_brakes(
            CURVE,
            lambda context: None,
            10,
            return_controller=True,
            name="Second",
            position=0.0,
        )
    assert isinstance(brakes, AirBrakes)
    assert controller is calisto_robust.controllers[-1]

    # Without the argument nothing is deprecated, and only the brakes return
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        third = calisto_robust.add_air_brakes(
            CURVE, lambda context: None, 10, name="Third", position=0.0
        )
    assert isinstance(third, AirBrakes)


def test_info_prints_the_settings_and_the_drag_table(capsys, tmp_path):
    brakes = AirBrakes(CURVE, 0.01, name="Main Brakes", override_rocket_drag=True)
    brakes.info()
    out = capsys.readouterr().out
    assert "Control input: main_brakes.deployment_level" in out
    assert "Reference area: 0.01 m^2" in out
    assert "the drag of the whole rocket at each deployment level" in out
    assert "Drag coefficient by deployment level" in out
    assert f"{brakes.drag_coefficient(0.5, 0.3):8.4f}" in out

    brakes.plots.drag_coefficient(filename=str(tmp_path / "cd.png"))
    assert (tmp_path / "cd.png").exists()
    brakes.plots.drag_coefficient(
        levels=(0.5,), mach=(0.1, 0.5), filename=str(tmp_path / "cd_one.png")
    )
    assert (tmp_path / "cd_one.png").exists()


def test_a_curve_given_as_a_function_is_read_as_level_then_mach(tmp_path):
    brakes = AirBrakes(lambda level, mach: 0.5 * level + mach, 0.01)
    assert brakes.drag_coefficient(0.4, 0.3) == pytest.approx(0.5)
    assert brakes.cA(0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4) == pytest.approx(0.5)
    assert brakes.cA(0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0
    # No data points: the plot falls back to default levels and Mach range
    brakes.plots.drag_coefficient(filename=str(tmp_path / "cd.png"))
    assert (tmp_path / "cd.png").exists()


def test_the_initial_level_is_limited_or_warned_about():
    assert (
        AirBrakes(CURVE, 0.01, initial_deployment_level=1.5).initial_deployment_level
        == 1
    )
    # Without clamp there is no limit at all
    loose = AirBrakes(CURVE, 0.01, clamp=False, initial_deployment_level=1.5)
    assert loose.initial_deployment_level == 1.5
    assert loose.control_limits["deployment_level"] is None
    with pytest.warns(DeprecationWarning, match="initial_deployment_level"):
        old = AirBrakes(CURVE, 0.01, deployment_level=0.3)
    assert old.initial_deployment_level == 0.3


def test_air_brakes_alone_are_saved_and_loaded():
    brakes = AirBrakes(
        CURVE,
        0.01,
        clamp=False,
        override_rocket_drag=True,
        initial_deployment_level=0.2,
        name="Main Brakes",
        active_during="power_off",
        active=False,
    )
    loaded = json.loads(json.dumps(brakes, cls=RocketPyEncoder), cls=RocketPyDecoder)
    for attribute in (
        "reference_area",
        "clamp",
        "override_rocket_drag",
        "initial_deployment_level",
        "name",
        "active_during",
        "active",
    ):
        assert getattr(loaded, attribute) == getattr(brakes, attribute)
    assert loaded.drag_coefficient(0.5, 0.3) == brakes.drag_coefficient(0.5, 0.3)

    # A file from before v1.15 names the level "deployment_level" and lacks
    # the newer fields
    data = json.loads(json.dumps(brakes, cls=RocketPyEncoder))
    data["deployment_level"] = data.pop("initial_deployment_level")
    for key in ("active_during", "active", "clamp", "name"):
        del data[key]
    old = json.loads(json.dumps(data), cls=RocketPyDecoder)
    assert old.initial_deployment_level == 0.2
    assert old.clamp and old.active and old.active_during == "always"
    assert old.name == "AirBrakes"


def _still_air_flight(rocket, speed, level, rows=5):
    """A flight stub flying ``rocket`` at ``speed`` m/s through still
    sea-level air, the air brakes held at ``level``."""
    times = np.linspace(0, 1, rows)
    constant = lambda value: Function(lambda t: value)  # noqa: E731
    return SimpleNamespace(
        rocket=rocket,
        time=times,
        free_stream_speed=constant(speed),
        mach_number=constant(speed / 340.0),
        density=constant(1.225),
        aerodynamic_drag=Function(np.column_stack((times, np.full(rows, 100.0)))),
        controls={"AirBrakes": {"deployment_level": constant(level)}},
        _surface_switches={},
    )


def test_the_drag_over_a_flight_is_the_curve_at_the_air_met(calisto_robust, tmp_path):
    brakes = _brakes()
    calisto_robust.add_surfaces(brakes, 0.0)
    flight = _still_air_flight(calisto_robust, 102.0, 0.7)

    drag = brakes._drag_in(flight)
    expected = 0.5 * 1.225 * 102.0**2 * 0.01 * brakes.drag_coefficient(0.7, 0.3)
    assert drag.x_array == pytest.approx(flight.time)
    assert drag(0.5) == pytest.approx(expected)
    assert brakes._drag_in(_still_air_flight(calisto_robust, 102.0, 0.0))(0.5) == 0

    brakes.plots.drag(flight, filename=str(tmp_path / "drag.png"))
    assert (tmp_path / "drag.png").exists()
    with pytest.raises(ValueError, match="not on the rocket"):
        _brakes(name="Other")._drag_in(flight)


def test_bad_inputs_are_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="reference area must be positive"):
        AirBrakes(CURVE, 0)
    with pytest.raises(ValueError, match="deployment level and the Mach number"):
        AirBrakes(lambda mach: mach, 0.01)
    with pytest.raises(ValueError, match="three columns"):
        AirBrakes(np.zeros((3, 2)), 0.01)


def test_a_number_is_a_constant_drag_coefficient():
    brakes = AirBrakes(0.4, 0.01)
    assert brakes.cA(0, 0, 0.5, 0, 0, 0, 0, 1.0) == pytest.approx(0.4)
    assert brakes.cA(0, 0, 0.5, 0, 0, 0, 0, 0.0) == 0.0


def test_replacing_the_curve_changes_the_drag_flown():
    """The stochastic air brakes scale ``drag_coefficient`` after building."""
    brakes = _brakes()
    before = brakes.cA(0, 0, 0.5, 0, 0, 0, 0, 1.0)
    brakes.drag_coefficient *= 2
    assert brakes.cA(0, 0, 0.5, 0, 0, 0, 0, 1.0) == pytest.approx(2 * before)


@pytest.mark.parametrize("override", [False, True])
def test_a_deep_copy_of_the_air_brakes_reads_its_own_drag_curve(override):
    """A copied rocket is often changed and flown again: the copy's drag is
    that of the copy's curve, and a change to either leaves the other alone."""
    state = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0)
    brakes = _brakes(override)
    drag = brakes.cA(*state)
    assert drag > 0

    copied = deepcopy(brakes)
    assert copied.cA(*state) == pytest.approx(drag)
    copied.drag_coefficient = copied.drag_coefficient * 4
    assert copied.cA(*state) == pytest.approx(4 * drag)
    assert brakes.cA(*state) == pytest.approx(drag)

    brakes.drag_coefficient = brakes.drag_coefficient * 10
    assert copied.cA(*state) == pytest.approx(4 * drag)


def test_a_deep_copy_of_a_rocket_flies_its_own_air_brakes(calisto_robust):
    calisto_robust.add_surfaces(_brakes(), None)
    copied = deepcopy(calisto_robust)
    (brakes,) = calisto_robust.air_brakes
    (copied_brakes,) = copied.air_brakes
    assert copied_brakes is not brakes

    state = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0)
    drag = brakes.cA(*state)
    copied_brakes.drag_coefficient = copied_brakes.drag_coefficient * 2
    assert copied_brakes.cA(*state) == pytest.approx(2 * drag)
    assert brakes.cA(*state) == pytest.approx(drag)
