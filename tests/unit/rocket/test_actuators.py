"""Actuators that are not aerodynamic surfaces: adding them to a rocket, the
generic effector, and the hooks that bring them into the equations of
motion."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import GenericEffector
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control.control_signal import initial_controls
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.simulation.helpers.flight_derivatives import (
    _added_mass_at,
    _effectors_at,
    _thrust_at,
    _with_added_mass,
)
from tests.fixtures.actuators.actuator_fixtures import Gimbal, MovingMass


def _flight_of(rocket):
    """A stand-in flight with the control inputs of ``rocket``."""
    controls, channels, limits = initial_controls(rocket.actuators)
    return SimpleNamespace(
        rocket=rocket,
        control_signal=controls,
        _control_channels=channels,
        _control_limits=limits,
    )


class _Both(MovingMass):
    """Moves mass and pushes the rocket: filed under both kinds."""

    def force_and_moment(self, t, state, controls):  # pylint: disable=unused-argument
        return [0, 0, 0], [0, 0, 0]


def test_add_actuator_files_the_part_by_what_it_does(
    calisto, gimbal, moving_mass, push_effector
):
    calisto.add_actuator(gimbal)
    calisto.add_actuator(push_effector)
    calisto.add_actuator(moving_mass)

    assert calisto.thrust_actuators == [gimbal]
    assert calisto.effectors == [push_effector]
    assert calisto.mass_actuators == [moving_mass]
    assert calisto.control_channels[-3:] == (
        "gimbal.pitch",
        "push.on",
        "moving_mass.position",
    )


def test_a_part_that_acts_in_two_ways_has_its_channels_once(calisto):
    both = _Both(name="Ballast")
    calisto.add_actuator(both)
    assert calisto.mass_actuators == [both]
    assert calisto.effectors == [both]
    assert calisto.control_channels.count("ballast.position") == 1


def test_add_actuator_refuses_a_part_that_does_nothing(calisto):
    with pytest.raises(TypeError, match="none of the methods"):
        calisto.add_actuator(SimpleNamespace(name="x", controls=()))


def test_add_actuator_refuses_a_name_already_used(calisto, gimbal):
    calisto.add_actuator(gimbal)
    with pytest.raises(ValueError, match="both have controls"):
        calisto.add_actuator(Gimbal())
    assert calisto.thrust_actuators == [gimbal]


def test_effectors_take_no_part_in_stability(calisto, push_effector, side_effector):
    margin = calisto.static_margin(0)
    center = calisto.aerodynamic_center(0.3)
    calisto.add_actuator(push_effector)
    calisto.add_actuator(side_effector)
    assert calisto.static_margin(0) == margin
    assert calisto.aerodynamic_center(0.3) == center
    assert all(
        not isinstance(surface, GenericEffector)
        for surface, _ in calisto.aerodynamic_surfaces
    )


def test_effectors_are_not_in_the_surface_of_the_whole_rocket(
    calisto_robust, push_effector
):
    before = calisto_robust.to_surface()
    calisto_robust.add_actuator(push_effector)
    after = calisto_robust.to_surface()
    arguments = (0.05, 0.0, 0.3, 1e6, 0.0, 0.0, 0.0)
    for old, new in zip(before, after):
        assert new.controls == ()
        assert new.cN(*arguments) == old.cN(*arguments)


def test_a_generic_effector_returns_vectors():
    effector = GenericEffector(
        lambda t, state, a, b: ([a, b, 0], [0, 0, a * b]),
        controls=("a", "b"),
        control_defaults={"b": 2.0},
        name="RCS Pair",
    )
    force, moment = effector.force_and_moment(0.0, [0.0] * 13, [3.0, 2.0])
    assert isinstance(force, Vector) and isinstance(moment, Vector)
    assert list(force) == [3.0, 2.0, 0] and list(moment) == [0, 0, 6.0]
    assert effector.control_defaults == {"a": 0.0, "b": 2.0}


@pytest.mark.parametrize(
    "function", [lambda t, state: None, lambda t: None, "not a function"]
)
def test_a_generic_effector_checks_its_function(function):
    with pytest.raises((TypeError, ValueError)):
        GenericEffector(function, controls=("on",))


def test_a_generic_effector_is_saved_and_loaded(calisto, push_effector):
    calisto.add_actuator(push_effector)
    loaded = json.loads(json.dumps(calisto, cls=RocketPyEncoder), cls=RocketPyDecoder)
    (effector,) = loaded.effectors
    assert isinstance(effector, GenericEffector)
    assert effector.name == "Push"
    assert effector.control_limits == {"on": (0, 1)}
    assert list(effector.force_and_moment(0, [0] * 13, [1.0])[0]) == [0, 0, 10.0]
    assert loaded.control_channels == calisto.control_channels


def test_the_gimbal_turns_the_thrust_and_makes_a_moment(calisto, gimbal):
    calisto.add_actuator(gimbal)
    flight = _flight_of(calisto)
    flight.control_signal["gimbal.pitch"] = 0.1

    force, moment = _thrust_at(flight, 1.0, 1000.0, 101325.0, flight.control_signal)

    assert list(force) == pytest.approx([1000 * math.sin(0.1), 0, 1000 * math.cos(0.1)])
    # the nozzle is behind the center of dry mass, so a thrust tipped toward
    # +x turns the nose toward -x: a negative moment about y
    lever = calisto.nozzle_to_cdm
    assert list(moment) == pytest.approx([0, -lever * 1000 * math.sin(0.1), 0])


def test_the_thrust_moment_includes_the_eccentricity(calisto):
    calisto.add_thrust_eccentricity(x=0.01, y=-0.02)
    calisto.add_actuator(Gimbal())
    flight = _flight_of(calisto)
    _, moment = _thrust_at(flight, 1.0, 1000.0, 101325.0, flight.control_signal)
    # as the equations without actuators add it: M1 += ey T, M2 -= ex T
    assert list(moment) == pytest.approx(
        [
            calisto.thrust_eccentricity_y * 1000,
            -calisto.thrust_eccentricity_x * 1000,
            0,
        ]
    )


def test_effectors_add_up(calisto, push_effector, side_effector):
    calisto.add_actuator(push_effector)
    calisto.add_actuator(side_effector)
    flight = _flight_of(calisto)
    flight.control_signal.update({"push.on": 1.0, "side.on": 0.5})
    force, moment = _effectors_at(flight, 0.0, [0.0] * 13, flight.control_signal)
    assert list(force) == [5.0, 0.0, 10.0]
    assert list(moment) == [0.0, 0.0, 0.5]


def test_a_moving_mass_moves_the_center_of_mass(calisto, moving_mass):
    calisto.add_actuator(moving_mass)
    flight = _flight_of(calisto)
    flight.control_signal["moving_mass.position"] = 0.4

    mass, first_moment, inertia = _added_mass_at(flight, 0.0, flight.control_signal)
    assert mass == 1.0
    assert list(first_moment) == [0.0, 0.0, 0.4]
    # a point mass on the axis adds m d^2 about the two transverse axes
    assert np.allclose(
        np.array(inertia, dtype=float), np.diag([0.16, 0.16, 0.0]), atol=1e-15
    )

    total, r_CM, tensor = _with_added_mass(
        flight,
        0.0,
        flight.control_signal,
        20.0,
        Vector([0, 0, -0.1]),
        Matrix([[6, 0, 0], [0, 6, 0], [0, 0, 0.1]]),
    )
    assert total == 21.0
    # (20 * -0.1 + 1 * 0.4) / 21
    assert list(r_CM) == pytest.approx([0, 0, -1.6 / 21])
    assert tensor[0][0] == pytest.approx(6.16)
