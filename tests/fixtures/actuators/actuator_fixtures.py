"""Test actuators for the hooks in the equations of motion: a gimbal that
turns the thrust, a moving mass, and effectors that push or turn the rocket.
They are the smallest classes that satisfy each hook, not models of real
hardware."""

import math

import pytest

from rocketpy import GenericEffector


class Gimbal:
    """Turns the thrust by ``pitch`` radians about the body y axis."""

    def __init__(self, name="Gimbal"):
        self.name = name
        self.controls = ("pitch",)
        self.control_defaults = {"pitch": 0.0}
        self.control_limits = {"pitch": (-0.2, 0.2)}
        self.clamp = True

    def thrust_vector(self, t, thrust, pressure, controls):  # pylint: disable=unused-argument
        angle = controls[0]
        return [
            thrust[0] * math.cos(angle) + thrust[2] * math.sin(angle),
            thrust[1],
            -thrust[0] * math.sin(angle) + thrust[2] * math.cos(angle),
        ]


class PassThrough:
    """A thrust actuator that leaves the thrust as it is."""

    name = "Pass Through"
    controls = ()
    control_defaults = {}
    control_limits = {}
    clamp = True

    def thrust_vector(self, t, thrust, pressure, controls):  # pylint: disable=unused-argument
        return thrust


class MovingMass:
    """A point mass that slides along the rocket's axis: ``position`` is its
    distance ahead of the center of dry mass, in meters."""

    def __init__(self, mass=1.0, name="Moving Mass"):
        self.name = name
        self.mass = mass
        self.controls = ("position",)
        self.control_defaults = {"position": 0.0}
        self.control_limits = {"position": (-0.5, 0.5)}
        self.clamp = True

    def mass_properties(self, t, controls):  # pylint: disable=unused-argument
        return self.mass, [0.0, 0.0, controls[0]], None


@pytest.fixture
def gimbal():
    return Gimbal()


@pytest.fixture
def moving_mass():
    return MovingMass()


def _push(t, state, on):  # pylint: disable=unused-argument
    """10 N along the rocket's axis and 0.5 N m about it, when on."""
    return [0.0, 0.0, 10.0 * on], [0.0, 0.0, 0.5 * on]


def _side_push(t, state, on):  # pylint: disable=unused-argument
    """10 N sideways, through the center of dry mass, when on."""
    return [10.0 * on, 0.0, 0.0], [0.0, 0.0, 0.0]


@pytest.fixture
def push_effector():
    return GenericEffector(
        _push, controls=("on",), control_limits={"on": (0, 1)}, name="Push"
    )


@pytest.fixture
def side_effector():
    return GenericEffector(
        _side_push, controls=("on",), control_limits={"on": (0, 1)}, name="Side"
    )
