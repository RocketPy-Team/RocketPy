"""Test effectors for the hooks in the equations of motion: a gimbal that
turns the thrust, and effectors that push or turn the rocket.
They are the smallest classes that satisfy each hook, not models of real
hardware."""

import math

import pytest

from rocketpy import ForceEffector, ThrustEffector


class Gimbal(ThrustEffector):
    """Turns the thrust by ``pitch`` radians about the body y axis, at the
    nozzle exit: the exit itself stays where it is."""

    def __init__(self, name="Gimbal"):
        super().__init__(
            controls=("pitch",), control_limits={"pitch": (-0.2, 0.2)}, name=name
        )

    def thrust_and_nozzle(self, thrust, nozzle, controls, conditions):  # pylint: disable=unused-argument
        angle = controls[0]
        turned = [
            thrust[0] * math.cos(angle) + thrust[2] * math.sin(angle),
            thrust[1],
            -thrust[0] * math.sin(angle) + thrust[2] * math.cos(angle),
        ]
        return turned, nozzle


class PassThrough(ThrustEffector):
    """A thrust effector that leaves the thrust as it is."""

    def __init__(self):
        super().__init__(name="Pass Through")

    def thrust_and_nozzle(self, thrust, nozzle, controls, conditions):  # pylint: disable=unused-argument
        return thrust, nozzle


@pytest.fixture
def gimbal():
    return Gimbal()


def _push(on, conditions):  # pylint: disable=unused-argument
    """10 N along the rocket's axis and 0.5 N m about it, when on."""
    return [0.0, 0.0, 10.0 * on], [0.0, 0.0, 0.5 * on]


def _side_push(on, conditions):  # pylint: disable=unused-argument
    """10 N sideways, through the center of dry mass, when on."""
    return [10.0 * on, 0.0, 0.0], [0.0, 0.0, 0.0]


@pytest.fixture
def push_effector():
    return ForceEffector(
        _push, controls=("on",), control_limits={"on": (0, 1)}, name="Push"
    )


@pytest.fixture
def side_effector():
    return ForceEffector(
        _side_push, controls=("on",), control_limits={"on": (0, 1)}, name="Side"
    )
