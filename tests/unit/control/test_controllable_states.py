"""Controllables with states of their own: the flight integrates them after the
phase's states, and the controllable's physics receives the values the controllable acts
with, not the ones its controller wrote. They use stand-ins for the rocket
and the flight, so no simulation is run."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Flight
from rocketpy.control import ContinuousController, Controllable
from rocketpy.control.controllable import ForceEffector, ThrustEffector
from rocketpy.simulation.helpers.dynamics import (
    CANONICAL_STATE_NAMES,
    SIX_DOF_DYNAMICS,
    _PhaseDynamics,
    _with_controllables,
)
from rocketpy.simulation.solution import _PhaseSolution


class Servo(Controllable):
    """A control that reaches the commanded angle with a first-order lag."""

    states = ("angle",)

    def __init__(self, name="Servo", start=0.1):
        super().__init__(name, ("angle",), control_limits={"angle": (-1, 1)})
        self.start = start

    def initial_states(self):
        return [self.start]

    def state_derivative(self, states, commands, conditions):
        return [(commands[0] - states[0]) / 0.05]

    def actual_controls(self, states, commands, conditions):
        return [states[0]]


def _rocket(*controllables):
    """A rocket with these controllables, sorted as Rocket sorts them."""
    laws = tuple(c for c in controllables if isinstance(c, ContinuousController))
    parts = tuple(c for c in controllables if c not in laws)
    return SimpleNamespace(
        _parts_with_states=tuple(p for p in parts if p._acts_on_its_own()),
        _continuous_controllers=laws,
        effectors=SimpleNamespace(
            get_components=lambda: [
                p for p in parts if isinstance(p, (ThrustEffector, ForceEffector))
            ]
        ),
        controllables=controllables,
    )


# What FlightConditions reads of a flight
_FLIGHT = SimpleNamespace(rocket=None, env=None)


def _reads_controls(flight, t, u, controls, post_processing=False):  # pylint: disable=unused-argument
    """A phase whose derivative and post-process value are what it read."""
    if post_processing:
        return [controls["servo.angle"]]
    return [controls["servo.angle"], *[0.0] * 12]


_PHASE = _PhaseDynamics(
    "toy", _reads_controls, CANONICAL_STATE_NAMES, post_process_vars=("read",)
)


def test_a_rocket_without_controllable_states_flies_the_phase_as_it_is():
    plain = Controllable("Plain", ("on",))
    assert _with_controllables(_PHASE, _rocket(plain)) is _PHASE
    assert _with_controllables(_PHASE, _rocket()) is _PHASE


def test_the_controllable_states_come_after_the_phase_states():
    servo, other = Servo(), Servo("Other Servo")
    extended = _with_controllables(SIX_DOF_DYNAMICS, _rocket(servo, other))
    assert extended.states == (
        *CANONICAL_STATE_NAMES,
        "servo_angle",
        "other_servo_angle",
    )
    assert extended.name == SIX_DOF_DYNAMICS.name
    # Extending twice changes nothing
    assert _with_controllables(extended, _rocket(servo, other)) is extended


class Counter(Controllable):
    """A state of its own and no controls, so only the state's name can clash."""

    states = ("count",)

    def __init__(self, name="Counter"):
        super().__init__(name, ())

    def initial_states(self):
        return [0.0]

    def state_derivative(self, states, commands, conditions):
        return [1.0]


def test_two_controllables_cannot_share_a_state_name(calisto):
    with pytest.raises(ValueError, match="state named 'counter_count'"):
        calisto._check_new_controllables(Counter(), Counter())


def test_the_phase_reads_the_values_the_controllable_acts_with():
    """The phase's derivative receives the servo's angle, a state, not the
    command, and the state's derivative follows the command."""
    extended = _with_controllables(_PHASE, _rocket(Servo()))
    controls = {"servo.angle": 0.6}
    u = [0.0] * 13 + [0.2]
    derivative = extended.derivative(_FLIGHT, 0.0, u, controls)
    assert len(derivative) == 14
    assert derivative[0] == 0.2
    assert derivative[-1] == pytest.approx((0.6 - 0.2) / 0.05)
    assert extended.derivative(_FLIGHT, 0.0, u, controls, post_processing=True) == [0.2]
    # The controls the controller wrote are left as they were
    assert controls == {"servo.angle": 0.6}


def test_controllable_states_start_from_the_controllable_then_carry_over():
    extended = _with_controllables(_PHASE, _rocket(Servo(start=0.1)))
    canonical = list(range(13))
    assert extended.initial_state(None, 0.0, canonical) == [*canonical, 0.1]
    assert extended.initial_state(None, 0.0, canonical, {"servo_angle": 0.4}) == [
        *canonical,
        0.4,
    ]


def test_a_controllable_whose_initial_states_do_not_match_is_refused(calisto):
    servo = Servo()
    servo.initial_states = lambda: [0.0, 0.0]
    with pytest.raises(ValueError, match="1 states, but its initial_states"):
        calisto._check_new_controllables(servo)


def test_a_state_that_is_not_a_plain_name_is_refused(calisto):
    """A state is read by name in the flight, and a name with a dot would
    look like a control input."""
    servo = Servo()
    servo.states = ("angle.fast",)
    with pytest.raises(ValueError, match="letters, digits and underscores"):
        calisto._check_new_controllables(servo)


def test_a_state_of_its_own_carries_over_between_phases():
    phase = _PhaseDynamics("own", None, (*CANONICAL_STATE_NAMES, "spin"))
    assert phase.initial_state(None, 0.0, [0.0] * 13, {"spin": 3.0})[-1] == 3.0


def test_the_canonical_state_is_the_first_thirteen_values():
    extended = _with_controllables(SIX_DOF_DYNAMICS, _rocket(Servo()))
    state = list(range(14))
    assert not extended.is_canonical
    assert extended.canonicalize(state, None) == list(range(13))
    assert extended.canonicalize_derivative(state) == list(range(13))


def test_a_phase_extended_for_is_saved_and_read_back():
    extended = _with_controllables(SIX_DOF_DYNAMICS, _rocket(Servo()))
    phase = _PhaseSolution(extended, [0.0] * 13, t_start=0.0, name="free")
    loaded = _PhaseSolution.from_dict(phase.to_dict())
    assert loaded.dynamics.name == SIX_DOF_DYNAMICS.name
    assert loaded.dynamics.states == extended.states
    assert loaded.dynamics.post_process_vars == SIX_DOF_DYNAMICS.post_process_vars
    assert loaded.state_dict(list(range(14)))["servo_angle"] == 13


def test_the_phase_also_reads_the_controllable_states_by_name():
    seen = {}

    def reads(flight, t, u, controls):  # pylint: disable=unused-argument
        seen.update(controls)
        return [0.0] * 13

    phase = _PhaseDynamics("reads", reads, CANONICAL_STATE_NAMES)
    extended = _with_controllables(phase, _rocket(Servo()))
    extended.derivative(_FLIGHT, 0.0, [0.0] * 13 + [0.25], {"servo.angle": 0.6})
    assert seen == {"servo.angle": 0.25, "servo_angle": 0.25}


def test_the_states_are_read_after_the_flight_by_controllable_and_name():
    flight = SimpleNamespace(
        rocket=_rocket(Servo("Fin Servo"), Servo("Idle", start=0)),
        solution={
            "fin_servo_angle": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]),
            "idle_angle": np.array([[0.0, 0.0], [2.0, 0.0]]),
        },
    )
    history = Flight.state_history.func(flight)
    assert list(history) == ["Fin Servo", "Idle"]
    assert list(history.fin_servo) == ["angle"]
    assert history["Fin Servo"].angle(0.5) == pytest.approx(0.5)  # interpolated
    assert history.fin_servo.angle(1.5) == pytest.approx(1.0)
    with pytest.raises(AttributeError, match="Nothing here is named 'position'"):
        _ = history.fin_servo.position
