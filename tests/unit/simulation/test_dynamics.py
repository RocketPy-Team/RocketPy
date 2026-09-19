"""Tests for the _PhaseDynamics class, its bound form and the post-process build.

These tests use stub flights and stub derivatives, so they never run a Flight.
"""

import numpy as np
import pytest

from rocketpy.simulation.flight import Flight
from rocketpy.simulation.helpers.dynamics import (
    CANONICAL_STATE_NAMES,
    FULL_POST_PROCESS_VARS,
    PARACHUTE_DYNAMICS,
    RAIL_DYNAMICS,
    SIX_DOF_DYNAMICS,
    SOLID_PROPULSION_DYNAMICS,
    THREE_DOF_DYNAMICS,
    _PhaseDynamics,
)
from rocketpy.simulation.helpers.event_commands import apply_rollback_command
from rocketpy.simulation.solution import Solution


class StubFlight:
    """Minimal stand-in for a Flight, recording derivative calls."""

    def __init__(self):
        self.calls = []
        self.atol = 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3]


def stub_derivative(flight, t, u, post_processing=False):
    flight.calls.append((t, list(u), post_processing))
    if post_processing:
        return [t]
    return [value * 2 for value in u]


# Every phase RocketPy ships integrates the full canonical state, so the
# reduced-state machinery is exercised against a purpose-built phase instead of
# whichever preset happens to be short at the moment.
TRANSLATION_STATES = ("x", "y", "z", "vx", "vy", "vz")
TRANSLATION_DYNAMICS = _PhaseDynamics(
    "translation", stub_derivative, TRANSLATION_STATES, ("ax", "ay", "az")
)


def stub_flight_shell(live=False):
    """Return a Flight shell with only what post-processing reads."""
    flight = Flight.__new__(Flight)
    flight.calls = []
    flight.solution = Solution()
    flight._has_change_dynamics_events = live
    flight.solution.records_post_values = live
    return flight


# ---------------------------------------------------------------------------
# The built-in dynamics
# ---------------------------------------------------------------------------


def test_dynamics_names_and_states():
    assert RAIL_DYNAMICS.name == "rail"
    assert SOLID_PROPULSION_DYNAMICS.name == "solid_propulsion"
    assert SIX_DOF_DYNAMICS.name == "six_dof"
    assert THREE_DOF_DYNAMICS.name == "three_dof"
    assert PARACHUTE_DYNAMICS.name == "parachute"
    assert SIX_DOF_DYNAMICS.states == CANONICAL_STATE_NAMES
    assert SIX_DOF_DYNAMICS.is_canonical
    assert RAIL_DYNAMICS.is_canonical
    assert THREE_DOF_DYNAMICS.is_canonical
    assert SOLID_PROPULSION_DYNAMICS.is_canonical
    # The parachute descent moves only position and velocity, but it still
    # carries the whole canonical state, reporting zero for the rest. Keeping
    # every shipped phase canonical is what spares the event loop a state
    # reconstruction on each check.
    assert PARACHUTE_DYNAMICS.states == CANONICAL_STATE_NAMES
    assert PARACHUTE_DYNAMICS.is_canonical
    assert PARACHUTE_DYNAMICS.width == 13
    # It still reports only the variables a descent has.
    assert PARACHUTE_DYNAMICS.post_process_vars == ("ax", "ay", "az", "R1", "R2", "R3")


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_returns_a_canonical_state_untouched():
    state = list(range(13))
    assert SIX_DOF_DYNAMICS.canonicalize(state, None) is state


def test_canonicalize_freezes_the_states_a_phase_does_not_integrate():
    frozen = [float(i) for i in range(13)]
    state = [10, 11, 12, 13, 14, 15]
    result = TRANSLATION_DYNAMICS.canonicalize(state, frozen)
    assert result[:6] == state
    # quaternion and angular rates keep their start-of-phase values
    assert result[6:] == frozen[6:]


def heading_to_canonical(values):
    """Rebuild e0 from the heading; every other canonical state is copied."""
    values["e0"] = values["heading"] * 2
    return [values[name] for name in CANONICAL_STATE_NAMES]


def heading_to_canonical_dot(values, values_dot):
    result = [0.0] * 13
    for name, value in values_dot.items():
        if name in CANONICAL_STATE_NAMES:
            result[CANONICAL_STATE_NAMES.index(name)] = value
    result[6] = 2 * values_dot["heading"]
    return result


def heading_dynamics(with_dot=True):
    """A parafoil-style phase: integrates a heading and rebuilds e0 from it."""
    return _PhaseDynamics(
        "heading",
        stub_derivative,
        (*TRANSLATION_STATES, "heading"),
        to_canonical=heading_to_canonical,
        to_canonical_dot=heading_to_canonical_dot if with_dot else None,
    )


def test_to_canonical_rebuilds_states_and_keeps_the_rest():
    heading = heading_dynamics()
    frozen = [float(i) for i in range(13)]
    result = heading.canonicalize([1, 2, 3, 4, 5, 6, 0.25], frozen)
    assert result[:6] == [1, 2, 3, 4, 5, 6]
    assert result[6] == 0.5  # rebuilt from the heading
    assert result[7:] == frozen[7:]  # copied from the start of the phase


def test_canonicalize_derivative_zeroes_held_states():
    state_dot = [1, 2, 3, 4, 5, 6]
    result = TRANSLATION_DYNAMICS.canonicalize_derivative(state_dot)
    assert result == [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0]
    canonical_dot = list(range(13))
    assert SIX_DOF_DYNAMICS.canonicalize_derivative(canonical_dot) is canonical_dot


def test_canonicalize_derivative_uses_to_canonical_dot():
    """A rebuilt state changes, so its derivative must not read as zero."""
    heading = heading_dynamics()
    state = [1, 2, 3, 4, 5, 6, 0.25]
    state_dot = [0, 0, 0, 0, 0, 0, 3.0]  # heading turning at 3 rad/s
    result = heading.canonicalize_derivative(state_dot, state, [0.0] * 13)
    assert result[6] == 6.0  # de0/dt = 2 * dheading/dt, not 0
    assert result[7:] == [0.0] * 6  # e1..e3, w1..w3 are held, so genuinely zero


def test_canonicalize_derivative_without_the_state_raises():
    heading = heading_dynamics()
    with pytest.raises(ValueError, match="needs the phase's state"):
        heading.canonicalize_derivative([0, 0, 0, 0, 0, 0, 3.0])


def test_state_values_merges_start_of_phase_and_own_states():
    frozen = [float(i) for i in range(13)]
    values = TRANSLATION_DYNAMICS.state_values([9, 9, 9, 9, 9, 9], frozen)
    assert values["x"] == 9  # own state wins
    assert values["e0"] == 6.0  # the start of the phase supplies the rest


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def test_bound_dynamics_calls_free_function():
    dynamics = _PhaseDynamics("stub", stub_derivative, CANONICAL_STATE_NAMES, ("ax",))
    flight = StubFlight()
    bound = dynamics.bind(flight)
    # Calling gives the solver just the state derivative.
    assert bound(1.5, [1.0, 2.0, 3.0]) == [2.0, 4.0, 6.0]
    assert flight.calls == [(1.5, [1.0, 2.0, 3.0], False)]
    assert bound.dynamics is dynamics
    assert bound.__name__ == "stub_derivative"


def test_bound_dynamics_post_process_at_returns_reported_variables():
    dynamics = _PhaseDynamics("stub", stub_derivative, CANONICAL_STATE_NAMES, ("ax",))
    flight = StubFlight()
    bound = dynamics.bind(flight)
    assert bound.post_process_at(1.5, [1.0, 2.0, 3.0]) == [1.5]
    assert flight.calls == [(1.5, [1.0, 2.0, 3.0], True)]


def test_bound_dynamics_forwards_phase_arguments():
    """A phase argument fixed at bind time reaches both call paths."""

    def with_extra(flight, t, u, post_processing=False, *, parachute):
        flight.calls.append((t, post_processing, parachute))
        return [parachute]

    flight = StubFlight()
    bound = _PhaseDynamics("chute", with_extra, TRANSLATION_STATES, ("ax",)).bind(
        flight, parachute="main"
    )
    assert bound(0.0, [0] * 6) == ["main"]
    assert bound.post_process_at(0.0, [0] * 6) == ["main"]
    assert flight.calls == [(0.0, False, "main"), (0.0, True, "main")]


def test_a_phase_reporting_no_variables_never_calls_the_derivative():
    # This derivative has no post_processing parameter, so it raises TypeError if
    # post-processing asks it for variables the phase never declared.
    dynamics = _PhaseDynamics(
        "silent", lambda flight, t, u: list(u), TRANSLATION_STATES
    )
    bound = dynamics.bind(StubFlight())
    assert bound(0.0, [1] * 6) == [1] * 6
    assert bound.post_process_at(2.0, [1] * 6) == []


def test_bound_dynamics_default_initial_state():
    dynamics = _PhaseDynamics("chute", stub_derivative, TRANSLATION_STATES, ("ax",))
    bound = dynamics.bind(StubFlight())
    canonical = list(range(13))
    # default seeding picks this phase's states out of the canonical state
    assert bound.initial_state(0.0, canonical) == [0, 1, 2, 3, 4, 5]


def test_bound_dynamics_custom_initial_state():
    def seed(flight, t, canonical_state):
        return [canonical_state[2]]  # only altitude

    dynamics = _PhaseDynamics(
        "z_only", stub_derivative, ("z",), ("ax",), initial_state=seed
    )
    bound = dynamics.bind(StubFlight())
    assert bound.initial_state(0.0, list(range(13))) == [2]


# ---------------------------------------------------------------------------
# Absolute tolerance
# ---------------------------------------------------------------------------


def test_select_atol_reduces_a_canonical_vector():
    flight = StubFlight()
    assert SIX_DOF_DYNAMICS.select_atol(flight.atol) == flight.atol
    assert TRANSLATION_DYNAMICS.select_atol(flight.atol) == [1e-3] * 6


def test_scalar_atol_passthrough():
    assert SIX_DOF_DYNAMICS.select_atol(1e-5) == 1e-5


def test_atol_matching_the_phase_width_passes_through():
    custom = [1e-4] * 6
    assert TRANSLATION_DYNAMICS.select_atol(custom) == custom


def test_non_canonical_states_use_the_largest_atol():
    heading = _PhaseDynamics(
        "heading", stub_derivative, (*TRANSLATION_STATES, "heading")
    )
    atol = 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-2]
    assert heading.select_atol(atol) == [1e-3] * 6 + [1e-2]


def test_bad_atol_length_raises():
    with pytest.raises(ValueError, match="matches neither"):
        TRANSLATION_DYNAMICS.select_atol([1e-3, 1e-3, 1e-3])


# ---------------------------------------------------------------------------
# Post-process rows
# ---------------------------------------------------------------------------


def test_post_process_values_from_a_sequence():
    dynamics = _PhaseDynamics("d", stub_derivative, ("z",), ("ax", "ay"))
    assert dynamics.post_process_values([7, 8]) == [7, 8]


def test_post_process_values_rejects_the_wrong_number_of_values():
    dynamics = _PhaseDynamics("d", stub_derivative, ("z",), ("ax", "ay"))
    with pytest.raises(ValueError, match="computes 2 post-process variables"):
        dynamics.post_process_values([1])


# ---------------------------------------------------------------------------
# The one-pass post-process build
# ---------------------------------------------------------------------------


def test_build_replays_the_stored_states():
    """With no dynamics-changing event, each phase is replayed after the flight."""
    flight = stub_flight_shell()
    dynamics = _PhaseDynamics("stub", stub_derivative, TRANSLATION_STATES, ("ax",))
    flight.solution._start_phase(
        dynamics.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    flight.solution._append([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight.solution._append([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # The stub reports the time as its single post-process variable.
    assert flight.solution.post["ax"].tolist() == [[0.0, 0.0], [0.5, 0.5]]
    # Every stored state was replayed, in post-processing mode.
    assert [call[0] for call in flight.calls] == [0.0, 0.5]
    assert all(call[2] is True for call in flight.calls)


def test_build_prefers_values_recorded_during_the_simulation():
    """Live rows win, since a replay cannot reproduce a controller's changes."""
    flight = stub_flight_shell(live=True)
    dynamics = _PhaseDynamics("stub", stub_derivative, TRANSLATION_STATES, ("ax",))
    flight.solution._start_phase(
        dynamics.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    flight.solution._append([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight.solution._set_post_values(-1, [42.0])

    assert flight.solution.post["ax"].tolist() == [[0.0, 42.0]]
    assert flight.calls == []  # nothing was replayed


def test_a_row_records_the_values_of_its_own_state():
    """A row added to mark an exact event time gets values of its own.

    The row is written out of order, before the step the solver already took
    past it, so recording it against the most recent row would put the values
    on the wrong row and leave this one with nothing.
    """
    flight = stub_flight_shell(live=True)
    dynamics = _PhaseDynamics("stub", stub_derivative, TRANSLATION_STATES, ("ax",))
    flight.solution._start_phase(
        dynamics.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    flight.solution._append([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight._Flight__post_process_step()
    # a step, with an exact-time row wedged in before its end by an event of
    # that step: the wedged row is recorded as soon as it is inserted, the
    # step's own row once its events are handled
    flight.solution._append([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight.solution._insert_before_last([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight._post_process_row(-2)
    assert flight.solution._post_values[-2] == [0.5]
    assert flight.solution._post_values[-1] is None
    flight._Flight__post_process_step()

    # the stub reports the row's own time as its post-process variable
    assert flight.solution.post["ax"].tolist() == [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ]


def test_a_replaced_row_records_the_values_of_its_new_state():
    """Rolling a row back clears its values; the loop then records them anew."""
    flight = stub_flight_shell(live=True)
    dynamics = _PhaseDynamics("stub", stub_derivative, TRANSLATION_STATES, ("ax",))
    flight.solution._start_phase(
        dynamics.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    flight.solution._append([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight._Flight__post_process_step()
    flight.solution._append([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flight._Flight__post_process_step()
    apply_rollback_command(flight, 0.75, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # the rollback itself records nothing: the solver loop does, once per step
    assert flight.solution._post_values[-1] is None
    flight._Flight__post_process_step()
    recorded_calls = len(flight.calls)

    assert flight.solution.post["ax"].tolist() == [[0.0, 0.0], [0.75, 0.75]]
    assert len(flight.calls) == recorded_calls  # nothing was replayed


def test_a_phase_that_does_not_compute_a_variable_reports_zero():
    flight = stub_flight_shell()
    full = _PhaseDynamics(
        "full",
        lambda flight, t, u, post_processing=False: (
            [1.0, 2.0] if post_processing else list(u)
        ),
        TRANSLATION_STATES,
        ("ax", "M1"),
    )
    partial = _PhaseDynamics(
        "partial",
        lambda flight, t, u, post_processing=False: (
            [3.0] if post_processing else list(u)
        ),
        TRANSLATION_STATES,
        ("ax",),
    )
    flight.solution._start_phase(full.bind(flight), start_canonical=tuple([0.0] * 13))
    flight.solution._append([0.0, *[0.0] * 6])
    flight.solution._start_phase(
        partial.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    flight.solution._append([1.0, *[0.0] * 6])

    assert flight.solution.post["ax"].tolist() == [[0.0, 1.0], [1.0, 3.0]]
    # M1 is not computed by the second phase, so it is zero there.
    assert flight.solution.post["M1"].tolist() == [[0.0, 2.0], [1.0, 0.0]]


def test_unknown_post_process_variable_raises():
    flight = stub_flight_shell()
    dynamics = _PhaseDynamics("stub", stub_derivative, TRANSLATION_STATES, ("ax",))
    flight.solution._start_phase(
        dynamics.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    flight.solution._append([0.0, *[0.0] * 6])
    with pytest.raises(KeyError, match="No flight phase computed"):
        flight.solution.post["nope"]


def test_a_phase_with_no_live_dynamics_is_skipped():
    """A phase read back from a file cannot be replayed, so it contributes nothing."""
    flight = stub_flight_shell()
    flight.solution._start_phase(SIX_DOF_DYNAMICS, start_canonical=tuple([0.0] * 13))
    flight.solution._append([0.0, *[0.0] * 13])
    # The variable is one the phase declares, so the error says why it cannot be
    # produced rather than claiming the flight never computes it.
    with pytest.raises(KeyError, match="read back from a saved file"):
        flight.solution.post["ax"]


@pytest.mark.parametrize("name", FULL_POST_PROCESS_VARS)
def test_every_built_in_variable_has_a_flight_attribute(name):
    """The declared variables and the Flight attributes must stay in step.

    Each built-in variable is named twice: once in ``FULL_POST_PROCESS_VARS``,
    and once by the ``funcify_method`` attribute that exposes it. Nothing keeps
    the two in step, so this checks them against each other on a hand-built
    Flight shell, without simulating.
    """
    flight = stub_flight_shell()

    def report_everything(_flight, t, u, post_processing=False):
        return [t] * len(FULL_POST_PROCESS_VARS) if post_processing else list(u)

    dynamics = _PhaseDynamics(
        "all", report_everything, CANONICAL_STATE_NAMES, FULL_POST_PROCESS_VARS
    )
    flight.solution._start_phase(
        dynamics.bind(flight), start_canonical=tuple([0.0] * 13)
    )
    for t in (0.0, 0.5, 1.0, 1.5):
        flight.solution._append([t, *([0.0] * 13)])

    attribute = getattr(flight, name)
    assert np.allclose(attribute.y_array, [0.0, 0.5, 1.0, 1.5])
