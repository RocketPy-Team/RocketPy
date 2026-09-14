from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Environment, Event, Flight, Rocket, SolidMotor
from rocketpy.simulation.events import Commands, exact_time_solvers
from rocketpy.simulation.helpers.dynamics import SIX_DOF_DYNAMICS, _PhaseDynamics
from rocketpy.simulation.events.event_builders import (
    apogee_callback,
    apogee_event_exact_time_function,
    apogee_trigger,
    build_core_events,
    impact_callback,
    impact_event_exact_time_derivative,
    impact_event_exact_time_function,
    impact_trigger,
    out_of_rail_callback,
    out_of_rail_exact_time_derivative,
    out_of_rail_exact_time_function,
    out_of_rail_trigger,
)
from rocketpy.simulation.events.exact_time_solvers import (
    solve_brentq,
    solve_cubic_hermite,
    solve_linear,
)


def _callback_return_time(context):
    return {"time": context["time"]}


def _docs_root():
    return Path(__file__).resolve().parents[3]


def _docs_style_flight(custom_events):
    root = _docs_root()
    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=0)
    motor = SolidMotor(
        thrust_source=str(root / "data/motors/cesaroni/Cesaroni_M1670.eng"),
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        burn_time=3.9,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    rocket = Rocket(
        radius=127 / 2000,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=str(root / "data/rockets/calisto/powerOffDragCurve.csv"),
        power_on_drag=str(root / "data/rockets/calisto/powerOnDragCurve.csv"),
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.255)

    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=custom_events,
        name="Docs-style event flight",
    )


def _sample_state(time, vz):
    return np.array(
        [time, 0.0, 0.0, 0.0, 0.0, 0.0, vz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def _canonical_solution(*rows):
    """Build a Solution holding the given canonical rows in one phase."""
    from rocketpy.simulation.helpers.dynamics import SIX_DOF_DYNAMICS
    from rocketpy.simulation.solution import Solution

    solution = Solution()
    solution._start_phase(
        SIX_DOF_DYNAMICS, start_canonical=tuple(rows[0][1:]) if rows else None
    )
    for row in rows:
        solution._append(list(row))
    return solution


def _interpolator(time):
    return np.array(
        [1.0 - 2.0 * time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def _value_at(time):
    """Crosses zero at t = 0.5."""
    return 1.0 - 2.0 * time


def _rate_at(_time):
    return -2.0


@pytest.mark.parametrize(
    ("solver", "options"),
    [
        (solve_linear, {}),
        (solve_brentq, {}),
        (solve_cubic_hermite, {"rate_at": _rate_at}),
    ],
)
def test_exact_time_solvers_return_expected_event_time(solver, options):
    """All exact-time solvers should resolve the same simple root."""

    assert solver(_value_at, 0.0, 1.0, **options) == pytest.approx(0.5)


def test_exact_time_solvers_search_between_the_given_times():
    """The step does not have to start at zero."""

    def value_at(time):
        return 3.0 - time

    assert solve_linear(value_at, 2.0, 4.0) == pytest.approx(3.0)
    assert solve_brentq(value_at, 2.0, 4.0) == pytest.approx(3.0)
    assert solve_cubic_hermite(
        value_at, 2.0, 4.0, rate_at=lambda _t: -1.0
    ) == pytest.approx(3.0)


def test_exact_time_linear_raises_when_endpoint_values_match():
    """The linear solver should reject steps with identical endpoint values."""

    with pytest.raises(ValueError, match="same at both ends"):
        solve_linear(lambda _t: 1.0, 0.0, 1.0)


def test_exact_time_brentq_raises_when_no_sign_change_occurs():
    """Brent's method should fail cleanly when the value does not change sign."""

    with pytest.raises(ValueError, match="no crossing"):
        solve_brentq(lambda _t: 1.0, 0.0, 1.0)


def test_exact_time_brentq_wraps_runtime_errors_from_brentq(monkeypatch):
    """Brentq failures inside the solver should be wrapped in ValueError."""

    def failing_brentq(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(exact_time_solvers, "brentq", failing_brentq)

    with pytest.raises(ValueError, match="boom"):
        solve_brentq(_value_at, 0.0, 1.0)


def test_exact_time_cubic_hermite_raises_when_multiple_roots_are_found():
    """A cubic crossing zero more than once inside the step is rejected."""

    # (t - 0.2)(t - 0.5)(t - 0.8), with its exact rates at the ends
    def value_at(t):
        return (t - 0.2) * (t - 0.5) * (t - 0.8)

    def rate_at(t):
        return (t - 0.5) * (t - 0.8) + (t - 0.2) * (t - 0.8) + (t - 0.2) * (t - 0.5)

    with pytest.raises(ValueError, match="found 3"):
        solve_cubic_hermite(value_at, 0.0, 1.0, rate_at=rate_at)


def test_exact_time_cubic_hermite_raises_when_no_roots_are_found():
    """A cubic that stays away from zero inside the step is rejected."""

    with pytest.raises(ValueError, match="found 0"):
        solve_cubic_hermite(lambda _t: 1.0, 0.0, 1.0, rate_at=lambda _t: 0.0)


def test_commands_api_records_expected_payloads():
    """The command container should store all command types and reset cleanly."""

    commands = Commands()
    event = SimpleNamespace(name="other")

    commands.enable()
    commands.disable()
    commands.add_event(event)
    commands.disable_event(event)
    commands.set_dynamics(SIX_DOF_DYNAMICS, parachute="main")
    commands.start_flight_phase("descent", lag=1.5)
    commands.terminate_flight()

    assert commands._disabled is True
    assert commands.new_events == [event]
    assert commands.disable_events == [event]
    assert commands.new_dynamics is SIX_DOF_DYNAMICS
    assert commands.new_dynamics_kwargs == {"parachute": "main"}
    assert commands.changes_trajectory is True
    assert commands.new_flight_phase is True
    assert commands.new_flight_phase_name == "descent"
    assert commands.new_flight_phase_lag == 1.5
    assert commands._terminate is True

    commands.reset()

    assert commands._disabled is None
    assert commands.new_events == []
    assert commands.new_dynamics is None
    assert commands.new_dynamics_kwargs == {}
    assert commands.changes_trajectory is False
    assert commands._terminate is False


def test_core_event_builders_update_flight_state_and_commands():
    """The built-in event builders should mutate the flight state as expected."""

    out_of_rail_event, apogee_event, impact_event = build_core_events()

    flight = SimpleNamespace(
        env=SimpleNamespace(elevation=0.0),
        effective_1rl=1.0,
        out_of_rail_state=np.array([0.0]),
        out_of_rail_time=None,
        out_of_rail_time_index=None,
        apogee_state=np.array([0.0]),
        apogee_time=None,
        apogee_x=None,
        apogee_y=None,
        apogee=None,
        terminate_on_apogee=True,
        impact_state=np.array([0.0]),
        x_impact=None,
        y_impact=None,
        z_impact=None,
        impact_velocity=None,
        impact_time=None,
        solution=_canonical_solution(_sample_state(0.0, 1.0), _sample_state(1.0, -1.0)),
        u_dot_generalized=lambda *_args, **_kwargs: "new-derivative",
    )

    out_of_rail_state = _sample_state(0.5, 1.0)
    out_of_rail_state[1] = 1.0
    assert out_of_rail_trigger(dict(flight=flight, state=out_of_rail_state[1:]))
    out_of_rail_callback(
        dict(
            flight=flight,
            event=out_of_rail_event,
            time=0.5,
            state=out_of_rail_state[1:],
        )
    )
    assert flight.out_of_rail_time == pytest.approx(0.5)
    assert flight.out_of_rail_time_index == 1
    assert np.allclose(flight.out_of_rail_state, out_of_rail_state[1:])
    assert out_of_rail_event.commands.new_dynamics is flight.u_dot_generalized
    assert out_of_rail_event.commands.new_dynamics_kwargs == {}
    assert out_of_rail_event.commands.new_flight_phase is True
    assert out_of_rail_event.commands.new_flight_phase_name == "free_flight"

    assert apogee_trigger(dict(flight=flight, state=_sample_state(1.0, -1.0)[1:]))
    apogee_result = apogee_callback(
        dict(
            flight=flight,
            event=apogee_event,
            time=1.0,
            state=_sample_state(1.0, -1.0)[1:],
        )
    )
    assert apogee_result is False
    assert flight.apogee_time == pytest.approx(1.0)
    assert flight.apogee_x == pytest.approx(0.0)
    assert flight.apogee_y == pytest.approx(0.0)
    assert flight.apogee == pytest.approx(0.0)
    assert apogee_event.commands._terminate is True

    impact_state = _sample_state(2.0, -1.0)
    impact_state[1] = 2.0
    impact_state[2] = -3.0
    impact_state[3] = -4.0
    impact_state[6] = -4.0
    assert impact_trigger(dict(flight=flight, state=impact_state[1:]))
    impact_callback(
        dict(
            flight=flight,
            event=impact_event,
            time=2.0,
            state=impact_state[1:],
        )
    )
    assert flight.impact_time == pytest.approx(2.0)
    assert flight.x_impact == pytest.approx(2.0)
    assert flight.y_impact == pytest.approx(-3.0)
    assert flight.z_impact == pytest.approx(-4.0)
    assert flight.impact_velocity == pytest.approx(-4.0)
    assert impact_event.commands._terminate is True

    assert out_of_rail_exact_time_function(
        dict(state=out_of_rail_state[1:], flight=flight)
    ) == pytest.approx(0.0)
    assert out_of_rail_exact_time_derivative(
        dict(state=out_of_rail_state[1:], flight=flight)
    ) == pytest.approx(0.0)
    assert apogee_event_exact_time_function(
        dict(state=_sample_state(1.0, -1.0)[1:])
    ) == pytest.approx(-1.0)
    assert impact_event_exact_time_function(dict(height_agl=-4.0)) == pytest.approx(
        -4.0
    )
    assert impact_event_exact_time_derivative(
        dict(state=impact_state[1:], flight=flight)
    ) == pytest.approx(-4.0)


@pytest.mark.parametrize(
    "apogee_state, solution",
    [
        (np.array([0.0, 0.0]), [_sample_state(0.0, 1.0), _sample_state(1.0, -1.0)]),
        (np.array([0.0]), [_sample_state(0.0, 1.0)]),
    ],
)
def test_apogee_trigger_returns_false_without_complete_history(
    apogee_state,
    solution,
):
    """Apogee should not trigger until a prior sample and a pending crossing exist."""

    flight = SimpleNamespace(apogee_state=apogee_state, solution=solution)

    assert (
        apogee_trigger(dict(flight=flight, state=_sample_state(1.0, -1.0)[1:])) is False
    )


def test_flight_with_disable_and_enable_examples_from_docs():
    """The docs-style disable_on and enable_on examples should work in Flight."""

    def my_callback(context):
        return {"time": context["time"]}

    def disable_above_altitude(context):
        return context["height_agl"] > 700.0

    def enable_above_altitude(context):
        return context["height_agl"] > 500.0

    time_gated = Event(
        callback=my_callback,
        name="Disabled at t=3s",
        disable_on=3.0,
    )
    burnout_gated = Event(
        callback=my_callback,
        name="Disabled at burnout",
        disable_on="burnout",
        sampling_rate=10,
    )
    altitude_gated = Event(
        callback=my_callback,
        name="Disabled above 700m",
        disable_on=disable_above_altitude,
    )
    time_enabled = Event(
        callback=my_callback,
        name="Re-enabled at t=2s",
        enabled=False,
        enable_on=2.0,
        sampling_rate=10,
    )
    altitude_enabled = Event(
        callback=my_callback,
        name="Enabled above 500m",
        enabled=False,
        enable_on=enable_above_altitude,
        sampling_rate=10,
    )

    flight = _docs_style_flight(
        [
            time_gated,
            burnout_gated,
            altitude_gated,
            time_enabled,
            altitude_enabled,
        ]
    )

    assert time_gated.disabled_times
    assert burnout_gated.disabled_times
    assert altitude_gated.disabled_times
    assert time_enabled.enabled_times
    assert altitude_enabled.enabled_times
    assert time_gated.enabled is False
    assert burnout_gated.enabled is False
    assert altitude_gated.enabled is False
    assert time_enabled.enabled is True
    assert altitude_enabled.enabled is True
    assert time_gated.disabled_times[0] >= 3.0
    assert burnout_gated.disabled_times[0] >= flight.rocket.motor.burn_out_time
    assert altitude_gated.disabled_times[0] > 0.0


def test_flight_with_exact_time_example_from_docs():
    """The docs-style exact-time event should be more precise than sampling."""

    def altitude_trigger(context):
        state = context["state"]
        flight = context["flight"]
        target_altitude = context["event"].memory["target_altitude"]
        return state[2] - flight.env.elevation > target_altitude

    def altitude_exact_time_function(context):
        return context["height_agl"]

    exact_time_event = Event(
        callback=_callback_return_time,
        trigger=altitude_trigger,
        exact_time_function=altitude_exact_time_function,
        exact_time_config={"target": 543.21},
        name="Exact-time altitude detector",
        memory={"target_altitude": 543.21},
        trigger_only_once=True,
    )
    sampled_altitude_event = Event(
        callback=_callback_return_time,
        trigger=altitude_trigger,
        name="Sampled altitude detector",
        memory={"target_altitude": 543.21},
        trigger_only_once=True,
    )

    flight = _docs_style_flight([exact_time_event, sampled_altitude_event])

    assert exact_time_event.triggered_times
    assert sampled_altitude_event.triggered_times

    target_altitude = exact_time_event.memory["target_altitude"]
    exact_error = abs(flight.z(exact_time_event.triggered_times[0]) - target_altitude)
    sampled_error = abs(
        flight.z(sampled_altitude_event.triggered_times[0]) - target_altitude
    )

    assert exact_error <= sampled_error
    assert exact_error < 1.0
    assert exact_time_event.enabled is False
    assert sampled_altitude_event.enabled is False


# ---------------------------------------------------------------------------
# Phases narrower and wider than the canonical state, in a running flight
# ---------------------------------------------------------------------------
#
# Every set of equations RocketPy ships integrates the full canonical state, so
# the two capabilities below are exercised against purpose-built ones handed to
# a flight through the set_dynamics command, which is how a phase reaches a
# simulation in the first place.

GRAVITY = 9.80665

BALLISTIC_STATES = ("x", "y", "z", "vx", "vy", "vz")


def _ballistic_derivative(_flight, _t, u, post_processing=False):
    """Free fall: position follows velocity, velocity follows gravity."""
    if post_processing:
        return ()
    return [u[3], u[4], u[5], 0.0, 0.0, -GRAVITY]


def _parafoil_derivative(_flight, _t, u, post_processing=False):
    """Free fall that also turns, carrying a heading the canonical state lacks."""
    if post_processing:
        return ()
    return [u[3], u[4], u[5], 0.0, 0.0, -GRAVITY, 0.5]


BALLISTIC_DYNAMICS = _PhaseDynamics(
    "test_ballistic", _ballistic_derivative, BALLISTIC_STATES
)

PARAFOIL_DYNAMICS = _PhaseDynamics(
    "test_parafoil",
    _parafoil_derivative,
    (*BALLISTIC_STATES, "heading"),
    # A heading is not a canonical state, so it cannot be picked out of the
    # state that ended the previous phase. The phase says where it starts.
    initial_state=lambda _flight, _t, canonical: [*canonical[:6], 0.0],
)


def _flight_switching_dynamics_at(time, dynamics, phase_name):
    """Run a flight that switches to ``dynamics`` once ``time`` is reached."""

    def switch(context):
        context["event"].commands.set_dynamics(dynamics)
        context["event"].commands.start_flight_phase(phase_name)
        return f"switched at {context['time']:.2f} s"

    switch_event = Event(
        callback=switch,
        trigger=lambda context: context["time"] >= time,
        name=f"Switch to {phase_name}",
        sampling_rate=10,
        trigger_only_once=True,
        changes_dynamics=True,
    )
    return _docs_style_flight([switch_event]), switch_event


def test_a_phase_narrower_than_the_canonical_state_flies():
    """A six-state phase integrates, stores and reads back correctly."""
    flight, switch_event = _flight_switching_dynamics_at(
        5.0, BALLISTIC_DYNAMICS, "ballistic"
    )
    solution = flight.solution

    assert switch_event.callback_log, "the flight never switched dynamics"
    ballistic = solution.phases[-1]
    assert ballistic.name == "ballistic"
    assert ballistic.dynamics.states == BALLISTIC_STATES

    # rows of that phase are stored at its own width, not the canonical one
    assert len(solution.raw_row(-1)) == 7
    assert len(solution.raw_row(0)) == 14
    # but the flight still reads as 14-value canonical rows throughout
    assert len(solution[-1]) == 14
    assert np.array(solution).shape == (len(solution), 14)

    # a canonical state the phase does not integrate is held at the value it
    # had when the phase began, so its history still covers the whole flight
    switch_index = ballistic.start - 1
    assert solution["e0"].shape == (len(solution), 2)
    assert solution.value_at(-1, "e0") == pytest.approx(
        solution.value_at(switch_index, "e0")
    )
    # and one it does integrate keeps moving
    assert solution.value_at(-1, "vz") != solution.value_at(switch_index, "vz")

    # the phase reports no post-process variables, so it contributes zeros
    # rather than breaking the flight's own accelerations
    assert np.isfinite(flight.az(flight.t_final))


def test_a_phase_wider_than_the_canonical_state_flies():
    """A phase carrying a state of its own integrates and reads back by name."""
    flight, switch_event = _flight_switching_dynamics_at(
        5.0, PARAFOIL_DYNAMICS, "parafoil"
    )
    solution = flight.solution

    assert switch_event.callback_log, "the flight never switched dynamics"
    parafoil = solution.phases[-1]
    assert parafoil.dynamics.states == (*BALLISTIC_STATES, "heading")

    # time plus six canonical states plus the phase's own heading
    assert len(solution.raw_row(-1)) == 8
    # the extra state is not part of the canonical row
    assert len(solution[-1]) == 14
    assert np.array(solution).shape == (len(solution), 14)

    # the heading exists only in this phase, and reading it says so
    with pytest.warns(UserWarning, match="not defined during"):
        heading = solution["heading"]
    assert len(heading) == len(solution) - parafoil.start
    # The phase's own rule seeds the heading at zero when the phase begins, and
    # it turns at 0.5 rad/s from there. The first row is stored one solver step
    # in, so both rows are checked against the time the phase started.
    assert heading[0, 1] == pytest.approx(
        0.5 * (heading[0, 0] - parafoil.t_start), abs=1e-9
    )
    assert heading[-1, 1] == pytest.approx(
        0.5 * (heading[-1, 0] - parafoil.t_start), rel=1e-9
    )
    assert heading[-1, 1] > heading[0, 1] > 0.0
    # and it reads through the single-row accessors as well
    assert solution.at_index(-1)["heading"] == pytest.approx(heading[-1, 1])
    assert solution.value_at(-1, "heading") == pytest.approx(heading[-1, 1])
    # while the phases before it have no such state
    with pytest.raises(KeyError, match="not defined in this flight phase"):
        solution.value_at(0, "heading")
