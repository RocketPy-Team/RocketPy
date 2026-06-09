from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Environment, Event, Flight, Rocket, SolidMotor
from rocketpy.simulation.events import Commands
import rocketpy.simulation.events.exact_time_solvers as exact_time_solvers
from rocketpy.simulation.events.event_builders import (
    apogee_callback,
    apogee_event_exact_time_function,
    apogee_trigger,
    build_core_events,
    impact_callback,
    impact_event_exact_time_derivative,
    impact_event_exact_time_function,
    impact_step_end_function,
    impact_trigger,
    out_of_rail_callback,
    out_of_rail_exact_time_derivative,
    out_of_rail_exact_time_function,
    out_of_rail_trigger,
)
from rocketpy.simulation.events.exact_time_solvers import (
    filter_roots_by_policy,
    solve_cubic_hermite_step_roots,
    solve_exact_time_brentq,
    solve_exact_time_cubic_hermite,
    solve_exact_time_linear,
)


def _callback_return_time(**kwargs):
    return {"time": kwargs["time"], "sampled_time": kwargs.get("sampled_time")}


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


def _interpolator(time):
    return np.array(
        [1.0 - 2.0 * time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


class _FakePhase:
    def __init__(self, interpolator):
        self.solver = SimpleNamespace(dense_output=lambda: interpolator)

    def derivative(self, _time, state, post_processing=False):
        _ = post_processing
        return np.zeros_like(state, dtype=float)


@pytest.mark.parametrize(
    ("roots", "lower_bound", "upper_bound", "expected"),
    [
        ([0.1 + 0.0j, 0.6 + 0.0005j, 0.9 + 0.1j], 0.0, 1.0, [0.1, 0.6]),
        ([0.1 + 0.0j, 1.5 + 0.0j], 0.2, 1.0, []),
    ],
)
def test_filter_roots_by_policy_filters_complex_and_out_of_interval_roots(
    roots,
    lower_bound,
    upper_bound,
    expected,
):
    """The root policy should keep real roots inside the requested interval."""

    valid_roots = filter_roots_by_policy(
        roots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_abs_imag=1e-3,
        strict_interval=False,
    )

    assert valid_roots == pytest.approx(expected)


def test_solve_cubic_hermite_step_roots_returns_expected_root():
    """The cubic-Hermite helper should return the root inside the interval."""

    roots = solve_cubic_hermite_step_roots(
        step_end=1.0,
        y0=1.0,
        yp0=-1.5,
        y1=-1.0,
        yp1=-1.5,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    assert roots == pytest.approx([0.5])


@pytest.mark.parametrize(
    "solver",
    [solve_exact_time_linear, solve_exact_time_brentq, solve_exact_time_cubic_hermite],
)
def test_exact_time_solvers_return_expected_event_time(solver):
    """All exact-time solvers should resolve the same simple root."""

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, -1.0)
    previous_state[1] = 1.0
    current_state[1] = -1.0

    kwargs = {
        "previous_state": previous_state,
        "current_state": current_state,
        "interpolator": _interpolator,
        "event_function": lambda state, **_kwargs: state[0],
        "no_root_error_message": "no root",
    }
    if solver is solve_exact_time_cubic_hermite:
        kwargs["derivative_function"] = lambda state, **_kwargs: -1.5

    result = solver(**kwargs)

    assert result["event_time"] == pytest.approx(0.5)
    assert result["event_state"][0] == pytest.approx(0.0)


def test_exact_time_linear_raises_when_endpoint_values_match():
    """The linear solver should reject steps with identical endpoint values."""

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, -1.0)

    with pytest.raises(ValueError, match="no root"):
        solve_exact_time_linear(
            previous_state=previous_state,
            current_state=current_state,
            interpolator=_interpolator,
            event_function=lambda state, **_kwargs: 1.0,
            no_root_error_message="no root",
        )


def test_exact_time_brentq_raises_when_no_sign_change_occurs():
    """Brent's method should fail cleanly when the event function does not
    bracket a root."""

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, 1.0)

    with pytest.raises(ValueError, match="no root"):
        solve_exact_time_brentq(
            previous_state=previous_state,
            current_state=current_state,
            interpolator=_interpolator,
            event_function=lambda state, **_kwargs: state[5],
            no_root_error_message="no root",
        )


def test_exact_time_brentq_wraps_runtime_errors_from_brentq(monkeypatch):
    """Brentq failures inside the solver should be wrapped in ValueError."""

    def failing_brentq(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(exact_time_solvers, "brentq", failing_brentq)

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, 1.0)

    with pytest.raises(ValueError, match="no root"):
        solve_exact_time_brentq(
            previous_state=previous_state,
            current_state=current_state,
            interpolator=lambda time: np.array([time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            event_function=lambda state, **_kwargs: state[0] - 0.5,
            no_root_error_message="no root",
        )


def test_exact_time_cubic_hermite_raises_when_multiple_roots_are_found(monkeypatch):
    """Multiple valid Hermite roots should be rejected by the wrapper."""

    monkeypatch.setattr(
        exact_time_solvers,
        "solve_cubic_hermite_step_roots",
        lambda **_kwargs: [0.25, 0.75],
    )

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, -1.0)

    with pytest.raises(ValueError, match="no root"):
        solve_exact_time_cubic_hermite(
            previous_state=previous_state,
            current_state=current_state,
            interpolator=_interpolator,
            event_function=lambda state, **_kwargs: state[0],
            derivative_function=lambda state, **_kwargs: -1.5,
            no_root_error_message="no root",
        )


def test_exact_time_cubic_hermite_raises_when_no_roots_are_found(monkeypatch):
    """Empty Hermite root sets should be rejected by the wrapper."""

    monkeypatch.setattr(
        exact_time_solvers,
        "solve_cubic_hermite_step_roots",
        lambda **_kwargs: [],
    )

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, -1.0)

    with pytest.raises(ValueError, match="no root"):
        solve_exact_time_cubic_hermite(
            previous_state=previous_state,
            current_state=current_state,
            interpolator=_interpolator,
            event_function=lambda state, **_kwargs: state[0],
            derivative_function=lambda state, **_kwargs: -1.5,
            no_root_error_message="no root",
        )


def test_commands_api_records_expected_payloads():
    """The command container should store all command types and reset cleanly."""

    commands = Commands()
    event = SimpleNamespace(name="other")
    controller = SimpleNamespace(name="controller")

    commands.enable()
    commands.disable()
    commands.add_event(event)
    commands.disable_event(event)
    commands.set_derivative(lambda *_args, **_kwargs: None)
    commands.start_flight_phase("descent", lag=1.5, parachute="main")
    commands.terminate_flight()

    assert commands._disabled is True
    assert commands.new_events == [event]
    assert commands.disable_events == [event]
    assert commands.new_derivative_set is True
    assert commands.new_flight_phase is True
    assert commands.new_flight_phase_name == "descent"
    assert commands.new_flight_phase_lag == 1.5
    assert commands.new_flight_phase_parachute == "main"
    assert commands._terminate is True

    commands.reset()

    assert commands._disabled is None
    assert commands.new_events == []
    assert commands.new_derivative is None
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
        solution=[_sample_state(0.0, 1.0), _sample_state(1.0, -1.0)],
        u_dot_generalized=lambda *_args, **_kwargs: "new-derivative",
    )

    out_of_rail_state = _sample_state(0.5, 1.0)
    out_of_rail_state[1] = 1.0
    assert out_of_rail_trigger(flight=flight, state=out_of_rail_state[1:])
    out_of_rail_callback(
        flight=flight,
        event=out_of_rail_event,
        time=0.5,
        state=out_of_rail_state[1:],
    )
    assert flight.out_of_rail_time == pytest.approx(0.5)
    assert flight.out_of_rail_time_index == 1
    assert np.allclose(flight.out_of_rail_state, out_of_rail_state[1:])
    assert out_of_rail_event.commands.new_derivative == flight.u_dot_generalized
    assert out_of_rail_event.commands.new_flight_phase is True
    assert out_of_rail_event.commands.new_flight_phase_name == "free_flight"

    assert apogee_trigger(flight=flight, state=_sample_state(1.0, -1.0)[1:])
    apogee_result = apogee_callback(
        flight=flight,
        event=apogee_event,
        time=1.0,
        state=_sample_state(1.0, -1.0)[1:],
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
    assert impact_trigger(flight=flight, state=impact_state[1:])
    impact_callback(
        flight=flight,
        event=impact_event,
        time=2.0,
        state=impact_state[1:],
    )
    assert flight.impact_time == pytest.approx(2.0)
    assert flight.x_impact == pytest.approx(2.0)
    assert flight.y_impact == pytest.approx(-3.0)
    assert flight.z_impact == pytest.approx(-4.0)
    assert flight.impact_velocity == pytest.approx(-4.0)
    assert impact_event.commands._terminate is True

    assert out_of_rail_exact_time_function(out_of_rail_state[1:], flight=flight) == pytest.approx(0.0)
    assert out_of_rail_exact_time_derivative(out_of_rail_state[1:], flight=flight) == pytest.approx(0.0)
    assert apogee_event_exact_time_function(_sample_state(1.0, -1.0)[1:]) == pytest.approx(-1.0)
    assert impact_event_exact_time_function(impact_state[1:], flight=flight) == pytest.approx(-4.0)
    assert impact_event_exact_time_derivative(impact_state[1:], flight=flight) == pytest.approx(-4.0)
    assert impact_step_end_function(step_size=0.25) == pytest.approx(0.25)


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

    assert apogee_trigger(flight=flight, state=_sample_state(1.0, -1.0)[1:]) is False


def test_flight_with_disable_and_enable_examples_from_docs():
    """The docs-style disable_on and enable_on examples should work in Flight."""

    def my_callback(**kwargs):
        return {"time": kwargs["time"]}

    def disable_above_altitude(**kwargs):
        return kwargs["height_above_ground_level"] > 700.0

    def enable_above_altitude(**kwargs):
        return kwargs["height_above_ground_level"] > 500.0

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

    def altitude_trigger(**kwargs):
        state = kwargs["state"]
        flight = kwargs["flight"]
        target_altitude = kwargs["event"].context["target_altitude"]
        return state[2] - flight.env.elevation > target_altitude

    def altitude_exact_time_function(state, **kwargs):
        flight = kwargs["flight"]
        return state[2] - flight.env.elevation

    exact_time_event = Event(
        callback=_callback_return_time,
        trigger=altitude_trigger,
        exact_time_function=altitude_exact_time_function,
        exact_time_config={"target": 543.21},
        name="Exact-time altitude detector",
        context={"target_altitude": 543.21},
        trigger_only_once=True,
    )
    sampled_altitude_event = Event(
        callback=_callback_return_time,
        trigger=altitude_trigger,
        name="Sampled altitude detector",
        context={"target_altitude": 543.21},
        trigger_only_once=True,
    )

    flight = _docs_style_flight([exact_time_event, sampled_altitude_event])

    assert exact_time_event.triggered_times
    assert sampled_altitude_event.triggered_times
    assert exact_time_event.callback_log[0]["sampled_time"] is not None
    assert sampled_altitude_event.callback_log[0]["sampled_time"] is None

    target_altitude = exact_time_event.context["target_altitude"]
    exact_error = abs(flight.z(exact_time_event.triggered_times[0]) - target_altitude)
    sampled_error = abs(
        flight.z(sampled_altitude_event.triggered_times[0]) - target_altitude
    )

    assert exact_error <= sampled_error
    assert exact_error < 1.0
    assert exact_time_event.enabled is False
    assert sampled_altitude_event.enabled is False