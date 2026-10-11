"""Flights with controls: the control inputs stored with the rows, read back
after the flight, recorded values, and saving."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import Event, Flight, GenericSurface
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder


def _roll_damped_flight(rocket, environment, **event_options):
    """Calisto with canards whose deflection a 50 Hz event sets against the
    roll rate, and records."""
    canards = GenericSurface(
        rocket.area,
        2 * rocket.radius,
        {"cl": lambda mach, deflection: 2.0 * deflection},
        name="Canards",
        controls=("deflection",),
        control_limits={"deflection": (-0.15, 0.15)},
    )
    rocket.add_surfaces(canards, 0.7)

    def damper(context):
        context.controls["canards.deflection"] = -0.05 * context.state.w3
        context.record(roll_rate=context.state.w3)

    damper_event = Event(
        callback=damper, sampling_rate=50, name="Roll damper", **event_options
    )
    flight = Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        custom_events=[damper_event],
    )
    return flight, damper_event


@pytest.fixture
def damped_flight(calisto_robust, example_plain_env):
    calisto_robust.parachutes = []
    return _roll_damped_flight(calisto_robust, example_plain_env)


def test_the_control_history_holds_each_value(damped_flight):
    flight, damper_event = damped_flight
    history = flight.control_history["canards.deflection"]
    stored = flight.solution["canards.deflection"]

    assert list(flight.control_history) == ["canards.deflection"]
    assert stored.shape == (len(flight.solution), 2)
    # between two rows the value in force is the earlier row's
    t0, t1 = stored[100, 0], stored[101, 0]
    assert history((t0 + t1) / 2) == stored[100, 1]
    # the deflection only ever changes at a time the damper ran
    changes = stored[1:, 1] != stored[:-1, 1]
    sampled = set(np.round(damper_event.triggered_times, 9))
    assert set(np.round(stored[1:, 0][changes], 9)) <= sampled
    # the canards damp the roll: the deflection stays inside its limits
    assert np.all(np.abs(stored[:, 1]) <= 0.15)


def test_a_write_inside_a_step_acts_from_the_event_time(damped_flight):
    """The damper runs on interpolated states inside solver steps; each write
    cuts the step there, so the row at the sampling time carries the value."""
    flight, damper_event = damped_flight
    stored = flight.solution["canards.deflection"]
    # the first row whose deflection differs from the one before it
    row = int(np.argmax(stored[1:, 1] != stored[:-1, 1])) + 1
    assert stored[row - 1, 1] == 0.0 != stored[row, 1]
    # it sits exactly at a time the damper ran, not at a solver step's end
    assert stored[row, 0] in damper_event.triggered_times


def test_records_are_read_after_the_flight(damped_flight):
    flight, damper_event = damped_flight
    roll_rate = flight.records["roll_damper.roll_rate"]
    assert list(roll_rate.x_array) == list(damper_event.records["roll_rate"].x_array)
    # one record per time the damper ran: the times it ran again after the
    # flight went back to rail exit replaced what it had recorded then
    assert len(roll_rate.x_array) == len(set(damper_event.triggered_times))
    # the recorded roll rate is the flight's, at the recorded times
    t = roll_rate.x_array[len(roll_rate.x_array) // 2]
    assert roll_rate(t) == pytest.approx(flight.w3(t), rel=1e-3, abs=1e-6)


def test_recorded_outputs_equal_the_rows_worked_out_again(flight_calisto_air_brakes):
    """An air-brakes flight records its outputs while flying. Working each
    row out again with the controls stored with it gives the same values,
    so the stored controls are the ones the row was flown with."""
    solution = flight_calisto_air_brakes.solution
    assert solution.records_post_values
    assert max(solution["air_brakes.deployment_level"][:, 1]) > 0
    for index in range(len(solution)):
        phase = solution.phases[solution.phase_index_at(index)]
        if not phase.dynamics.post_process_vars:
            continue
        row = solution.raw_row(index)
        replayed = phase.dynamics.post_process_values(
            phase.bound_dynamics.post_process_at(
                row[0], row[1:], solution._controls[index]
            )
        )
        assert replayed == solution._post_values[index], index


def test_controls_and_records_are_saved_with_the_flight(damped_flight):
    flight, _ = damped_flight
    loaded = json.loads(json.dumps(flight, cls=RocketPyEncoder), cls=RocketPyDecoder)

    assert loaded.solution.control_names == ("canards.deflection",)
    for t in (2.0, 5.0, 10.0):
        assert loaded.control_history["canards.deflection"](t) == (
            flight.control_history["canards.deflection"](t)
        )
    assert list(loaded.records) == ["roll_damper.roll_rate"]


@patch("matplotlib.pyplot.show")
def test_controls_are_plotted_and_printed(mock_show, damped_flight, capsys):  # pylint: disable=unused-argument
    flight, _ = damped_flight
    assert flight.plots.controls() is None
    assert flight.plots.records() is None
    flight.prints.controls()
    assert "canards.deflection" in capsys.readouterr().out
