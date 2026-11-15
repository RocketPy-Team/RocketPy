"""Flights with controls: the control inputs stored with the rows, read back
after the flight, recorded values, and saving."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import ContinuousController, Controller, Event, Flight, GenericSurface
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
        context.controls.canards.deflection = -0.05 * context.state.w3
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


def test_the_controls_hold_each_value(damped_flight):
    flight, damper_event = damped_flight
    history = flight.controls["Canards"].deflection
    stored = flight.solution.controls.canards.deflection

    assert list(flight.controls) == ["Canards"]
    assert stored.tolist() == flight.solution["canards.deflection"].tolist()
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
    stored = flight.solution.controls.canards.deflection
    # the first row whose deflection differs from the one before it
    row = int(np.argmax(stored[1:, 1] != stored[:-1, 1])) + 1
    assert stored[row - 1, 1] == 0.0 != stored[row, 1]
    # it sits exactly at a time the damper ran, not at a solver step's end
    assert stored[row, 0] in damper_event.triggered_times


def test_records_are_read_after_the_flight(damped_flight):
    flight, damper_event = damped_flight
    roll_rate = flight.records.roll_damper.roll_rate
    assert roll_rate == damper_event.records["roll_rate"]
    # one record per time the damper ran: the times it ran again after the
    # flight went back to rail exit replaced what it had recorded then
    assert len(roll_rate) == len(set(damper_event.triggered_times))
    # the recorded roll rate is the flight's, at the recorded times
    t, recorded = roll_rate[len(roll_rate) // 2]
    assert recorded == pytest.approx(flight.w3(t), rel=1e-3, abs=1e-6)


def test_controls_and_records_are_saved_with_the_flight(damped_flight):
    flight, _ = damped_flight
    loaded = json.loads(json.dumps(flight, cls=RocketPyEncoder), cls=RocketPyDecoder)

    assert loaded.solution.control_names == ("canards.deflection",)
    for t in (2.0, 5.0, 10.0):
        assert loaded.controls["Canards"].deflection(t) == (
            flight.controls["Canards"].deflection(t)
        )
    assert list(loaded.records) == ["Roll damper"]
    assert list(loaded.records["Roll damper"]) == ["roll_rate"]


@patch("matplotlib.pyplot.show")
def test_controls_are_plotted_and_printed(mock_show, damped_flight, capsys):  # pylint: disable=unused-argument
    flight, _ = damped_flight
    assert flight.plots.controls() is None
    assert flight.plots.records() is None
    flight.prints.controls()
    # named as ``flight.controls`` reads it: the name as given, then the control
    assert "Canards.deflection" in capsys.readouterr().out


def _integrate_roll(states, controls, conditions):  # pylint: disable=unused-argument
    return [conditions.state.w3]


def _write_level(context):
    context.outputs.every_step.level = 0.1


@patch("matplotlib.pyplot.show")
def test_every_kind_of_controller_is_printed_and_plotted(
    mock_show,  # pylint: disable=unused-argument
    calisto_robust,
    example_plain_env,
    capsys,
):
    """A controller run at every solver step, a continuous controller with a
    state and an output are all in the flight's prints and plots."""
    calisto_robust.parachutes = []
    calisto_robust.add_controller(
        Controller(_write_level, None, outputs=("level",), name="Every Step")
    )
    calisto_robust.add_controller(
        ContinuousController(
            states=("roll",), state_derivative=_integrate_roll, name="Observer"
        )
    )
    flight = _fly(calisto_robust, example_plain_env)

    flight.prints.controllers()
    out = capsys.readouterr().out
    assert "Sampling Rate: every solver step" in out
    assert "Continuous Controller: Observer" in out and "State roll" in out
    assert flight.plots.state_history() is None
    assert flight.plots.event_outputs() is None
    assert flight.all_info() is None


def _canards(rocket):
    canards = GenericSurface(
        rocket.area,
        2 * rocket.radius,
        {"cl": lambda mach, deflection: 2.0 * deflection},
        name="Canards",
        controls=("deflection",),
        control_limits={"deflection": (-0.15, 0.15)},
    )
    rocket.add_surfaces(canards, 0.7)
    return canards


def _fly(rocket, environment, **options):
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        **options,
    )


def _profile_deflection(time):
    return 0.01 * time


def _follow_profile(context):
    context.controls.canards.deflection = _profile_deflection(context.time)


def test_an_open_loop_profile_is_flown_at_the_sampling_times(
    calisto_robust, example_plain_env
):
    calisto_robust.parachutes = []
    _canards(calisto_robust)
    controller = Controller(_follow_profile, sampling_rate=10)
    calisto_robust.add_controller(controller)
    flight = _fly(calisto_robust, example_plain_env)

    assert flight.controllers == [controller]
    history = flight.controls["Canards"].deflection
    times = sorted(set(controller.triggered_times))
    assert times[:2] == pytest.approx([0.1, 0.2])
    for t in times[:30]:
        # the value flown from each sampling time on, and held until the next
        assert history(t) == _profile_deflection(t)
        assert history(t + 0.05) == _profile_deflection(t)


def test_a_rocket_with_a_controller_flies_the_same_after_saving(
    calisto_robust, example_plain_env
):
    calisto_robust.parachutes = []
    _canards(calisto_robust)
    calisto_robust.add_controller(
        Controller(_follow_profile, sampling_rate=10, name="Profile")
    )
    flight = _fly(calisto_robust, example_plain_env)
    loaded = json.loads(
        json.dumps(calisto_robust, cls=RocketPyEncoder), cls=RocketPyDecoder
    )
    again = _fly(loaded, example_plain_env)

    assert [c.name for c in loaded.controllers] == ["Profile"]
    assert again.apogee == flight.apogee
    assert np.array_equal(
        again.solution.controls.canards.deflection,
        flight.solution.controls.canards.deflection,
    )


def test_records_stay_with_the_flight_that_made_them(damped_flight, example_plain_env):
    flight, damper_event = damped_flight
    first = flight.records.roll_damper.roll_rate
    # the same event flies again in a second flight
    _fly(flight.rocket, example_plain_env, custom_events=[damper_event])
    assert flight.records.roll_damper.roll_rate == first
