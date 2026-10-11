"""Unit tests for a flight's control inputs: a dictionary from channel name to
value. They use stand-ins for the flight, so no simulation is run."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Actuator, GenericSurface
from rocketpy.control.control_signal import (
    check_actuator,
    check_writes,
    control_channels,
    control_key,
    initial_controls,
)
from rocketpy.simulation.events import Event
from rocketpy.simulation.events.commands import Commands
from rocketpy.simulation.events.event_context import EventContext
from rocketpy.simulation.events.event_execution import restore_controls
from rocketpy.simulation.helpers.flight_derivatives import u_dot_generalized


class Gimbal:
    """A minimal actuator that is not a surface: it only has to define the
    attributes of :class:`rocketpy.Actuator`."""

    def __init__(self, name="Gimbal", clamp=True):
        self.name = name
        self.controls = ("pitch", "yaw")
        self.control_defaults = {"pitch": 0.0, "yaw": 0.01}
        self.control_limits = {"pitch": (-0.1, 0.1), "yaw": None}
        self.clamp = clamp


def _surface(name, **kwargs):
    return GenericSurface(1.0, 1.0, {}, name=name, **kwargs)


def _canard(name="Canards", **kwargs):
    return _surface(
        name,
        controls=("deflection",),
        control_limits={"deflection": (-0.2, 0.2)},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------


def test_surfaces_and_any_object_with_the_attributes_are_actuators():
    assert isinstance(_surface("Fins"), Actuator)
    assert isinstance(Gimbal(), Actuator)
    assert not isinstance(object(), Actuator)
    check_actuator(_canard())
    check_actuator(_surface("Fins"))
    check_actuator(Gimbal())


def test_an_object_without_the_attributes_is_refused():
    incomplete = Gimbal()
    del incomplete.clamp
    with pytest.raises(TypeError, match="has no clamp"):
        check_actuator(incomplete)


@pytest.mark.parametrize(
    "change, error, match",
    [
        ({"controls": ("pitch", "pitch")}, ValueError, "more than once"),
        ({"controls": ("pitch", "yaw", "roll")}, ValueError, "'roll' .* no default"),
        ({"controls": ("pitch", "2yaw")}, ValueError, "must be made of letters"),
        ({"control_defaults": {"pitch": "up", "yaw": 0}}, TypeError, "a number"),
        (
            {"control_limits": {"pitch": (0.1, -0.1), "yaw": None}},
            ValueError,
            "must not be larger",
        ),
        ({"control_limits": {"pitch": 0.1, "yaw": None}}, ValueError, "low, high"),
        (
            {"control_defaults": {"pitch": 0, "yaw": 0, "roll": 0}},
            ValueError,
            "not among its controls",
        ),
    ],
)
def test_a_wrong_actuator_is_refused(change, error, match):
    gimbal = Gimbal()
    for name, value in change.items():
        setattr(gimbal, name, value)
    with pytest.raises(error, match=match):
        check_actuator(gimbal)


# ---------------------------------------------------------------------------
# Channel names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, key",
    [
        ("Canards", "canards"),
        ("AirBrakes", "air_brakes"),
        ("calisto_nose_cone", "calisto_nose_cone"),
        ("TVCGimbal", "tvc_gimbal"),
        ("Fin #2 (left)", "fin_2_left"),
        ("1st stage fins", "actuator_1st_stage_fins"),
        ("", "actuator"),
    ],
)
def test_a_key_is_the_name_in_snake_case(name, key):
    assert control_key(name) == key


def test_only_actuators_with_controls_get_channels():
    canard, gimbal, fins = _canard(), Gimbal(), _surface("Fins")
    channels = control_channels([canard, fins, gimbal])
    assert channels == {
        canard: ("canards.deflection",),
        gimbal: ("gimbal.pitch", "gimbal.yaw"),
    }


def test_two_actuators_with_controls_and_the_same_key_are_refused():
    with pytest.raises(ValueError, match="'Canards' and 'canards' both"):
        control_channels([_canard("Canards"), _canard("canards")])
    # surfaces without controls may share any name
    control_channels([_canard("Fins"), _surface("Fins"), _surface("Fins")])


def test_a_flight_starts_with_every_channel_at_its_default():
    canard, gimbal = _canard(control_defaults={"deflection": 0.1}), Gimbal()
    controls, channels, limits = initial_controls([canard, _surface("Fins"), gimbal])

    assert controls == {
        "canards.deflection": 0.1,
        "gimbal.pitch": 0.0,
        "gimbal.yaw": 0.01,
    }
    assert channels[gimbal] == ("gimbal.pitch", "gimbal.yaw")
    assert limits["canards.deflection"] == ((-0.2, 0.2), True)
    assert limits["gimbal.yaw"] == (None, True)
    # a rocket without controls flies with no channels
    assert initial_controls([_surface("Fins")]) == ({}, {}, {})


# ---------------------------------------------------------------------------
# Checking writes
# ---------------------------------------------------------------------------


def _controls(*actuators):
    controls, _, limits = initial_controls(actuators)
    return controls, limits


def test_a_write_outside_the_limits_is_clamped_or_warned():
    controls, limits = _controls(_canard(), Gimbal(clamp=False))
    before = dict(controls)

    controls["canards.deflection"] = 0.5
    controls["gimbal.yaw"] = -3.0  # no limits
    assert check_writes(controls, before, limits) == {
        "canards.deflection": 0.2,
        "gimbal.yaw": -3.0,
    }
    assert controls["canards.deflection"] == 0.2
    assert controls["gimbal.yaw"] == -3.0

    before = dict(controls)
    controls["gimbal.pitch"] = 0.5
    with pytest.warns(UserWarning, match="gimbal.pitch = 0.5 is outside"):
        check_writes(controls, before, limits)
    assert controls["gimbal.pitch"] == 0.5


def test_writing_the_same_value_changes_nothing():
    controls, limits = _controls(_canard())
    before = dict(controls)
    controls["canards.deflection"] = 0.0
    assert check_writes(controls, before, limits) == {}


def test_a_misspelled_channel_is_refused_and_undone():
    controls, limits = _controls(_canard())
    before = dict(controls)
    controls["canard.deflection"] = 0.1
    with pytest.raises(KeyError, match="no control channel 'canard.deflection'"):
        check_writes(controls, before, limits)
    assert controls == before

    del controls["canards.deflection"]
    with pytest.raises(KeyError, match="cannot be removed"):
        check_writes(controls, before, limits)
    assert controls == before


# ---------------------------------------------------------------------------
# Writing from an event
# ---------------------------------------------------------------------------


def _run_callback(callback, controls, limits, **event_options):
    """Run an event's callback once, as the flight does, and return it."""
    event = Event(callback=callback, name="Writer", **event_options)
    flight = SimpleNamespace(control_signal=controls, _control_limits=limits)
    assert event(EventContext(flight=flight, time=1.0), callback_only=True)
    return event


def test_an_event_writes_the_controls_and_cuts_the_step():
    controls, limits = _controls(_canard())

    def deploy(context):
        context.controls["canards.deflection"] = 0.5

    event = _run_callback(deploy, controls, limits)

    assert controls["canards.deflection"] == 0.2  # clamped
    assert event.commands.wrote_controls is True
    assert event.commands.changes_trajectory is True
    event.commands.reset()
    assert event.commands.changes_trajectory is False


def test_an_event_that_reads_or_rewrites_a_value_cuts_nothing():
    controls, limits = _controls(_canard())
    seen = []

    def hold(context):
        seen.append(context.controls["canards.deflection"])
        context.controls["canards.deflection"] = 0.0

    assert _run_callback(hold, controls, limits).commands.wrote_controls is False
    assert seen == [0.0]


def test_an_event_that_changes_the_dynamics_is_not_marked():
    """Such an event already runs at its exact time, as it always did."""
    controls, limits = _controls(_canard())

    def deploy(context):
        context.controls["canards.deflection"] = 0.1

    event = _run_callback(deploy, controls, limits, changes_dynamics=True)
    assert controls["canards.deflection"] == 0.1
    assert event.commands.wrote_controls is False


# ---------------------------------------------------------------------------
# The rocket's channels
# ---------------------------------------------------------------------------


def test_a_rocket_without_controls_has_no_channels(calisto_robust):
    assert not calisto_robust.control_channels


def test_air_brakes_and_controlled_surfaces_add_their_channels(
    calisto_air_brakes_clamp_on,
):
    rocket = calisto_air_brakes_clamp_on
    rocket.add_surfaces(_canard(), 0.5)
    assert rocket.control_channels == (
        "canards.deflection",
        "air_brakes.deployment_level",
    )


def test_a_failed_add_leaves_the_rocket_as_it_was(calisto_robust):
    calisto_robust.add_surfaces(_canard("Canards"), 0.5)
    before = calisto_robust.actuators

    with pytest.raises(ValueError, match="both have controls"):
        calisto_robust.add_surfaces(_canard("canards"), 0.6)
    with pytest.raises(ValueError, match="both have controls"):
        calisto_robust.add_surfaces(
            [_surface("Strake"), _canard("CANARDS")], [0.4, 0.6]
        )
    with pytest.raises(TypeError, match="cannot be added to the rocket"):
        calisto_robust.add_surfaces(object(), 0.6)
    assert calisto_robust.actuators == before


# ---------------------------------------------------------------------------
# The equations of motion read the controls they are given
# ---------------------------------------------------------------------------


def _coasting_forces(rocket, environment, controls):
    """The aerodynamic forces of a coasting state with some sideslip,
    evaluated with a stand-in flight, so no simulation is run."""
    _, channels, _ = initial_controls(rocket.actuators)
    flight = SimpleNamespace(
        rocket=rocket,
        env=environment,
        _surface_switches={},
        _control_channels=channels,
    )
    upright = [1.0, 0.0, 0.0, 0.0]
    state = [0.0, 0.0, 1500.0, 10.0, 0.0, 200.0, *upright, 0.0, 0.0, 0.0]
    out = u_dot_generalized(flight, 10.0, state, controls, post_processing=True)
    return np.array(out[6:9])


def test_a_control_of_a_surface_moves_its_forces(calisto_robust, example_plain_env):
    rocket = calisto_robust
    canard = GenericSurface(
        rocket.area,
        2 * rocket.radius,
        {"cY": lambda deflection: 2.0 * deflection},
        name="Canards",
        controls=("deflection",),
    )
    rocket.add_surfaces(canard, 0.5)
    controls, _, _ = initial_controls(rocket.actuators)
    neutral = _coasting_forces(rocket, example_plain_env, controls)

    controls["canards.deflection"] = 0.1
    deflected = _coasting_forces(rocket, example_plain_env, controls)
    assert deflected[0] != neutral[0]
    # and the surface itself was not changed
    assert canard.control_defaults == {"deflection": 0.0}


def test_the_air_brakes_drag_follows_their_channel(
    calisto_air_brakes_clamp_on, example_plain_env
):
    rocket = calisto_air_brakes_clamp_on
    controls, _, _ = initial_controls(rocket.actuators)
    retracted = _coasting_forces(rocket, example_plain_env, controls)[2]

    controls["air_brakes.deployment_level"] = 0.8
    deployed = _coasting_forces(rocket, example_plain_env, controls)[2]
    assert deployed < retracted  # more drag, toward the tail
    assert rocket.air_brakes[0].deployment_level == 0


# ---------------------------------------------------------------------------
# Rollbacks and records
# ---------------------------------------------------------------------------


def _flight_in_a_step(controls, events=(), t=2.0):
    """A stand-in flight at the end of a solver step that began with
    ``controls``, holding only what a rollback reads."""
    return SimpleNamespace(
        t=t,
        control_signal=dict(controls),
        _step_controls=dict(controls),
        events=list(events),
    )


def test_a_rollback_undoes_writes_made_after_the_time_it_goes_back_to():
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    flight.control_signal["canards.deflection"] = 0.1  # written at the step end

    restore_controls(flight, 1.5)

    assert flight.control_signal == {"canards.deflection": 0.0}


def test_a_rollback_keeps_the_writes_of_the_event_that_asks_for_it():
    """The event wrote at the time the flight goes back to, so its writes
    stand."""
    flight = _flight_in_a_step({"canards.deflection": 0.0, "tab.trim": 0.0})
    flight.control_signal.update({"canards.deflection": 0.1, "tab.trim": 0.3})
    commands = Commands()
    commands.control_changes = {"tab.trim": 0.3}

    restore_controls(flight, 1.5, commands)

    assert flight.control_signal == {"canards.deflection": 0.0, "tab.trim": 0.3}


def test_going_back_to_the_current_time_cuts_nothing():
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    flight.control_signal["canards.deflection"] = 0.1

    restore_controls(flight, flight.t)

    assert flight.control_signal == {"canards.deflection": 0.1}


def test_the_controls_object_is_kept_through_a_rollback():
    """``context.controls`` is the same dict before and after."""
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    controls = flight.control_signal
    controls["canards.deflection"] = 0.1
    restore_controls(flight, 1.0)
    assert flight.control_signal is controls


def test_an_event_records_named_values_with_their_time():
    def callback(context):
        context.record(error=context.time * 2, gain=1.0)

    event = Event(callback=callback, name="Recorder")
    for time in (0.0, 0.5, 1.0):
        event(EventContext(time=time), callback_only=True)

    error = event.records["error"]
    assert list(error.x_array) == [0.0, 0.5, 1.0]
    assert list(error.y_array) == [0.0, 1.0, 2.0]
    assert error(0.7) == 1.0  # held until the next record
    assert set(event.records) == {"error", "gain"}

    event.reset()
    assert event.records == {}


def test_a_rollback_drops_the_records_made_after_its_time():
    event = Event(callback=lambda context: context.record(x=1.0), name="Recorder")
    for time in (0.5, 1.0, 1.8):
        event(EventContext(time=time), callback_only=True)
    flight = _flight_in_a_step({}, events=[event])

    restore_controls(flight, 1.5)

    assert list(event.records["x"].x_array) == [0.5, 1.0]


def test_an_event_keeps_what_its_callback_changed():
    controls, limits = _controls(_canard())

    def deploy(context):
        context.controls["canards.deflection"] = 0.1

    event = _run_callback(deploy, controls, limits)
    assert event.commands.control_changes == {"canards.deflection": 0.1}
    event.commands.reset()
    assert event.commands.control_changes == {}
