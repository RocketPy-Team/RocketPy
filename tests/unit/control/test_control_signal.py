"""Unit tests for a flight's control inputs: a dictionary from channel name to
value. They use stand-ins for the flight, so no simulation is run."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import ForceEffector, GenericSurface, ThrustEffector
from rocketpy.control._helpers import check_writes, initial_controls
from rocketpy.control.controllable import Controllable
from rocketpy.simulation.events import Event
from rocketpy.simulation.events.commands import Commands
from rocketpy.simulation.events.event_context import EventContext
from rocketpy.simulation.events.event_execution import (
    call_event,
    rewind,
    save_for_rollback,
)
from rocketpy.simulation.events.views import ControlsView
from rocketpy.simulation.helpers.flight_derivatives import (
    u_dot_generalized,
)
from rocketpy.tools import snake_case_key


class Gimbal(Controllable):
    """A minimal controllable with controls that is not a surface."""

    def __init__(self, name="Gimbal"):
        super().__init__(
            name,
            ("pitch", "yaw"),
            control_defaults={"pitch": 0.0, "yaw": 0.01},
            control_limits={"pitch": (-0.1, 0.1)},
        )


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
# Controllables with controls
# ---------------------------------------------------------------------------


def test_surfaces_and_effectors_are_controllable():
    effector = ForceEffector(lambda conditions: ([0, 0, 0], [0, 0, 0]))
    for controllable in (_surface("Fins"), _canard(), Gimbal(), effector):
        assert isinstance(controllable, Controllable)
        controllable._check_controls()


def test_an_object_that_is_not_controllable_is_refused(calisto):
    class LookAlike:  # pylint: disable=too-few-public-methods
        name = "Look Alike"
        controls = ()
        control_defaults = {}
        control_limits = {}

    for controllable in (object(), LookAlike()):
        with pytest.raises(TypeError, match="not one of RocketPy's controllables"):
            calisto._check_new_controllables(controllable)


def test_a_controllable_checks_its_controls_when_built():
    with pytest.raises(ValueError, match="more than once"):
        Controllable("Gimbal", ("pitch", "pitch"))
    with pytest.raises(ValueError, match="not among its controls"):
        Controllable("Gimbal", ("pitch",), control_defaults={"yaw": 0})
    with pytest.raises(ValueError, match="must not be larger"):
        Controllable("Gimbal", ("pitch",), control_limits={"pitch": (1, -1)})


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
def test_a_wrong_part_is_refused(change, error, match):
    gimbal = Gimbal()
    for name, value in change.items():
        setattr(gimbal, name, value)
    with pytest.raises(error, match=match):
        gimbal._check_controls()


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
        ("1st stage fins", "controllable_1st_stage_fins"),
        ("", "controllable"),
    ],
)
def test_a_key_is_the_name_in_snake_case(name, key):
    assert snake_case_key(name) == key


def test_only_parts_with_controls_get_channels():
    canard, gimbal, fins = _canard(), Gimbal(), _surface("Fins")
    _, channels, _ = initial_controls([canard, fins, gimbal])
    assert channels == {
        canard: ("canards.deflection",),
        gimbal: ("gimbal.pitch", "gimbal.yaw"),
    }


def test_two_parts_with_controls_and_the_same_key_are_refused():
    with pytest.raises(ValueError, match="'Canards' and 'canards' both"):
        initial_controls([_canard("Canards"), _canard("canards")])
    # surfaces without controls may share any name
    initial_controls([_canard("Fins"), _surface("Fins"), _surface("Fins")])


def test_a_flight_starts_with_every_channel_at_its_default():
    canard, gimbal = _canard(control_defaults={"deflection": 0.1}), Gimbal()
    controls, channels, limits = initial_controls([canard, _surface("Fins"), gimbal])

    assert controls == {
        "canards.deflection": 0.1,
        "gimbal.pitch": 0.0,
        "gimbal.yaw": 0.01,
    }
    assert channels[gimbal] == ("gimbal.pitch", "gimbal.yaw")
    assert limits["canards.deflection"] == (-0.2, 0.2)
    assert limits["gimbal.yaw"] is None
    # a rocket without controls flies with no channels
    assert initial_controls([_surface("Fins")]) == ({}, {}, {})


# ---------------------------------------------------------------------------
# Checking writes
# ---------------------------------------------------------------------------


def _controls(*controllables):
    controls, _, limits = initial_controls(controllables)
    return controls, limits


def test_a_write_outside_the_limits_is_clamped():
    controls, limits = _controls(_canard(), Gimbal())
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
    assert check_writes(controls, before, limits) == {"gimbal.pitch": 0.1}
    assert controls["gimbal.pitch"] == 0.1


def test_writing_the_same_value_changes_nothing():
    controls, limits = _controls(_canard())
    before = dict(controls)
    controls["canards.deflection"] = 0.0
    assert check_writes(controls, before, limits) == {}


def test_a_write_beyond_the_limit_the_control_sits_at_changes_nothing():
    """A saturated controller keeps writing past the limit: the control stays
    at the limit, so the trajectory does not change and no step is cut."""
    controls, limits = _controls(_canard())
    controls["canards.deflection"] = 0.2
    before = dict(controls)
    controls["canards.deflection"] = 0.7
    assert check_writes(controls, before, limits) == {}
    assert controls["canards.deflection"] == 0.2


def test_a_misspelled_channel_is_refused_and_undone():
    controls, limits = _controls(_canard())
    before = dict(controls)
    controls["canard.deflection"] = 0.1
    with pytest.raises(KeyError, match="no control input 'canard.deflection'"):
        check_writes(controls, before, limits)
    assert controls == before

    del controls["canards.deflection"]
    with pytest.raises(KeyError, match="cannot be removed"):
        check_writes(controls, before, limits)
    assert controls == before


def test_a_controllable_is_found_by_its_name_or_by_itself():
    canards, gimbal = _canard(), Gimbal(name="Main Gimbal")
    controls, _ = _controls(canards, gimbal)
    view = ControlsView(controls, (canards, gimbal))

    view[canards].deflection = 0.1
    view["Main Gimbal"].yaw = 0.02
    assert controls["canards.deflection"] == 0.1
    assert controls["main_gimbal.yaw"] == 0.02
    assert view[gimbal].pitch == view.main_gimbal.pitch == 0.0
    assert list(view) == ["Canards", "Main Gimbal"] and len(view) == 2
    assert "Main Gimbal" in view and "main_gimbal" not in view

    with pytest.raises(TypeError, match="not the controllable itself"):
        view["Canards"] = 0.1
    with pytest.raises(KeyError, match="named 'main_gimbal'.*'Canards', 'Main"):
        _ = view["main_gimbal"]  # the name as given, not in snake_case
    with pytest.raises(KeyError, match="named 'Main Gimbal.yaw'"):
        _ = view["Main Gimbal.yaw"]
    with pytest.raises(KeyError, match="named 'Fins'"):
        _ = view[_canard("Fins")]


# ---------------------------------------------------------------------------
# Writing from an event
# ---------------------------------------------------------------------------


def _run_callback(callback, controls, limits, **event_options):
    """Run an event's callback once, as the flight does, and return it."""
    event = Event(callback=callback, name="Writer", **event_options)
    flight = SimpleNamespace(
        t=1.0,
        _controls=controls,
        _control_limits=limits,
        rocket=SimpleNamespace(_refresh_aerodynamics=lambda: None),
    )
    context = EventContext(
        controls=controls,
        time=1.0,
        canonical_state=[0.0] * 13,
        raw_state=[0.0] * 13,
        phase_names=(),
    )
    assert call_event(flight, event, context, None, 0, 0)
    return event


def test_an_event_writes_the_controls_and_cuts_the_step():
    controls, limits = _controls(_canard())

    def deploy(context):
        context.controls.canards.deflection = 0.5

    event = _run_callback(deploy, controls, limits)

    assert controls["canards.deflection"] == 0.2  # clamped
    assert event.commands.control_changes == {"canards.deflection": 0.2}
    assert event.commands.changes_trajectory is True
    event.commands.reset()
    assert event.commands.changes_trajectory is False


def test_an_event_that_reads_or_rewrites_a_value_cuts_nothing():
    controls, limits = _controls(_canard())
    seen = []

    def hold(context):
        seen.append(context.controls.canards.deflection)
        context.controls.canards.deflection = 0.0

    commands = _run_callback(hold, controls, limits).commands
    assert commands.control_changes == {}
    assert commands.changes_trajectory is False
    assert seen == [0.0]


# ---------------------------------------------------------------------------
# The rocket's channels
# ---------------------------------------------------------------------------


def test_a_rocket_without_controls_has_no_channels(calisto_robust):
    assert not calisto_robust.control_names


def test_air_brakes_and_controlled_surfaces_add_their_channels(
    calisto_air_brakes_clamp_on,
):
    rocket = calisto_air_brakes_clamp_on
    rocket.add_surfaces(_canard(), 0.5)
    # air brakes are surfaces: the channels follow the order they were added
    assert rocket.control_names == (
        "air_brakes.deployment_level",
        "canards.deflection",
    )


def test_a_failed_add_leaves_the_rocket_as_it_was(calisto_robust):
    calisto_robust.add_surfaces(_canard("Canards"), 0.5)
    before = calisto_robust.controllables

    with pytest.raises(ValueError, match="both have controls"):
        calisto_robust.add_surfaces(_canard("canards"), 0.6)
    with pytest.raises(ValueError, match="both have controls"):
        calisto_robust.add_surfaces(
            [_surface("Strake"), _canard("CANARDS")], [0.4, 0.6]
        )
    with pytest.raises(TypeError, match="cannot be added to the rocket"):
        calisto_robust.add_surfaces(object(), 0.6)
    assert calisto_robust.controllables == before


# ---------------------------------------------------------------------------
# The equations of motion read the controls they are given
# ---------------------------------------------------------------------------


def _coasting_forces(rocket, environment, controls):
    """The aerodynamic forces of a coasting state with some sideslip,
    evaluated with a stand-in flight, so no simulation is run."""
    _, channels, _ = initial_controls(rocket.controllables)
    flight = SimpleNamespace(
        rocket=rocket,
        env=environment,
        _surface_switches={},
        _control_names=channels,
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
    controls, _, _ = initial_controls(rocket.controllables)
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
    controls, _, _ = initial_controls(rocket.controllables)
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
        _controls=dict(controls),
        _control_limits=dict.fromkeys(controls),
        _step_log={},
        events=list(events),
    )


def _run(event, flight, time):
    """Run ``event``'s callback at ``time`` in ``flight``'s step, as the flight
    does, without applying its commands."""
    controls = flight._controls
    before = dict(controls)
    save_for_rollback(flight, event, time, before, None)
    event(EventContext(time=time, controls=controls), callback_only=True)
    changes = check_writes(controls, before, flight._control_limits)
    event.commands.control_changes.update(changes)


def _writer(name, value, **kwargs):
    key, channel = name.split(".")

    def callback(context):
        setattr(getattr(context.controls, key), channel, value)

    return Event(callback=callback, name=f"Write {value}", **kwargs)


def test_a_rollback_undoes_writes_made_after_the_time_it_goes_back_to():
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    _run(_writer("canards.deflection", 0.1), flight, 2.0)  # at the step end

    rewind(flight, 1.5)

    assert flight._controls == {"canards.deflection": 0.0}


def test_a_rollback_keeps_writes_made_before_the_time_it_goes_back_to():
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    _run(_writer("canards.deflection", 0.1), flight, 1.0)
    _run(_writer("canards.deflection", 0.2), flight, 2.0)

    rewind(flight, 1.5)

    assert flight._controls == {"canards.deflection": 0.1}


def test_a_rollback_keeps_the_writes_of_the_event_that_asks_for_it():
    """The event wrote at the time the flight goes back to, so its writes
    stand."""
    flight = _flight_in_a_step({"canards.deflection": 0.0, "tab.trim": 0.0})
    _run(_writer("canards.deflection", 0.1), flight, 2.0)
    cutter = _writer("tab.trim", 0.3)
    _run(cutter, flight, 2.0)

    rewind(flight, 1.5, cutter.commands)

    assert flight._controls == {"canards.deflection": 0.0, "tab.trim": 0.3}


def test_going_back_to_the_current_time_cuts_nothing():
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    _run(_writer("canards.deflection", 0.1), flight, 2.0)

    rewind(flight, flight.t)

    assert flight._controls == {"canards.deflection": 0.1}


def test_the_controls_object_is_kept_through_a_rollback():
    """``context.controls`` is the same dict before and after."""
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    controls = flight._controls
    _run(_writer("canards.deflection", 0.1), flight, 2.0)
    rewind(flight, 1.0)
    assert flight._controls is controls


def _counter():
    def callback(context):
        memory = context.memory
        memory["runs"] = memory.get("runs", 0) + 1
        memory.setdefault("times", []).append(context.time)

    return Event(callback=callback, name="Counter", memory={"runs": 0})


def test_a_rollback_puts_the_memory_back_as_it_was_at_its_time():
    counter = _counter()
    flight = _flight_in_a_step({}, events=[counter])
    for time in (0.5, 1.0, 1.8, 2.0):
        _run(counter, flight, time)
    memory = counter.memory

    rewind(flight, 1.5)

    assert counter.memory == {"runs": 2, "times": [0.5, 1.0]}
    assert counter.memory is memory


def test_a_rollback_keeps_the_memory_of_the_event_that_asks_for_it():
    counter = _counter()
    flight = _flight_in_a_step({}, events=[counter])
    _run(counter, flight, 2.0)

    rewind(flight, 1.5, counter.commands)

    assert counter.memory == {"runs": 1, "times": [2.0]}


def test_an_event_checked_twice_at_one_time_is_saved_once():
    """A trigger check and then the callback at the same time: the memory to
    go back to is the one from before the check."""
    counter = _counter()
    flight = _flight_in_a_step({}, events=[counter])
    _run(counter, flight, 2.0)
    _run(counter, flight, 2.0)

    rewind(flight, 1.5)

    assert counter.memory == {"runs": 0}


def test_an_event_records_named_values_with_their_time():
    def callback(context):
        context.record(error=context.time * 2, gain=1.0)

    event = Event(callback=callback, name="Recorder")
    for time in (0.0, 0.5, 1.0):
        event(EventContext(time=time), callback_only=True)

    assert event.records["error"] == [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0)]
    assert set(event.records) == {"error", "gain"}

    event.reset()
    assert event.records == {}


def test_a_record_that_is_not_only_numbers_is_a_list_of_copies():
    command = [0.1, 0.2]

    def callback(context):
        context.record(command=command, mode="coast" if context.time else 1.0)

    event = Event(callback=callback, name="Recorder")
    for time in (0.0, 0.5):
        event(EventContext(time=time), callback_only=True)
        command[0] += 1.0  # changed after it was recorded

    assert event.records["command"] == [(0.0, [0.1, 0.2]), (0.5, [1.1, 0.2])]
    assert event.records["mode"] == [(0.0, 1.0), (0.5, "coast")]


def test_a_record_keeps_a_number_as_it_was_given():
    def callback(context):
        context.record(count=np.int64(3), flag=True, rate=np.float32(0.5))

    event = Event(callback=callback, name="Recorder")
    event(EventContext(time=0.0), callback_only=True)

    assert event.records["count"] == [(0.0, 3)]
    assert event.records["flag"] == [(0.0, True)]
    assert event.records["flag"][0][1] is True
    assert event.records["rate"] == [(0.0, np.float32(0.5))]


def test_a_rollback_drops_the_records_made_after_its_time():
    event = Event(callback=lambda context: context.record(x=1.0), name="Recorder")
    for time in (0.5, 1.0, 1.8):
        event(EventContext(time=time), callback_only=True)
    flight = _flight_in_a_step({}, events=[event])

    rewind(flight, 1.5)

    assert event.records["x"] == [(0.5, 1.0), (1.0, 1.0)]


def test_an_event_keeps_what_its_callback_changed():
    controls, limits = _controls(_canard())

    def deploy(context):
        context.controls.canards.deflection = 0.1

    event = _run_callback(deploy, controls, limits)
    assert event.commands.control_changes == {"canards.deflection": 0.1}
    event.commands.reset()
    assert event.commands.control_changes == {}


def _view():
    from rocketpy.simulation.events.views import (  # pylint: disable=import-outside-toplevel
        ControlsView,
    )

    controls = {"canards.fin_1": 0.0, "canards.fin_2": 0.0, "air_brakes.level": 0.5}
    return controls, ControlsView(controls)


def test_a_view_shows_only_control_inputs_and_can_be_pointed_at_other_values():
    """What a continuous law reads: a dictionary that also holds the states
    of the controllables, by names without a dot, through one view that is
    pointed at the values of each evaluation of the equations of motion."""
    controls, view = _view()
    controls["servo_angle"] = 0.3
    assert list(view) == ["canards", "air_brakes"]
    assert "servo_angle" not in view
    other = {**controls, "canards.fin_1": 0.7}
    view._point_at(other)
    assert view.canards.fin_1 == 0.7
    view.canards.fin_2 = 0.2
    assert other["canards.fin_2"] == 0.2
    assert controls["canards.fin_2"] == 0.0


def test_a_channel_reads_and_writes_by_attribute():
    controls, view = _view()
    view.canards.fin_1 = 0.1
    view["canards"].fin_2 = -0.1  # without its controllable, a key is its own name

    assert controls == {
        "canards.fin_1": 0.1,
        "canards.fin_2": -0.1,
        "air_brakes.level": 0.5,
    }
    assert view.canards.fin_2 == -0.1
    assert view.air_brakes.level == 0.5


def test_a_name_that_is_not_a_channel_says_which_exist():
    controls, view = _view()
    with pytest.raises(AttributeError, match="canards, air_brakes|air_brakes, canards"):
        view.canard.fin_1 = 0.1
    with pytest.raises(AttributeError, match="fin_1, fin_2"):
        view.canards.fin_3 = 0.1
    with pytest.raises(AttributeError):
        view.canards = 0.1
    assert len(controls) == 3


def test_context_controls_writes_the_flight_controls():
    controls = {"canards.fin_1": 0.0}
    context = EventContext(controls=controls)
    context.controls.canards.fin_1 = 0.3
    assert controls == {"canards.fin_1": 0.3}


def test_the_context_does_not_hold_the_flight():
    """Users' functions get no way to the flight: its results read before it
    ends would be kept, covering only part of it."""
    assert not hasattr(EventContext, "flight")
    assert not hasattr(EventContext, "phase")


def test_context_memory_is_the_memory_of_the_event_being_checked():
    first = Event(callback=lambda context: None, memory={"a": 1})
    second = Event(callback=lambda context: None, memory={"b": 2})
    context = EventContext(time=0.0)
    context.bind(first)
    assert context.memory is first.memory
    context.bind(second)
    assert context.memory is second.memory


def test_context_commands_are_the_commands_of_the_event_being_checked():
    event = Event(callback=lambda context: context.commands.terminate_flight())
    context = EventContext(time=0.0)
    assert event(context, callback_only=True)
    assert context.commands is event.commands
    assert event.commands._terminate


def test_a_control_is_also_read_and_written_by_name():
    """For a name built in a loop: ``controls.canards[fin]``."""
    controls, view = _view()
    for fin in ("fin_1", "fin_2"):
        view.canards[fin] = 0.1
    assert controls["canards.fin_1"] == 0.1 and controls["canards.fin_2"] == 0.1
    assert view["canards"]["fin_1"] == 0.1
    with pytest.raises(KeyError, match="no control named 'fin_9'"):
        view.canards["fin_9"] = 0.1
    with pytest.raises(KeyError, match="no control named 'fin_9'"):
        _ = view.canards["fin_9"]


def test_a_rollback_restores_the_previous_check_and_the_enabled_flag():
    """An event run in the part of the step that is cut is as it was before:
    its next run does not see a check that never happened."""
    flight = _flight_in_a_step({"canards.deflection": 0.0})
    event = _writer("canards.deflection", 0.1, sampling_rate=10)
    _run(event, flight, 2.0)
    event._previous_time = 2.0
    event._enabled = False  # as a flight turns it off

    rewind(flight, 1.5)

    assert event._previous_time is None
    assert event.enabled is True


def test_a_control_named_as_a_method_of_the_view_is_read_by_name():
    part = Controllable("Valve", controls=("update",))
    view = ControlsView({"valve.update": 0.3}, [part])
    assert view.valve["update"] == 0.3
    assert "0.3" in repr(view.valve)


def test_one_control_name_given_alone_is_one_control():
    assert Controllable("Valve", controls="opening").controls == ("opening",)


def test_enable_event_queues_the_event_to_turn_on():
    commands, other = Commands(), object()
    commands.enable_event(other)
    assert commands.enable_events == [other]


def _context_at(time, vz, height=100.0):
    state = [0.0] * 13
    state[5] = vz
    return EventContext(
        time=time,
        canonical_state=state,
        raw_state=state,
        phase_names=(),
        height_agl=height,
    )


def test_the_apogee_trigger_holds_from_the_first_check_past_apogee():
    """Whatever the time between checks: it does not need the check to fall
    in the solver step where the apogee was crossed. Like the other named
    triggers it holds from then on. On the pad, at rest, it does not fire."""
    event = Event(lambda context: None, trigger="apogee", sampling_rate=1)
    checks = ((0, 0.0, 0.0), (1, 50.0, 30.0), (2, 5.0, 90.0), (3, -4.0, 91.0))
    fired = [event(_context_at(*check)) for check in (*checks, (4, -9.0, 85.0))]
    assert fired == [False, False, False, True, True]


def test_the_apogee_gates_hold_from_the_apogee_on():
    event = Event(lambda context: None, enabled=False, enable_on="apogee")
    assert event.enable_on(_context_at(1.0, 50.0)) is False
    assert event.enable_on(_context_at(3.0, -4.0)) is True
    assert event.enable_on(_context_at(9.0, -30.0)) is True
