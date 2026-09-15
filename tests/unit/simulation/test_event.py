import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.simulation.events import Event
from rocketpy.simulation.helpers.dynamics import (
    CANONICAL_STATE_NAMES,
    SIX_DOF_DYNAMICS,
    _PhaseDynamics,
)
from rocketpy.simulation.helpers.event_calling import (
    EventContext,
    build_event_kwargs,
)
from rocketpy.simulation.helpers.event_commands import apply_event_list_updates
from rocketpy.simulation.solution import Solution


def _canonical_solution(*rows):
    """Build a Solution holding the given canonical rows in one phase."""
    solution = Solution()
    solution._start_phase(
        SIX_DOF_DYNAMICS, start_canonical=tuple(rows[0][1:]) if rows else None
    )
    for row in rows:
        solution._append(list(row))
    return solution


def _context(**values):
    """Build the context an event is called with, from plain keyword values."""
    return EventContext(values)


def _always_true(_context):
    return True


def _trigger_on_positive_vz(context):
    return context["state"][5] > 0


def _callback_record_kwargs(context):
    return {
        "time": context["time"],
        "state": context["state"],
    }


def _callback_returns_int(_context) -> int:
    return 1


def _trigger_returns_int(_context) -> int:
    return 1


def _enable_on_raises(_context):
    raise RuntimeError("enable gate failed")


def _disable_on_raises(_context):
    raise RuntimeError("disable gate failed")


def _exact_time_function(context):
    return context["state"][5]


def _linear_interpolator(time):
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - 2.0 * time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


class _FakePhase:
    def __init__(self, interpolator):
        self.solver = SimpleNamespace(dense_output=lambda: interpolator)
        self.dynamics = SIX_DOF_DYNAMICS


def _make_exact_time_flight(previous_state, current_state, interpolator):
    flight = SimpleNamespace(
        solution=_canonical_solution(previous_state, current_state),
        env=SimpleNamespace(elevation=0.0),
    )
    phase = _FakePhase(interpolator)
    return flight, phase


def _sample_state(time, vz):
    return np.array(
        [time, 0.0, 0.0, 0.0, 0.0, 0.0, vz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_initialization_sets_sampling_interval_and_gate_presets():
    """Event construction should normalize sampling and gate conditions."""

    event = Event(
        callback=_callback_record_kwargs,
        sampling_rate=2.0,
        time_overshootable=True,
        disable_on=3.0,
        enable_on="burnout",
        memory={"count": 1},
    )

    assert event.is_discrete is True
    assert event.sampling_interval == pytest.approx(0.5)
    assert event.time_overshootable is True
    assert event.disable_on({"time": 2.9}) is False
    assert event.disable_on({"time": 3.0}) is True
    assert (
        event.enable_on(
            {
                "time": 1.0,
                "rocket": SimpleNamespace(motor=SimpleNamespace(burn_out_time=2.0)),
            }
        )
        is False
    )
    assert (
        event.enable_on(
            {
                "time": 2.0,
                "rocket": SimpleNamespace(motor=SimpleNamespace(burn_out_time=2.0)),
            }
        )
        is True
    )

    continuous_event = Event(callback=_callback_record_kwargs)
    assert continuous_event.time_overshootable is False


def test_trigger_accepts_string_presets_and_numeric_thresholds():
    """Trigger should support apogee/burn out presets and numeric time gates."""

    apogee_event = Event(
        callback=_callback_record_kwargs,
        trigger="apogee",
    )
    burnout_event = Event(
        callback=_callback_record_kwargs,
        trigger="burnout",
    )
    time_event = Event(
        callback=_callback_record_kwargs,
        trigger=3.0,
    )

    apogee_flight = SimpleNamespace(
        solution=_canonical_solution(_sample_state(0.0, 1.0), _sample_state(1.0, -1.0))
    )
    burnout_flight = SimpleNamespace(motor=SimpleNamespace(burn_out_time=2.0))

    assert apogee_event.trigger(
        _context(flight=apogee_flight, state=_sample_state(1.0, -1.0)[1:])
    )
    assert burnout_event.trigger(_context(time=2.0, rocket=burnout_flight))
    assert time_event.trigger(_context(time=3.0))
    assert time_event.trigger(_context(time=2.9)) is False


def test_trigger_rejects_unknown_preset():
    """Unknown trigger presets should fail fast during construction."""

    with pytest.raises(ValueError, match="Unknown trigger preset"):
        Event(callback=_callback_record_kwargs, trigger="unknown")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"callback": 1}, "Callback must be a callable."),
        (
            {"callback": lambda time, state: None},
            "Callback function must take the event context as its single",
        ),
        (
            {"callback": _callback_returns_int},
            "Callback function return annotation must be None, dict, or unspecified",
        ),
    ],
)
def test_callback_validation_rejects_invalid_callbacks(kwargs, message):
    """Invalid callbacks should fail fast during construction."""

    with pytest.raises(ValueError, match=message):
        Event(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"trigger": [], "callback": _callback_record_kwargs},
            "Trigger must be a callable, preset string, or number.",
        ),
        (
            {"trigger": lambda: True, "callback": _callback_record_kwargs},
            "Trigger function must take the event context as its single",
        ),
        (
            {"trigger": _trigger_returns_int, "callback": _callback_record_kwargs},
            "Trigger function return annotation must be bool when provided",
        ),
    ],
)
def test_trigger_validation_rejects_invalid_triggers(kwargs, message):
    """Invalid triggers should fail fast during construction."""

    with pytest.raises(ValueError, match=message):
        Event(**kwargs)


@pytest.mark.parametrize(
    ("exact_time_function", "message"),
    [
        (1, "exact_time_function must be callable or None."),
        (
            lambda time, state: state,
            "exact_time_function must take the event context as its single",
        ),
    ],
)
def test_exact_time_function_validation_rejects_invalid_signatures(
    exact_time_function,
    message,
):
    """Exact-time functions must follow the documented signature."""

    with pytest.raises(ValueError, match=message):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=exact_time_function,
        )


def test_exact_time_configuration_rejects_invalid_solver_and_sampling_mix():
    """Invalid exact-time configuration should be rejected."""

    with pytest.raises(ValueError, match="Unknown disable_on or enable_on preset"):
        Event(callback=_callback_record_kwargs, disable_on="invalid")

    with pytest.raises(
        TypeError,
        match="disable_on must be None, a string preset, a number, or a callable",
    ):
        Event(callback=_callback_record_kwargs, disable_on=[])

    with pytest.raises(ValueError, match="Unknown exact-time solver"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=_exact_time_function,
            exact_time_config={"solver": "unknown"},
        )

    Event(
        callback=_callback_record_kwargs,
        exact_time_function=_exact_time_function,
        exact_time_config={
            "solver": "cubic_hermite",
            "derivative_function": lambda context: 0.0,
        },
    )

    with pytest.raises(ValueError, match="requires \\['derivative_function'\\]"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=_exact_time_function,
            exact_time_config={"solver": "cubic_hermite"},
        )

    with pytest.raises(ValueError, match="Unknown exact_time_config keys"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=_exact_time_function,
            exact_time_config={"solver": "linear", "xtol": 1e-9},
        )

    with pytest.raises(ValueError, match="must take the event context"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=_exact_time_function,
            exact_time_config={
                "solver": "cubic_hermite",
                "derivative_function": lambda: 0.0,
            },
        )

    with pytest.raises(ValueError, match="only supported for continuous hooks"):
        Event(
            callback=_callback_record_kwargs,
            trigger=_always_true,
            sampling_rate=1.0,
            exact_time_function=_exact_time_function,
        )


def test_reset_restores_initial_runtime_state():
    """Reset should restore the construction-time snapshot."""

    event = Event(
        callback=_callback_record_kwargs,
        memory={"count": 1},
        enabled=False,
        sampling_rate=2.0,
    )
    event.memory["count"] = 99
    event.verbose_log.append({"time": 1.0})
    event.callback_log.append({"time": 1.0})
    event.triggered_times.append(1.0)
    event.enabled_times.append(2.0)
    event.disabled_times.append(3.0)
    event.commands.disable()
    event._trigger_checked = True
    event.enabled = True

    event.reset()

    assert event.enabled is False
    assert event.memory == {"count": 1}
    assert event.verbose_log == []
    assert event.callback_log == []
    assert event.triggered_times == []
    assert event.enabled_times == []
    assert event.disabled_times == []
    assert event.commands._disabled is None
    assert event._trigger_checked is False


def test_call_supports_trigger_only_callback_only_and_disable_commands():
    """Trigger-only calls should skip the callback, while callback-only calls
    should execute the callback and queue command results."""

    triggered = []

    def callback(context):
        triggered.append(context["time"])
        context["event"].commands.disable()
        return {"time": context["time"]}

    event = Event(
        callback=callback,
        trigger=_trigger_on_positive_vz,
        sampling_rate=5.0,
        trigger_only_once=True,
        disable_on=3.0,
    )

    trigger_only_result = event(
        _context(
            flight=SimpleNamespace(solution=[]),
            phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
            time=1.0,
            state=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
        ),
        trigger_only=True,
    )

    assert trigger_only_result is True
    assert triggered == []
    assert event.callback_log == []

    callback_only_result = event(
        _context(
            flight=SimpleNamespace(solution=[]),
            phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
            time=3.0,
            state=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
        ),
        callback_only=True,
    )

    assert callback_only_result is True
    assert triggered == [3.0]
    assert event.callback_log[0]["time"] == 3.0
    assert event.triggered_times == [3.0]
    assert event.commands._disabled is True


def test_enable_on_is_checked_before_the_trigger_and_disable_on_after():
    """When both gates are true on one check, the event fires and ends disabled."""

    event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
        enabled=False,
        enable_on=_always_true,
        disable_on=_always_true,
    )

    assert event(_context(time=1.0, state=[0.0] * 13)) is True
    assert len(event.callback_log) == 1
    # enable_on queued an enable, then disable_on queued a disable: the disable
    # is what the simulation applies
    assert event.commands._disabled is True


def test_call_returns_false_when_enable_gate_is_absent_or_raises():
    """A disabled event should stay disabled when no enable gate exists, and
    gate exceptions should be handled without crashing."""

    disabled_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
    )
    assert (
        disabled_event(
            _context(
                flight=SimpleNamespace(solution=[]),
                phase=SimpleNamespace(
                    derivative=lambda *_args, **_kwargs: np.zeros(13)
                ),
                time=1.0,
                state=np.zeros(13),
            )
        )
        is False
    )

    gated_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
        enable_on=_enable_on_raises,
    )
    with pytest.warns(UserWarning, match="Error evaluating enable_on"):
        assert (
            gated_event._call_enable_on(
                _context(
                    time=1.0,
                    flight=SimpleNamespace(solution=[]),
                    phase=SimpleNamespace(
                        derivative=lambda *_args, **_kwargs: np.zeros(13)
                    ),
                    state=np.zeros(13),
                )
            )
            is False
        )

    enabled_gate_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
        enable_on=lambda context: True,
    )
    assert (
        enabled_gate_event._call_enable_on(
            _context(
                time=1.0,
                flight=SimpleNamespace(solution=[]),
                phase=SimpleNamespace(
                    derivative=lambda *_args, **_kwargs: np.zeros(13)
                ),
                state=np.zeros(13),
            )
        )
        is None
    )
    assert enabled_gate_event.commands._disabled is False

    blocked_gate_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
        enable_on=lambda context: False,
    )
    assert (
        blocked_gate_event._call_enable_on(
            _context(
                time=1.0,
                flight=SimpleNamespace(solution=[]),
                phase=SimpleNamespace(
                    derivative=lambda *_args, **_kwargs: np.zeros(13)
                ),
                state=np.zeros(13),
            )
        )
        is False
    )

    trigger_phase_blocked_event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
    )
    trigger_phase_blocked_event._call_enable_on = lambda _context: False
    assert (
        trigger_phase_blocked_event(
            _context(
                flight=SimpleNamespace(solution=[]),
                phase=SimpleNamespace(
                    derivative=lambda *_args, **_kwargs: np.zeros(13)
                ),
                time=1.0,
                state=np.zeros(13),
            )
        )
        is False
    )


def test_call_returns_false_when_trigger_fails_and_disable_gate_handles_errors():
    """Trigger failures and disable-gate exceptions should be handled locally."""

    event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
        disable_on=_disable_on_raises,
    )

    assert (
        event._call_trigger(
            _context(
                time=1.0,
                flight=SimpleNamespace(solution=[]),
                phase=None,
                state=np.zeros(13),
            )
        )
        is True
    )

    false_trigger_event = Event(
        callback=_callback_record_kwargs,
        trigger=lambda context: False,
    )
    assert (
        false_trigger_event(
            _context(
                flight=SimpleNamespace(solution=[]),
                phase=SimpleNamespace(
                    derivative=lambda *_args, **_kwargs: np.zeros(13)
                ),
                time=1.0,
                state=np.zeros(13),
            )
        )
        is False
    )

    with pytest.warns(UserWarning, match="Error evaluating disable_on"):
        event._call_disable_on(
            _context(
                time=1.0,
                flight=SimpleNamespace(solution=[]),
                phase=None,
                state=np.zeros(13),
            )
        )


def test_call_refines_exact_time():
    """Successful exact-time solving should update callback kwargs with the
    refined time and state."""

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, -1.0)
    flight, phase = _make_exact_time_flight(
        previous_state=previous_state,
        current_state=current_state,
        interpolator=_linear_interpolator,
    )

    event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
        exact_time_function=_exact_time_function,
        exact_time_config={"solver": "linear"},
    )

    result = event(
        _context(flight=flight, phase=phase, time=1.0, state=current_state[1:])
    )

    assert result is True
    assert event.triggered_times == [pytest.approx(0.5)]
    assert event.callback_log[0]["time"] == pytest.approx(0.5)
    assert event.callback_log[0]["state"][5] == pytest.approx(0.0)
    assert event.commands.exact_time == pytest.approx(0.5)
    assert event.commands.exact_state[5] == pytest.approx(0.0)


def test_call_falls_back_to_sampled_time_when_exact_time_solver_fails():
    """If exact-time solving fails, the event should still fire at the sampled
    step time and emit warnings."""

    previous_state = _sample_state(0.0, 1.0)
    current_state = _sample_state(1.0, 1.0)
    flight, phase = _make_exact_time_flight(
        previous_state=previous_state,
        current_state=current_state,
        interpolator=_linear_interpolator,
    )

    event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
        exact_time_function=_exact_time_function,
        exact_time_config={"solver": "linear"},
    )

    with pytest.warns(UserWarning):
        result = event(
            _context(flight=flight, phase=phase, time=1.0, state=current_state[1:])
        )

    assert result is True
    assert event.triggered_times == [1.0]
    assert event.callback_log[0]["time"] == 1.0
    assert event.commands.exact_time is None
    assert event.commands.exact_state is None


def test_exact_time_falls_back_when_solution_history_is_short():
    """Without a stored step there is nothing to search, so the event fires at
    the sampled time with a warning."""

    event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
        exact_time_function=_exact_time_function,
        exact_time_config={"solver": "linear"},
    )
    flight = SimpleNamespace(solution=[_sample_state(0.0, 1.0)])
    phase = _FakePhase(_linear_interpolator)

    with pytest.warns(UserWarning, match="no solver step to search"):
        event(_context(flight=flight, phase=phase, time=0.0, state=np.zeros(13)))

    assert event.triggered_times == [0.0]
    assert event.commands.exact_time is None


def test_repr_and_str_include_key_configuration():
    """The string representations should expose the user-facing settings."""

    event = Event(
        callback=_callback_record_kwargs,
        name="Example",
        sampling_rate=4.0,
        trigger_only_once=True,
        time_overshootable=False,
    )

    representation = repr(event)
    string_value = str(event)

    assert "Example" in representation
    assert "sampling_rate=4.0" in representation
    assert "trigger_only_once=True" in representation
    assert "Example" in string_value
    assert "sampling_rate=4.0" in string_value
    assert "trigger_only_once=True" in string_value


def test_key_errors_inside_a_trigger_are_reraised():
    """A KeyError raised by the user's own code reaches the user unchanged."""

    def trigger_accesses_missing_dict_key(_context):
        d = {}
        return d["some_user_key"]  # unrelated KeyError

    event = Event(
        callback=_callback_record_kwargs,
        trigger=trigger_accesses_missing_dict_key,
    )

    with pytest.raises(KeyError, match="some_user_key"):
        event._call_trigger(_context(time=0.0, state=np.zeros(13)))


# ---------------------------------------------------------------------------
# previous_state: comparing an event against its own previous check
# ---------------------------------------------------------------------------


def _build_event_call_kwargs(time, state):
    """Minimal kwargs for calling an Event directly, without a Flight."""
    return {
        "time": time,
        "state": state,
        "raw_state": state,
        "flight": SimpleNamespace(),
        "rocket": SimpleNamespace(),
        "environment": SimpleNamespace(),
        "phase": SimpleNamespace(),
        "step_size": 0.01,
        "height_agl": state[2],
        "sensors": [],
        "sensors_by_name": {},
    }


def test_previous_state_is_none_on_first_check():
    seen = []

    def trigger(context):
        seen.append((context["previous_state"], context["previous_time"]))
        return False

    event = Event(callback=lambda context: None, trigger=trigger)
    event(EventContext(_build_event_call_kwargs(1.0, [0.0] * 13)))

    assert seen == [(None, None)]


def test_previous_state_tracks_the_previous_evaluation():
    seen = []

    def trigger(context):
        seen.append((context["previous_time"], context["previous_state"]))
        return False

    event = Event(callback=lambda context: None, trigger=trigger)
    first = [1.0] * 13
    second = [2.0] * 13
    event(EventContext(_build_event_call_kwargs(1.0, first)))
    event(EventContext(_build_event_call_kwargs(2.0, second)))

    assert seen[0] == (None, None)
    assert seen[1] == (1.0, first)


def test_previous_state_records_even_when_trigger_is_false():
    """A check that does not fire is still part of the sequence.

    Otherwise a crossing could be compared against a state from much earlier.
    """
    seen = []

    def trigger(context):
        seen.append(context["previous_time"])
        return False

    event = Event(callback=lambda context: None, trigger=trigger)
    for t in (1.0, 2.0, 3.0):
        event(EventContext(_build_event_call_kwargs(t, [t] * 13)))

    assert seen == [None, 1.0, 2.0]


def test_previous_state_detects_a_crossing_across_checks():
    """The apogee pattern: vertical velocity turning from positive to negative."""
    fired = []

    def trigger(context):
        previous = context["previous_state"]
        if previous is None:
            return False
        return previous[5] > 0 >= context["state"][5]

    event = Event(
        callback=lambda context: fired.append(context["time"]),
        trigger=trigger,
    )
    for t, vz in ((1.0, 5.0), (2.0, 1.0), (3.0, -1.0), (4.0, -5.0)):
        state = [0.0] * 13
        state[5] = vz
        event(EventContext(_build_event_call_kwargs(t, state)))

    assert fired == [3.0]


def test_reset_clears_previous_state():
    event = Event(callback=lambda context: None, trigger=lambda context: False)
    event(EventContext(_build_event_call_kwargs(1.0, [1.0] * 13)))
    assert event._previous_state is not None

    event.reset()

    assert event._previous_state is None
    assert event._previous_time is None


# ---------------------------------------------------------------------------
# Sampled checks on a step boundary
# ---------------------------------------------------------------------------


def test_sample_on_a_step_boundary_is_checked_exactly_once():
    """A sampled check landing on a step boundary must run once, in one step.

    Drives the real scheduling code in :class:`Flight` rather than restating
    what it is supposed to do, so that reverting either half of the fix fails
    here: the node list no longer carrying a skipped sentinel at the step end,
    or the consumer walking every node instead of dropping the last one.

    Uses a 4 Hz rate so the sampling times are exact in binary and the boundary
    genuinely coincides with a check.
    """
    # imported here only to keep Flight out of this module's import graph
    from rocketpy.simulation.flight import Flight  # pylint: disable=import-outside-toplevel

    checked = []

    def trigger(context):
        checked.append(round(context["time"], 7))
        return False

    event = Event(
        callback=lambda context: None,
        trigger=trigger,
        sampling_rate=4,  # a check every 0.25 s
    )

    def run_one_step(start, end):
        """Run Flight's sampled-event pass over a single solver step."""
        solution = Solution()
        solution._start_phase(
            SIX_DOF_DYNAMICS, tuple([0.0] * 13), t_start=0.0, name="test"
        )
        solution._append([start, *[start] * 13])
        solution._append([end, *[end] * 13])

        flight = SimpleNamespace(
            _overshootable_events=[event],
            solution=solution,
            t=end,
            y_sol=[end] * 13,
            sensors=[],
            sensors_by_name={},
            env=SimpleNamespace(elevation=0.0),
            rocket=SimpleNamespace(),
        )
        phase = SimpleNamespace(
            dynamics=SIX_DOF_DYNAMICS,
            solver=SimpleNamespace(
                step_size=end - start,
                dense_output=lambda: lambda t: [t] * 13,
            ),
        )
        # The real node builder, so both halves of the fix are under test: the
        # nodes this produces, and how the caller below walks them.
        flight._Flight__build_overshootable_nodes = lambda: (
            Flight._Flight__build_overshootable_nodes(flight)
        )
        Flight._Flight__process_overshootable_nodes(flight, phase, 0, 0)

    checked.clear()
    run_one_step(0.25, 0.5)
    earlier = list(checked)

    checked.clear()
    run_one_step(0.5, 0.75)
    later = list(checked)

    assert 0.5 in earlier, "boundary check lost by the step that ends on it"
    assert 0.5 not in later, "boundary check ran twice"
    assert earlier + later == [0.5, 0.75]


# ---------------------------------------------------------------------------
# Adding an event partway through the flight
# ---------------------------------------------------------------------------


def _flight_shell_for_added_events(records_post_values=False):
    """A stand-in Flight holding only what apply_event_list_updates reads."""
    solution = Solution()
    solution.records_post_values = records_post_values
    return SimpleNamespace(
        solution=solution,
        events=[],
        custom_events=[],
        _overshootable_events=[],
        _non_overshootable_events=[],
        time_overshoot=False,
    )


def _phase_shell():
    """A stand-in flight phase whose time nodes accept a new event."""
    return SimpleNamespace(
        time_nodes=SimpleNamespace(
            add_event=lambda *_args, **_kwargs: None,
            sort=lambda: None,
            merge=lambda: None,
        ),
        time_bound=10.0,
    )


def test_adding_an_event_that_changes_the_rocket_warns():
    """Recording is settled before the flight starts, so this arrives too late."""
    flight = _flight_shell_for_added_events()
    added = Event(
        callback=lambda context: None,
        trigger=_always_true,
        name="Late controller",
        changes_dynamics=True,
    )

    with pytest.warns(
        UserWarning, match="added with add_event after the flight started"
    ):
        apply_event_list_updates(
            flight,
            SimpleNamespace(new_events=[added]),
            _phase_shell(),
            1.0,
        )

    # the event is still added: the trajectory it produces is correct
    assert flight.events == [added]
    assert flight.custom_events == [added]


def test_adding_an_event_that_changes_the_rocket_is_quiet_when_already_recording():
    """Another such event was there from the start, so recording is already on."""
    flight = _flight_shell_for_added_events(records_post_values=True)
    added = Event(
        callback=lambda context: None,
        trigger=_always_true,
        name="Another controller",
        changes_dynamics=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_event_list_updates(
            flight, SimpleNamespace(new_events=[added]), _phase_shell(), 1.0
        )

    assert flight.events == [added]


def test_adding_an_ordinary_event_does_not_warn():
    flight = _flight_shell_for_added_events()
    added = Event(callback=lambda context: None, trigger=_always_true, name="Plain")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_event_list_updates(
            flight, SimpleNamespace(new_events=[added]), _phase_shell(), 1.0
        )

    assert flight.events == [added]


# ---------------------------------------------------------------------------
# The phase's own states, alongside the canonical ones
# ---------------------------------------------------------------------------


def test_phase_state_reports_the_states_the_phase_integrates():
    """A canonical phase reports the same thirteen, by name."""
    solution = _canonical_solution([0.0, *range(13)], [1.0, *range(1, 14)])
    flight = SimpleNamespace(
        solution=solution,
        sensors=[],
        sensors_by_name={},
        env=SimpleNamespace(elevation=0.0),
        rocket=None,
    )
    phase = SimpleNamespace(dynamics=lambda t, u: [2.0] * 13)
    context = build_event_kwargs(flight, 1.0, list(range(1, 14)), phase)

    assert context["phase_state"]["vz"] == 6.0
    assert set(context["phase_state"]) == set(CANONICAL_STATE_NAMES)
    assert context["phase_state_dot"]["vz"] == 2.0
    # the canonical view is unchanged and still a plain sequence
    assert list(context["state"]) == list(range(1, 14))


def test_phase_state_carries_a_state_the_canonical_view_cannot_hold():
    """A parafoil heading reaches the callback even though ``state`` has no slot."""
    parafoil = _PhaseDynamics(
        "parafoil",
        lambda _flight, _t, u, post_processing=False: [0.0] * 6 + [0.5],
        ("x", "y", "z", "vx", "vy", "vz", "heading"),
        initial_state=lambda _flight, _t, canonical: [*canonical[:6], 0.0],
    )
    solution = Solution()
    solution._start_phase(parafoil, start_canonical=tuple([0.0] * 13))
    solution._append([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25])
    flight = SimpleNamespace(
        solution=solution,
        sensors=[],
        sensors_by_name={},
        env=SimpleNamespace(elevation=0.0),
        rocket=None,
    )
    phase = SimpleNamespace(dynamics=lambda t, u: [0.0] * 6 + [0.5])
    raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]
    context = build_event_kwargs(flight, 0.0, raw, phase)

    # the heading is absent from the canonical state but present by name
    assert len(context["state"]) == 13
    assert context["phase_state"]["heading"] == 0.25
    assert context["phase_state_dot"]["heading"] == 0.5
    # and the states it does share are still there
    assert context["phase_state"]["vz"] == 6.0


def test_phase_state_dot_is_not_computed_unless_read():
    """It costs a full evaluation of the equations of motion, so it waits."""
    solution = _canonical_solution([0.0, *range(13)])
    flight = SimpleNamespace(
        solution=solution,
        sensors=[],
        sensors_by_name={},
        env=SimpleNamespace(elevation=0.0),
        rocket=None,
    )
    calls = []

    def dynamics(t, _u):
        calls.append(t)
        return [2.0] * 13

    phase = SimpleNamespace(dynamics=dynamics)
    context = build_event_kwargs(flight, 0.0, list(range(13)), phase)

    assert "phase_state_dot" not in context
    assert calls == []  # the equations of motion were never evaluated
    assert context["phase_state_dot"]["vz"] == 2.0
    assert calls == [0.0]  # evaluated once, when first read
    assert "phase_state_dot" in context  # and kept for the rest of the step


def test_both_derivative_views_share_one_evaluation():
    solution = _canonical_solution([0.0, *range(13)])
    flight = SimpleNamespace(
        solution=solution,
        sensors=[],
        sensors_by_name={},
        env=SimpleNamespace(elevation=0.0),
        rocket=None,
    )
    calls = []

    def dynamics(t, _u):
        calls.append(t)
        return [2.0] * 13

    phase = SimpleNamespace(dynamics=dynamics)
    context = build_event_kwargs(flight, 0.0, list(range(13)), phase)

    assert context["phase_state_dot"]["vz"] == 2.0
    assert list(context["state_dot"]) == [2.0] * 13
    assert len(calls) == 1  # reading both costs no more than reading one


def test_moving_the_context_drops_what_was_worked_out_for_the_old_moment():
    solution = _canonical_solution([0.0, *range(13)])
    flight = SimpleNamespace(
        solution=solution,
        sensors=[],
        sensors_by_name={},
        env=SimpleNamespace(
            elevation=0.0, pressure=SimpleNamespace(get_value_opt=lambda z: 10.0 * z)
        ),
        rocket=None,
    )
    phase = SimpleNamespace(dynamics=lambda t, u: [t] * 13)
    context = build_event_kwargs(flight, 0.0, list(range(13)), phase)
    assert context["pressure"] == 20.0
    assert context["state_dot"][0] == 0.0

    moved = context.at(1.0, [5.0] * 13)

    assert moved["pressure"] == 50.0
    assert moved["state_dot"][0] == 1.0
    assert moved["phase_state"]["x"] == 5.0
    # the original is untouched
    assert context["pressure"] == 20.0


# ---------------------------------------------------------------------------
# The shared event context
# ---------------------------------------------------------------------------


def _tracking_event(name):
    return Event(
        callback=lambda context: None,
        trigger=lambda context: False,
        name=name,
    )


def test_binding_swaps_the_values_that_belong_to_one_event():
    """Two events at the same node must not read each other's history."""
    context = _context(time=1.0, state=[0.0] * 13)
    watcher = _tracking_event("watcher")
    watcher._previous_time = 0.5
    plain = _tracking_event("plain")

    context.bind(watcher)
    assert context["event"] is watcher
    assert context["previous_time"] == 0.5

    context.bind(plain)
    assert context["event"] is plain
    # the previous event's history is gone, not inherited
    assert context["previous_time"] is None
    assert "previous_time" not in context


def test_owned_values_leave_with_the_event_that_added_them():
    context = _context(time=1.0, state=[0.0] * 13)
    first = _tracking_event("first")
    second = _tracking_event("second")

    context.bind(first)
    context.own(controller="mine")
    assert context["controller"] == "mine"

    context.bind(second)
    assert "controller" not in context


def test_expanding_the_context_gives_the_event_exactly_its_own_keys():
    seen = {}

    def trigger(context):
        seen.update(context)
        return False

    event = Event(callback=lambda context: None, trigger=trigger, name="reader")
    context = _context(time=1.0, state=[0.0] * 13)
    event(context)

    assert seen["time"] == 1.0
    assert seen["event"] is event
    assert seen["sampling_rate"] is event.sampling_rate
    # answered by the event when read, never stored in the shared context
    assert "previous_time" not in seen


def test_an_exact_time_event_does_not_rewrite_the_shared_context():
    """The refined time belongs to one event; its neighbours still see the step."""
    solution = _canonical_solution(_sample_state(0.0, 1.0), _sample_state(1.0, -1.0))
    flight = SimpleNamespace(solution=solution, env=SimpleNamespace(elevation=0.0))
    phase = _FakePhase(_linear_interpolator)
    seen = {}

    event = Event(
        callback=lambda context: seen.update(context),
        trigger=_always_true,
        name="refined",
        exact_time_function=lambda context: context["state"][5],
    )
    context = _context(
        time=1.0, state=_sample_state(1.0, -1.0)[1:], flight=flight, phase=phase
    )
    event(context)

    # the callback saw the refined time
    assert seen["time"] == pytest.approx(0.5)
    # but the context the next event will be given is untouched
    assert context["time"] == 1.0


def test_exact_time_function_sees_the_context_of_each_candidate_time():
    """Values worked out from the state, such as height_agl, follow the search
    and reach the callback at the exact time."""

    def altitude_interpolator(time):
        state = np.zeros(13)
        state[2] = 10.0 - 20.0 * time
        return state

    solution = _canonical_solution(
        [0.0, *altitude_interpolator(0.0)], [1.0, *altitude_interpolator(1.0)]
    )
    flight = SimpleNamespace(solution=solution, env=SimpleNamespace(elevation=0.0))
    phase = _FakePhase(altitude_interpolator)
    seen = {}

    event = Event(
        callback=lambda context: seen.update(context),
        trigger=_always_true,
        exact_time_function=lambda context: context["height_agl"],
        exact_time_config={"target": 4.0},
    )
    event(
        _context(
            time=1.0,
            state=altitude_interpolator(1.0),
            height_agl=-10.0,
            flight=flight,
            phase=phase,
        )
    )

    assert seen["time"] == pytest.approx(0.3)
    assert seen["height_agl"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Remembering the phase's own states between checks
# ---------------------------------------------------------------------------


def _phase_context(time, phase, heading):
    """A context carrying a phase and that phase's own states."""
    flight = SimpleNamespace(
        solution=SimpleNamespace(
            phases=[SimpleNamespace(dynamics=SimpleNamespace(states=("x", "heading")))]
        )
    )
    return _context(
        time=time,
        state=[0.0] * 13,
        raw_state=[0.0, heading],
        phase=phase,
        flight=flight,
    )


def test_previous_phase_state_follows_the_phase_states():
    """A parafoil can watch its own heading the way vz is watched canonically."""
    seen = []
    event = Event(
        callback=lambda context: None,
        trigger=lambda context: seen.append(context["previous_phase_state"]) or False,
    )
    phase = SimpleNamespace(name="parafoil")
    for time, heading in ((0.0, 0.25), (1.0, 0.75)):
        event(_phase_context(time, phase, heading))

    assert seen[0] is None  # nothing to compare against on the first check
    assert seen[1] == {"x": 0.0, "heading": 0.25}


def test_a_new_phase_starts_the_sequence_again():
    """The states a phase follows mean nothing in the phase that follows it."""
    seen = []
    event = Event(
        callback=lambda context: None,
        trigger=lambda context: seen.append(context["previous_phase_state"]) or False,
    )
    ascent = SimpleNamespace(name="ascent")
    parafoil = SimpleNamespace(name="parafoil")
    event(_phase_context(0.0, ascent, 0.25))
    event(_phase_context(1.0, ascent, 0.5))
    event(_phase_context(2.0, parafoil, 0.75))
    event(_phase_context(3.0, parafoil, 1.0))

    assert seen[0] is None  # first check of the flight
    assert seen[1] == {"x": 0.0, "heading": 0.25}
    assert seen[2] is None  # first check of the new phase
    assert seen[3] == {"x": 0.0, "heading": 0.75}


def test_the_canonical_previous_state_survives_a_phase_change():
    """Every phase reports the canonical states, so that sequence never breaks."""
    seen = []
    event = Event(
        callback=lambda context: None,
        trigger=lambda context: seen.append(context["previous_state"]) or False,
    )
    event(_phase_context(0.0, SimpleNamespace(name="ascent"), 0.25))
    event(_phase_context(1.0, SimpleNamespace(name="parafoil"), 0.75))

    assert seen[0] is None
    assert seen[1] == [0.0] * 13


def test_resetting_an_event_forgets_the_phase_it_was_in():
    event = Event(
        callback=lambda context: None,
        trigger=lambda context: False,
    )
    phase = SimpleNamespace(name="ascent")
    event(_phase_context(0.0, phase, 0.25))
    assert event._previous_raw_state is not None

    event.reset()

    assert event._previous_raw_state is None
    assert event._previous_phase is None


# ---------------------------------------------------------------------------
# Exact-time solving on a state the canonical thirteen cannot hold
# ---------------------------------------------------------------------------


PARAFOIL_EXACT_STATES = ("x", "y", "z", "vx", "vy", "vz", "heading")


def _parafoil_phase():
    """A phase whose raw state carries a heading that turns through zero."""

    def raw_interpolator(t):
        # position and velocity hold still; the heading crosses zero at t = 1.5
        return [0.0, 0.0, 100.0, 0.0, 0.0, -1.0, 1.5 - t]

    def bound_dynamics(_t, _u):
        return [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]

    parafoil = _PhaseDynamics(
        "parafoil_exact",
        lambda _flight, t, u, post_processing=False: bound_dynamics(t, u),
        PARAFOIL_EXACT_STATES,
        initial_state=lambda _flight, _t, canonical: [*canonical[:6], 0.0],
    )
    solution = Solution()
    solution._start_phase(parafoil, start_canonical=tuple([0.0] * 13))
    solution._append([1.0, *raw_interpolator(1.0)])
    solution._append([2.0, *raw_interpolator(2.0)])

    flight = SimpleNamespace(solution=solution, env=SimpleNamespace(elevation=0.0))
    phase = SimpleNamespace(
        solver=SimpleNamespace(dense_output=lambda: raw_interpolator),
        dynamics=bound_dynamics,
    )
    return flight, phase


def test_exact_time_can_solve_on_a_phase_state():
    """A parafoil heading is not in ``state``, but it can still time an event."""
    event = Event(
        callback=lambda context: None,
        trigger=_always_true,
        name="Heading zero",
        exact_time_function=lambda context: context["phase_state"]["heading"],
    )
    flight, phase = _parafoil_phase()
    result = event._compute_exact_time(
        _context(time=2.0, state=[0.0] * 13, flight=flight, phase=phase)
    )

    assert result["time"] == pytest.approx(1.5, abs=1e-9)
    assert result["phase_state"]["heading"] == pytest.approx(0.0, abs=1e-9)


def test_the_phase_view_moves_with_the_search():
    """A frozen snapshot would answer the same at every candidate time."""
    seen = []

    def heading(context):
        seen.append(context["phase_state"]["heading"])
        return context["phase_state"]["heading"]

    event = Event(
        callback=lambda context: None,
        trigger=_always_true,
        name="Heading zero",
        exact_time_function=heading,
        exact_time_config={"solver": "brentq"},
    )
    flight, phase = _parafoil_phase()
    event._compute_exact_time(
        _context(time=2.0, state=[0.0] * 13, flight=flight, phase=phase)
    )

    assert len(set(seen)) > 2, "the heading did not follow the root finder"


def test_exact_time_rates_are_only_worked_out_when_read():
    """Their rates cost an evaluation of the equations of motion each time."""
    flight, phase = _parafoil_phase()
    calls = []
    bound_dynamics = phase.dynamics
    phase.dynamics = lambda t, u: calls.append(t) or bound_dynamics(t, u)

    without = Event(
        callback=lambda context: None,
        trigger=_always_true,
        exact_time_function=lambda context: context["phase_state"]["heading"],
    )
    without._compute_exact_time(
        _context(time=2.0, state=[0.0] * 13, flight=flight, phase=phase)
    )
    assert calls == []

    seen = []
    with_rates = Event(
        callback=lambda context: None,
        trigger=_always_true,
        exact_time_function=lambda context: (
            seen.append(context["phase_state_dot"]["heading"])
            or context["phase_state"]["heading"]
        ),
    )
    with_rates._compute_exact_time(
        _context(time=2.0, state=[0.0] * 13, flight=flight, phase=phase)
    )
    assert seen and all(rate == -1.0 for rate in seen)
    assert len(calls) == len(seen)


def test_the_ends_of_the_step_are_read_from_the_stored_rows():
    """The ends were flown and saved, so they are read rather than estimated."""
    seen = []

    def heading(context):
        seen.append(context["phase_state"]["heading"])
        return context["phase_state"]["heading"]

    event = Event(
        callback=lambda context: None,
        trigger=_always_true,
        name="Heading zero",
        exact_time_function=heading,
        exact_time_config={"solver": "linear"},
    )
    flight, phase = _parafoil_phase()
    # make the interpolator disagree with what was stored, so the two sources
    # can be told apart
    stored = [
        row[-1] for row in (flight.solution.raw_row(-2), flight.solution.raw_row(-1))
    ]
    phase.solver.dense_output = lambda: (
        lambda t: [0.0, 0.0, 100.0, 0.0, 0.0, -1.0, 99.0]
    )

    event._compute_exact_time(
        _context(time=2.0, state=[0.0] * 13, flight=flight, phase=phase)
    )

    assert seen == stored, "the ends came from the interpolator, not the rows"
