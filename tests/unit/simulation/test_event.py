from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.simulation.events import Event


def _always_true(**_kwargs):
    return True


def _trigger_on_positive_vz(**kwargs):
    return kwargs["state"][5] > 0


def _callback_record_kwargs(**kwargs):
    return {
        "time": kwargs["time"],
        "sampled_time": kwargs.get("sampled_time"),
        "state": kwargs["state"],
    }


def _callback_returns_int(**_kwargs) -> int:
    return 1


def _trigger_returns_int(**_kwargs) -> int:
    return 1


def _enable_on_raises(**_kwargs):
    raise RuntimeError("enable gate failed")


def _disable_on_raises(**_kwargs):
    raise RuntimeError("disable gate failed")


def _exact_time_function(state, **_kwargs):
    return state[5]


def _linear_interpolator(time):
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - 2.0 * time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


class _FakePhase:
    def __init__(self, interpolator):
        self.solver = SimpleNamespace(dense_output=lambda: interpolator)

    def derivative(self, _time, state, post_processing=False):
        _ = post_processing
        return np.zeros_like(state, dtype=float)


def _make_exact_time_flight(previous_state, current_state, interpolator):
    flight = SimpleNamespace(solution=[previous_state, current_state])
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
        context={"count": 1},
    )

    assert event.is_discrete is True
    assert event.sampling_interval == pytest.approx(0.5)
    assert event.time_overshootable is True
    assert event.disable_on(time=2.9) is False
    assert event.disable_on(time=3.0) is True
    assert event.enable_on(
        time=1.0,
        rocket=SimpleNamespace(motor=SimpleNamespace(burn_out_time=2.0)),
    ) is False
    assert event.enable_on(
        time=2.0,
        rocket=SimpleNamespace(motor=SimpleNamespace(burn_out_time=2.0)),
    ) is True

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

    apogee_flight = SimpleNamespace(solution=[_sample_state(0.0, 1.0), _sample_state(1.0, -1.0)])
    burnout_flight = SimpleNamespace(motor=SimpleNamespace(burn_out_time=2.0))

    assert apogee_event.trigger(flight=apogee_flight, state=_sample_state(1.0, -1.0)[1:])
    assert burnout_event.trigger(time=2.0, rocket=burnout_flight)
    assert time_event.trigger(time=3.0)
    assert time_event.trigger(time=2.9) is False


def test_trigger_rejects_unknown_preset():
    """Unknown trigger presets should fail fast during construction."""

    with pytest.raises(ValueError, match="Unknown trigger preset"):
        Event(callback=_callback_record_kwargs, trigger="unknown")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"callback": 1}, "Callback must be a callable."),
        (
            {"callback": lambda time: None},
            "Callback function must accept arbitrary keyword arguments",
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
            {"trigger": lambda time: True, "callback": _callback_record_kwargs},
            "Trigger function must accept arbitrary keyword arguments",
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
            lambda time, **kwargs: time,
            "exact_time_function must accept 'state' as its first argument.",
        ),
        (
            lambda state: state,
            "exact_time_function must accept arbitrary keyword arguments",
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

    with pytest.raises(TypeError, match="disable_on must be None, a string preset, a number, or a callable"):
        Event(callback=_callback_record_kwargs, disable_on=[])

    with pytest.raises(ValueError, match="Unknown exact-time solver"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=_exact_time_function,
            exact_time_config={"solver": "unknown"},
        )

    cubic_hermite_event = Event(
        callback=_callback_record_kwargs,
        exact_time_function=_exact_time_function,
        exact_time_config={
            "solver": "cubic_hermite",
            "derivative_function": lambda state, **kwargs: 0.0,
        },
    )
    assert cubic_hermite_event.exact_time_solver.__name__ == "solve_exact_time_cubic_hermite"

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
        context={"count": 1},
        enabled=False,
        sampling_rate=2.0,
    )
    event.context["count"] = 99
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
    assert event.context == {"count": 1}
    assert event.verbose_log == []
    assert event.callback_log == []
    assert event.triggered_times == []
    assert event.enabled_times == []
    assert event.disabled_times == []
    assert event.commands.results["disable"] is None
    assert event._trigger_checked is False


def test_call_supports_trigger_only_callback_only_and_disable_commands():
    """Trigger-only calls should skip the callback, while callback-only calls
    should execute the callback and queue command results."""

    triggered = []

    def callback(**kwargs):
        triggered.append(kwargs["time"])
        kwargs["event"].commands.disable()
        return {"time": kwargs["time"]}

    event = Event(
        callback=callback,
        trigger=_trigger_on_positive_vz,
        sampling_rate=5.0,
        trigger_only_once=True,
        disable_on=3.0,
    )

    trigger_only_result = event(
        trigger_only=True,
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        time=1.0,
        state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    assert trigger_only_result is True
    assert triggered == []
    assert event.callback_log == []

    callback_only_result = event(
        callback_only=True,
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        time=3.0,
        state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    assert callback_only_result is True
    assert triggered == [3.0]
    assert event.callback_log[0]["time"] == 3.0
    assert event.triggered_times == [3.0]
    assert event.commands.results["disable"] is True


def test_call_returns_false_when_enable_gate_is_absent_or_raises():
    """A disabled event should stay disabled when no enable gate exists, and
    gate exceptions should be handled without crashing."""

    disabled_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
    )
    assert disabled_event(
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        time=1.0,
        state=np.zeros(13),
    ) is False

    gated_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
        enable_on=_enable_on_raises,
    )
    with pytest.warns(UserWarning, match="Error evaluating enable_on"):
        assert gated_event._call_enable_on(
            time=1.0,
            flight=SimpleNamespace(solution=[]),
            phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
            state=np.zeros(13),
        ) is False

    enabled_gate_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
        enable_on=lambda **kwargs: True,
    )
    assert enabled_gate_event._call_enable_on(
        time=1.0,
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        state=np.zeros(13),
    ) is None
    assert enabled_gate_event.commands.results["disable"] is False

    blocked_gate_event = Event(
        callback=_callback_record_kwargs,
        enabled=False,
        enable_on=lambda **kwargs: False,
    )
    assert blocked_gate_event._call_enable_on(
        time=1.0,
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        state=np.zeros(13),
    ) is False

    trigger_phase_blocked_event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
    )
    trigger_phase_blocked_event._call_enable_on = lambda **kwargs: False
    assert trigger_phase_blocked_event(
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        time=1.0,
        state=np.zeros(13),
    ) is False


def test_call_returns_false_when_trigger_fails_and_disable_gate_handles_errors():
    """Trigger failures and disable-gate exceptions should be handled locally."""

    event = Event(
        callback=_callback_record_kwargs,
        trigger=_always_true,
        disable_on=_disable_on_raises,
    )

    assert event._call_trigger(time=1.0, flight=SimpleNamespace(solution=[]), phase=None, state=np.zeros(13)) is True

    false_trigger_event = Event(
        callback=_callback_record_kwargs,
        trigger=lambda **kwargs: False,
    )
    assert false_trigger_event(
        flight=SimpleNamespace(solution=[]),
        phase=SimpleNamespace(derivative=lambda *_args, **_kwargs: np.zeros(13)),
        time=1.0,
        state=np.zeros(13),
    ) is False

    with pytest.warns(UserWarning, match="Error evaluating disable_on"):
        event._call_disable_on(time=1.0, flight=SimpleNamespace(solution=[]), phase=None, state=np.zeros(13))


def test_call_refines_exact_time_and_tracks_sampled_values():
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
        flight=flight,
        phase=phase,
        time=1.0,
        state=current_state[1:],
    )

    assert result is True
    assert event.triggered_times == [pytest.approx(0.5)]
    assert event.callback_log[0]["time"] == pytest.approx(0.5)
    assert event.callback_log[0]["sampled_time"] == pytest.approx(1.0)
    assert event.callback_log[0]["state"][5] == pytest.approx(0.0)
    assert event.commands.results["exact_time"] == pytest.approx(0.5)
    assert event.commands.results["exact_state"][5] == pytest.approx(0.0)


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
            flight=flight,
            phase=phase,
            time=1.0,
            state=current_state[1:],
        )

    assert result is True
    assert event.triggered_times == [1.0]
    assert event.callback_log[0]["time"] == 1.0
    assert event.callback_log[0]["sampled_time"] is None
    assert event.commands.results["exact_time"] is None
    assert event.commands.results["exact_state"] is None


def test_compute_exact_time_returns_none_when_solution_history_is_short():
    """Exact-time solving should be skipped until there are at least two solution points."""

    event = Event(
        callback=_callback_record_kwargs,
        exact_time_function=_exact_time_function,
        exact_time_config={"solver": "linear"},
        verbose=True,
    )
    flight = SimpleNamespace(solution=[_sample_state(0.0, 1.0)])
    phase = _FakePhase(_linear_interpolator)

    assert event._compute_exact_time(flight=flight, phase=phase, time=0.0, state=np.zeros(13)) is None
    assert event.verbose_log[-1]["skip_reason"].startswith(
        "Trigger condition met, but callback was not executed"
    )


def test_exact_time_validation_rejects_no_params_and_non_positional_state():
    """Exact-time functions must provide a state parameter and kwargs."""

    with pytest.raises(ValueError, match="must accept a mandatory 'state' argument"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=lambda: 0.0,
        )

    def keyword_only_state(*, state, **kwargs):
        return state

    with pytest.raises(ValueError, match="must accept 'state' as its first argument"):
        Event(
            callback=_callback_record_kwargs,
            exact_time_function=keyword_only_state,
        )


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