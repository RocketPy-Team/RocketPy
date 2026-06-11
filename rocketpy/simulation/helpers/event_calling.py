from .event_commands import apply_event_commands, apply_rollback_command


def compute_needs_union(events):
    """Return the union of ``Event.needs`` for all enabled events."""
    result = frozenset()
    for event in events:
        if not event.enabled:
            continue
        result = result | event.needs
    return result

def infer_step_size(flight, time):
    """Infer the elapsed step size for an event at ``time``.

    The solver's internal ``step_size`` is not reliable after rollback or
    interpolation-driven callbacks, so derive it from the latest accepted
    solution history instead.
    """
    if len(flight.solution) < 2:
        return 0.0
    return max(0.0, time - flight.solution[-2][0])

def build_event_kwargs(flight, time, state, step_size, phase, rollback=False, needs=frozenset()):
    """Build the keyword arguments shared by event triggers and callbacks.

    Parameters
    ----------
    needs : frozenset of str, optional
        Union of ``Event.needs`` across all events that will consume the
        returned dict. Only keys present in ``needs`` are computed for the
        expensive values: ``state_dot``, ``pressure``, ``state_history``.
        Defaults to empty (compute nothing expensive).
    """
    kwargs = {
        "time": time,
        "state": state,
        "sensors": flight.sensors,
        "sensors_by_name": flight.sensors_by_name,
        "environment": flight.env,
        "rocket": flight.rocket,
        "flight": flight,
        "phase": phase,
        "step_size": step_size,
        "height_above_ground_level": state[2] - flight.env.elevation,
    }
    if "state_dot" in needs:
        kwargs["state_dot"] = phase.derivative(time, state)
    if "pressure" in needs:
        kwargs["pressure"] = flight.env.pressure.get_value_opt(state[2])
    if "state_history" in needs:
        index = 2 if rollback else 1
        kwargs["state_history"] = SafeStateHistory(
            [sol[1:] for sol in flight.solution[:-index]]
        )
    return kwargs


def update_overshootable_event_kwargs(
    flight,
    phase,
    event_kwargs,
    interpolated_time,
    interpolated_state,
    needs=frozenset(),
):
    """Refresh event kwargs for a specific overshootable node.

    Parameters
    ----------
    needs : frozenset of str, optional
        Union of ``Event.needs`` across all overshootable events at this node.
        Expensive values are skipped when absent from ``needs``. Defaults to
        empty (compute nothing expensive).
    """
    event_kwargs["time"] = interpolated_time
    event_kwargs["state"] = interpolated_state
    event_kwargs["step_size"] = infer_step_size(flight, interpolated_time)
    event_kwargs["height_above_ground_level"] = interpolated_state[2] - flight.env.elevation
    if "state_dot" in needs:
        event_kwargs["state_dot"] = phase.derivative(interpolated_time, interpolated_state)
    if "pressure" in needs:
        event_kwargs["pressure"] = flight.env.pressure.get_value_opt(interpolated_state[2])
    # state_history does not change per node — already set by build_event_kwargs
    return event_kwargs

def process_overshootable_event(
    flight,
    event,
    event_kwargs,
    phase,
    phase_index,
    node_index,
    rolled_back,
):
    """Evaluate one overshootable event and apply its side effects."""
    trigger_result = event(trigger_only=True, **event_kwargs)

    if not trigger_result:
        event._trigger_checked = False
        return rolled_back, False

    event._trigger_checked = True

    if not event.changes_dynamics:
        event(callback_only=True, reset=False, **event_kwargs)
        apply_event_commands(
            flight=flight,
            event=event,
            event_results=event.commands,
            phase=phase,
            phase_index=phase_index,
            node_index=node_index,
            command_time=event_kwargs["time"],
        )
        event._trigger_checked = False
        return rolled_back, False

    if not rolled_back:
        apply_rollback_command(flight, event_kwargs["time"], event_kwargs["state"])
        return True, True

    return rolled_back, False


def call_events(flight, events, phase, phase_index, node_index, time, state, step_size, needs=frozenset()):
    event_kwargs = build_event_kwargs(
        flight=flight,
        time=time,
        state=state,
        step_size=step_size,
        phase=phase,
        needs=needs,
    )

    nodes_modified = False
    for event in events:
        trigger_result = event._trigger_checked
        trigger_result = event(callback_only=trigger_result, **event_kwargs)
        if trigger_result:
            nodes_modified |= apply_event_commands(
                flight=flight,
                event=event,
                event_results=event.commands,
                phase=phase,
                phase_index=phase_index,
                node_index=node_index,
                command_time=event_kwargs["time"],
            )
        event._trigger_checked = False
    return nodes_modified


class SafeStateHistory(list):
    """Wrapper around state history list that provides helpful error messages
    when trying to access states that are not yet available.

    This is useful for callback functions that may try to access backward
    states (e.g., state_history[-2]) early in the simulation when insufficient
    history has been accumulated.
    """

    def __getitem__(self, index):
        """Override indexing to provide helpful error messages."""
        try:
            return super().__getitem__(index)
        except IndexError as e:
            if isinstance(index, int) and index < 0:
                # User tried to access a negative index (backward lookup)
                available = len(self)
                requested = abs(index)
                raise IndexError(
                    f"state_history does not have enough prior states. "
                    f"Requested state_history[{index}] (need {requested} prior states), "
                    f"but only {available} state(s) available. "
                    f"This commonly occurs early in simulation (near t=0). "
                    f"Consider checking len(state_history) first or using "
                    f"min(abs(index), len(state_history)) to safely access historical states."
                ) from e
            raise e
