from .event_commands import apply_event_commands, apply_rollback_command

def infer_step_size(flight, time):
    """Infer the elapsed step size for an event at ``time``.

    The solver's internal ``step_size`` is not reliable after rollback or
    interpolation-driven callbacks, so derive it from the latest accepted
    solution history instead.
    """
    if len(flight.solution) < 2:
        return 0.0
    return max(0.0, time - flight.solution[-2][0])

def build_event_kwargs(flight, time, state, step_size, phase, rollback=False):
    """Build the keyword arguments shared by event triggers and callbacks."""
    state_dot = phase.derivative(time, state)
    pressure = flight.env.pressure.get_value_opt(state[2])
    height_above_ground_level = flight.env.height_above_ground_level.get_value_opt(
        pressure
    )
    # Extract previous state vectors from solution history.
    # If rolling back, keep one extra sample so the history excludes the step
    # currently being re-evaluated.
    index = 2 if rollback else 1
    state_history = SafeStateHistory([sol[1:] for sol in flight.solution[:-index]])
    return {
        "time": time,
        "state": state,
        "state_dot": state_dot,
        "state_history": state_history,
        "sensors": flight.sensors,
        "sensors_by_name": flight.sensors_by_name,
        "environment": flight.env,
        "rocket": flight.rocket,
        "flight": flight,
        "phase": phase,
        "step_size": step_size,
        "pressure": pressure,
        "height_above_ground_level": height_above_ground_level,
    }

def update_event_kwargs(
    event_kwargs,
    time,
    state,
    state_dot,
    step_size,
    pressure,
    height_above_ground_level,
):
    """Update shared event kwargs with step-specific values."""
    event_kwargs.update(
        time=time,
        state=state,
        state_dot=state_dot,
        step_size=step_size,
        pressure=pressure,
        height_above_ground_level=height_above_ground_level,
    )
    return event_kwargs


def update_overshootable_event_kwargs(
    flight,
    phase,
    event_kwargs,
    interpolated_time,
    interpolated_state,
):
    """Refresh event kwargs for a specific overshootable node."""
    pressure = flight.env.pressure.get_value_opt(interpolated_state[2])
    height_above_ground_level = flight.env.height_above_ground_level.get_value_opt(
        pressure
    )
    state_dot = phase.derivative(interpolated_time, interpolated_state)
    return update_event_kwargs(
        event_kwargs,
        time=interpolated_time,
        state=interpolated_state,
        state_dot=state_dot,
        step_size=infer_step_size(flight, interpolated_time),
        pressure=pressure,
        height_above_ground_level=height_above_ground_level,
    )

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
            event_results=event.commands.results,
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


def call_events(flight, events, phase, phase_index, node_index, time, state, step_size):
    event_kwargs = build_event_kwargs(
        flight=flight,
        time=time,
        state=state,
        step_size=step_size,
        phase=phase,
    )

    for event in events:
        trigger_result = event._trigger_checked
        trigger_result = event(callback_only=trigger_result, **event_kwargs)
        if trigger_result:
            apply_event_commands(
                flight=flight,
                event=event,
                event_results=event.commands.results,
                phase=phase,
                phase_index=phase_index,
                node_index=node_index,
                command_time=event_kwargs["time"],
            )
        event._trigger_checked = False


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