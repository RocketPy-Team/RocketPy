import warnings

from ..events import Event
from .dynamics import _BoundDynamics


def apply_event_commands(
    flight,
    event,
    event_results,
    phase,
    phase_index,
    node_index,
    command_time,
):
    """Apply the command results returned by an event solver.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event : Event
        Event that produced the results.
    event_results : dict
        Result payload returned by the event system.
    phase : _FlightPhase
        Current flight phase.
    phase_index : int
        Index of the current flight phase.
    node_index : int
        Index of the current time node.
    command_time : float
        The time the commands apply at, in seconds. Replaced by the event's
        exact time when it solved one.
    """
    t_apply = command_time
    if event_results.exact_time is not None:
        t_apply = event_results.exact_time
        apply_exact_time_result(flight, event_results)

    apply_new_phase_or_dynamics(
        flight, event_results, phase, phase_index, node_index, time=t_apply
    )
    apply_termination(
        flight, event_results, phase, phase_index, node_index, time=t_apply
    )

    apply_event_list_updates(flight, event_results, phase, time=t_apply)
    apply_enable_commands(flight, event_results, node_index, event, phase, time=t_apply)
    apply_disable_commands(
        flight, event_results, node_index, event, phase, time=t_apply
    )


def apply_rollback_command(flight, time, state):
    """Apply a rollback request returned by an event trigger.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    time : float
        Interpolated simulation time to restore.
    state : array_like
        Interpolated flight state vector to restore.

    Returns
    -------
    None
        This function updates flight state/history in place.
    """
    flight.t = time
    flight.y_sol = state
    flight.solution._replace_last([time, *state])


def apply_disable_commands(_, event_results, node_index, event, phase, time):
    """Apply disable commands returned by an event solver.

    Parameters
    ----------
    _ : Flight
        Flight instance (unused; accepted for a uniform command signature).
    event_results : dict
        Result payload returned by the event system.
    node_index : int
        Index of the current time node.
    event : Event
        Event currently being processed.
    phase : _FlightPhase
        Current flight phase.
    time : float
        Simulation time at which the events are disabled.

    """
    if event_results.disable_events:
        disable_events = event_results.disable_events
        if not isinstance(disable_events, (list, tuple)):
            disable_events = [disable_events]

        for event_to_disable in disable_events:
            event_to_disable.enabled = False
            event_to_disable.disabled_times.append(time)
            _safe_disable_time_nodes_event(
                phase=phase,
                node_index=node_index,
                event=event_to_disable,
                time=time,
            )

    if event_results._disabled:
        event.enabled = False
        event.disabled_times.append(time)
        _safe_disable_time_nodes_event(
            phase=phase,
            node_index=node_index,
            event=event,
            time=time,
        )


def apply_enable_commands(flight, event_results, node_index, event, phase, time):
    """Apply enable commands returned by an event solver.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : dict
        Result payload returned by the event system.
    node_index : int
        Index of the current time node.
    event : Event
        Event currently being processed.
    phase : _FlightPhase
        Current flight phase.
    time : float
        Simulation time at which the events are enabled.

    """
    if event_results.enable_events:
        enable_events = event_results.enable_events
        if not isinstance(enable_events, (list, tuple)):
            enable_events = [enable_events]

        for event_to_enable in enable_events:
            event_to_enable.enabled = True
            event_to_enable.enabled_times.append(time)
            if event_to_enable in flight._non_overshootable_events:
                _safe_enable_time_nodes_event(
                    phase=phase,
                    node_index=node_index,
                    event=event_to_enable,
                    time=time,
                )

    # Commands API uses `_disabled` flag: True -> disable, False -> enable
    if event_results._disabled is False:
        event.enabled = True
        event.enabled_times.append(time)

        # if the event is non-overshootable, we need to create discrete nodes
        if event in flight._non_overshootable_events:
            _safe_enable_time_nodes_event(
                phase=phase,
                node_index=node_index,
                event=event,
                time=time,
            )


def apply_exact_time_result(flight, event_results):
    """Store the row at the exact time an event happened.

    When the event changes what happens after it (a new phase, new equations of
    motion, or the end of the flight), the part of the step flown past the event
    no longer happened, so the exact row takes the place of the step's end and
    the flight carries on from it. Otherwise the step stands as flown and the
    exact row is added just before its end.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : Commands
        The commands the event queued, holding its exact time and raw state.
    """
    time = event_results.exact_time
    state = event_results.exact_state
    solution = flight.solution
    previous_time = solution.raw_row(-2)[0] if len(solution) > 1 else None
    if event_results.changes_trajectory:
        if time == previous_time:
            # The event happened right where the step began, which is stored.
            solution._drop_last()
        else:
            solution._replace_last([time, *state])
        flight.t = time
        flight.y_sol = state
    elif previous_time < time < solution.raw_row(-1)[0]:
        solution._insert_before_last([time, *state])
        # Recorded now, with the rocket as it was when the step was flown: an
        # event later in this step may change it before the step is recorded.
        flight._post_process_row(-2)  # pylint: disable=protected-access


def bind_new_dynamics(flight, event_results):
    """Attach the dynamics an event asked for to the flight that is running.

    An event names the equations it wants to switch to without knowing anything
    about binding, so the two accepted forms are resolved here, where the flight
    is at hand:

    - a :class:`_PhaseDynamics`, such as one of the built-in phases, is bound to
      this flight;
    - dynamics already bound to a flight, such as ``flight.u_dot_generalized``
      (which is whichever ascent equations this flight was configured with), is
      re-bound so that any values the event supplied come along.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : Commands
        The commands the event queued.

    Returns
    -------
    Bound dynamics, or ``None`` if the event did not ask for a change.
    """
    dynamics = event_results.new_dynamics
    if dynamics is None:
        return None
    if isinstance(dynamics, _BoundDynamics):
        dynamics = dynamics.dynamics
    return dynamics.bind(flight, **event_results.new_dynamics_kwargs)


def apply_new_phase_or_dynamics(
    flight, event_results, phase, phase_index, node_index, time
):
    """Apply a flight-phase transition or a change of the equations of motion.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : dict
        Result payload returned by the event system.
    phase : _FlightPhase
        Current flight phase.
    phase_index : int
        Index of the current flight phase.
    node_index : int
        Index of the current time node.
    time : float
        Simulation time at which the new phase or derivative takes effect.

    Returns
    -------
    None
        This function mutates the flight phase list and time-node state in place.
    """
    if event_results.new_flight_phase is None and event_results.new_dynamics is None:
        return

    when_time = time
    lag = event_results.new_flight_phase_lag

    # No new dynamics means the new phase keeps flying the current equations.
    dynamics = bind_new_dynamics(flight, event_results) or phase.dynamics

    # Inserting the phase also moves the current phase's end to its start.
    flight.flight_phases.add_phase(
        when_time + lag,
        dynamics=dynamics,
        index=phase_index + 1,
        name=event_results.new_flight_phase_name,
    )

    # Rollback solution to the trigger time
    apply_rollback_command(flight, when_time, flight.solution.raw_row(-1)[1:])

    if lag == 0:
        # Prepare to leave loops and start new flight phase
        phase.time_nodes.flush_after(node_index)
        phase.time_nodes.add_node(when_time, [])
        phase.solver.status = "finished"
    else:
        # The current phase keeps flying its own equations until the new one
        # begins. Its schedule now ends there; the solver loop restarts the
        # solver from the rolled-back state.
        phase.time_nodes.truncate(phase.time_bound)


def apply_termination(flight, event_results, phase, phase_index, node_index, time):
    """Apply flight termination results.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : dict
        Result payload returned by the event system.
    phase : _FlightPhase
        Current flight phase.
    phase_index : int
        Index of the current flight phase.
    node_index : int
        Index of the current time node.
    time : float
        Simulation time at which the flight is terminated.

    Returns
    -------
    None
        This function mutates the flight and phase objects in place when the
        event requests termination.
    """
    if not event_results._terminate:
        return

    when_time = time

    flight.t_final = when_time
    phase.solver.status = "finished"

    # Set last flight phase
    flight.flight_phases.flush_after(phase_index)
    flight.flight_phases.add_phase(when_time, name="event_termination_phase")

    # Prepare to leave loops and start new flight phase
    phase.time_nodes.flush_after(node_index)
    phase.time_nodes.add_node(when_time, [])
    phase.solver.status = "finished"


def apply_event_list_updates(flight, event_results, phase, time):
    """Update the event lists based on event solver results.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : dict
        Result payload returned by the event system.
    phase : _FlightPhase
        Current flight phase.
    time : float
        Simulation time at which the new events are scheduled.

    """
    if event_results.new_events:
        # Normalize new_events to always be a list for consistent iteration
        new_events = event_results.new_events
        if isinstance(new_events, Event):
            new_events = [new_events]

        for new_event in new_events:
            if new_event.changes_dynamics and not flight.solution.records_post_values:
                warnings.warn(
                    f"Event '{new_event.name}' has changes_dynamics=True but was "
                    f"added with add_event after the flight started. The "
                    f"trajectory is correct, but the accelerations, aerodynamic "
                    f"forces and moments and net thrust (flight.ax, flight.R1, "
                    f"flight.net_thrust, ...) will be wrong. Fix: build the "
                    f"Flight with an event that has changes_dynamics=True in "
                    f"custom_events (a disabled placeholder event is enough).",
                    UserWarning,
                )
            flight.events.append(new_event)
            flight.custom_events.append(new_event)
            is_new_event_overshootable = (
                flight.time_overshoot
                and new_event.sampling_rate is not None
                and new_event.time_overshootable
            )
            if is_new_event_overshootable:
                flight._overshootable_events.append(new_event)
            else:
                flight._non_overshootable_events.append(new_event)
                phase.time_nodes.add_event(new_event, time, phase.time_bound)
                phase.time_nodes.sort()
                phase.time_nodes.merge()

        flight._overshootable_events.sort(key=lambda x: x.priority)
        flight._non_overshootable_events.sort(key=lambda x: x.priority)
        return True

    return False


def _safe_disable_time_nodes_event(phase, node_index, event, time):
    """Disable an event in the time-node schedule without raising on repeats."""
    if event.sampling_rate is None:
        if event not in phase.time_nodes.continuous_events:
            warnings.warn(
                (
                    f"Event '{event.name}' was requested to disable at t={time}, "
                    "but it was already disabled."
                ),
                UserWarning,
            )
            return

    try:
        phase.time_nodes.disable_event(node_index, event)
    except ValueError:
        warnings.warn(
            (
                f"Event '{event.name}' was requested to disable at t={time}, "
                "but it was already disabled."
            ),
            UserWarning,
        )


def _safe_enable_time_nodes_event(phase, node_index, event, time):
    """Enable an event in the time-node schedule without duplicating nodes."""
    if event.sampling_rate is None:
        if event in phase.time_nodes.continuous_events:
            warnings.warn(
                (
                    f"Event '{event.name}' was requested to enable at t={time}, "
                    "but it was already enabled."
                ),
                UserWarning,
            )
            return
    else:
        active = any(
            event in node.events for node in phase.time_nodes.list[node_index + 1 :]
        )
        if active:
            warnings.warn(
                (
                    f"Event '{event.name}' was requested to enable at t={time}, "
                    "but it was already enabled."
                ),
                UserWarning,
            )
            return

    try:
        phase.time_nodes.enable_event(node_index, event)
    except ValueError:
        warnings.warn(
            (
                f"Event '{event.name}' was requested to enable at t={time}, "
                "but it could not be scheduled."
            ),
            UserWarning,
        )
