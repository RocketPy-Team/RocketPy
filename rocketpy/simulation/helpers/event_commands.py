import warnings

from ..events import Event

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

    Returns
    -------
    None
        This function updates the flight, phase, and event objects in place.
    """
    apply_exact_time_result(flight, event_results)
    t_apply = event_results.exact_time
    if t_apply is None:
        t_apply = command_time

    apply_new_phase_or_derivative(
        flight, event_results, phase, phase_index, node_index, time=t_apply
    )
    apply_termination(
        flight, event_results, phase, phase_index, node_index, time=t_apply
    )

    # Track whether time_nodes was modified by the following functions
    nodes_modified = False
    nodes_modified |= apply_event_list_updates(
        flight, event_results, phase, time=t_apply
    )
    nodes_modified |= apply_enable_commands(
        flight, event_results, node_index, event, phase, time=t_apply
    )
    nodes_modified |= apply_disable_commands(
        flight, event_results, node_index, event, phase, time=t_apply
    )

    # Synchronize solver bounds only when processing discrete events in the
    # main node loop, not during overshoot processing. Overshootable events are
    # evaluated on interpolated step points with rolled-back solver state and
    # should not trigger bound updates mid-overshoot context.
    # TODO: this sould not be here? this is for addition of events
    if (
        event.sampling_rate is not None
        and not event.time_overshootable
        and nodes_modified
    ):
        node = phase.time_nodes[node_index]
        next_node = phase.time_nodes[node_index + 1]
        # Determine time bound for this time node
        node.time_bound = next_node.t
        # Update solver time bound and status to run until next node
        phase.solver.t_bound = node.time_bound
        if flight._Flight__is_lsoda:
            phase.solver._lsoda_solver._integrator.rwork[0] = phase.solver.t_bound
            phase.solver._lsoda_solver._integrator.call_args[4] = (
                phase.solver._lsoda_solver._integrator.rwork
            )


def apply_rollback_command(flight, time, state):
    """Apply a rollback request returned by an event trigger.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : dict
        Result payload returned by the event system.
    phase : _FlightPhase
        Current flight phase.
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
    flight.solution[-1] = [time, *state]
    return None


def apply_disable_commands(_, event_results, node_index, event, phase, time):
    """Apply disable commands returned by an event solver.

    Parameters
    ----------
    event_results : dict
        Result payload returned by the event system.
    node_index : int
        Index of the current time node.
    event : Event
        Event currently being processed.
    phase : _FlightPhase
        Current flight phase.

    Returns
    -------
    bool
        ``True`` if time_nodes structure was modified, ``False`` otherwise.
    """
    nodes_modified = False
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
        nodes_modified = True

    if event_results._disabled:
        event.enabled = False
        event.disabled_times.append(time)
        _safe_disable_time_nodes_event(
            phase=phase,
            node_index=node_index,
            event=event,
            time=time,
        )
        nodes_modified = True

    return nodes_modified


def apply_enable_commands(flight, event_results, node_index, event, phase, time):
    """Apply enable commands returned by an event solver.

    Parameters
    ----------
    event_results : dict
        Result payload returned by the event system.
    node_index : int
        Index of the current time node.
    event : Event
        Event currently being processed.
    phase : _FlightPhase
        Current flight phase.

    Returns
    -------
    bool
        ``True`` if time_nodes structure was modified, ``False`` otherwise.
    """
    nodes_modified = False
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
        nodes_modified = True

    # Commands API uses `_disabled` flag: True -> disable, False -> enable
    if event_results._disabled is False:
        event.enabled = True
        event.enabled_times.append(time)
        nodes_modified = True

        # if the event is non-overshootable, we need to create discrete nodes
        if event in flight._non_overshootable_events:
            _safe_enable_time_nodes_event(
                phase=phase,
                node_index=node_index,
                event=event,
                time=time,
            )

    return nodes_modified


def apply_exact_time_result(flight, event_results):
    """Apply an exact-time correction to the current flight state.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : dict
        Result payload returned by the event system.

    Returns
    -------
    None
        This function updates ``flight.t``, ``flight.y_sol``, and the last stored
        solution row in place when an exact-time result is available.
    """
    # TODO: CONTINUOS EVENTS WITH CHANGE DUNAMICS SHOULD NOT JUST INSERT INTO
    # SOLUTION. THEY HAVE TO ROLLBACK ALSO. HOWEVER, THIS IS NOT A VALID USE
    # CASE I BELIVE. SO IF CONTINUOUS EVENTS WITH DYNAMICS CHANGES ARE NEEDED,
    # THE EVENT SHOULD NOT ACCEPT EXACT TIME FUNCTION.
    if (
        event_results.exact_time is not None
        and event_results.exact_state is not None
    ):
        t_exact = event_results.exact_time
        y_exact = event_results.exact_state

        # Prefer inserting the exact point between the previous and last
        # stored solution points so we don't clobber the step-end data.
        if len(flight.solution) >= 2:
            t_prev = flight.solution[-2][0]
            t_last = flight.solution[-1][0]

            # Only insert if the exact time lies strictly between the two
            # stored times to avoid duplicate timestamps.
            if t_prev < t_exact < t_last:
                flight.solution.insert(-1, [t_exact, *y_exact])
            elif t_exact == t_last or t_exact == t_prev:
                # exact time matches previous or last point: nothing to insert
                pass
            else:
                # Unexpected: exact time outside bracket; warn and fall back
                warnings.warn(
                    "Exact event time outside last solver interval; replacing last point.",
                    UserWarning,
                )
                flight.solution[-1] = [t_exact, *y_exact]
        else:
            # Not enough history: replace last entry (best effort)
            flight.solution[-1] = [t_exact, *y_exact]

        # Update current flight time/state to the exact values
        flight.t = t_exact
        flight.y_sol = y_exact


def apply_new_phase_or_derivative(
    flight, event_results, phase, phase_index, node_index, time
):
    """Apply a flight-phase transition or derivative change.

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

    Returns
    -------
    None
        This function mutates the flight phase list and time-node state in place.
    """
    if (
        event_results.new_flight_phase is None
        and event_results.new_derivative is None
    ):
        return

    when_time = time

    # Check if there was exact time point insertion in solution for this event
    # If so, remove the point after the exact time point, so the solution vector
    # and the new phase start time are consistent
    if (event_results.exact_time is not None
        and event_results.exact_state is not None
        and flight.solution[-1][0] > when_time
    ):
        flight.solution.pop(-1)

    derivative = phase.derivative
    if event_results.new_derivative is not None:
        derivative = event_results.new_derivative

    flight.flight_phases.add_phase(
        when_time + event_results.new_flight_phase_lag,
        derivatives=derivative,
        index=phase_index + 1,
        name=event_results.new_flight_phase_name,
        parachute=event_results.new_flight_phase_parachute,
    )

    # Prepare to leave loops and start new flight phase
    phase.time_nodes.flush_after(node_index)
    phase.time_nodes.add_node(when_time, [])
    phase.solver.status = "finished"

    # Rollback solution to time when new phase starts
    apply_rollback_command(flight, when_time, flight.solution[-1][1:])


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

    Returns
    -------
    None
        This function mutates the flight and phase objects in place when the
        event requests termination.
    """
    if not event_results._terminate:
        return

    when_time = time

    if (event_results.exact_time is not None
        and event_results.exact_state is not None
        and flight.solution[-1][0] > when_time
    ):
        flight.solution.pop(-1)

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
    node_index : int
        Index of the current time node.
    event : Event
        Event currently being processed.
    phase : _FlightPhase
        Current flight phase.

    Returns
    -------
    bool
        ``True`` if time_nodes structure was modified, ``False`` otherwise.
    """
    if event_results.new_events:
        # Normalize new_events to always be a list for consistent iteration
        new_events = event_results.new_events
        if isinstance(new_events, Event):
            new_events = [new_events]

        for new_event in new_events:
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


