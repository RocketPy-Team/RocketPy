import pickle
import warnings
from copy import deepcopy

from ...control._helpers import check_writes
from ..helpers.dynamics import _BoundDynamics
from .event import Event
from .event_context import build_event_kwargs
from .views import output_events


def call_event(
    flight,
    event,
    context,
    phase,
    phase_index,
    node_index,
):
    """Run one event inside ``flight`` and apply what it asked for.

    Before the event runs, what a rollback needs is saved. After it runs, what
    it wrote to the control inputs is checked: a changed value is kept inside
    its limits and marks the event's commands, since the
    trajectory changes from then on (see ``Commands.changes_trajectory``).

    If the event fired, its commands are applied. When they change the
    trajectory and ``context`` describes a moment before the flight's current
    time (a sampling time the solver step went past), the step is first cut
    back to that moment, so the commands act from there.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event : Event
        The event to run.
    context : EventContext
        The values handed to the event, shared with the other events checked
        at the same moment.
    phase : _FlightPhase
        Current flight phase.
    phase_index : int
        Index of the current flight phase.
    node_index : int
        Index of the current time node.

    Returns
    -------
    bool
        Whether the event fired.
    """
    controls = flight._controls
    before = dict(controls)
    save_for_rollback(flight, event, context["time"], before, phase)
    fired = event(context)
    limits = flight._control_limits  # pylint: disable=protected-access
    changes = check_writes(controls, before, limits)
    event.commands.control_changes.update(changes)
    if changes:
        # The derivatives were worked out with the controls as they were
        for key in ("state_dot", "canonical_state_dot", "raw_state_dot"):
            context.pop(key, None)

    if not fired:
        # An event turned on by its enable_on stays on, even if its trigger
        # did not fire in the same check
        if event.commands._disabled is False:
            apply_enable_commands(
                flight, event.commands, node_index, event, phase, context["time"]
            )
        return fired
    # The rollback writes into the solution, so it uses the raw phase-layout
    # state, not the reconstructed canonical one
    if event.commands.changes_trajectory and context["time"] < flight.t:
        apply_rollback_command(
            flight, context["time"], context["raw_state"], event.commands
        )
    apply_event_commands(
        flight=flight,
        event=event,
        event_results=event.commands,
        phase=phase,
        phase_index=phase_index,
        node_index=node_index,
        command_time=context["time"],
    )
    return fired


def save_for_rollback(flight, event, time, controls, phase):
    """Save what ``event`` is before its first run at ``time`` in the solver
    step (its memory, whether it is enabled and its previous check), with
    ``controls``, a copy of the flight's controls then, so a rollback to an
    earlier time can put them back (see :func:`rewind`)."""
    # There is no log before the first solver step, when nothing can be cut
    log = getattr(flight, "_step_log", None)
    if log is None:
        return
    # Keyed by time and event, so finding an earlier run of the event at this
    # time does not take a pass over every run of the step
    key = (time, id(event))
    if key in log:
        return
    memory = _copy_of(event.memory) if event.memory else {}
    log[key] = (time, event, controls, memory, event._check_state(), phase)


def _copy_of(memory):
    """Return a deep copy of an event's memory. Plain data is copied through
    pickle, several times faster than ``deepcopy`` for a long history."""
    try:
        return pickle.loads(pickle.dumps(memory, pickle.HIGHEST_PROTOCOL))
    except Exception:  # pylint: disable=broad-exception-caught
        return deepcopy(memory)


def process_overshootable_event(
    flight,
    event,
    event_kwargs,
    phase,
    phase_index,
    node_index,
    rolled_back,
):
    """Run one event at a sampling time inside the solver step just taken.

    The callback runs at the sampling time. If its commands change the
    trajectory (a new flight phase, new equations of motion, termination, a
    surface switch, a change to the rocket or a control write), the rest of the
    step is no longer valid: :func:`call_event` cuts it back to this time
    before applying them, so their effects start exactly here.

    Returns
    -------
    bool
        Whether the step was cut, by this event or by an earlier one
        (``rolled_back``). After a cut, later sampling times in the step are not
        checked, and the solver is restarted from here if the phase goes on.
    """
    fired = call_event(flight, event, event_kwargs, phase, phase_index, node_index)
    return rolled_back or (fired and event.commands.changes_trajectory)


def call_events(
    flight,
    events,
    phase,
    phase_index,
    node_index,
    time,
    state,
):
    """Run ``events`` in order at ``time``, applying what each asks for.

    An event that happened at an exact time before ``time`` and changed the
    trajectory takes the flight back to that time. The events after it then
    run there, at the flight's new time and state, since the stretch up to
    ``time`` was never flown.

    Returns
    -------
    bool
        Whether a fired event changed the trajectory.
    """
    event_kwargs = build_event_kwargs(
        flight=flight, time=time, state=state, phase=phase
    )

    trajectory_changed = False
    for event in events:
        if call_event(flight, event, event_kwargs, phase, phase_index, node_index):
            trajectory_changed |= event.commands.changes_trajectory
        if flight.t < event_kwargs["time"]:
            event_kwargs = build_event_kwargs(
                flight=flight, time=flight.t, state=flight.y_sol, phase=phase
            )
    return trajectory_changed


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
    event_results : Commands
        The commands the event queued.
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

    # Before a new phase is started, so its solver begins with these surfaces
    apply_surface_switches(flight, event_results, time=t_apply)

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


def apply_surface_switches(flight, event_results, time):
    """Record the surfaces an event switched on or off, with the time.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    event_results : Commands
        The commands the event queued.
    time : float
        The time the switches happen at, in seconds.

    Raises
    ------
    ValueError
        If a surface is not one of the rocket's aerodynamic surfaces.
    """
    for surface, active in event_results.surface_switches:
        if all(surface is not known for known, _ in flight.rocket.aerodynamic_surfaces):
            raise ValueError(
                f"Cannot switch the surface {getattr(surface, 'name', surface)!r} "
                "on or off: it is not one of the rocket's aerodynamic "
                "surfaces. Add it to the rocket before the flight, with "
                "`active=False` if it should start switched off."
            )
        flight._surface_switches.setdefault(surface, []).append((time, active))


def apply_rollback_command(flight, time, state, event_results=None):
    """Apply a rollback request returned by an event trigger.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    time : float
        Interpolated simulation time to restore.
    state : array_like
        Interpolated flight state vector to restore.
    event_results : Commands, optional
        The commands of the event that asked for the rollback. The controls
        its callback wrote and its memory are kept.

    Returns
    -------
    None
        This function updates flight state/history in place.
    """
    rewind(flight, time, event_results)
    flight.t = time
    flight.y_sol = state
    flight.solution._replace_last([time, *state])


def rewind(flight, time, event_results=None):
    """Undo what the events did after ``time`` in the step being cut.

    Nothing is done when ``time`` is not before the flight's current time,
    since nothing is cut then. Otherwise the control inputs go back to what
    they were before the first event run after ``time`` in this step, and
    every event run after ``time`` gets back its memory, whether it was
    enabled and its previous check (all saved by :func:`save_for_rollback`).
    The records, the outputs and the sensor readings made after ``time`` are
    dropped, with the times events ran or were turned on or off after it. The
    event that cuts the step happened at ``time``, so its control writes, its
    memory and its own state stand.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    time : float
        The time the flight goes back to, in seconds.
    event_results : Commands, optional
        The commands of the event that cuts the step.
    """
    log = getattr(flight, "_step_log", None)
    if log is None or not time < flight.t:
        return
    # In the order the events ran, as the log was filled
    later = [entry for entry in log.values() if entry[0] > time]
    for entry in later:
        del log[(entry[0], id(entry[1]))]
    controls = flight._controls
    if later:
        controls.update(later[0][2])
    restored = set()
    for _, event, _, memory, check_state, phase in later:
        if id(event) in restored or event.commands is event_results:
            continue
        restored.add(id(event))
        # In place, so the dictionary a callback reads stays the same object
        event.memory.clear()
        event.memory.update(memory)
        was_enabled = event._enabled
        event._restore_check_state(check_state)  # pylint: disable=protected-access
        if event._enabled and not was_enabled and event.sampling_rate is None:
            # Turned off after ``time``: checked at every step again
            continuous = phase.time_nodes.continuous_events
            if event not in continuous:
                continuous.append(event)
    if event_results is not None:
        controls.update(event_results.control_changes)
    for event in flight.events:
        event._drop_after(time)  # pylint: disable=protected-access
    rewind_sensors(flight, time)


def rewind_sensors(flight, time):
    """Take the sensors back to ``time``: the solution and their
    ``measured_data`` lose the readings made after it, their ``measurement``
    is again the last one kept, so a controller reading a sensor never sees a
    reading from the stretch of trajectory the flight went back over, and the
    random walk of their noise loses the steps of those readings.

    Parameters
    ----------
    flight : Flight
        Flight instance being updated.
    time : float
        The time the flight goes back to, in seconds.
    """
    positions_by_key = getattr(flight, "_sensors_by_key", None)
    if positions_by_key is None:
        return
    solution = flight.solution
    solution._drop_readings_after(time)  # pylint: disable=protected-access
    positions = {}
    latest = {}
    for sensor, event in positions_by_key.values():
        # A sensor added more than once keeps one list per position
        index = positions.get(id(sensor), 0)
        positions[id(sensor)] = index + 1
        data = sensor.measured_data
        rows = data[index] if data and isinstance(data[0], list) else data
        while rows and rows[-1][0] > time:
            rows.pop()
        sensor._drop_drift_after(time)  # pylint: disable=protected-access
        readings = [
            solution._latest_reading(event._position_name, label)  # pylint: disable=protected-access
            for label, _ in sensor.channels
        ]
        if not all(reading is not None for reading in readings):
            latest.setdefault(id(sensor), (None, sensor, None))
            continue
        read_at = readings[0][0]
        kept = latest.get(id(sensor))
        # Of a sensor added more than once, the position read last
        if kept is None or kept[0] is None or read_at >= kept[0]:
            values = tuple(value for _, value in readings)
            latest[id(sensor)] = (read_at, sensor, values)
    for _, sensor, values in latest.values():
        if values is None:
            sensor.measurement = None
        else:
            sensor.measurement = values if len(values) > 1 else values[0]


def apply_disable_commands(_, event_results, node_index, event, phase, time):
    """Apply disable commands returned by an event solver.

    Parameters
    ----------
    _ : Flight
        Flight instance (unused; accepted for a uniform command signature).
    event_results : Commands
        The commands the event queued.
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
        for event_to_disable in event_results.disable_events:
            event_to_disable._enabled = False
            event_to_disable.disabled_times.append(time)
            _safe_disable_time_nodes_event(
                phase=phase,
                node_index=node_index,
                event=event_to_disable,
                time=time,
            )

    if event_results._disabled:
        event._enabled = False
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
    event_results : Commands
        The commands the event queued.
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
        for event_to_enable in event_results.enable_events:
            event_to_enable._enabled = True
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
        event._enabled = True
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
        rewind(flight, time, event_results)
        if time == previous_time:
            # The event happened right where the step began, which is stored.
            solution._drop_last()
        else:
            solution._replace_last([time, *state])
        flight.t = time
        flight.y_sol = state
    elif previous_time is not None and previous_time < time < solution.raw_row(-1)[0]:
        solution._insert_before_last([time, *state])
        solution._set_controls(-2, dict(flight._controls))


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
    event_results : Commands
        The commands the event queued.
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

    # A parachute coming out: the flight notes it, with the time it happened
    parachute = event_results.new_dynamics_kwargs.get("parachute")
    if parachute is not None:
        flight.parachute_events.append([when_time, parachute])

    # Inserting the phase also moves the current phase's end to its start.
    flight.flight_phases.add_phase(
        when_time + lag,
        dynamics=dynamics,
        index=phase_index + 1,
        name=event_results.new_flight_phase_name,
    )

    # Rollback solution to the trigger time
    apply_rollback_command(
        flight, when_time, flight.solution.raw_row(-1)[1:], event_results
    )

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
    event_results : Commands
        The commands the event queued.
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
    event_results : Commands
        The commands the event queued.
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
            flight.events.append(new_event)
            # Its outputs are read and written like those of any other event
            flight._output_events.update(output_events([new_event]))
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
