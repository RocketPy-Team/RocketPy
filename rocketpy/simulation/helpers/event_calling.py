from ..solution import CANONICAL_INDEX
from .event_commands import apply_event_commands, apply_rollback_command

# Altitude's slot in the canonical state. Reading it by position avoids a
# by-name lookup on every event check.
_Z_SLOT = CANONICAL_INDEX["z"]

# Context keys worked out only when a function reads them. They depend on
# the time being evaluated, so they are dropped whenever the context is moved
# to another time inside the step, and worked out again if read there.
_LAZY_CONTEXT_KEYS = (
    "state_dot",
    "phase_state_dot",
    "pressure",
    "phase_state",
    "step_size",
)

# Keys answered from the event being evaluated, never stored in the context.
_PREVIOUS_KEYS = frozenset({"previous_state", "previous_time", "previous_phase_state"})


class EventContext(dict):
    """The values an event's trigger and callback are given.

    One of these is built per solver step and shared by every event checked in
    that step, so the values describing the step are written once. It is a
    dictionary, read with ``context["time"]``, ``context["state"]`` and so on.

    Some values are only worked out when they are first read: the time
    derivatives ``state_dot`` and ``phase_state_dot`` (one evaluation of the
    equations of motion serves both), ``pressure``, ``phase_state`` and
    ``step_size``. Once read they are kept for the rest of the step, so a second
    event reading the same value in the same step pays nothing. A value no
    function reads costs nothing at all.

    ``previous_state``, ``previous_time`` and ``previous_phase_state`` describe
    the previous check of the event being evaluated, so they are answered from
    that event and never stored here.

    A context is shared and rewritten as the step is worked through, so nothing
    may hold on to it, and a function should not add keys to it: whatever it
    writes is seen by the other events of the step. Keep an event's own data in
    ``context["event"].memory``.
    """

    __slots__ = ("_owned",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keys the event currently being evaluated owns, which are cleared
        # before the next event is bound. Keys every event writes are not
        # listed, since they are overwritten rather than left behind.
        self._owned = ()

    def __missing__(self, key):
        if key in _PREVIOUS_KEYS:
            # Belongs to one event, so it is answered rather than stored.
            event = self["event"]
            if key != "previous_phase_state":
                return getattr(event, "_" + key)
            if event._previous_raw_state is None:
                return None
            return dict(
                zip(
                    self["flight"].solution.phases[-1].dynamics.states,
                    event._previous_raw_state,
                )
            )
        if key in ("state_dot", "phase_state_dot"):
            self._compute_derivatives()
        elif key == "pressure":
            self["pressure"] = self["environment"].pressure.get_value_opt(
                self["state"][_Z_SLOT]
            )
        elif key == "phase_state":
            self["phase_state"] = dict(
                zip(
                    self["flight"].solution.phases[-1].dynamics.states,
                    self["raw_state"],
                )
            )
        elif key == "step_size":
            self["step_size"] = infer_step_size(self["flight"], self["time"])
        else:
            raise KeyError(key)
        return self[key]

    def get(self, key, default=None):
        """Return ``self[key]``, working it out if needed, else ``default``."""
        try:
            return self[key]
        except KeyError:
            return default

    def _compute_derivatives(self):
        """Evaluate the equations of motion once, for both derivative views."""
        current = self["flight"].solution.phases[-1]
        raw_state = self["raw_state"]
        raw_state_dot = self["phase"].dynamics(self["time"], raw_state)
        self["phase_state_dot"] = dict(zip(current.dynamics.states, raw_state_dot))
        self["state_dot"] = current.canonical_derivative(raw_state_dot, raw_state)

    def bind(self, event):
        """Point this context at the event about to be evaluated.

        Parameters
        ----------
        event : Event
            The event being checked.
        """
        for key in self._owned:
            del self[key]
        self._owned = ()
        self["event"] = event
        self["sampling_rate"] = event.sampling_rate

    def own(self, **values):
        """Add values that belong to the event being evaluated alone.

        They are removed again when the next event is bound, so no other event
        of the step sees them. A controller uses this for ``controller`` and
        the names of the objects it drives.
        """
        self.update(values)
        self._owned = self._owned + tuple(values)

    def at(self, time, raw_state):
        """Return a copy of this context describing another moment of the step.

        Everything that depends on the moment is worked out again for ``time``,
        or dropped so that it is worked out when next read. The context itself
        is not changed.

        Parameters
        ----------
        time : float
            The moment, in seconds, inside the current solver step.
        raw_state : sequence of float
            The current phase's raw state at ``time``.

        Returns
        -------
        EventContext
        """
        moved = EventContext(self)
        moved._owned = self._owned
        return refresh_event_kwargs(self["flight"], moved, time, raw_state)


def infer_step_size(flight, time):
    """Return how long the simulation has been inside the current solver step.

    This is ``time`` measured from the start of the step the solver is working
    on, which is the time of the second-to-last stored row. At the end of a step
    that is the whole step the solver just took; at a point interpolated inside
    a step it is the part of the step up to that point.

    It reaches user callbacks as ``context["step_size"]``. It is not the
    interval between checks of any one callback, and not ``1 / sampling_rate``:
    a sampled callback is checked on its own schedule, which has nothing to do
    with where the solver put its step boundaries.

    Worked out from the stored rows rather than read off the solver, because the
    solver's own ``step_size`` still describes the step it last attempted, which
    after a rollback is a step the flight no longer contains.
    """
    solution = flight.solution
    if len(solution) < 2:
        return 0.0
    return max(0.0, time - solution.raw_row(-2)[0])


def build_event_kwargs(flight, time, state, phase):
    """Build the context shared by the event triggers and callbacks of a step.

    ``state`` is the raw state of the current flight phase, which may not be the
    full canonical state. Triggers and callbacks are given the reconstructed
    canonical state (``state``) so that user-written and built-in hooks always
    see the familiar 13-variable layout, while the raw state is also provided
    (``raw_state``) for internal use such as rolling the solver back.

    Only what nearly every event reads is worked out here; the rest is worked
    out by the context itself when first read.

    Parameters
    ----------
    flight : Flight
        Flight instance whose rocket, environment and sensors are exposed.
    time : float
        Current simulation time, in seconds.
    state : sequence of float
        The current phase's raw state at ``time``.
    phase : FlightPhase
        Active flight phase.

    Returns
    -------
    EventContext
        The context handed to event triggers and callbacks.
    """
    current = flight.solution.phases[-1]
    # A phase that already integrates the full canonical state needs no
    # reconstruction, and this runs on every event check.
    canonical = (
        state if current.dynamics.is_canonical else current.canonical_state(state)
    )
    return EventContext(
        {
            "time": time,
            "state": canonical,
            "raw_state": state,
            "sensors": flight.sensors,
            "sensors_by_name": flight.sensors_by_name,
            "environment": flight.env,
            "rocket": flight.rocket,
            "flight": flight,
            "phase": phase,
            "height_agl": canonical[_Z_SLOT] - flight.env.elevation,
        }
    )


def refresh_event_kwargs(flight, event_kwargs, interpolated_time, interpolated_state):
    """Rewrite the event context to describe another moment, in place.

    Used for a moment inside the solver step: a sampling time the step
    overshot, or a candidate time of an exact-time search. Values worked out
    on demand for the old moment are dropped, to be worked out again if read.

    Parameters
    ----------
    flight : Flight
        Flight instance whose environment is used to recompute derived values.
    event_kwargs : EventContext
        Context (from :func:`build_event_kwargs`) updated in place.
    interpolated_time : float
        The moment to describe, in seconds.
    interpolated_state : sequence of float
        The current phase's raw state at ``interpolated_time``.

    Returns
    -------
    EventContext
        The updated ``event_kwargs``.
    """
    current = flight.solution.phases[-1]
    # A phase that already integrates the full canonical state needs no
    # reconstruction, and this runs on every overshootable node.
    if current.dynamics.is_canonical:
        canonical = interpolated_state
    else:
        canonical = current.canonical_state(interpolated_state)
    event_kwargs["time"] = interpolated_time
    event_kwargs["state"] = canonical
    event_kwargs["raw_state"] = interpolated_state
    event_kwargs["height_agl"] = canonical[_Z_SLOT] - flight.env.elevation
    for key in _LAZY_CONTEXT_KEYS:
        event_kwargs.pop(key, None)
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
    trigger_result = event(event_kwargs, trigger_only=True)

    if not trigger_result:
        event._trigger_checked = False
        return rolled_back, False

    event._trigger_checked = True

    if not event.changes_dynamics:
        event(event_kwargs, callback_only=True, reset=False)
        # If the callback queued commands that change the post-trigger
        # trajectory (a new flight phase, a new derivative, or termination),
        # the overshoot step-end state is no longer valid. Roll the flight
        # state back to the interpolated trigger crossing first, so every
        # queued command is applied on a consistent state and effects start
        # exactly at the crossing rather than at the overshoot step-end.
        # This lets any event (e.g. parachutes, or user-defined events that
        # add a phase / terminate) work without declaring changes_dynamics.
        # The rollback writes into the solution, so it must use the raw
        # phase-layout state, not the reconstructed canonical one.
        changed = event.commands.changes_trajectory
        if changed:
            apply_rollback_command(
                flight, event_kwargs["time"], event_kwargs["raw_state"]
            )
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
        # A rollback means the rest of the step no longer happened: later
        # sampling times in it are not checked, and the solver is restarted
        # from the crossing if the phase goes on (a new phase after a lag).
        return rolled_back or changed, False

    if not rolled_back:
        apply_rollback_command(flight, event_kwargs["time"], event_kwargs["raw_state"])
        return True, True

    return rolled_back, False


def call_events(
    flight,
    events,
    phase,
    phase_index,
    node_index,
    time,
    state,
):
    event_kwargs = build_event_kwargs(
        flight=flight, time=time, state=state, phase=phase
    )

    trajectory_changed = False
    for event in events:
        trigger_result = event._trigger_checked
        trigger_result = event(event_kwargs, callback_only=trigger_result)
        if trigger_result:
            trajectory_changed |= event.commands.changes_trajectory
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
    return trajectory_changed
