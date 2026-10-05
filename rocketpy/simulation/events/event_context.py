from ..helpers.dynamics import CANONICAL_INDEX
from .state import _State, _StateDot

# Altitude's slot in the canonical state. Reading it by position avoids a
# by-name lookup on every event check.
_Z_SLOT = CANONICAL_INDEX["z"]

# Context keys worked out only when a function reads them. They depend on
# the time being evaluated, so they are dropped whenever the context is moved
# to another time inside the step, and worked out again if read there.
_LAZY_CONTEXT_KEYS = (
    "state",
    "state_dot",
    "canonical_state_dot",
    "raw_state_dot",
    "pressure",
    "step_size",
)

# Keys answered from the event being evaluated, never stored in the context.
_PREVIOUS_KEYS = frozenset({"previous_state", "previous_time"})


class EventContext(dict):
    """The values an event's trigger and callback are given.

    One of these is built per solver step and shared by every event checked in
    that step, so the values describing the step are written once. Read a
    value as an attribute: ``context.time``, ``context.state.vz``,
    ``context.height_agl`` and so on. See :ref:`eventusage` for what each one
    holds.

    Some values are only worked out when they are first read: ``state``,
    ``state_dot``, ``pressure`` and ``step_size``. Once read they are kept for
    the rest of the step, so a second event reading the same value in the same
    step pays nothing. A value no function reads costs nothing at all.

    ``previous_state`` and ``previous_time`` describe the previous check of the
    event being evaluated, so they are answered from that event and never
    stored here.

    A context is shared and rewritten as the step is worked through, so nothing
    may hold on to it, and a function should not add values to it: whatever it
    writes is seen by the other events of the step. Keep an event's own data in
    ``context.event.memory``.
    """

    # The values are stored as dictionary items, which is what RocketPy's own
    # code reads (``context["canonical_state"]``), since an item read is faster
    # than a property. ``canonical_state`` and ``raw_state`` are the plain
    # arrays behind ``state``: the 13 canonical values and the values the
    # current phase integrates, which ``phase_names`` names in order. The
    # properties below are the documented way in.

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
            if key == "previous_time":
                return event._previous_time
            if event._previous_state is None:
                return None
            # The phase's own states are named after the phase they were
            # taken in, which may no longer be the current one.
            return _State(
                event._previous_state,
                event._previous_phase_names,
                event._previous_raw_state,
            )
        if key == "state":
            self["state"] = _State(
                self["canonical_state"], self["phase_names"], self["raw_state"]
            )
        elif key == "state_dot":
            self["state_dot"] = _StateDot(
                self["canonical_state_dot"],
                self["phase_names"],
                self["raw_state_dot"],
            )
        elif key in ("canonical_state_dot", "raw_state_dot"):
            self._compute_derivatives()
        elif key == "pressure":
            self["pressure"] = self["environment"].pressure.get_value_opt(
                self["canonical_state"][_Z_SLOT]
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
        self["raw_state_dot"] = raw_state_dot
        self["canonical_state_dot"] = current.canonical_derivative(
            raw_state_dot, raw_state
        )

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

    @property
    def time(self):
        """Current simulation time, in s."""
        return self["time"]

    @property
    def state(self):
        """The rocket's states, read by name as ``state.vz``."""
        return self["state"]

    @property
    def state_dot(self):
        """Time derivative of each state, read as ``state_dot.az``."""
        return self["state_dot"]

    @property
    def previous_state(self):
        """The ``state`` at this event's previous check, or ``None``."""
        return self["previous_state"]

    @property
    def previous_time(self):
        """The ``time`` of this event's previous check, in s, or ``None``."""
        return self["previous_time"]

    @property
    def height_agl(self):
        """Height of the rocket above ground level, in m."""
        return self["height_agl"]

    @property
    def pressure(self):
        """Atmospheric pressure at the rocket's altitude, in Pa."""
        return self["pressure"]

    @property
    def step_size(self):
        """How long the simulation has been inside this solver step, in s."""
        return self["step_size"]

    @property
    def sampling_rate(self):
        """How often the event being checked is sampled, in Hz, or ``None``."""
        return self["sampling_rate"]

    @property
    def event(self):
        """The event being checked."""
        return self["event"]

    @property
    def flight(self):
        """The flight being simulated."""
        return self["flight"]

    @property
    def rocket(self):
        """The rocket being flown."""
        return self["rocket"]

    @property
    def environment(self):
        """The environment the rocket flies in."""
        return self["environment"]

    @property
    def phase(self):
        """The current flight phase."""
        return self["phase"]

    @property
    def sensors(self):
        """The rocket's sensors, in the order they were added."""
        return self["sensors"]

    @property
    def sensors_by_name(self):
        """The rocket's sensors, by name."""
        return self["sensors_by_name"]

    @property
    def controller(self):
        """The controller being run, or ``None`` outside a controller."""
        return self.get("controller")

    @property
    def controlled(self):
        """The objects the controller drives, or ``None`` outside one."""
        return self.get("controlled")


def infer_step_size(flight, time):
    """Return how long the simulation has been inside the current solver step.

    This is ``time`` measured from the start of the step the solver is working
    on, which is the time of the second-to-last stored row. At the end of a step
    that is the whole step the solver just took; at a point interpolated inside
    a step it is the part of the step up to that point.

    It reaches user callbacks as ``context.step_size``. It is not the
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
    full canonical state. The reconstructed canonical state is stored as
    ``canonical_state``, so that built-in hooks always see the familiar
    13-variable layout, and the raw state as ``raw_state``, for internal use
    such as rolling the solver back. User-written hooks read ``state``, a
    ``_State`` over both that is only built when it is read.

    Only what nearly every event reads is worked out here; the rest is worked
    out by the context itself when first read.
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
            "canonical_state": canonical,
            "raw_state": state,
            "phase_names": current.dynamics.states,
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
    interpolated_state : sequence of float
        The current phase's raw state at ``interpolated_time``.
    """
    current = flight.solution.phases[-1]
    # A phase that already integrates the full canonical state needs no
    # reconstruction, and this runs on every overshootable node.
    if current.dynamics.is_canonical:
        canonical = interpolated_state
    else:
        canonical = current.canonical_state(interpolated_state)
    event_kwargs["time"] = interpolated_time
    event_kwargs["canonical_state"] = canonical
    event_kwargs["raw_state"] = interpolated_state
    event_kwargs["height_agl"] = canonical[_Z_SLOT] - flight.env.elevation
    for key in _LAZY_CONTEXT_KEYS:
        event_kwargs.pop(key, None)
    return event_kwargs
