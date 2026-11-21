from ..helpers.conditions import FlightConditions
from ..helpers.state import CANONICAL_INDEX, _State, _StateDot
from .views import ControlsView, OutputsView, SensorsView

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
    "conditions",
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
    ``state_dot``, ``conditions``, the air data (``pressure``, ``density``,
    ``mach``, ...) and ``phase_name``. Once read, a value is kept for the rest
    of the step, so a second event reading it in the same step does not work it
    out again.

    ``previous_state`` and ``previous_time`` describe the previous check of the
    event being evaluated, so they are answered from that event and never
    stored here.

    The same context is given to every event of the step, and it is rewritten
    as the step goes on. Do not keep a reference to it after the function
    returns, and do not add values to it, since the other events of the step
    would see them. Keep an event's own data in ``context.memory``.
    """

    # The values are stored as dictionary items, which is what RocketPy's own
    # code reads (``context["canonical_state"]``), since an item read is faster
    # than a property. ``canonical_state`` and ``raw_state`` are the plain
    # arrays behind ``state``: the 13 canonical values and the values the
    # current phase integrates, which ``phase_names`` names in order. The
    # properties below are the documented way in.

    __slots__ = ("_solution", "_phase", "_output_events")

    def __init__(self, *args, solution=None, phase=None, output_events=None, **values):
        """Build a context from its values, as a dict is built.

        Parameters
        ----------
        *args, **values
            The values, as given to ``dict``.
        solution : Solution, optional
            The flight's solution so far. Default ``None``, outside a flight.
        phase : _FlightPhase, optional
            The flight phase being flown. Default ``None``, outside a flight.
        output_events : dict, optional
            The flight's events with outputs, by their name in snake_case.
            Default ``None``, no outputs.
        """
        super().__init__(*args, **values)
        self._solution = solution
        self._phase = phase
        self._output_events = output_events

    def __dir__(self):
        # What a trigger or callback reads, without the dictionary's own
        # methods, so that completion in a notebook lists only these
        return [
            name
            for name, member in vars(EventContext).items()
            if isinstance(member, property) or name == "record"
        ]

    def __repr__(self):
        """Return a summary of what a trigger or callback can read and write
        here, with the names to use: ``print(context)`` inside a function
        shows them. The equations of motion and the air data are not
        evaluated for it."""
        lines = []

        def add(section):
            # A context built outside a flight may lack a part: skip it
            try:
                lines.extend(section())
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        def header():
            parts = [f"EventContext at t = {self['time']:.6g} s"]
            if self._phase is not None:
                parts.append(f"phase '{self._phase.name}'")
            if "event" in self:
                parts.append(f"event '{self['event'].name}'")
            return [", ".join(parts)]

        def state():
            return [f"  state (context.state.<name>): {', '.join(self['state'].names)}"]

        def height():
            return [f"  height_agl: {self['height_agl']:.6g} m"]

        def controls():
            view = self.controls
            groups = [
                f"    {key}: {', '.join(view[name])}" for name, key in _keys(view)
            ]
            return ["  controls (context.controls.<name>.<control>):", *groups]

        def sensors():
            view = self.sensors
            groups = [
                f"    {key}: {', '.join(view[name])}" for name, key in _keys(view)
            ]
            return ["  sensors (context.sensors.<name>.<channel>):", *groups]

        def outputs():
            view = self.outputs
            groups = [
                f"    {key}: {', '.join(view[name])}" for name, key in _keys(view)
            ]
            return ["  outputs (context.outputs.<name>.<output>):", *groups]

        def memory():
            keys = ", ".join(map(str, self["event"].memory)) or "empty"
            return [f"  memory (context.memory[<key>]): {keys}"]

        def _keys(view):
            # Each group of a view by its name as given, with its key; a view
            # with no groups raises, so its section is left out
            pairs = list(view._names().items())  # pylint: disable=protected-access
            if not pairs:
                raise LookupError
            return pairs

        for section in (header, state, height, controls, sensors, outputs, memory):
            add(section)
        shown = {"time", "state", "height_agl", "controls", "sensors", "outputs"}
        shown |= {"memory", "phase_name", "event"}
        rest = [name for name in dir(self) if name not in shown]
        lines.append(f"  also: {', '.join(rest)}")
        return "\n".join(lines)

    def __missing__(self, key):
        if key in _PREVIOUS_KEYS:
            # Belongs to one event, so it is answered rather than stored.
            event = self["event"]
            if key == "previous_time":
                return event._previous_time
            if event._previous_state is None:
                return None
            # One view per check: kept on the event while its previous check
            # is the same one, so reading it again gives the same object.
            source = event._previous_state
            kept = event._previous_view
            if kept is None or kept[0] is not source:
                # The phase's own states are named after the phase they were
                # taken in, which may no longer be the current one.
                view = _State(
                    source, event._previous_phase_names, event._previous_raw_state
                )
                kept = event._previous_view = (source, view)
            return kept[1]
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
        elif key == "conditions":
            self["conditions"] = FlightConditions(
                self["rocket"],
                self["environment"],
                self["time"],
                self["canonical_state"],
                self["phase_names"],
                self["raw_state"],
            )
        elif key == "phase_name":
            self["phase_name"] = self._phase.name
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
        current = self._solution.phases[-1]
        raw_state = self["raw_state"]
        raw_state_dot = self._phase.dynamics(self["time"], raw_state)
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
        self["event"] = event
        self["sampling_rate"] = event.sampling_rate

    def record(self, **values):
        """Record named values at the current time, to read after the flight.

        .. code-block:: python

            context.record(estimated_apogee=3012.0, error=0.4)
            context.record(command=[0.1, -0.2, 0.0], mode="coast")

        After the flight, ``event.records["estimated_apogee"]`` and
        ``flight.records.<event name>.estimated_apogee`` (the records of
        every event, by the event's name) give each name as the list of
        ``(time, value)`` pairs recorded, with the value exactly as given.
        Unlike ``context.memory``, which keeps what the next call needs, a
        record keeps every value with its time.

        Parameters
        ----------
        **values : float or object
            The values to record, by name. A value that is not a number is
            copied, so changing the object afterwards does not change what was
            recorded.
        """
        self["event"]._record(self["time"], values)  # pylint: disable=protected-access

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
        """Derivative of each state with respect to time, read by name as
        ``state_dot.az``."""
        return self["state_dot"]

    @property
    def previous_state(self):
        """The ``state`` at this event's previous check, or ``None`` on the
        first check."""
        return self["previous_state"]

    @property
    def previous_time(self):
        """The ``time`` of this event's previous check, in s, or ``None`` on
        the first check."""
        return self["previous_time"]

    @property
    def height_agl(self):
        """Height of the rocket above ground level, in m."""
        return self["height_agl"]

    @property
    def conditions(self):
        """The air data (``pressure``, ``density``, ``mach``, ...) as one
        object, a :class:`rocketpy.control.FlightConditions`. It is the object a
        controllable's methods receive inside the equations of motion, so use
        it to pass the air data on to one of them. To read a single value,
        ``context.pressure`` is shorter."""
        return self["conditions"]

    # The air data are worked out by the conditions object when first read,
    # and kept for the rest of the step with it.

    @property
    def pressure(self):
        """Air pressure at the rocket's altitude, in Pa."""
        return self["conditions"].pressure

    @property
    def density(self):
        """Air density at the rocket's altitude, in kg/m³."""
        return self["conditions"].density

    @property
    def speed_of_sound(self):
        """Speed of sound at the rocket's altitude, in m/s."""
        return self["conditions"].speed_of_sound

    @property
    def wind_velocity(self):
        """Wind velocity at the rocket's altitude, in m/s, in the Earth frame
        (x East, y North, z up)."""
        return self["conditions"].wind_velocity

    @property
    def stream_velocity(self):
        """Velocity of the air relative to the rocket, in m/s, in the rocket's
        body frame (z along the rocket's axis, toward the nose)."""
        return self["conditions"].stream_velocity

    @property
    def stream_speed(self):
        """Speed of the air relative to the rocket, in m/s."""
        return self["conditions"].stream_speed

    @property
    def mach(self):
        """Mach number of the air relative to the rocket."""
        return self["conditions"].mach

    @property
    def dt(self):
        """Time since this event was last checked, in s. For a controller, this
        is the time since its previous run. On the first check it is
        ``1 / sampling_rate``, or ``None`` for an event with no
        ``sampling_rate``, which is checked at every solver step."""
        previous_time = self["previous_time"]
        if previous_time is not None:
            return self["time"] - previous_time
        sampling_rate = self["sampling_rate"]
        return None if sampling_rate is None else 1 / sampling_rate

    @property
    def sampling_rate(self):
        """How often the event being checked is sampled, in Hz, or ``None`` for
        an event checked at every solver step."""
        return self["sampling_rate"]

    @property
    def event(self):
        """The event being checked."""
        return self["event"]

    @property
    def memory(self):
        """A dictionary that belongs to the event being checked and is kept
        from one check to the next. Store in it what the next call needs, such
        as the integral of a PID controller. The same as
        ``context.event.memory``."""
        return self["event"].memory

    @property
    def commands(self):
        """The requests the event being checked makes to the flight, such as
        ``context.commands.terminate_flight()`` or
        ``context.commands.set_dynamics(...)``. This is the only way a trigger
        or callback can change the course of the flight. A request takes
        effect once the function returns. The same as
        ``context.event.commands``. See
        :class:`rocketpy.simulation.events.commands.Commands`."""
        return self["event"].commands

    @property
    def controls(self):
        """The rocket's control inputs, read and written by the name of the
        controllable in snake_case, then the name of the control:

        .. code-block:: python

            context.controls.air_brakes.deployment_level = 0.5
            context.controls["Air Brakes"].deployment_level = 0.5
            context.controls[air_brakes].deployment_level = 0.5
            context.controls.canards[f"fin_{i}"] = 0.05

        The second and third lines find the controllable by its name as given
        and by the object itself. The last one writes a control whose name is
        built in a loop. ``list(context.controls)`` gives the names of the
        controllables.

        A value written outside the control's limits is moved to the nearest
        limit once the callback returns. The new value acts from the event's
        time on: if the solver step had already gone past that time, the step
        is cut there. See
        :class:`rocketpy.simulation.events.views.ControlsView`.
        """
        owners = self.get("control_owners")
        if owners is None:
            owners = getattr(self.get("rocket"), "controllables", ())
        return ControlsView(self["controls"], owners)

    @property
    def outputs(self):
        """The values the events of the flight share with each other. An
        event declares its outputs when it is built (``outputs=`` of
        :class:`rocketpy.Event` and :class:`rocketpy.Controller`), writes them
        under its own name in snake_case, and every other event reads them:

        .. code-block:: python

            context.outputs.guidance.pitch_setpoint = 0.1  # in "Guidance"
            setpoint = context.outputs.guidance.pitch_setpoint  # elsewhere

        An output can hold any value and keeps it until it is written again.
        Only the event that declared an output can write it.
        ``list(context.outputs)`` gives the names of the events with outputs.
        See :class:`rocketpy.simulation.events.views.OutputsView`.
        """
        return OutputsView(self._output_events or {}, self.get("event"), self["time"])

    @property
    def rocket(self):
        """The rocket being flown. Do not change it here, since the same rocket
        object is used by later flights."""
        return self["rocket"]

    @property
    def environment(self):
        """The environment the rocket flies in, such as
        ``context.environment.wind_velocity_x(z)``. Do not change it here."""
        return self["environment"]

    @property
    def phase_name(self):
        """Name of the current flight phase: ``"initial_phase"`` (from the
        start, on the rail), ``"free_flight"`` (after leaving the rail), or
        ``"<parachute name>_parachute_descent"`` (under that parachute)."""
        return self["phase_name"]

    @property
    def sensors(self):
        """The latest reading of every channel of the rocket's sensors. Find
        a sensor by its name as given, or by that name in snake_case as an
        attribute, then read its channels by attribute, such as ``ax``,
        ``wz``, ``pressure`` or ``latitude``. The two are the same:

        .. code-block:: python

            context.sensors["Accelerometer"].az
            context.sensors.accelerometer.az

        ``list(context.sensors)`` gives the names. Every sensor reads once on
        the pad when the flight starts, so there is always a reading. The
        readings cannot be written. The sensor objects themselves, with their
        settings, are in ``context.rocket.sensors``. See
        :class:`rocketpy.simulation.events.views.SensorsView`.
        """
        return SensorsView(self["sensors"], self._solution)


def build_event_kwargs(flight, time, state, phase):
    """Build the context shared by the event triggers and callbacks of a step.

    ``state`` is the raw state of the current flight phase, which may not be the
    full canonical state. The reconstructed canonical state is stored as
    ``canonical_state``, so that RocketPy's own events always see the 13
    standard states, and the raw state as ``raw_state``, for internal use such
    as rolling the solver back. Triggers and callbacks written by users read
    ``state``, a ``_State`` over both that is only built when it is read.

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
    phase : _FlightPhase
        The flight phase being flown.

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
            "canonical_state": canonical,
            "raw_state": state,
            "phase_names": current.dynamics.states,
            "sensors": flight._sensors_by_key,
            "environment": flight.env,
            "rocket": flight.rocket,
            "height_agl": canonical[_Z_SLOT] - flight.env.elevation,
            # The same dictionary for the whole flight, so it is kept when the
            # context moves to another moment of the step
            "controls": flight._controls,
            # The controllables the control inputs are named after
            "control_owners": getattr(flight, "_control_owners", None),
        },
        solution=flight.solution,
        phase=phase,
        output_events=getattr(flight, "_output_events", None),
    )


def refresh_event_kwargs(event_kwargs, interpolated_time, interpolated_state):
    """Rewrite the event context to describe another moment, in place.

    Used for a moment inside the solver step: a sampling time the step
    overshot, or a candidate time of an exact-time search. Values worked out
    on demand for the old moment are dropped, to be worked out again if read.

    Parameters
    ----------
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
    current = event_kwargs._solution.phases[-1]  # pylint: disable=protected-access
    # A phase that already integrates the full canonical state needs no
    # reconstruction, and this runs on every overshootable node.
    if current.dynamics.is_canonical:
        canonical = interpolated_state
    else:
        canonical = current.canonical_state(interpolated_state)
    event_kwargs["time"] = interpolated_time
    event_kwargs["canonical_state"] = canonical
    event_kwargs["raw_state"] = interpolated_state
    event_kwargs["height_agl"] = (
        canonical[_Z_SLOT] - event_kwargs["environment"].elevation
    )
    for key in _LAZY_CONTEXT_KEYS:
        event_kwargs.pop(key, None)
    return event_kwargs
