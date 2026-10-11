"""Controllers: events that set the rocket's control inputs while it flies."""

import inspect
import pickle
import warnings
from copy import deepcopy

from rocketpy.control.control_signal import control_key, limited_value
from rocketpy.simulation.events.event import Event
from rocketpy.tools import from_hex_decode, to_hex_encode

from ..prints.controller_prints import _ControllerPrints


class Controller(Event):
    """An event that runs at a fixed rate and sets the rocket's controls.

    A controller is how a flight computer is simulated: at every sample it
    reads what the rocket knows (the state, the sensors) and writes the
    control inputs, such as the deployment level of air brakes or the
    deflection of canards::

        def controller_function(context):
            if context.state.vz < 0:
                context.controls["air_brakes.deployment_level"] = 0.0
            else:
                context.controls["air_brakes.deployment_level"] = 0.5
            context.record(vertical_speed=context.state.vz)

        rocket.add_controller(Controller(controller_function, sampling_rate=10))

    The value written stays in force until the next write, and is stored with
    the flight: ``flight.control_history`` gives every control input as a
    function of time. ``rocket.control_channels`` lists the names to write.

    Everything an :class:`rocketpy.Event` offers works the same way here: the
    context the function receives, ``memory``, ``context.record``, and turning
    the controller off and on with ``disable_on`` and ``enable_on``.

    See :ref:`eventusage` for the full description of the context.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        controller_function,
        sampling_rate,
        writes=None,
        memory=None,
        name="Controller",
        enabled=True,
        disable_on=None,
        enable_on=None,
        priority=3,
    ):
        """Build a controller.

        Parameters
        ----------
        controller_function : callable
            The control logic, ``controller_function(context)``, called once
            per sample. It sets the controls by writing
            ``context.controls["<actuator>.<control>"]``, for example
            ``context.controls["air_brakes.deployment_level"] = 0.5``. It may
            return ``None``, or a dictionary of numbers, which is recorded as
            if passed to ``context.record``: ``return {"error": error}`` gives
            ``flight.records["<controller>.error"]`` after the flight.
        sampling_rate : float
            How many times per second the controller runs, in Hz. It runs
            every ``1 / sampling_rate`` seconds of flight.
        writes : str, Actuator or list, optional
            The controls this controller is allowed to set, as a channel name
            (``"air_brakes.deployment_level"``), an actuator name in snake_case
            (``"air_brakes"``, for all its controls), the actuator object
            itself, or a list of these. Writing any other control then raises
            an error, which keeps two controllers from fighting over one
            actuator. Defaults to ``None``: the controller may set any control.
        memory : dict, optional
            Values the controller keeps from one run to the next, read and
            written as ``context.event.memory``, such as the integral of a PID
            controller. It starts again from this dictionary at every flight.
            Defaults to an empty dictionary.
        name : str, optional
            Name of the controller. Its records are named after it, in
            snake_case: a controller named ``"Apogee Controller"`` gives
            ``flight.records["apogee_controller.<record>"]``. Defaults to
            ``"Controller"``.
        enabled : bool, optional
            Whether the controller runs from the start of the flight. Defaults
            to ``True``.
        disable_on : str, float or callable, optional
            When to stop running: ``"apogee"``, ``"burnout"``, a time in
            seconds, or a function ``f(context)`` that returns ``True`` when the
            controller should stop. Defaults to ``None``, never.
        enable_on : str, float or callable, optional
            When a stopped controller starts running, in the same forms as
            ``disable_on``. Defaults to ``None``, never.
        priority : int, optional
            Order among the events that run at the same time: lower runs first.
            Defaults to 3, after the sensors (1) and parachutes (2), so a
            controller reads this sample's measurements.
        """
        self.controller_function = _check_controller_function(controller_function)
        if not sampling_rate or sampling_rate <= 0:
            raise ValueError(
                f"The sampling rate of the controller '{name}' must be a "
                f"positive number of Hz, got {sampling_rate!r}."
            )
        self.writes = _writes_names(writes)
        # Kept as given, for saving: the event turns them into functions
        self._gates = (disable_on, enable_on)
        super().__init__(
            callback=self._run,
            sampling_rate=sampling_rate,
            memory=memory,
            disable_on=disable_on,
            enable_on=enable_on,
            trigger_only_once=False,
            # A controller never switches the flight phase. A write that
            # changes a control cuts the solver step at the sampling time,
            # like any event's (see Event.__call__)
            changes_dynamics=False,
            name=name,
            enabled=enabled,
            priority=priority,
        )
        # The channels ``writes`` allows, found at the first run of a flight
        self._allowed = None
        self.prints = _ControllerPrints(self)

    def reset(self):
        super().reset()
        self._allowed = None

    def _run(self, context):
        """The event's callback: run the controller function once."""
        if self.writes is None:
            result = self.controller_function(context)
        else:
            controls = context.controls
            if self._allowed is None:
                self._allowed = self._allowed_channels(controls)
            before = dict(controls)
            result = self.controller_function(context)
            self._check_allowed(controls, before)
        if isinstance(result, dict):
            context.record(**result)
        return result

    def _allowed_channels(self, controls):
        """Return the channels ``writes`` names, among those of the flight."""
        allowed = set()
        for entry in self.writes:
            if entry in controls:
                allowed.add(entry)
                continue
            channels = {name for name in controls if name.split(".")[0] == entry}
            if not channels:
                raise ValueError(
                    f"The controller '{self.name}' writes '{entry}', but the "
                    "rocket has no control channel or actuator by that name. "
                    f"The channels are: {', '.join(controls) or 'none'}."
                )
            allowed |= channels
        return frozenset(allowed)

    def _check_allowed(self, controls, before):
        """Raise if the function changed a channel ``writes`` does not allow,
        after putting the controls back as they were."""
        for name, value in controls.items():
            if name in self._allowed or name not in before:
                # A channel that did not exist is reported by the event
                continue
            if value != before[name]:
                controls.clear()
                controls.update(before)
                raise ValueError(
                    f"The controller '{self.name}' wrote '{name}', which is not "
                    "among the controls it writes: "
                    f"{', '.join(sorted(self._allowed))}. Add it to `writes`, "
                    "or leave `writes` as None to allow any control."
                )

    def __str__(self):
        return f"Controller '{self.name}' with sampling rate {self.sampling_rate} Hz."

    def info(self):
        """Print what the controller is and what it does."""
        self.prints.all()

    def all_info(self):
        """Print all the information about the controller."""
        self.info()

    def to_dict(self, **kwargs):
        """Return the controller as a dictionary, to be saved.

        Parameters
        ----------
        **kwargs : dict
            ``allow_pickle`` (default ``True``): save the controller function
            and the gate functions themselves. With ``False`` only their names
            are saved, and a loaded controller cannot run.

        Returns
        -------
        dict
        """
        allow_pickle = kwargs.get("allow_pickle", True)

        def encode(function):
            if not callable(function):
                return function
            if allow_pickle:
                return to_hex_encode(function)
            return getattr(function, "__name__", type(function).__name__)

        disable_on, enable_on = self._gates
        return {
            "controller_function": encode(self.controller_function),
            "sampling_rate": self.sampling_rate,
            "writes": None if self.writes is None else list(self.writes),
            "memory": deepcopy(self._initial_memory),
            "name": self.name,
            "enabled": self._initial_enabled,
            "disable_on": encode(disable_on),
            "enable_on": encode(enable_on),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data):
        """Build a controller from a dictionary made by :meth:`to_dict`.

        A controller whose function cannot be loaded, such as one saved with
        ``allow_pickle=False``, is loaded disabled, with a warning.

        Parameters
        ----------
        data : dict

        Returns
        -------
        Controller
        """
        name = data.get("name", "Controller")
        sampling_rate = data.get("sampling_rate")
        enabled = data.get("enabled", True)
        try:
            function = from_hex_decode(data["controller_function"])
        except _LOAD_ERRORS as error:
            warnings.warn(
                f"The function of the controller '{name}' could not be loaded "
                f"({type(error).__name__}: {error}), so the controller is "
                "loaded disabled. Build it again and add it to the rocket to "
                "fly it.",
                UserWarning,
            )
            function, enabled = _missing_controller_function, False
        if _positional_count(function) != 1:
            # Saved before controllers took the context: an air brakes
            # controller with positional arguments
            function = _PositionalController(function, sampling_rate)
        gates = []
        for key in ("disable_on", "enable_on"):
            gate = data.get(key)
            try:
                gate = from_hex_decode(gate)
            except (TypeError, ValueError):
                pass  # a preset name, a time or None
            gates.append(gate)
        return cls(
            controller_function=function,
            sampling_rate=sampling_rate,
            writes=data.get("writes"),
            memory=data.get("memory"),
            name=name,
            enabled=enabled,
            disable_on=gates[0],
            enable_on=gates[1],
            priority=data.get("priority", 3),
        )


# What a saved function that cannot be loaded raises: a name saved instead of
# the function, or a function whose module or code is not available here
_LOAD_ERRORS = (
    TypeError,
    ValueError,
    AttributeError,
    ImportError,
    EOFError,
    pickle.UnpicklingError,
)


def _missing_controller_function(context):  # pylint: disable=unused-argument
    """Stands in for a controller function that could not be loaded."""
    return None


def _positional_count(function):
    """Return how many positional arguments ``function`` takes, or -1 if it
    takes any number."""
    parameters = inspect.signature(function).parameters.values()
    if any(p.kind == p.VAR_POSITIONAL for p in parameters):
        return -1
    return sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
    )


def _check_controller_function(function):
    """Raise unless ``function`` takes the context as its one argument."""
    if not callable(function):
        raise TypeError(f"The controller function must be callable, got {function!r}.")
    try:
        inspect.signature(function).bind(None)
    except TypeError as error:
        raise ValueError(
            "The controller function must take the event context as its "
            "single argument, like `def controller(context): ...`."
        ) from error
    return function


def _writes_names(writes):
    """Return ``writes`` as a tuple of channel and actuator names, or None."""
    if writes is None:
        return None
    if isinstance(writes, str) or not isinstance(writes, (list, tuple, set)):
        writes = [writes]
    names = []
    for entry in writes:
        if isinstance(entry, str):
            if not entry:
                raise ValueError("An entry of `writes` is an empty name.")
            names.append(entry)
        elif hasattr(entry, "name") and hasattr(entry, "controls"):
            names.append(control_key(entry.name))
        else:
            raise TypeError(
                "`writes` takes channel names such as "
                "'air_brakes.deployment_level', actuator names such as "
                f"'air_brakes', or actuators, got {entry!r}."
            )
    return tuple(names)


class _PositionalController:
    """Run an air brakes controller function written the deprecated way, with
    positional arguments: ``(time, sampling_rate, state, observed_variables,
    air_brakes[, sensors[, environment]])``.

    The air brakes are found in the flight's rocket by name at every run, so
    nothing here points at an object and a saved copy runs the same.
    """

    def __init__(self, function, sampling_rate, actuator_name=None):
        self.function = function
        self.sampling_rate = sampling_rate
        # None: the rocket's only actuator with controls
        self.actuator_key = (
            None if actuator_name is None else control_key(actuator_name)
        )
        self.__name__ = getattr(function, "__name__", "controller_function")
        self._count = _positional_count(function)

    def __call__(self, context):
        flight = context["flight"]
        actuator = _with_controls(self._actuator(context["rocket"]), flight)
        arguments = [
            context["time"],
            self.sampling_rate,
            context["canonical_state"],
            context["event"].memory.get("observed_variables", []),
            actuator,
            context["sensors"],
            context["environment"],
        ]
        if self._count >= 0:
            arguments = arguments[: self._count]
        return self.function(*arguments)

    def _actuator(self, rocket):
        """Return the actuator this controller drives."""
        candidates = [
            actuator
            for actuator in rocket.actuators
            if actuator.controls
            and (
                self.actuator_key is None
                or control_key(actuator.name) == self.actuator_key
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"The controller function '{self.__name__}' moves one actuator, "
                f"but the rocket has {len(candidates)} that it could be. Write "
                "it as `controller_function(context)` and set "
                "`context.controls[...]` instead."
            )
        return candidates[0]


def _with_controls(obj, flight):
    """Return ``obj`` with its controls in the flight's control inputs, or
    ``obj`` itself when it has no channels in this flight."""
    channels = getattr(flight, "_control_channels", {}).get(obj)
    if channels is None:
        return obj
    return _ControlledObject(obj, dict(zip(obj.controls, channels)), flight)


class _ControlledObject:
    """An actuator, as a controller function written the deprecated way sees
    it.

    Its controls, such as the ``deployment_level`` of air brakes, are read and
    written in the flight's control inputs, so the controller moves the flight
    and never the object itself. Every other attribute is the object's own. A
    write is kept inside the control's limits right away, as the object did,
    so the controller reads back the value that flies.
    """

    __slots__ = ("_object", "_channels", "_flight")

    def __init__(self, obj, channels, flight):
        object.__setattr__(self, "_object", obj)
        object.__setattr__(self, "_channels", channels)
        object.__setattr__(self, "_flight", flight)

    def __getattr__(self, name):
        if name in _ControlledObject.__slots__:
            # Not set yet, as in an empty copy being filled in
            raise AttributeError(name)
        channel = self._channels.get(name)
        if channel is not None:
            return self._flight.control_signal[channel]
        return getattr(self._object, name)

    def __setattr__(self, name, value):
        if name in _ControlledObject.__slots__:
            object.__setattr__(self, name, value)
            return
        channel = self._channels.get(name)
        if channel is None:
            setattr(self._object, name, value)
            return
        flight = self._flight
        bounds, clamp = flight._control_limits[channel]  # pylint: disable=protected-access
        flight.control_signal[channel] = limited_value(value, bounds, clamp, channel)

    def __repr__(self):
        return repr(self._object)
