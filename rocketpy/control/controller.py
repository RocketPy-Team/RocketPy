import warnings
from inspect import signature
from keyword import iskeyword

from rocketpy.simulation.events.event import Event
from rocketpy.tools import from_hex_decode, to_hex_encode

from ..prints.controller_prints import _ControllerPrints
from .controlled import Controlled


class _Controller:
    """A controller that modifies rocket state during simulation.

    Controllers execute at a fixed sampling rate and can mutate rocket
    objects (e.g. air brakes, fins) during flight. Like :class:`Event`
    objects, controllers use a callback pattern with a persistent ``memory``,
    but they explicitly expect external object state to change.

    Internally a controller is a thin wrapper around an :class:`Event`: it
    builds an event (see :meth:`to_event`) whose callback invokes the
    user-supplied ``controller_function``. The wrapping event is created with
    ``changes_dynamics=True``, ``trigger_only_once=False``, and
    ``priority=3``, and it mirrors the controller's ``enabled`` flag,
    ``memory``, ``sampling_rate``, ``disable_on`` and ``enable_on`` settings.

    The controller function is responsible for:

    1. Reading simulation state and sensor data,
    2. Computing control actions,
    3. Mutating ``controlled_objects`` to apply those actions,
    4. Returning logging information (appended to :attr:`log`).

    The key difference from :class:`Event` is that object mutations are
    intentional and expected -- this is the controller's primary purpose.

    Attributes
    ----------
    event : Event
        The wrapping :class:`Event` consumed by the simulation loop.
    log : list
        Per-execution return values of ``controller_function`` (alias
        :attr:`return_log`). Backed by the wrapped event's ``callback_log``.
    enabled : bool
        Current enabled state, mirrored from the wrapped event.
    context : dict
        Persistent state shared across executions.
    """

    def __init__(
        self,
        controller_function,
        controlled_objects,
        sampling_rate,
        memory=None,
        name="Controller",
        controlled_objects_name=None,
        enabled=True,
        disable_on=None,
        enable_on=None,
    ):
        """Initialize the controller.

        Parameters
        ----------
        controller_function : callable
            Function that executes the control logic, with signature
            ``controller_function(context) -> dict or None``. Invoked once
            per sample; its return value is appended to :attr:`log`. Mutate
            ``controlled_objects`` directly to apply control actions.
            ``context`` holds the values listed in :class:`rocketpy.Event`
            (``context.time``, ``context.state``, ``context.height_agl``,
            ``context.sensors``, ``context.environment``, ``context.rocket``,
            ``context.flight``, ``context.state_dot``, ``context.pressure``,
            ``context.previous_state`` and so on) plus ``context.controller``
            (this :class:`_Controller` instance) and ``context.controlled``
            (the objects to mutate). Read a controlled object by the name
            given in ``controlled_objects_name``, such as
            ``context.controlled.air_brakes``, or by its position, such as
            ``context.controlled[0]``.
            For the trajectory itself, read ``context.flight.solution``.
        controlled_objects : object or list of object
            Object(s) the controller is allowed to modify (e.g. an air brakes
            instance). May be a single object or a list. They are held by
            reference, so mutations persist in the simulation.
        sampling_rate : float
            Rate in hertz at which the controller executes; it runs every
            ``1 / sampling_rate`` seconds.
        memory : dict, optional
            The controller's own dictionary, kept from one run to the next.
            Read and write it as ``context.controller.memory``. It is the
            same dict as the wrapped event's ``memory``. Defaults to an empty
            dict.
        name : str, optional
            Human-readable controller name, used for identification and
            logging. Defaults to ``"Controller"``.
        controlled_objects_name : str or list of str, optional
            Name(s) under which the controller function reads the controlled
            objects, as ``context.controlled.<name>``. For example, with
            ``controlled_objects_name="air_brakes"`` the function reads
            ``context.controlled.air_brakes``. Pass a single string for a
            single object, or a list/tuple of unique strings, one per object,
            when ``controlled_objects`` is a list. Each name must be usable
            after a dot in Python: letters, digits and underscores, not
            starting with a digit or an underscore. Defaults to ``None``, in
            which case the objects are read by position, as
            ``context.controlled[0]``.
        enabled : bool, optional
            Initial enabled state of the wrapped event. If ``False``, the
            controller does not execute until re-enabled, either via the
            ``enable`` command or the ``enable_on`` condition. Defaults to
            ``True``.
        disable_on : str or int or float or callable, optional
            Condition that automatically disables the controller. May be a
            string preset (``"apogee"`` or ``"burnout"``), a simulation time in
            seconds (int or float), or a callable ``function(context)`` that
            returns ``True`` when the controller should be disabled. The
            condition is forwarded to the wrapped event. Defaults to ``None``
            (no automatic disabling).
        enable_on : str or int or float or callable, optional
            Condition that automatically re-enables a disabled controller,
            using the same formats as ``disable_on``. When the condition is met
            while the controller is disabled, it re-enables before the next
            trigger evaluation. Defaults to ``None`` (no automatic enabling).

        See Also
        --------
        to_event : Builds the :class:`Event` that wraps this controller.
        :ref:`eventusage` : Description of the callback ``context``.
        """
        # TODO: rethink controllers
        self.controller_function = self.__evaluate_controller_function(
            controller_function
        )
        self.controlled_objects = controlled_objects
        # Optional name(s) the controller function reads the objects by.
        # Accept either a single string name or an iterable of string names
        self.controlled_objects_name = controlled_objects_name
        self._controlled = self.__build_controlled()
        self.sampling_rate = sampling_rate
        self.name = name
        self.memory = memory if memory is not None else {}
        self.prints = _ControllerPrints(self)
        self.enabled = enabled
        self.disable_on = disable_on
        self.enable_on = enable_on

        # Create the event during initialization
        self.event = self.to_event()
        self.log = self.event.callback_log

    def __evaluate_controller_function(self, controller_function):
        """Detect legacy positional-argument signatures and wrap them for compatibility."""
        sig = signature(controller_function)
        params = list(sig.parameters.values())
        positional_count = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params
        )
        accepts_var_positional = any(p.kind == p.VAR_POSITIONAL for p in params)

        if positional_count == 1 and not accepts_var_positional:
            return controller_function

        if positional_count == 0 and not accepts_var_positional:
            raise ValueError(
                "controller_function must take the event context as its single "
                "positional argument, like `def controller(context): ...`."
            )

        warnings.warn(
            "It is recommended not to use positional arguments when defining "
            "a controller function. Instead, define it as "
            "`controller_function(context)` and read values such as "
            "`context.time`, `context.state`, `context.sensors` and "
            "`context.environment`. See the controller documentation for "
            "the full list of available values.",
            UserWarning,
            stacklevel=3,
        )

        def wrapped(context):
            args = [
                context["time"],
                self.sampling_rate,
                context["canonical_state"],
                self.log,
                self.controlled_objects,
            ]
            if positional_count >= 6 or accepts_var_positional:
                args.append(context["sensors"])
            if positional_count >= 7 or accepts_var_positional:
                args.append(context["environment"])
            return controller_function(*args)

        return wrapped

    def to_event(self):
        """Create an Event that wraps this controller for simulation execution.

        Returns
        -------
        Event
            Event configured for controller sampling rate and callback.

        Notes
        -----
        The Event callback directly invokes the controller function with
        proper parameters. The controller is responsible for mutating
        controlled_objects to apply control actions.
        """

        def controller_callback(context):
            """Execute controller and handle mutations.

            Parameters
            ----------
            context : EventContext
                Event context including:
                - time: float, simulation time
                - state: list, state vector
                - sensors: dict, sensor measurements
                - environment: Environment, environmental model
                - event: Event, the event object itself
                - controller: _Controller, this controller instance
                - (other standard Event context keys)

            Returns
            -------
            dict or None
                callback_log dict from controller function, logged to callback_log.
            """
            # These belong to this controller alone; the context is shared
            # with the other events of the step, so they are owned rather
            # than written outright and are removed when the next event binds.
            context.own(controller=self, controlled=self._controlled)
            return self.controller_function(context)

        return Event(
            callback=controller_callback,
            name=f"{self.name}",
            sampling_rate=self.sampling_rate,
            memory=self.memory,
            changes_dynamics=True,
            trigger_only_once=False,
            enabled=self.enabled,
            disable_on=self.disable_on,
            enable_on=self.enable_on,
            priority=3,
        )

    @property
    def enabled(self):
        """Return the current enabled state mirrored from the wrapped event."""
        if hasattr(self, "event"):
            return self.event.enabled
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)
        if hasattr(self, "event"):
            self.event.enabled = self._enabled

    def __str__(self):
        return f"Controller '{self.name}' with sampling rate {self.sampling_rate} Hz."

    @property
    def log(self):
        """Return the controller callback log."""
        return self._log

    @log.setter
    def log(self, value):
        self._log = value
        if hasattr(self, "event"):
            self.event.callback_log = value

    @property
    def return_log(self):
        """Alias for :attr:`log`."""
        return self.log

    @return_log.setter
    def return_log(self, value):
        self.log = value

    def __build_controlled(self):
        """Validate controlled_objects_name and build ``context.controlled``."""
        objects = self.controlled_objects
        names = self.controlled_objects_name
        many = isinstance(objects, (list, tuple))
        if names is None:
            return Controlled(objects if many else (objects,))

        if isinstance(names, str):
            # A single name stands for whatever was passed, object or list.
            names, objects = (names,), (objects,)
        elif not isinstance(names, (list, tuple)):
            raise TypeError(
                "controlled_objects_name must be a string or list/tuple of strings"
            )
        elif not many:
            raise ValueError(
                "controlled_objects_name is a list but controlled_objects is not "
                "a list/tuple"
            )

        if not all(isinstance(name, str) for name in names):
            raise TypeError(
                "All entries in controlled_objects_name list must be strings"
            )
        if len(set(names)) != len(names):
            raise ValueError("controlled_objects_name entries must be unique")
        if len(names) != len(objects):
            raise ValueError(
                "Length of controlled_objects_name must match number of "
                "controlled_objects"
            )
        for name in names:
            if not name.isidentifier() or iskeyword(name) or name.startswith("_"):
                raise ValueError(
                    f"controlled_objects_name entry '{name}' cannot be read as "
                    f"`context.controlled.{name}`. Use letters, digits and "
                    "underscores only, not starting with a digit or an "
                    "underscore, and not a Python keyword."
                )
        return Controlled(objects, names)

    def info(self):
        """Prints out summarized information about the controller."""
        self.prints.all()

    def all_info(self):
        """Prints out all information about the controller."""
        self.info()

    def to_dict(self, **kwargs):
        """Serialize controller to dictionary.

        Parameters
        ----------
        **kwargs : dict
            allow_pickle : bool, optional
                If True, serialize controller_function, disable_on, and
                enable_on callables using hex encoding. If False, use function
                name. Default is True.

        Returns
        -------
        dict
            Serialized controller state.
        """
        allow_pickle = kwargs.get("allow_pickle", True)

        if allow_pickle:
            controller_function = to_hex_encode(self.controller_function)
        else:
            controller_function = self.controller_function.__name__

        # Serialize gate conditions: if callable, use hex encoding; if string or None, keep as-is
        disable_on = self.disable_on
        if allow_pickle and callable(disable_on):
            disable_on = to_hex_encode(disable_on)

        enable_on = self.enable_on
        if allow_pickle and callable(enable_on):
            enable_on = to_hex_encode(enable_on)

        return {
            "controller_function": controller_function,
            "sampling_rate": self.sampling_rate,
            "name": self.name,
            "controlled_objects_name": getattr(self, "controlled_objects_name", None),
            # Hash(es) identifying the controlled object(s), so Rocket
            # deserialization can reconnect the controller to the rocket's own
            # reconstructed objects (see Rocket.from_dict).
            "controlled_objects_hash": self._controlled_objects_hash(),
            "memory": self.memory.copy(),
            "enabled": self.enabled,
            "disable_on": disable_on,
            "enable_on": enable_on,
            # Note: controlled_objects are recovered in from_dict via
            # object reference matching in Rocket deserialization
        }

    def _controlled_objects_hash(self):
        """Return the identity hash of the controlled object(s), matching the
        shape of ``controlled_objects`` (a single hash for a single object, a
        list of hashes for a list). These hashes match the ones the encoder
        stores in each object's signature, letting Rocket deserialization find
        the reconstructed objects. Returns ``None`` for anything unhashable."""

        def safe_hash(obj):
            try:
                return hash(obj)
            except TypeError:
                return None

        controlled_objects = self.controlled_objects
        if isinstance(controlled_objects, (list, tuple)):
            return [safe_hash(obj) for obj in controlled_objects]
        return safe_hash(controlled_objects)

    @classmethod
    def from_dict(cls, data, controlled_objects=None):
        """Reconstruct controller from dictionary.

        Parameters
        ----------
        data : dict
            Serialized controller data from to_dict().
        controlled_objects : list or object, optional
            Objects the controller will mutate. If not provided,
            must be set manually after reconstruction.

        Returns
        -------
        _Controller
            Reconstructed controller instance.
        """
        controller_function = data.get("controller_function")
        sampling_rate = data.get("sampling_rate")
        name = data.get("name", "Controller")
        controlled_objects_name = data.get("controlled_objects_name")
        memory = data.get("memory", {})
        enabled = data.get("enabled", True)
        disable_on = data.get("disable_on")
        enable_on = data.get("enable_on")

        try:
            controller_function = from_hex_decode(controller_function)
        except (TypeError, ValueError):
            pass

        # Deserialize disable_on: try hex decoding for callables, keep strings and None
        try:
            disable_on = from_hex_decode(disable_on)
        except (TypeError, ValueError):
            # If not hex-encoded, keep as string or None
            pass

        try:
            enable_on = from_hex_decode(enable_on)
        except (TypeError, ValueError):
            pass

        if controlled_objects is None:
            controlled_objects = []

        controller = cls(
            controller_function=controller_function,
            controlled_objects=controlled_objects,
            sampling_rate=sampling_rate,
            name=name,
            memory=memory,
            controlled_objects_name=controlled_objects_name,
            enabled=enabled,
            disable_on=disable_on,
            enable_on=enable_on,
        )
        # Stash the serialized controlled-object hash(es) so Rocket.from_dict
        # can reconnect the controller to the rocket's reconstructed objects.
        controller._serialized_controlled_objects_hash = data.get(
            "controlled_objects_hash"
        )
        return controller

    def rebind_controlled_objects(self, controlled_objects):
        """Point the controller at reconstructed controlled object(s) and
        rebuild ``context.controlled`` from them.

        Used when a rocket is loaded from a file: the controller is rebuilt
        without its controlled objects (they are separate objects in the saved
        data), so this reconnects it to the rocket's own objects, ensuring the
        controller mutates them rather than orphaned copies.

        Parameters
        ----------
        controlled_objects : object or list of object
            The reconstructed object(s) the controller should control.
        """
        self.controlled_objects = controlled_objects
        self._controlled = self.__build_controlled()
