import warnings
from inspect import signature

from rocketpy.simulation.events.event import Event
from rocketpy.tools import from_hex_decode, to_hex_encode

from ..prints.controller_prints import _ControllerPrints


class _Controller:
    """A controller that modifies rocket state during simulation.

    Controllers execute at a fixed sampling rate and can mutate rocket
    objects (e.g. air brakes, fins) during flight. Like :class:`Event`
    objects, controllers use a callback pattern with persistent ``context``,
    but they explicitly expect external object state to change.

    Internally a controller is a thin wrapper around an :class:`Event`: it
    builds an event (see :meth:`to_event`) whose callback invokes the
    user-supplied ``controller_function``. The wrapping event is created with
    ``changes_dynamics=True``, ``trigger_only_once=False``, and
    ``priority=3``, and it mirrors the controller's ``enabled`` flag,
    ``context``, ``sampling_rate``, ``disable_on`` and ``enable_on`` settings.

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
        context=None,
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
            ``controller_function(**kwargs) -> dict or None``. It is invoked
            once per sample and receives the standard event ``**kwargs`` (see
            :ref:`eventusage` for the full list) plus controller-specific keys:

            - ``controller`` (:class:`_Controller`): this controller instance;
              read or write persistent state via ``controller.context`` and the
              targets via ``controller.controlled_objects``.
            - ``controlled_objects``: the same object(s) passed as
              ``controlled_objects``, for convenience.
            - one entry per name in ``controlled_objects_name`` (and, for a
              list of names, a ``controlled_objects_by_name`` mapping).

            Among the standard keys it also receives ``time``, ``state``,
            ``state_history``, ``sensors``, ``environment``, ``rocket``,
            ``flight`` and ``event``. The function should mutate
            ``controlled_objects`` directly to apply control actions; its
            return value (a dict of user-defined keys, or ``None``) is appended
            to :attr:`log`.
        controlled_objects : object or list of object
            Object(s) the controller is allowed to modify (e.g. an air brakes
            instance). May be a single object or a list. They are held by
            reference, so mutations persist in the simulation.
        sampling_rate : float
            Rate in hertz at which the controller executes; it runs every
            ``1 / sampling_rate`` seconds.
        context : dict, optional
            Initial persistent state, passed to ``controller_function`` and
            mutated in place to carry data across executions. The same dict is
            shared with the wrapped event's ``context``. Defaults to an empty
            dict.
        name : str, optional
            Human-readable controller name, used for identification and
            logging. Defaults to ``"Controller"``.
        controlled_objects_name : str or list of str, optional
            Friendly name(s) under which the controlled objects are exposed in
            the callback ``**kwargs``, so the function can access them as
            ``kwargs[name]`` instead of via ``controlled_objects``. Pass a
            single string for a single object, or a list/tuple of unique
            strings matching the length of ``controlled_objects`` for multiple
            objects (which also adds a ``controlled_objects_by_name`` mapping).
            Names must not collide with reserved callback keywords. Defaults to
            ``None`` (no friendly binding).
        enabled : bool, optional
            Initial enabled state of the wrapped event. If ``False``, the
            controller does not execute until re-enabled, either via the
            ``enable`` command or the ``enable_on`` condition. Defaults to
            ``True``.
        disable_on : str or int or float or callable, optional
            Condition that automatically disables the controller. May be a
            string preset (``"apogee"`` or ``"burnout"``), a simulation time in
            seconds (int or float), or a callable ``function(**kwargs)`` that
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
        :ref:`eventusage` : Description of the callback ``**kwargs``.
        """
        # TODO: undo the breaking change of the definition of controller function here
        # TODO: rethingk controllers
        self.controller_function = self.__evaluate_controller_function(
            controller_function
        )
        self.controlled_objects = controlled_objects
        # Optional friendly name(s) to expose controlled objects in callback kwargs
        # Accept either a single string name or an iterable of string names
        self.controlled_objects_name = controlled_objects_name
        self._controlled_objects_bindings = self.__verify_controlled_objects_name()
        self.sampling_rate = sampling_rate
        self.name = name
        self.context = context if context is not None else {}
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

        if positional_count > 0 or accepts_var_positional:
            warnings.warn(
                "It is recommended not to use positional arguments when defining "
                "a controller function. Instead, define the controller function "
                "to accept `**kwargs` only and read values such as "
                "`kwargs['time']`, `kwargs['state']`, `kwargs['sensors']` and "
                "`kwargs['environment']`. See the controller documentation for "
                "the full list of available keyword arguments.",
                UserWarning,
                stacklevel=3,
            )

            def wrapped(**kwargs):
                args = [
                    kwargs.get("time"),
                    self.sampling_rate,
                    kwargs.get("state"),
                    kwargs.get("state_history"),
                    self.log,
                    self.controlled_objects,
                ]
                if positional_count >= 7 or accepts_var_positional:
                    args.append(kwargs.get("sensors"))
                if positional_count >= 8 or accepts_var_positional:
                    args.append(kwargs.get("environment"))
                return controller_function(*args)

            return wrapped

        return controller_function

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

        def controller_callback(**kwargs):
            """Execute controller and handle mutations.

            Parameters
            ----------
            **kwargs : dict
                Event context including:
                - time: float, simulation time
                - state: list, state vector
                - state_history: list, state trajectory
                - sensors: dict, sensor measurements
                - environment: Environment, environmental model
                - event: Event, the event object itself
                - controller: _Controller, this controller instance
                - (other standard Event kwargs)

            Returns
            -------
            dict or None
                callback_log dict from controller function, logged to callback_log.
            """
            # Inject controller reference into kwargs (like kwargs["event"] for events)
            kwargs["controller"] = self
            kwargs["controlled_objects"] = self.controlled_objects

            # Also expose controlled objects under a user-provided name
            if self._controlled_objects_bindings:
                kwargs.update(self._controlled_objects_bindings)

            # Call controller function with kwargs directly
            # The function can access context via kwargs["controller"].context
            # and controlled_objects via kwargs["controlled_objects"] or
            # via the provided friendly name.
            callback_log = self.controller_function(**kwargs)
            return callback_log

        return Event(
            callback=controller_callback,
            name=f"{self.name}",
            sampling_rate=self.sampling_rate,
            context=self.context,  # Pass context to Event
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

    def __verify_controlled_objects_name(self):
        """Validate controlled_objects_name and build callback bindings."""
        if self.controlled_objects_name is None:
            return None

        single_name = isinstance(self.controlled_objects_name, str)
        list_names = isinstance(self.controlled_objects_name, (list, tuple))
        if not (single_name or list_names):
            raise TypeError(
                "controlled_objects_name must be a string or list/tuple of strings"
            )

        reserved = {
            "time",
            "state",
            "state_history",
            "sensors",
            "environment",
            "rocket",
            "flight",
            "event",
            "controller",
            "controlled_objects",
            "step_size",
            "state_dot",
            "sensors_by_name",
            "pressure",
            "height_above_ground_level",
            "callback_log",
            "triggered_times",
            "commands",
            "context",
        }

        if single_name:
            if self.controlled_objects_name in reserved:
                raise ValueError(
                    f"controlled_objects_name '{self.controlled_objects_name}' conflicts with reserved callback keywords"
                )
            return {self.controlled_objects_name: self.controlled_objects}

        if not all(isinstance(n, str) for n in self.controlled_objects_name):
            raise TypeError(
                "All entries in controlled_objects_name list must be strings"
            )
        if len(set(self.controlled_objects_name)) != len(self.controlled_objects_name):
            raise ValueError("controlled_objects_name entries must be unique")
        for n in self.controlled_objects_name:
            if n in reserved:
                raise ValueError(
                    f"controlled_objects_name entry '{n}' conflicts with reserved callback keywords"
                )
        if not isinstance(self.controlled_objects, (list, tuple)):
            raise ValueError(
                "controlled_objects_name is a list but controlled_objects is not a list/tuple"
            )
        if len(self.controlled_objects_name) != len(self.controlled_objects):
            raise ValueError(
                "Length of controlled_objects_name must match number of controlled_objects"
            )

        controlled_objects_by_name = dict(
            zip(self.controlled_objects_name, self.controlled_objects)
        )
        controlled_objects_bindings = dict(controlled_objects_by_name)
        controlled_objects_bindings["controlled_objects_by_name"] = (
            controlled_objects_by_name
        )
        return controlled_objects_bindings

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
            "context": self.context.copy(),  # Preserve context state
            "enabled": self.enabled,
            "disable_on": disable_on,
            "enable_on": enable_on,
            # Note: controlled_objects are recovered in from_dict via
            # object reference matching in Rocket deserialization
        }

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
        context = data.get("context", {})
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

        return cls(
            controller_function=controller_function,
            controlled_objects=controlled_objects,
            sampling_rate=sampling_rate,
            name=name,
            context=context,
            controlled_objects_name=controlled_objects_name,
            enabled=enabled,
            disable_on=disable_on,
            enable_on=enable_on,
        )
