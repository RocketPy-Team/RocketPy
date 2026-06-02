from rocketpy.simulation.events.event import Event
from rocketpy.tools import from_hex_decode, to_hex_encode

from ..prints.controller_prints import _ControllerPrints


class _Controller:
    """A controller that modifies rocket state during simulation.

    Controllers execute at specified sampling rates and can mutate rocket
    objects (e.g., air brakes, fins) during flight. Like Event objects,
    controllers use a callback pattern with persistent context, but
    explicitly expect external object state to change.

    The controller function is responsible for:
    1. Reading simulation state and sensor data
    2. Computing control actions
    3. Mutating controlled_objects to apply those actions
    4. Returning callback_log/logging information

    The key difference from Event: object mutations are intentional and
    expected. This is the controller's primary purpose.
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
        controller_function : callable # TODO: undo breaking change, and fix docs
            Function that executes control logic. Signature:

            .. code-block:: python

                def controller_function(**kwargs) -> dict:
                    '''
                    Computes and applies control actions.

                    Parameters
                    ----------
                    **kwargs : dict
                        time : float
                            Current simulation time (seconds).
                        state : list
                            State vector [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].
                        state_history : list
                            History of previous state vectors.
                        sensors : dict
                            Sensor data (by sensor name).
                        environment : Environment
                            Environmental object.
                        rocket : Rocket
                            Rocket object.
                        flight : Flight
                            Flight object.
                        event : Event
                            The event object. Access context via event.context.
                        controller : _Controller
                            This controller instance. Access controlled_objects via
                            controller.controlled_objects and persistent state via
                            controller.context.
                        (+ other standard Event kwargs)

                    Returns
                    -------
                    dict or None
                        callback_log/logging dict. Keys are user-defined.
                        Return None if no callback_log needed.

                    Example
                    -------
                    .. code-block:: python

                        def my_controller(**kwargs):
                            time = kwargs["time"]
                            state = kwargs["state"]
                            controller = kwargs["controller"]

                            # Access persistent state
                            context = controller.context
                            if "counter" not in context:
                                context["counter"] = 0
                            context["counter"] += 1

                            # Mutate controlled objects
                            air_brakes = controller.controlled_objects
                            if time > 5.0:
                                air_brakes.open()

                            return {"counter": context["counter"]}
                    '''

            The function should mutate controlled_objects directly.

        controlled_objects : list or object
            Object(s) that the controller can modify. Can be a single object
            or list of objects. These are passed by reference; mutations
            persist in the simulation.

        sampling_rate : float
            Rate (Hz) at which the controller executes. Controller runs
            every 1/sampling_rate seconds.

        name : str, optional
            Controller name (for identification and logging).

        context : dict, optional
            Initial persistent state. Passed to controller_function and
            modified in-place to track state across executions.
            Defaults to empty dict.

        enabled : bool, optional
            Whether the wrapped event is initially enabled. If False, the
            controller will not execute during simulation until the wrapped
            event is enabled. Can be enabled via commands.enable().
            Defaults to True.

        disable_on : str or callable, optional
            Condition to automatically disable this controller. Can be:
            - A string preset: "apogee" or "burnout"
            - A callable that returns a boolean: True when controller should be disabled
            When the condition is met, the controller disables after the
            current execution. Defaults to None (no automatic disabling).

        enable_on : str or callable, optional
            Condition to automatically re-enable this controller. Can be:
            - A string preset: "apogee" or "burnout"
            - A callable that returns a boolean: True when controller should be enabled
            When the condition is met while the wrapped event is disabled,
            the wrapped event re-enables before trigger evaluation.
            Defaults to None (no automatic enabling).

        Returns
        -------
        None
        """
        self.controller_function = controller_function
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
