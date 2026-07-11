import warnings
from importlib import import_module
from inspect import signature

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.simulation.events.event import Event
from rocketpy.tools import from_hex_decode, to_hex_encode

from ..plots.controller_plots import _ControllerPlots
from ..prints.controller_prints import _ControllerPrints


class Controller(Event):
    """A controller that modifies rocket state during simulation.

    A ``Controller`` is an :class:`Event` that executes at a fixed sampling
    rate, always fires (no trigger), and is expected to change the simulation
    dynamics (``changes_dynamics=True``, ``priority=3``). At each execution
    the user-supplied ``controller_function`` reads the simulation state and
    mutates the ``controlled_objects`` (e.g. an air brakes instance) to apply
    control actions.

    After every execution, the controller automatically records the
    ``control_state`` of each controlled object that exposes one (see
    :class:`rocketpy.control.controlled_object.ControlledObject`) into
    :attr:`control_history`, timestamped with the simulation time. This
    history powers post-flight prints/plots and open-loop replay through
    ``ScheduledController``.

    The controller function is responsible for:

    1. Reading simulation state and sensor data,
    2. Computing control actions,
    3. Mutating ``controlled_objects`` to apply those actions,
    4. Returning logging information (appended to :attr:`log`).

    Attributes
    ----------
    Controller.controlled_objects : object or list
        Object(s) the controller modifies, held by reference.
    Controller.control_history : dict
        ``{object_name: {variable: [(time, value), ...]}}`` recorded during
        the last simulation. Cleared at the start of each flight.
    Controller.log : list
        Per-execution return values of ``controller_function`` (alias
        :attr:`return_log`); same list as ``callback_log``.
    Controller.context : dict
        Persistent state shared across executions, restored to its
        construction-time snapshot at the start of each flight.
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
        needs=None,
    ):
        """Initialize the controller.

        Parameters
        ----------
        controller_function : callable
            Function that executes the control logic, with signature
            ``controller_function(**kwargs) -> dict or None``. Invoked once
            per sample; its return value is appended to :attr:`log`. Mutate
            ``controlled_objects`` directly to apply control actions.
            Functions with positional parameters are rejected.
            The following keys are always available in ``kwargs``:
            ``time`` (float, s),
            ``state`` (list ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``),
            ``sensors`` (list of sensor objects),
            ``sensors_by_name`` (dict of sensor objects),
            ``environment`` (:class:`rocketpy.Environment`),
            ``rocket`` (:class:`rocketpy.Rocket`),
            ``flight`` (:class:`rocketpy.Flight`),
            ``phase`` (current flight phase),
            ``step_size`` (float, s),
            ``height_agl`` (float, m),
            ``event`` (this :class:`Controller` instance),
            ``sampling_rate`` (float, Hz),
            ``controller`` (this :class:`Controller` instance),
            ``controlled_objects`` (the object(s) to mutate).
            If ``controlled_objects_name`` was set, those friendly names are
            also injected (plus ``controlled_objects_by_name`` for lists).
            The following keys are only injected when declared via ``needs``:
            ``pressure`` (float, Pa),
            ``state_dot`` (list, time derivative of ``state``),
            ``state_history`` (list of past state vectors).
        controlled_objects : object or list of object
            Object(s) the controller is allowed to modify (e.g. an air brakes
            instance). May be a single object or a list. They are held by
            reference, so mutations persist in the simulation. Objects that
            expose a ``control_state`` are automatically tracked in
            :attr:`control_history` and reset at the start of each flight.
        sampling_rate : float
            Rate in hertz at which the controller executes; it runs every
            ``1 / sampling_rate`` seconds.
        context : dict, optional
            Initial persistent state, accessed inside the controller function
            via ``kwargs["controller"].context`` and mutated in place to carry
            data across executions. Restored to this initial snapshot at the
            start of each flight. Defaults to an empty dict.
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
            Initial enabled state. If ``False``, the controller does not
            execute until re-enabled, either via the ``enable`` command or the
            ``enable_on`` condition. Defaults to ``True``.
        disable_on : str or int or float or callable, optional
            Condition that automatically disables the controller. May be a
            string preset (``"apogee"`` or ``"burnout"``), a simulation time in
            seconds (int or float), or a callable ``function(**kwargs)`` that
            returns ``True`` when the controller should be disabled. Defaults
            to ``None`` (no automatic disabling).
        enable_on : str or int or float or callable, optional
            Condition that automatically re-enables a disabled controller,
            using the same formats as ``disable_on``. Defaults to ``None``.
        needs : list or frozenset of str or None, optional
            Declares which expensive simulation values the controller function
            accesses. Valid keys: ``'state_dot'``, ``'pressure'``,
            ``'state_history'``. The default ``None`` is treated as an empty
            set and no expensive kwargs are computed.

        See Also
        --------
        :ref:`eventusage` : Description of the callback ``**kwargs``.
        """
        self.controller_function = self.__validate_controller_function(
            controller_function
        )
        self.bind_controlled_objects(controlled_objects, controlled_objects_name)
        super().__init__(
            callback=self._controller_callback,
            trigger=None,
            sampling_rate=sampling_rate,
            context=context,
            disable_on=disable_on,
            enable_on=enable_on,
            trigger_only_once=False,
            changes_dynamics=True,
            name=name,
            enabled=enabled,
            priority=3,
            needs=needs,
        )
        self.prints = _ControllerPrints(self)
        self.plots = _ControllerPlots(self)

    def __validate_controller_function(self, controller_function):
        """Require a keyword-only callable; reject positional signatures."""
        if not callable(controller_function):
            raise ValueError("controller_function must be a callable.")
        try:
            parameters = signature(controller_function).parameters.values()
        except (TypeError, ValueError):
            # Builtins / callables without an inspectable signature: allow.
            return controller_function
        has_positional = any(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
            for p in parameters
        )
        if has_positional:
            raise ValueError(
                "controller_function must accept keyword arguments only. "
                "Define it as `def controller_function(**kwargs):` and read "
                "values such as kwargs['time'], kwargs['state'], "
                "kwargs['sensors'] and kwargs['environment']. Support for "
                "positional controller signatures was removed; see the "
                "controller documentation for the full list of available "
                "keyword arguments."
            )
        return controller_function

    def bind_controlled_objects(self, controlled_objects, controlled_objects_name=None):
        """Bind (or re-bind) the objects this controller drives.

        Normalizes and validates ``controlled_objects`` /
        ``controlled_objects_name``, rebuilds the friendly-name callback
        bindings, and rebuilds the control-state tracking table used by
        :attr:`control_history`. Used at construction and when re-wiring a
        deserialized controller to freshly loaded objects.

        Parameters
        ----------
        controlled_objects : object or list of object
            Object(s) the controller is allowed to modify.
        controlled_objects_name : str or list of str, optional
            Friendly name(s) for the callback kwargs; same semantics as in
            the constructor.
        """
        self.controlled_objects = controlled_objects
        self.controlled_objects_name = controlled_objects_name
        self._controlled_objects_bindings = self.__verify_controlled_objects_name()

        if isinstance(controlled_objects, (list, tuple)):
            objects_list = list(controlled_objects)
        else:
            objects_list = [controlled_objects]
        self._controlled_objects_list = objects_list

        # Friendly names double as tracking names when they map one-to-one.
        if isinstance(controlled_objects_name, str) and len(objects_list) == 1:
            names = [controlled_objects_name]
        elif isinstance(controlled_objects_name, (list, tuple)):
            names = list(controlled_objects_name)
        else:
            names = [None] * len(objects_list)

        tracked = []
        used_names = set()
        for i, obj in enumerate(objects_list):
            if not hasattr(obj, "control_state"):
                continue
            base_name = (names[i] if i < len(names) else None) or getattr(
                obj, "name", f"object_{i}"
            )
            unique_name = base_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(unique_name)
            tracked.append((unique_name, obj))
        self._tracked = tracked
        self.control_history = {
            name: {var: [] for var in self.__control_variables(obj)}
            for name, obj in tracked
        }

    @staticmethod
    def __control_variables(obj):
        """Ordered control-variable names of a controlled object."""
        return getattr(obj, "control_variables", None) or list(obj.control_state)

    def _controller_callback(self, **kwargs):
        """Event callback: run the controller function, then snapshot the
        control state of every tracked object at the execution time."""
        kwargs["controller"] = self
        kwargs["controlled_objects"] = self.controlled_objects
        if self._controlled_objects_bindings:
            kwargs.update(self._controlled_objects_bindings)
        result = self.controller_function(**kwargs)
        self._record_control_state(kwargs.get("time"))
        return result

    def _record_control_state(self, time):
        """Append the current control state of each tracked object to
        :attr:`control_history`, timestamped with ``time``."""
        for name, obj in self._tracked:
            history = self.control_history[name]
            state = obj.control_state
            for variable in history:
                history[variable].append((time, state[variable]))

    @property
    def recorded_schedule(self):
        """Recorded control history as ``Function`` objects of time.

        Returns
        -------
        dict
            ``{object_name: {variable: Function}}`` built from
            :attr:`control_history` (linear interpolation, constant
            extrapolation). Objects/variables without samples are omitted.
        """
        schedule = {}
        for object_name, variables in self.control_history.items():
            object_schedule = {}
            for variable, samples in variables.items():
                if not samples:
                    continue
                if len(samples) == 1:
                    source = samples[0][1]
                else:
                    source = np.array(samples)
                object_schedule[variable] = Function(
                    source,
                    inputs="Time (s)",
                    outputs=variable,
                    interpolation="linear",
                    extrapolation="constant",
                )
            if object_schedule:
                schedule[object_name] = object_schedule
        return schedule

    def reset(self):
        """Reset controller runtime state.

        In addition to the :class:`Event` reset (commands, logs, enabled
        flag, ``context`` snapshot), clears the recorded
        :attr:`control_history` and restores every tracked controlled object
        to its initial control state. Called by ``Flight`` at the start of
        each simulation.
        """
        super().reset()
        for variables in self.control_history.values():
            for variable in variables:
                variables[variable] = []
        for _, obj in self._tracked:
            if hasattr(obj, "_reset"):
                obj._reset()

    @property
    def log(self):
        """Per-execution return values of ``controller_function`` (same list
        as ``callback_log``)."""
        return self.callback_log

    @log.setter
    def log(self, value):
        self.callback_log = value

    @property
    def return_log(self):
        """Alias for :attr:`log`."""
        return self.log

    @return_log.setter
    def return_log(self, value):
        self.log = value

    def __str__(self):
        return f"Controller '{self.name}' with sampling rate {self.sampling_rate} Hz."

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
            "height_agl",
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
        """Prints and plots all information about the controller."""
        self.info()
        self.plots.all()

    def to_dict(self, **kwargs):
        """Serialize controller to dictionary.

        Parameters
        ----------
        **kwargs : dict
            allow_pickle : bool, optional
                If True, serialize controller_function, disable_on, and
                enable_on callables using hex encoding. If False, use function
                name. Default is True.
            include_outputs : bool, optional
                If True, include the recorded ``control_history`` so a loaded
                controller can expose its schedule (and be replayed) without
                re-simulating. Default is False.

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

        data = {
            "controller_function": controller_function,
            "sampling_rate": self.sampling_rate,
            "name": self.name,
            "controlled_objects_name": self.controlled_objects_name,
            "context": self.context.copy(),  # Preserve context state
            "enabled": self.enabled,
            "disable_on": disable_on,
            "enable_on": enable_on,
            "needs": sorted(self.needs),
            # Controlled objects are not serialized here; their object names
            # are recorded so Rocket deserialization can rewire them to the
            # freshly decoded surfaces by name.
            "controlled_objects_ref": [
                getattr(obj, "name", tracked_name)
                for tracked_name, obj in self._tracked
            ]
            or getattr(self, "_controlled_objects_ref", []),
        }
        if kwargs.get("include_outputs", False):
            data["control_history"] = {
                object_name: {
                    variable: [list(sample) for sample in samples]
                    for variable, samples in variables.items()
                }
                for object_name, variables in self.control_history.items()
            }
        return data

    @classmethod
    def from_dict(cls, data, controlled_objects=None):
        """Reconstruct controller from dictionary.

        Parameters
        ----------
        data : dict
            Serialized controller data from to_dict().
        controlled_objects : list or object, optional
            Objects the controller will mutate. If not provided, must be
            re-bound after reconstruction via ``bind_controlled_objects``.

        Returns
        -------
        Controller
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
        needs = data.get("needs") or None
        control_history = data.get("control_history")

        try:
            controller_function = from_hex_decode(controller_function)
        except (TypeError, ValueError):
            pass
        if not callable(controller_function):
            # The controller function could not be restored (e.g. it was
            # serialized by name only, or unpickling failed across
            # environments). If a recorded control history is available, fall
            # back to replaying it open-loop.
            if control_history:
                warnings.warn(
                    f"The controller function of '{name}' could not be "
                    "restored; building a ScheduledController that replays "
                    "the recorded control schedule open-loop instead.",
                    UserWarning,
                )
                return cls._scheduled_from_history(data, controlled_objects)
            raise ValueError(
                f"Could not restore the controller function of '{name}', and "
                "no recorded control history is available to replay. "
                "Re-create the controller manually from its original "
                "function."
            )

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

        pending_name = None
        if controlled_objects is None:
            # No objects to bind yet: construct unbound and keep the friendly
            # name around so a later ``bind_controlled_objects`` (e.g. during
            # Rocket deserialization) can restore it.
            controlled_objects = []
            pending_name = controlled_objects_name
            controlled_objects_name = None

        controller = cls._construct(
            controller_function,
            controlled_objects,
            sampling_rate,
            name=name,
            context=context,
            controlled_objects_name=controlled_objects_name,
            enabled=enabled,
            disable_on=disable_on,
            enable_on=enable_on,
            needs=needs,
        )
        if pending_name is not None:
            controller.controlled_objects_name = pending_name
        cls._restore_serialized_state(controller, data)
        return controller

    @classmethod
    def _construct(
        cls, controller_function, controlled_objects, sampling_rate, **kwargs
    ):
        """Constructor hook for ``from_dict``; subclasses whose ``__init__``
        renames or drops parameters override this."""
        return cls(controller_function, controlled_objects, sampling_rate, **kwargs)

    @staticmethod
    def _restore_serialized_state(controller, data):
        """Restore serialization-only attributes: the controlled-object name
        references used for rewiring, and any recorded control history (so
        ``recorded_schedule`` works on a loaded controller)."""
        controller._controlled_objects_ref = data.get("controlled_objects_ref", [])
        control_history = data.get("control_history")
        if control_history:
            controller.control_history = {
                object_name: {
                    variable: [tuple(sample) for sample in samples]
                    for variable, samples in variables.items()
                }
                for object_name, variables in control_history.items()
            }

    @classmethod
    def _scheduled_from_history(cls, data, controlled_objects):
        """Build a ScheduledController replaying the recorded control history
        of a serialized controller whose function could not be restored."""
        # Resolved dynamically to avoid a circular import between this module
        # and scheduled_controller (which subclasses Controller).
        scheduled_controller_class = import_module(
            "rocketpy.control.scheduled_controller"
        ).ScheduledController

        schedule = {
            object_name: {
                variable: np.array(samples)
                for variable, samples in variables.items()
                if samples
            }
            for object_name, variables in data["control_history"].items()
        }
        schedule = {
            object_name: variables
            for object_name, variables in schedule.items()
            if variables
        }
        controller = scheduled_controller_class(
            schedule=schedule,
            controlled_objects=(
                controlled_objects if controlled_objects is not None else []
            ),
            sampling_rate=data.get("sampling_rate"),
            context=data.get("context", {}),
            name=data.get("name", "Controller"),
            enabled=data.get("enabled", True),
        )
        controller.controlled_objects_name = data.get("controlled_objects_name")
        Controller._restore_serialized_state(controller, data)
        return controller
