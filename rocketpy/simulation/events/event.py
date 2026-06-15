import inspect
import warnings
from copy import deepcopy
from numbers import Real

from ..._logging import logger
from .commands import Commands
from .exact_time_solvers import (
    solve_exact_time_brentq,
    solve_exact_time_cubic_hermite,
    solve_exact_time_linear,
)

NEEDS_KEYS = frozenset({"state_dot", "pressure", "state_history"})

PRESETS = {
    "apogee": lambda **kwargs: (
        len(kwargs["flight"].solution) >= 2
        and kwargs["flight"].solution[-2][6] > 0 >= kwargs["state"][5]
    ),
    "burnout": lambda **kwargs: (
        kwargs.get("time") >= kwargs["rocket"].motor.burn_out_time
    ),
}


class Event:
    """Event helper with trigger/callback execution and exact-time support.

    An ``Event`` is the main way RocketPy reacts to conditions during a
    flight. It pairs a ``trigger`` predicate with a ``callback`` action: at
    each evaluation the trigger is checked and, when it returns ``True``,
    the callback runs. Callbacks can inspect the simulation state, store
    persistent data in ``context``, log return values, and queue commands
    (through ``event.commands``) that modify the simulation, such as
    starting a new flight phase, replacing the derivative, scheduling other
    events, or terminating the flight.

    See :ref:`eventusage` for a full guide with runnable examples.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        callback,
        trigger=None,
        sampling_rate=None,
        context=None,
        disable_on=None,
        enable_on=None,
        exact_time_function=None,
        exact_time_config=None,
        trigger_only_once=False,
        time_overshootable=True,
        changes_dynamics=False,
        name="Custom Event",
        enabled=True,
        verbose=False,
        priority=4,
        needs=None,
    ):
        """Initialize an Event object.

        Parameters
        ----------
        callback : function
            Required callable executed when the event triggers. It must accept
            ``**kwargs`` and return ``None`` or a ``dict`` (appended to
            ``self.callback_log``). Queue commands via
            ``kwargs["event"].commands`` and access persistent state via
            ``kwargs["event"].context``.
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
            ``event`` (this :class:`Event` instance),
            ``sampling_rate`` (float, Hz or ``None``).
            The following keys are only injected when declared via ``needs``:
            ``pressure`` (float, Pa),
            ``state_dot`` (list, time derivative of ``state``),
            ``state_history`` (list of past state vectors).
        trigger : function, optional
            Predicate that returns ``True`` when the event should fire. It
            receives the same ``**kwargs`` as ``callback`` (see above), but
            its return value is interpreted as a boolean rather than logged.
            If ``None`` (default), the event fires every time it is evaluated.
        sampling_rate : float, optional
            Evaluation frequency in hertz. If ``None`` (default), the event is
            evaluated continuously at every solver time step. If a float (e.g.
            ``10``), the event is sampled at that rate, i.e. every
            ``1 / sampling_rate`` seconds.
        context : dict, optional
            Dictionary of persistent, mutable per-event state, exposed through
            ``kwargs["event"].context`` inside the trigger and callback. Useful
            for counters, thresholds, and data shared between trigger and
            callback. Each key is also unpacked into the ``**kwargs`` passed to
            the trigger and callback. Defaults to an empty dict. Note that
            ``context`` is not persisted to output logs or files.
        disable_on : str or int or float or callable, optional
            Condition that automatically disables the event. May be a string
            preset (``"apogee"`` or ``"burnout"``), a simulation time in seconds
            (int or float), or a callable ``function(**kwargs)`` that returns
            ``True`` when the event should be disabled. The times at which the
            event is disabled are recorded in ``self.disabled_times``.
        enable_on : str or int or float or callable, optional
            Condition that automatically (re-)enables a disabled event. Uses the
            same formats as ``disable_on`` (string preset, time threshold, or
            callable predicate). The times at which the event is enabled are
            recorded in ``self.enabled_times``.
        exact_time_function : function, optional
            Callable used to refine the trigger instant to an exact time between
            solver steps, with signature
            ``exact_time_function(state, **kwargs) -> float``. The mandatory
            ``state`` argument receives the interpolated solver state vector
            (without time); additional keyword arguments are the usual event
            kwargs. The function must return a scalar whose root (zero crossing,
            or the configured ``target``) defines the event instant, and it must
            derive that quantity directly from ``state`` rather than from derived
            kwargs such as ``height_agl``. Only supported for
            continuous events (``sampling_rate=None``).
        exact_time_config : dict, optional
            Configuration for the exact-time solver. The ``"solver"`` key selects
            the algorithm: ``"linear"``, ``"cubic_hermite"``, or ``"brentq"``
            (the default when omitted). All solvers accept a ``target`` float
            (default ``0.0``); ``brentq`` additionally accepts ``xtol``, ``rtol``,
            and ``maxiter``, and ``cubic_hermite`` requires a
            ``derivative_function``. If ``None`` or empty (default), the
            ``brentq`` solver is used with its defaults. See :ref:`eventusage`
            for the full list of keys.
        trigger_only_once : bool, optional
            If ``True``, the event disables itself after the first successful
            trigger. Useful for one-shot actions such as deployment or
            separation. Defaults to ``False``.
        time_overshootable : bool, optional
            Enables overshoot-path evaluation for sampled events. Only relevant
            when ``sampling_rate`` is a float. When ``True`` (default) the
            simulation may integrate past the next sampling time and step back to
            evaluate the event at the correct instant, allowing fewer integration
            steps and faster simulation. When ``False`` the solver places strict
            time nodes at multiples of the sampling interval, which is much
            slower with no gain in accuracy. Automatically forced to ``False``
            when ``sampling_rate`` is ``None``.
        changes_dynamics : bool, optional
            Set to ``True`` when the callback changes the simulation dynamics or
            any parameter affecting the ODE derivative. This includes mutating an
            attribute of any simulation object, and using the
            ``set_derivative``, ``start_flight_phase``, or ``terminate_flight``
            commands. Defaults to ``False``.
        name : str, optional
            Human-readable identifier used in logs and debugging. Defaults to
            ``"Custom Event"``.
        enabled : bool, optional
            Initial enabled state. Disabled events can be re-enabled through the
            ``enable`` command or via the ``enable_on`` parameter. Defaults to
            ``True``.
        verbose : bool, optional
            When ``True``, the event prints a message and stores extra execution
            logs in ``self.verbose_log`` whenever it triggers. Defaults to
            ``False``.
        priority : int, optional
            Integer event evaluation priority; lower numbers are evaluated
            earlier. String aliases are not supported. Recommended mapping (used
            by built-in events):

            - 0: Core events (out of rail, apogee, landing)
            - 1: Sensor events
            - 2: Parachute events
            - 3: Controller events
            - 4: Custom / user-defined events (default)
        needs : list of str or None, optional
            Declares which expensive simulation values the event's trigger and
            callback actually access. Valid keys are ``'state_dot'``,
            ``'pressure'``, and ``'state_history'``. The default``None`` is
            treated as an empty set and no expensive kwargs are computed.
            Supply a list with the keys this event accesses so the runtime
            computes them.

        See Also
        --------
        :ref:`eventusage` : User guide for building and using events.
        """
        self.callback = self.__validate_callback(callback)
        self.name = name
        needs_set = frozenset(needs) if needs is not None else frozenset()
        invalid = needs_set - NEEDS_KEYS
        if invalid:
            raise ValueError(
                f"Unknown needs keys: {invalid!r}. Valid keys: {sorted(NEEDS_KEYS)!r}. "
                "Note: 'height_agl' is always computed and does "
                "not need to be declared."
            )
        self.needs = needs_set
        self.trigger = self.__validate_trigger(trigger) if trigger is not None else None
        self.context = context if context is not None else {}
        self.sampling_rate = sampling_rate
        self.trigger_only_once = trigger_only_once
        self.changes_dynamics = bool(changes_dynamics)
        self.priority = priority
        self.time_overshootable = bool(time_overshootable)
        if self.time_overshootable and self.sampling_rate is None:
            self.time_overshootable = False
        self._initial_enabled = bool(enabled)
        self.enabled = bool(enabled)
        self.verbose = verbose
        self.verbose_log = []
        self.callback_log = []
        self.triggered_times = []
        self._trigger_checked = False

        self.is_discrete = self.sampling_rate is not None
        self.sampling_interval = (
            None if sampling_rate is None else 1.0 / float(sampling_rate)
        )

        # Track times when the event was enabled/disabled during a run.
        # These are lists of timestamps (floats) in simulation time.
        self.enabled_times = []
        self.disabled_times = []
        self._initial_context = deepcopy(self.context)
        self._initial_enabled_times = list(self.enabled_times)
        self._initial_disabled_times = list(self.disabled_times)

        self.commands = Commands()

        self.disable_on = self.__validate_gate_condition(disable_on, "disable_on")
        self.enable_on = self.__validate_gate_condition(enable_on, "enable_on")

        self.exact_time_function = self.__validate_exact_time_function(
            exact_time_function
        )
        self.exact_time_config = exact_time_config if exact_time_config else {}

        exact_time_solver_name = self.exact_time_config.get("solver", "brentq")
        if exact_time_solver_name == "linear":
            self.exact_time_solver = solve_exact_time_linear
        elif exact_time_solver_name == "cubic_hermite":
            self.exact_time_solver = solve_exact_time_cubic_hermite
        elif exact_time_solver_name == "brentq":
            self.exact_time_solver = solve_exact_time_brentq
        else:
            raise ValueError(f"Unknown exact-time solver: {exact_time_solver_name}")

        if self.exact_time_function is not None and self.sampling_rate is not None:
            raise ValueError(
                "exact_time_function is only supported for continuous hooks "
                "with sampling_rate=None."
            )

    def _reset_commands(self):
        self.commands.reset()

    def reset(self):
        """Reset event runtime state.

        This clears per-run command/results state and internal logging buffers,
        restores the initial ``enabled`` flag, and restores ``context`` to its
        construction-time snapshot.

        Returns
        -------
        None
        """
        self._reset_commands()
        self.enabled = self._initial_enabled
        self.verbose_log.clear()
        self.callback_log.clear()
        self.triggered_times.clear()
        self._trigger_checked = False
        self.context = deepcopy(self._initial_context)

        # Restore enable/disable time history to initial snapshot
        self.enabled_times = list(self._initial_enabled_times)
        self.disabled_times = list(self._initial_disabled_times)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled}, "
            f"sampling_rate={self.sampling_rate}, "
            f"time_overshootable={self.time_overshootable}, "
            f"trigger_only_once={self.trigger_only_once}, "
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} '{self.name}' (enabled={self.enabled}, "
            f"sampling_rate={self.sampling_rate}, "
            f"time_overshootable={self.time_overshootable}, "
            f"trigger_only_once={self.trigger_only_once}, "
        )

    def __call__(self, trigger_only=False, callback_only=False, reset=True, **kwargs):
        """Evaluate the event trigger and execute the callback if triggered.

        Parameters
        ----------
        trigger_only : bool, optional
            If True, only evaluate the trigger condition without executing the
            callback. The enable_on function is also called.
        callback_only : bool, optional
            If True, only execute the callback without evaluating the trigger
            condition. The exact time function and disable_on function are also
            called.
        kwargs : dict
            Keyword arguments passed to the trigger and callback functions.

        Returns
        -------
        bool
            True if the event was triggered, False otherwise.
        """
        if self.enabled is False:
            # If event is disabled, only evaluate enable_on function if it exists.
            if self._call_enable_on(**kwargs) is False:
                return False

        if reset:
            self._reset_commands()

        kwargs["event"] = self
        kwargs["sampling_rate"] = self.sampling_rate

        # --- Trigger Phase ---
        # Skip evaluating triggers if we are only running the callback.
        if callback_only is False:
            if self._call_enable_on(**kwargs) is False:
                return False

            if self._call_trigger(**kwargs) is False:
                return False

            if trigger_only:
                return True

        # --- Callback Phase ---
        kwargs = self._call_exact_time(**kwargs)

        try:
            callback_log = self.callback(**kwargs)
        except KeyError as exc:
            key = exc.args[0] if exc.args else None
            if isinstance(key, str) and key in NEEDS_KEYS:
                raise KeyError(
                    f"{key!r} is not available in event '{self.name}' callback kwargs. "
                    f"Add it to the event's needs parameter: Event(..., needs=[{key!r}])"
                ) from exc
            raise

        self.callback_log.append(callback_log)
        self.triggered_times.append(kwargs.get("time"))

        self._call_disable_on(**kwargs)

        if self.trigger_only_once:
            self.commands.disable()

        self._log(
            triggered=True,
            kwargs=kwargs,
        )
        return True

    def _call_enable_on(self, **kwargs):
        if not self.enabled:
            # No enable_on function. Event stays disabled
            if self.enable_on is None:
                self._log(triggered=False, kwargs=kwargs)
                return False
            try:
                if not self.enable_on(**kwargs):
                    self._log(triggered=False, kwargs=kwargs)
                    return False
                self.commands.enable()
            except Exception as e:  # pylint: disable=W0718
                warnings.warn(
                    f"Error evaluating enable_on for event '{self.name}': {e}",
                    UserWarning,
                )
                self._log(triggered=False, kwargs=kwargs)
                return False

    def _call_trigger(self, **kwargs):
        if self.trigger is not None:
            try:
                result = self.trigger(**kwargs)
            except KeyError as exc:
                key = exc.args[0] if exc.args else None
                if isinstance(key, str) and key in NEEDS_KEYS:
                    raise KeyError(
                        f"{key!r} is not available in event '{self.name}' trigger kwargs. "
                        f"Add it to the event's needs parameter: Event(..., needs=[{key!r}])"
                    ) from exc
                raise
            if not result:
                self._log(triggered=False, kwargs=kwargs)
                return False
        return True

    def _call_exact_time(self, **kwargs):
        if self.exact_time_function is not None:
            try:
                exact_time_result = self._compute_exact_time(**kwargs)
            except (ValueError, RuntimeError) as e:
                # Raise warning, and show error
                warnings.warn(
                    f"Event '{self.name}' trigger condition met, but exact-time "
                    "solving failed. Event trigger time will be approximated as "
                    "current step time."
                )
                warnings.warn(f"Exact-time solving error: {e}", UserWarning)
                exact_time_result = None
            if exact_time_result is not None:
                # Store original sampled time/state for callback access if needed
                kwargs["sampled_time"] = kwargs.get("time")
                kwargs["sampled_state"] = kwargs.get("state")
                # Update time and state in kwargs to exact values for callback
                kwargs["time"] = exact_time_result["event_time"]
                kwargs["state"] = exact_time_result["event_state"]
        return kwargs

    def _compute_exact_time(self, **kwargs):
        """Compute the exact trigger time and corresponding state if
        exact_time_function is set."""
        flight = kwargs.get("flight")
        phase = kwargs.get("phase")
        if len(flight.solution) < 2:
            self._log(
                triggered=True,
                kwargs=kwargs,
                callback_executed=False,
                skip_reason=(
                    "Trigger condition met, but callback was not executed "
                    "because exact-time solving requires at least two "
                    "solution points."
                ),
            )
            return None

        exact_time_result = self.exact_time_solver(
            previous_state=flight.solution[-2],
            current_state=flight.solution[-1],
            interpolator=phase.solver.dense_output(),
            event_function=self.exact_time_function,
            no_root_error_message=(
                "No valid roots found when solving exact event time for "
                f"event {self.name}"
            ),
            **self.exact_time_config,
            **kwargs,
        )

        self.commands.exact_time = exact_time_result["event_time"]
        self.commands.exact_state = exact_time_result["event_state"]

        return exact_time_result

    def _call_disable_on(self, **kwargs):
        if self.disable_on is not None:
            try:
                if self.disable_on(**kwargs):
                    self.commands.disable()
            except Exception as e:  # pylint: disable=W0718
                warnings.warn(
                    f"Error evaluating disable_on for event '{self.name}': {e}",
                    UserWarning,
                )

    def _log(
        self,
        triggered,
        kwargs,
        callback_executed=None,
        skip_reason=None,
    ):
        if self.verbose:
            self.verbose_log.append(
                {
                    "time": kwargs.get("time"),
                    "triggered": triggered,
                    "callback_executed": callback_executed,
                    "skip_reason": skip_reason,
                }
            )
        logger.debug(
            "Event '%s' at t=%s: triggered=%s, callback_executed=%s, skip_reason=%s",
            self.name,
            kwargs.get("time"),
            triggered,
            callback_executed,
            skip_reason,
        )

    def __validate_trigger(self, trigger):
        if isinstance(trigger, str):
            if trigger not in PRESETS:
                raise ValueError(
                    f"Unknown trigger preset: {trigger!r}. Supported presets: "
                    f"{list(PRESETS.keys())}"
                )
            return PRESETS[trigger]

        if isinstance(trigger, Real) and not isinstance(trigger, bool):
            return lambda **kwargs: kwargs.get("time") >= float(trigger)

        if not callable(trigger):
            raise ValueError("Trigger must be a callable, preset string, or number.")

        signature = inspect.signature(trigger)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            raise ValueError(
                "Trigger function must accept arbitrary keyword arguments (**kwargs)."
            )

        return_annotation = signature.return_annotation
        if return_annotation not in (inspect.Signature.empty, bool, "bool"):
            raise ValueError(
                "Trigger function return annotation must be bool when provided."
            )

        return trigger

    def __validate_callback(self, callback):
        if not callable(callback):
            raise ValueError("Callback must be a callable.")

        signature = inspect.signature(callback)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            raise ValueError(
                "Callback function must accept arbitrary keyword arguments (**kwargs)."
            )

        return_annotation = signature.return_annotation
        valid_return_annotations = (
            inspect.Signature.empty,
            type(None),
            None,
            dict,
            "dict",
        )

        if return_annotation not in valid_return_annotations:
            raise ValueError(
                "Callback function return annotation must be None, dict, or unspecified when provided."
            )

        return callback

    def __validate_gate_condition(self, condition, parameter_name):
        """Normalize a gate condition to a callable or None."""
        if condition is None:
            return None
        if isinstance(condition, str):
            if condition not in PRESETS:
                raise ValueError(
                    f"Unknown disable_on or enable_on preset: {condition!r}. "
                    f"Supported presets: {list(PRESETS.keys())}"
                )
            return PRESETS[condition]
        if isinstance(condition, Real) and not isinstance(condition, bool):
            return lambda **kwargs: kwargs.get("time") >= float(condition)
        if callable(condition):
            return condition
        raise TypeError(
            f"{parameter_name} must be None, a string preset, a number, or a callable"
        )

    def __validate_exact_time_function(self, exact_time_function):
        if exact_time_function is None:
            return None

        if not callable(exact_time_function):
            raise ValueError("exact_time_function must be callable or None.")

        signature = inspect.signature(exact_time_function)
        parameters = list(signature.parameters.values())
        if not parameters:
            raise ValueError(
                "exact_time_function must accept a mandatory 'state' argument and "
                "arbitrary keyword arguments (**kwargs)."
            )

        first_parameter = parameters[0]
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )

        if first_parameter.name != "state" or first_parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                "exact_time_function must accept 'state' as its first argument."
            )

        if not accepts_var_kwargs:
            raise ValueError(
                "exact_time_function must accept arbitrary keyword arguments (**kwargs)."
            )

        return exact_time_function
