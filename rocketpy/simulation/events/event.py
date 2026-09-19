import inspect
import warnings
from copy import deepcopy
from numbers import Real

from ..._logging import logger
from .commands import Commands
from .exact_time_solvers import SOLVERS

PRESETS = {
    "apogee": lambda context: (
        len(context["flight"].solution) >= 2
        and context["flight"].solution.at_index(-2)["vz"] > 0 >= context["state"][5]
    ),
    "burnout": lambda context: context["time"] >= context["rocket"].motor.burn_out_time,
}


class Event:
    """Event helper with trigger/callback execution and exact-time support.

    An ``Event`` is the main way RocketPy reacts to conditions during a
    flight. It pairs a ``trigger`` predicate with a ``callback`` action: at
    each evaluation the trigger is checked and, when it returns ``True``,
    the callback runs. Callbacks can inspect the simulation state, store
    persistent data in ``memory``, log return values, and queue commands
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
        memory=None,
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
    ):
        """Initialize an Event object.

        Parameters
        ----------
        callback : function
            Required callable executed when the event triggers, with signature
            ``callback(context) -> None or dict``. A returned ``dict`` is
            appended to ``self.callback_log``. Queue commands via
            ``context["event"].commands`` and keep the event's own data in
            ``context["event"].memory``.
            ``context`` is a dictionary with the following keys:
            ``time`` (float, s),
            ``state`` (list ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``),
            ``height_agl`` (float, m),
            ``sensors`` (list of sensor objects),
            ``sensors_by_name`` (dict of sensor objects),
            ``environment`` (:class:`rocketpy.Environment`),
            ``rocket`` (:class:`rocketpy.Rocket`),
            ``flight`` (:class:`rocketpy.Flight`),
            ``phase`` (current flight phase),
            ``event`` (this :class:`Event` instance),
            ``sampling_rate`` (float, Hz or ``None``),
            ``step_size`` (float, s, how long the simulation has been inside
            the solver step being evaluated; not the interval between checks
            of this callback, and not ``1 / sampling_rate``),
            ``phase_state`` (dict, this flight phase's own states by name,
            which may include states the 13 canonical ones cannot hold),
            ``phase_state_dot`` (dict, time derivative of each entry in
            ``phase_state``, by the same names),
            ``state_dot`` (list, time derivative of ``state``),
            ``pressure`` (float, Pa, at the rocket's altitude),
            ``previous_state`` (the ``state`` from the previous time this
            event was checked, or ``None`` on the first check),
            ``previous_time`` (float, s, or ``None`` on the first check),
            ``previous_phase_state`` (the ``phase_state`` from that previous
            check, or ``None`` on the first check and whenever the flight has
            just entered a new phase, since the states a phase follows are
            only comparable within that phase).
            The time derivatives and ``pressure`` are only worked out when a
            function reads them, and once per solver step, so reading them
            costs nothing unless they are used. Do not add keys to
            ``context``: it is shared by every event checked in the step.
        trigger : function, optional
            Predicate that returns ``True`` when the event should fire. It
            receives the same ``context`` as ``callback`` (see above), but
            its return value is interpreted as a boolean rather than logged.
            If ``None`` (default), the event fires every time it is evaluated.
        sampling_rate : float, optional
            Evaluation frequency in hertz. If ``None`` (default), the event is
            evaluated continuously at every solver time step. If a float (e.g.
            ``10``), the event is sampled at that rate, i.e. every
            ``1 / sampling_rate`` seconds.
        memory : dict, optional
            The event's own dictionary, kept from one check to the next and
            from trigger to callback. Read and write it as
            ``context["event"].memory``. Useful for counters, thresholds and
            data shared between trigger and callback. It is restored to the
            values given here when the event is reset for a new flight.
            Defaults to an empty dict.
        disable_on : str or int or float or callable, optional
            Condition that automatically disables the event. May be a string
            preset (``"apogee"`` or ``"burnout"``), a simulation time in seconds
            (int or float), or a callable ``function(context)`` that returns
            ``True`` when the event should be disabled. The times at which the
            event is disabled are recorded in ``self.disabled_times``.
        enable_on : str or int or float or callable, optional
            Condition that automatically (re-)enables a disabled event. Uses the
            same formats as ``disable_on`` (string preset, time threshold, or
            callable predicate). The times at which the event is enabled are
            recorded in ``self.enabled_times``.
        exact_time_function : function, optional
            Function that pins down the exact moment the event happened, between
            two solver steps. It receives the same ``context`` as ``trigger``,
            worked out again at every moment the search tries, and returns a
            number that crosses zero (or the configured ``target``) at the
            event. For example, to fire exactly at 500 m above ground level::

                def altitude(context):
                    return context["height_agl"] - 500

            When a crossing is found, ``callback`` runs with the ``context``
            of that exact moment. Only supported for continuous events
            (``sampling_rate=None``).
        exact_time_config : dict, optional
            How the exact moment is searched for. The ``"solver"`` key selects
            the method: ``"brentq"`` (default), ``"linear"`` or
            ``"cubic_hermite"``. Every solver accepts ``target`` (float, default
            ``0.0``); ``"brentq"`` also accepts ``xtol``, ``rtol`` and
            ``maxiter``; ``"cubic_hermite"`` requires ``derivative_function``
            (the time derivative of ``exact_time_function``, taking the same
            ``context``). Any other key raises a ``ValueError``. See
            :ref:`eventusage` for what each key does.
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
            ``set_dynamics``, ``start_flight_phase``, or ``terminate_flight``
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

        See Also
        --------
        :ref:`eventusage` : User guide for building and using events.
        """
        self.callback = self.__validate_callback(callback)
        self.name = name
        self.trigger = self.__validate_trigger(trigger) if trigger is not None else None
        self.memory = memory if memory is not None else {}
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
        self._forget_previous_check()

        self.is_discrete = self.sampling_rate is not None
        self.sampling_interval = (
            None if sampling_rate is None else 1.0 / float(sampling_rate)
        )

        # Track times when the event was enabled/disabled during a run.
        # These are lists of timestamps (floats) in simulation time.
        self.enabled_times = []
        self.disabled_times = []
        self._initial_memory = deepcopy(self.memory)
        self._initial_enabled_times = list(self.enabled_times)
        self._initial_disabled_times = list(self.disabled_times)

        self.commands = Commands()

        self.disable_on = self.__validate_gate_condition(disable_on, "disable_on")
        self.enable_on = self.__validate_gate_condition(enable_on, "enable_on")

        self.exact_time_function = (
            None
            if exact_time_function is None
            else self.__validate_context_function(
                exact_time_function, "exact_time_function"
            )
        )
        self.exact_time_config = dict(exact_time_config or {})
        self.__validate_exact_time_config(self.exact_time_config)

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
        restores the initial ``enabled`` flag, and restores ``memory`` to its
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
        self._forget_previous_check()
        self.memory = deepcopy(self._initial_memory)

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

    def _forget_previous_check(self):
        """Drop what this event saw the last time it was checked.

        Called when the event is built and again when it is reset, so a second
        flight does not compare its first check against the last check of the
        one before it. The context answers ``previous_state``, ``previous_time``
        and ``previous_phase_state`` from these.
        """
        self._previous_state = None
        self._previous_time = None
        self._previous_raw_state = None
        self._previous_phase = None

    def __call__(self, context, trigger_only=False, callback_only=False, reset=True):
        """Evaluate the event trigger and execute the callback if triggered.

        Parameters
        ----------
        context : EventContext
            The values to hand the trigger and callback. It is shared with the
            other events checked in this solver step, so this event is bound to
            it here and unbound again when the next event is bound.
        trigger_only : bool, optional
            If True, only evaluate the trigger condition without executing the
            callback. The enable_on function is also called.
        callback_only : bool, optional
            If True, only execute the callback without evaluating the trigger
            condition. The exact time function and disable_on function are also
            called.

        Returns
        -------
        bool
            True if the event was triggered, False otherwise.
        """
        # A new phase may integrate different states, so forget the previous
        # raw state. The previous canonical state is kept.
        if context.get("phase") is not self._previous_phase:
            self._previous_raw_state = None
        context.bind(self)

        if self.enabled is False:
            # If event is disabled, only evaluate enable_on function if it exists.
            if self._call_enable_on(context) is False:
                return False

        if reset:
            self._reset_commands()

        # --- Trigger Phase ---
        # Skip evaluating triggers if we are only running the callback.
        if callback_only is False:
            if self._call_enable_on(context) is False:
                return False

            triggered = self._call_trigger(context)

            # Remember this check, whether or not it triggered, so the next one
            # can be compared against it. Recorded before the early return so a
            # trigger that did not fire is still part of the sequence. The
            # phase's own states are kept raw; the context names them if asked.
            self._previous_state = context["state"]
            self._previous_time = context["time"]
            self._previous_phase = context.get("phase")
            self._previous_raw_state = context.get("raw_state")

            if triggered is False:
                return False

            if trigger_only:
                return True

        # --- Callback Phase ---
        context = self._call_exact_time(context)

        callback_log = self.callback(context)

        self.callback_log.append(callback_log)
        self.triggered_times.append(context.get("time"))

        self._call_disable_on(context)

        if self.trigger_only_once:
            self.commands.disable()

        self._log(
            triggered=True,
            context=context,
        )
        return True

    def _call_enable_on(self, context):
        if not self.enabled:
            # No enable_on function. Event stays disabled
            if self.enable_on is None:
                self._log(triggered=False, context=context)
                return False
            try:
                if not self.enable_on(context):
                    self._log(triggered=False, context=context)
                    return False
                self.commands.enable()
            except Exception as e:  # pylint: disable=W0718
                warnings.warn(
                    f"Error evaluating enable_on for event '{self.name}': {e}",
                    UserWarning,
                )
                self._log(triggered=False, context=context)
                return False

    def _call_trigger(self, context):
        if self.trigger is not None:
            if not self.trigger(context):
                self._log(triggered=False, context=context)
                return False
        return True

    def _call_exact_time(self, context):
        """Return the context at the exact event time, or ``context`` unchanged
        if there is no exact-time function or no crossing was found."""
        if self.exact_time_function is None:
            return context
        try:
            return self._compute_exact_time(context)
        except (ValueError, RuntimeError) as error:
            warnings.warn(
                f"Event '{self.name}': the exact moment it happened could not be "
                f"found, so it fires at t = {context['time']:.6g} s, when its "
                f"trigger was checked.\n{error}",
                UserWarning,
            )
            return context

    def _compute_exact_time(self, context):
        """Search the last solver step for the exact event time.

        The step runs from the second-to-last stored row to the last one. At
        every candidate time the whole context is worked out again, so the
        exact-time function sees ``state``, ``height_agl``, ``phase_state`` and
        the rest as they were at that moment.

        Returns
        -------
        EventContext
            A copy of ``context`` at the event time, which the callback is given.
            ``context`` itself is shared with the other events of this step and
            is left as it is.
        """
        flight = context["flight"]
        solution = flight.solution
        if len(solution) < 2:
            raise ValueError(
                "It fired before the first solver step, so there is no solver "
                "step to search."
            )

        t0, *state0 = solution.raw_row(-2)
        t1, *state1 = solution.raw_row(-1)
        # The ends of the step were stored, so they are read rather than
        # estimated. The start row belongs to the phase before when this is the
        # first step of a phase, and its states may not match this phase's.
        stored = {t1: state1}
        if len(solution) - 2 >= solution.phases[-1].start:
            stored[t0] = state0
        dense_output = context["phase"].solver.dense_output()

        # A solver may ask for the value and its rate at the same moment, so the
        # last context built is kept.
        cached = {}

        def context_at(t):
            if t not in cached:
                raw_state = stored[t] if t in stored else dense_output(t)
                cached.clear()
                cached[t] = context.at(t, raw_state)
            return cached[t]

        options = dict(self.exact_time_config)
        solver_name = options.pop("solver", "brentq")
        solver = SOLVERS[solver_name][0]
        target = options.pop("target", 0.0)
        function = self.exact_time_function
        derivative = options.pop("derivative_function", None)
        if derivative is not None:
            options["rate_at"] = lambda t: derivative(context_at(t))

        try:
            event_time = solver(
                lambda t: function(context_at(t)) - target, t0, t1, **options
            )
        except (ValueError, RuntimeError) as error:
            # The solver only says what went wrong. Add where, with which
            # values, and what the user can do about it.
            start = float(function(context_at(t0)))
            end = float(function(context_at(t1)))
            if min(start, end) <= target <= max(start, end):
                hint = (
                    "A smaller max_time_step on the Flight makes the solver steps "
                    "shorter, which usually avoids this."
                )
            else:
                hint = (
                    "The trigger fired in a step where exact_time_function did "
                    "not cross its target: check that the trigger and "
                    "exact_time_function describe the same condition."
                )
            raise ValueError(
                f"Over the solver step from t = {t0:.6g} s to {t1:.6g} s, "
                f"exact_time_function went from {start:.6g} to {end:.6g}; its "
                f"target is {target:.6g}. The {solver_name!r} solver reports: "
                f"{error}.\n{hint}"
            ) from error

        exact_context = context_at(event_time)
        self.commands.exact_time = event_time
        self.commands.exact_state = exact_context["raw_state"]
        return exact_context

    def _call_disable_on(self, context):
        if self.disable_on is not None:
            try:
                if self.disable_on(context):
                    self.commands.disable()
            except Exception as e:  # pylint: disable=W0718
                warnings.warn(
                    f"Error evaluating disable_on for event '{self.name}': {e}",
                    UserWarning,
                )

    def _log(
        self,
        triggered,
        context,
        callback_executed=None,
        skip_reason=None,
    ):
        if self.verbose:
            self.verbose_log.append(
                {
                    "time": context.get("time"),
                    "triggered": triggered,
                    "callback_executed": callback_executed,
                    "skip_reason": skip_reason,
                }
            )
        logger.debug(
            "Event '%s' at t=%s: triggered=%s, callback_executed=%s, skip_reason=%s",
            self.name,
            context.get("time"),
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
            return lambda context: context["time"] >= float(trigger)

        if not callable(trigger):
            raise ValueError("Trigger must be a callable, preset string, or number.")

        trigger = self.__validate_context_function(trigger, "Trigger function")

        return_annotation = inspect.signature(trigger).return_annotation
        if return_annotation not in (inspect.Signature.empty, bool, "bool"):
            raise ValueError(
                "Trigger function return annotation must be bool when provided."
            )

        return trigger

    def __validate_callback(self, callback):
        if not callable(callback):
            raise ValueError("Callback must be a callable.")

        callback = self.__validate_context_function(callback, "Callback function")

        return_annotation = inspect.signature(callback).return_annotation
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
            return lambda context: context["time"] >= float(condition)
        if callable(condition):
            return self.__validate_context_function(condition, parameter_name)
        raise TypeError(
            f"{parameter_name} must be None, a string preset, a number, or a callable"
        )

    def __validate_context_function(self, function, parameter_name):
        """Check that ``function`` is callable and takes the context as its one
        positional argument."""
        if not callable(function):
            raise ValueError(f"{parameter_name} must be callable or None.")
        try:
            inspect.signature(function).bind(None)
        except TypeError as error:
            raise ValueError(
                f"{parameter_name} must take the event context as its single "
                f"positional argument, like `def f(context): ...`."
            ) from error
        return function

    def __validate_exact_time_config(self, config):
        """Check the solver name and that every key belongs to that solver."""
        solver_name = config.get("solver", "brentq")
        if solver_name not in SOLVERS:
            raise ValueError(
                f"Unknown exact-time solver: {solver_name!r}. "
                f"Supported solvers: {sorted(SOLVERS)!r}."
            )
        _, accepted, required = SOLVERS[solver_name]
        unknown = set(config) - accepted - {"solver", "target"}
        if unknown:
            raise ValueError(
                f"Unknown exact_time_config keys for the {solver_name!r} solver: "
                f"{sorted(unknown)!r}. Accepted keys: "
                f"{sorted(accepted | {'solver', 'target'})!r}."
            )
        missing = required - set(config)
        if missing:
            raise ValueError(
                f"The {solver_name!r} exact-time solver requires "
                f"{sorted(missing)!r} in exact_time_config."
            )
        if "derivative_function" in config:
            self.__validate_context_function(
                config["derivative_function"], "derivative_function"
            )
