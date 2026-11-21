import inspect
import keyword
import logging
import warnings
from copy import deepcopy
from numbers import Real

from ..._logging import logger
from .commands import Commands
from .event_context import EventContext, refresh_event_kwargs
from .exact_time_solvers import SOLVERS


# The conditions a trigger, ``enable_on`` and ``disable_on`` can name. Each
# holds from that moment on.
PRESETS = {
    # Coming down and above the ground
    "apogee": lambda context: (
        context["canonical_state"][5] <= 0 < context["height_agl"]
    ),
    "burnout": lambda context: context["time"] >= context["rocket"].motor.burn_out_time,
}


def _copied(value):
    """Return a number as it is and anything else as a copy, or as it is when
    it cannot be copied."""
    if isinstance(value, Real):
        return value
    try:
        return deepcopy(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return value


class Event:
    """An action that runs when a condition is met during a flight.

    An event pairs a ``trigger``, the condition, with a ``callback``, the
    action. Each time the event is checked, the trigger is evaluated and, when
    it returns ``True``, the callback runs. Both receive the ``context``, a
    :class:`rocketpy.EventContext` with the time, the rocket's state, the
    sensor readings and the rest of what is known at that moment.

    A callback can keep data from one check to the next in
    ``context.memory``, record values to read after the flight with
    ``context.record``, share values with other events through
    ``context.outputs``, and change the course of the flight through
    ``context.commands``: start a new flight phase, change the equations of
    motion, add other events or end the flight.

    See :ref:`eventusage` for a full guide with runnable examples.

    Attributes
    ----------
    name : str
        Name of the event.
    enabled : bool
        Whether the event is currently checked.
    memory : dict
        The event's own data, as the last check left it.
    triggered_times : list of float
        The times the event fired in the last flight, in s.
    enabled_times, disabled_times : list of float
        The times the event was turned on and off in the last flight, in s.
    records : dict
        What the event recorded in the last flight.
    outputs : tuple of str
        Names of the values the event shares with other events.
    output_history : dict
        What the event wrote to its outputs in the last flight.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        callback,
        trigger=None,
        sampling_rate=None,
        memory=None,
        *,
        outputs=None,
        name="Custom Event",
        enabled=True,
        enable_on=None,
        disable_on=None,
        trigger_only_once=False,
        time_overshootable=True,
        exact_time_function=None,
        exact_time_config=None,
        priority=4,
        verbose=False,
    ):
        """Build an event.

        Parameters
        ----------
        callback : callable
            The action, ``callback(context)``, run when the event triggers.
            ``context`` is a :class:`rocketpy.EventContext`, read by attribute:
            ``context.time``, ``context.state.vz``, ``context.height_agl``,
            ``context.sensors.barometer.pressure``. What the callback returns
            is ignored.
        trigger : callable, str or float, optional
            The condition that fires the event. One of:

            - a function ``trigger(context)`` that returns ``True`` when the
              event should fire, given the same ``context`` as ``callback``;
            - ``"apogee"`` (the rocket is coming down and above the ground)
              or ``"burnout"`` (the motor has burned out);
            - a number, the simulation time in s from which it fires.

            ``"apogee"``, ``"burnout"`` and a time stay ``True`` from that
            moment on, so the event fires at every check after it. Set
            ``trigger_only_once=True`` to fire once.
            Default ``None``: the event fires every time it is checked.
        sampling_rate : float, optional
            How many times per second the event is checked, in Hz: with
            ``10``, every 0.1 s. Default ``None``: the event is checked at the
            end of every solver step.
        memory : dict, optional
            The event's own dictionary, kept from one check to the next and
            from trigger to callback. Read and write it as
            ``context.memory``. It starts every flight with the values given
            here. Defaults to an empty dictionary.
        outputs : sequence of str or dict, optional
            Values this event computes for other events to read. Give the name
            of each output. It is then written and read through
            ``context.outputs``, under the event's name in snake_case. For
            example, an event named ``"Guidance"`` with
            ``outputs=("pitch_setpoint",)`` writes its output in its callback,
            and any other event reads it:

            .. code-block:: python

                context.outputs.guidance.pitch_setpoint = 0.1  # in "Guidance"
                setpoint = context.outputs.guidance.pitch_setpoint  # elsewhere

            An output can hold any value: a number, a vector, a text. It keeps
            its value until it is written again, and only the event that
            declared it can write it. Every output starts each flight as
            ``None``, or at the value given with a dictionary, such as
            ``outputs={"pitch_setpoint": 0.05}``. After the flight,
            ``flight.event_outputs.guidance.pitch_setpoint`` is the list of
            ``(time, value)`` pairs written. Two events with outputs cannot
            have the same name. Default ``None``, no outputs.
        name : str, optional
            Name of the event. It names the event's records and outputs
            (``flight.records.<name in snake_case>``) and appears in the log
            messages. Defaults to ``"Custom Event"``.
        enabled : bool, optional
            Whether the event is checked from the start of the flight. An
            event that is not can be turned on later by its ``enable_on``, or
            by another event with ``context.commands.enable_event(event)``.
            Defaults to ``True``.
        enable_on : str, float or callable, optional
            When a disabled event is turned on: ``"apogee"``, ``"burnout"``, a
            simulation time in s, or a function ``f(context)`` that returns
            ``True`` when the event should be turned on. The times it happens
            are kept in ``event.enabled_times``. Default ``None``, never.
        disable_on : str, float or callable, optional
            When the event is turned off, in the same forms as ``enable_on``.
            The times it happens are kept in ``event.disabled_times``. Default
            ``None``, never.
        trigger_only_once : bool, optional
            If ``True``, the event turns itself off after the first time it
            fires. Defaults to ``False``.
        time_overshootable : bool, optional
            How the flight reaches each sample of a sampled event. When
            ``True`` (default) the solver may step past the sample and the
            event runs at the interpolated state; if its commands or control
            writes change the trajectory, the step is cut back to the sample
            and the solver restarted there. When ``False`` every sample is a
            point the solver stops at, so the event runs on the solver's own
            state and no restart follows a change. The default is the faster
            one, even for a controller that changes a control at every sample;
            ``False`` is for an event that must run on the solver's own state
            rather than an interpolated one. Always ``False`` for an event
            without a ``sampling_rate``. Ignored when the flight has
            ``time_overshoot=False``.
        exact_time_function : callable, optional
            Function that finds the exact moment the event happened, between
            two solver steps. It receives the same ``context`` as ``trigger``,
            worked out again at every moment the search tries, and returns a
            number that crosses zero (or the configured ``target``) at the
            event. For example, to fire exactly at 500 m above ground level:

            .. code-block:: python

                def altitude(context):
                    return context.height_agl - 500

            When a crossing is found, ``callback`` runs with the ``context``
            of that exact moment. Only for an event without a
            ``sampling_rate``.
        exact_time_config : dict, optional
            How the exact moment is searched for. The ``"solver"`` key selects
            the method: ``"brentq"`` (default), ``"linear"`` or
            ``"cubic_hermite"``. Every solver accepts ``target`` (float, default
            ``0.0``); ``"brentq"`` also accepts ``xtol``, ``rtol`` and
            ``maxiter``; ``"cubic_hermite"`` requires ``derivative_function``
            (the derivative of ``exact_time_function`` with respect to time,
            taking the same ``context``). Any other key raises a ``ValueError``. See
            :ref:`eventusage` for what each key does.
        priority : int, optional
            The order events are checked in when several are due at the same
            time: lower numbers first. RocketPy's own events use:

            - 0: out of rail, apogee and landing
            - 1: sensors
            - 2: parachutes
            - 3: controllers
            - 4: events given to the flight (default)
        verbose : bool, optional
            When ``True``, every check of the event is reported by RocketPy's
            logger at the ``INFO`` level instead of ``DEBUG``, which is shown
            with ``Flight(..., verbose=True)`` or ``rocketpy.enable_logging()``.
            Defaults to ``False``.

        Raises
        ------
        ValueError
            If ``callback``, ``trigger`` or ``exact_time_function`` is not a
            function of the context alone, if a name given to ``trigger``,
            ``enable_on`` or ``disable_on`` is not ``"apogee"`` or
            ``"burnout"``, if an output name cannot be used, if
            ``exact_time_config`` has a key its solver does not take, or if
            ``exact_time_function`` is given with a ``sampling_rate``.
        TypeError
            If ``enable_on`` or ``disable_on`` is not one of the accepted
            forms.

        See Also
        --------
        rocketpy.EventContext : The values a trigger and a callback are given.
        :ref:`eventusage` : User guide for building and using events.
        """
        self.callback = self.__validate_callback(callback)
        self.name = name
        self.trigger = self.__validate_trigger(trigger) if trigger is not None else None
        self.memory = memory if memory is not None else {}
        self._output_defaults = self.__validate_outputs(outputs)
        self._outputs = deepcopy(self._output_defaults)
        self._output_history = {}
        self.sampling_rate = sampling_rate
        self.trigger_only_once = trigger_only_once
        self.priority = priority
        self.time_overshootable = bool(time_overshootable)
        if self.time_overshootable and self.sampling_rate is None:
            self.time_overshootable = False
        self._initial_enabled = bool(enabled)
        self.enabled = bool(enabled)
        self.verbose = verbose
        self.triggered_times = []
        self._records = {}
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
                "exact_time_function is only supported for an event without a "
                "sampling_rate."
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled}, "
            f"sampling_rate={self.sampling_rate}, "
            f"time_overshootable={self.time_overshootable}, "
            f"trigger_only_once={self.trigger_only_once})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} '{self.name}' (enabled={self.enabled}, "
            f"sampling_rate={self.sampling_rate}, "
            f"time_overshootable={self.time_overshootable}, "
            f"trigger_only_once={self.trigger_only_once})"
        )

    # Running the event

    def __call__(self, context, trigger_only=False, callback_only=False, reset=True):
        """Check the event: evaluate its trigger and, if it fires, run its
        callback.

        Parameters
        ----------
        context : EventContext
            The values the trigger and the callback are given. It is shared
            with the other events checked in this solver step, and points at
            this event until the next one is checked.
        trigger_only : bool, optional
            If ``True``, the trigger and ``enable_on`` are evaluated and the
            callback is not run. Defaults to ``False``.
        callback_only : bool, optional
            If ``True``, the callback is run without evaluating the trigger.
            ``exact_time_function`` and ``disable_on`` are still evaluated.
            Defaults to ``False``.
        reset : bool, optional
            If ``True`` (default), the commands left by the previous check are
            cleared first.

        Returns
        -------
        bool
            ``True`` if the event fired, ``False`` otherwise.
        """
        context.bind(self)

        if reset:
            self.commands.reset()

        # If event is disabled, only evaluate enable_on function if it exists.
        if self._call_enable_on(context) is False:
            return False

        # --- Trigger Phase ---
        # Skip evaluating triggers if we are only running the callback.
        if callback_only is False:
            triggered = self._call_trigger(context)

            # Kept for the next check, triggered or not. Read from the context
            # now, since it is rewritten during the step, and stored only after
            # the callback, which must still see the check before this one.
            check = (
                context["canonical_state"],
                context["time"],
                context["raw_state"],
                context["phase_names"],
            )
            if triggered is False or trigger_only:
                (
                    self._previous_state,
                    self._previous_time,
                    self._previous_raw_state,
                    self._previous_phase_names,
                ) = check
                return triggered

        # --- Callback Phase ---
        context = self._call_exact_time(context)

        self.callback(context)
        if callback_only is False:
            (
                self._previous_state,
                self._previous_time,
                self._previous_raw_state,
                self._previous_phase_names,
            ) = check
        self.triggered_times.append(context.get("time"))

        self._call_disable_on(context)

        if self.trigger_only_once:
            self.commands.disable()

        self._log(triggered=True, context=context)
        return True

    def _call_enable_on(self, context):
        """Return ``False`` if the event is disabled and stays so. If its
        ``enable_on`` condition is met, ask the flight to enable it."""
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
        """Return whether the trigger fires. An event without one always
        does."""
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
        exact-time function sees ``state``, ``height_agl`` and the rest as they
        were at that moment.

        Returns
        -------
        EventContext
            A copy of ``context`` at the event time, which the callback is given.
            ``context`` itself is shared with the other events of this step and
            is left as it is.
        """
        solution = context._solution  # pylint: disable=protected-access
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
        phase = context._phase  # pylint: disable=protected-access
        dense_output = phase.solver.dense_output()

        # A solver may ask for the value and its rate at the same moment, so the
        # last context built is kept.
        cached = {}

        def context_at(t):
            if t not in cached:
                raw_state = stored[t] if t in stored else dense_output(t)
                cached.clear()
                # A copy, since the other events of this step share ``context``
                moved = EventContext(
                    context,
                    solution=solution,
                    phase=phase,
                    output_events=context._output_events,  # pylint: disable=protected-access
                )
                cached[t] = refresh_event_kwargs(moved, t, raw_state)
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
        """Ask the flight to disable the event if its ``disable_on`` condition
        is met."""
        if self.disable_on is not None:
            try:
                if self.disable_on(context):
                    self.commands.disable()
            except Exception as e:  # pylint: disable=W0718
                warnings.warn(
                    f"Error evaluating disable_on for event '{self.name}': {e}",
                    UserWarning,
                )

    def _log(self, triggered, context):
        """Report one check of the event to RocketPy's logger, at the ``INFO``
        level for a verbose event and ``DEBUG`` otherwise."""
        logger.log(
            logging.INFO if self.verbose else logging.DEBUG,
            "Event '%s' at t=%s: triggered=%s",
            self.name,
            context.get("time"),
            triggered,
        )

    # What the event keeps: its outputs and records

    @property
    def outputs(self):
        """The names of the values this event computes for other events to
        read, as ``context.outputs.<event name in snake_case>.<output>``.
        Empty for an event without outputs.

        Returns
        -------
        tuple of str
        """
        return tuple(self._output_defaults)

    @property
    def output_defaults(self):
        """The value each output starts every flight with, by name.

        Returns
        -------
        dict
        """
        return dict(self._output_defaults)

    @property
    def output_history(self):
        """What this event wrote to its outputs, by name.

        Each output is the list of ``(time, value)`` pairs written, with the
        value as it was when written. The list starts empty at every flight.
        ``flight.event_outputs`` holds the outputs of every event of a flight.

        Returns
        -------
        dict
            Output name to list of ``(time, value)``.
        """
        return {name: list(points) for name, points in self._output_history.items()}

    def _write_output(self, time, name, value):
        """Set an output at ``time``. The value is kept as given for the events
        that read it, and a copy is kept with its time, so a later change to
        the object does not change what was written."""
        self._outputs[name] = value
        self._output_history.setdefault(name, []).append((time, _copied(value)))

    @property
    def records(self):
        """What this event recorded with ``context.record``, by name, exactly
        as recorded.

        Each record is the list of ``(time, value)`` pairs the event stored,
        with ``value`` the object passed to ``context.record``: a number, a
        vector, a text, a dictionary. Records exist only at the times the
        event ran, and start empty at every flight. ``flight.records`` holds
        the records of every event of a flight.

        Returns
        -------
        dict
            Record name to list of ``(time, value)``.
        """
        return {name: list(points) for name, points in self._records.items()}

    def _record(self, time, values):
        """Store named values at ``time``, as given: a number as it is,
        anything else as a copy, so a later change to the object does not
        change what was recorded."""
        for name, value in values.items():
            self._records.setdefault(name, []).append((time, _copied(value)))

    # Starting a new flight, and going back in time

    def reset(self):
        """Prepare the event for a new flight.

        The event is enabled or not as it was built, its memory and outputs
        are back at the values it was built with, and its records and the
        times it fired or was turned on and off are cleared. A flight calls
        this on each of its events when it starts.
        """
        self.commands.reset()
        self.enabled = self._initial_enabled
        self._records = {}
        self._outputs = deepcopy(self._output_defaults)
        self._output_history = {}
        self.triggered_times.clear()
        self._forget_previous_check()
        self.memory = deepcopy(self._initial_memory)

        self.enabled_times = []
        self.disabled_times = []

    def _forget_previous_check(self):
        """Drop what this event saw the last time it was checked.

        Called when the event is built and again when it is reset, so a second
        flight does not compare its first check against the last check of the
        one before it. The context answers ``previous_state`` and
        ``previous_time`` from these.
        """
        self._previous_view = None
        self._previous_state = None
        self._previous_time = None
        self._previous_raw_state = None
        self._previous_phase_names = ()

    def _check_state(self):
        """Whether the event is enabled and what it saw at its previous check:
        what a rollback puts back with :meth:`_restore_check_state`."""
        return (
            self.enabled,
            self._previous_state,
            self._previous_time,
            self._previous_raw_state,
            self._previous_phase_names,
        )

    def _restore_check_state(self, saved):
        """Put back what :meth:`_check_state` returned."""
        (
            self.enabled,
            self._previous_state,
            self._previous_time,
            self._previous_raw_state,
            self._previous_phase_names,
        ) = saved

    def _drop_after(self, time):
        """Forget what was recorded, what was written to the outputs, and when
        the event ran or was turned on or off, after ``time``, a moment the
        flight went back to. An output is again the last value written up to
        ``time``, or the value it started the flight with."""
        for points in self._records.values():
            while points and points[-1][0] > time:
                points.pop()
        for name, points in self._output_history.items():
            dropped = False
            while points and points[-1][0] > time:
                points.pop()
                dropped = True
            if dropped:
                self._outputs[name] = _copied(
                    points[-1][1] if points else self._output_defaults[name]
                )
        for times in (self.triggered_times, self.enabled_times, self.disabled_times):
            while times and times[-1] > time:
                times.pop()

    # Checks of the arguments given to the event

    def __validate_callback(self, callback):
        """Check that ``callback`` is a function of the context alone."""
        if not callable(callback):
            raise ValueError("Callback must be a callable.")

        # What it returns is ignored, so its return annotation is not checked
        return self.__validate_context_function(callback, "Callback function")

    def __validate_trigger(self, trigger):
        """Return the trigger as a function of the context: the one given, the
        condition a preset names, or "the time has reached this number"."""
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

    def __validate_outputs(self, outputs):
        """Return the event's outputs as a dictionary of name to the value
        each starts a flight with (``None`` unless given)."""
        if not outputs:
            return {}
        defaults = dict(outputs) if isinstance(outputs, dict) else {}
        names = tuple(outputs)
        for name in names:
            # Read as an attribute, ``context.outputs.<event>.<name>``
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or keyword.iskeyword(name)
                or name.startswith("_")
            ):
                raise ValueError(
                    f"The output name {name!r} of the event '{self.name}' cannot "
                    "be used: a name is made of letters, digits and underscores, "
                    "and starts with a letter."
                )
        if len(set(names)) != len(names):
            raise ValueError(
                f"The event '{self.name}' has two outputs with the same name."
            )
        return {name: defaults.get(name) for name in names}

    def __validate_gate_condition(self, condition, parameter_name):
        """Return an ``enable_on`` or ``disable_on`` condition as a function of
        the context, in the same forms as a trigger, or ``None``."""
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
