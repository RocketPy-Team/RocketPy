"""Controllers: events that set the rocket's control inputs while it flies."""

import warnings
from copy import deepcopy

from rocketpy.control._helpers import (
    _LOAD_ERRORS,
    _check_controller_function,
    _missing_controller_function,
)
from rocketpy.control._legacy import _legacy_controller, _PositionalController
from rocketpy.simulation.events.event import PRESETS, Event
from rocketpy.tools import from_hex_decode, to_hex_encode

from ..prints.controller_prints import _ControllerPrints


class Controller(Event):
    """Adds a controller to the rocket: a flight computer that runs at a fixed
    sampling rate and sets the rocket's controls. Add it with
    :meth:`rocketpy.Rocket.add_controller`.

    At every sample the controller reads what the rocket knows, its state
    and its sensors, and decides the controls: the deployment level of the
    air brakes, the deflection of the canards, the firing of a thruster. A
    value it sets stays in force until the next sample, as a real flight
    computer holds its outputs between cycles, and the rocket responds to it
    through the part it controls.

    The controller can keep values from one sample to the next, such as the
    integral of a PID loop, record values to look at after the flight, and
    be switched off and on during the flight, at apogee, at burnout or when
    a condition it defines is met. After the flight, what it set is read
    from ``flight.controls`` and what it recorded from
    ``flight.records``. A controller can also share values with other
    controllers, such as the setpoint an outer loop gives an inner one, as
    outputs.

    See :ref:`controllers` for examples and :ref:`eventusage` for what the
    controller is told about the flight.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        controller_function,
        sampling_rate,
        memory=None,
        *,
        outputs=None,
        name="Controller",
        enabled=True,
        enable_on=None,
        disable_on=None,
        time_overshootable=True,
    ):
        """Build a controller.

        Parameters
        ----------
        controller_function : callable
            The control logic, ``controller_function(context)``, called once
            per sample. It writes the controls as
            ``context.controls.<controllable>.<control>``, among the names
            ``rocket.control_names`` lists, and may keep values for after
            the flight with ``context.record``. What it returns is ignored.
        sampling_rate : float
            How many times per second the controller runs, in Hz. Must be
            positive. For a law that acts at every instant instead of at a
            fixed rate, use :class:`rocketpy.ContinuousController`.
        memory : dict, optional
            Values the controller keeps from one run to the next, read and
            written as ``context.memory``, such as the integral of a PID
            controller. It starts again from this dictionary at every flight.
            Defaults to an empty dictionary.
        outputs : sequence of str or dict, optional
            Values this controller computes for other controllers or events
            to read. Give the name of each output. It is then written and
            read through ``context.outputs``, under the controller's name in
            snake_case: a controller with ``name="Guidance"`` and
            ``outputs=("setpoint",)`` writes
            ``context.outputs.guidance.setpoint = 0.1``, and any other
            controller or event reads the same name. An output can hold any
            value, keeps it until it is written again, and can be written
            only by the controller that declared it. It starts every flight
            as ``None``, or at the value given with a dictionary, such as
            ``outputs={"setpoint": 0.05}``. After the flight,
            ``flight.event_outputs.guidance.setpoint`` is the list of
            ``(time, value)`` pairs written. Default ``None``, no outputs.
        name : str, optional
            Name of the controller, which names its records and outputs:
            ``"Apogee Controller"`` gives ``flight.records.apogee_controller``.
            Defaults to ``"Controller"``.
        enabled : bool, optional
            Whether the controller runs from the start of the flight. Defaults
            to ``True``.
        enable_on : str, float or callable, optional
            When a stopped controller starts running: ``"apogee"``,
            ``"burnout"``, a time in seconds, or a function ``f(context)`` that
            returns ``True`` when the controller should start. Defaults to
            ``None``, never.
        disable_on : str, float or callable, optional
            When to stop running, in the same forms as ``enable_on``. Defaults
            to ``None``, never. To stop from inside ``controller_function``
            instead, such as after detecting a fault, call
            ``context.commands.disable()``.
        time_overshootable : bool, optional
            ``True`` (the default) lets the solver step past each sample and
            runs the controller on the interpolated state, restarting the
            solver at the sample only when a control changes. ``False`` stops
            the solver at every sample. The default is faster; the results
            differ only by the solver's choice of steps. Ignored when the
            flight has ``time_overshoot=False``.

        See Also
        --------
        rocketpy.Event : What the context offers and how events are gated.
        rocketpy.ContinuousController : A controller that runs with the
            dynamics instead of at a fixed rate.
        rocketpy.Rocket.add_controller : Add the controller to a rocket.
        rocketpy.Flight.controls : The control inputs after the flight.
        rocketpy.Flight.records : The values recorded with ``context.record``.
        rocketpy.Flight.event_outputs : The outputs written during the flight.
        """
        controller_function = _check_controller_function(controller_function)
        # A controller with no rate would run once per solver step, so how
        # often it ran would depend on the solver's tolerances
        if sampling_rate is None or not sampling_rate > 0:
            raise ValueError(
                f"The sampling rate of the controller '{name}' must be a "
                f"positive number of Hz, got {sampling_rate!r}. For a law that "
                "acts at every instant, use rocketpy.ContinuousController."
            )
        # Kept as given, for saving: the event turns them into functions
        self._gates = (disable_on, enable_on)
        super().__init__(
            callback=controller_function,
            sampling_rate=sampling_rate,
            memory=memory,
            disable_on=disable_on,
            enable_on=enable_on,
            trigger_only_once=False,
            time_overshootable=time_overshootable,
            name=name,
            enabled=enabled,
            priority=3,
            outputs=outputs,
        )
        self.prints = _ControllerPrints(self)

    @property
    def controller_function(self):
        """The control logic, ``controller_function(context)``: the event's
        callback."""
        return self.callback

    @controller_function.setter
    def controller_function(self, function):
        self.callback = _check_controller_function(function)

    def reset(self):
        """Reset the controller for a new flight: what :class:`rocketpy.Event`
        resets, and the ``observed_variables`` of a deprecated positional
        controller function."""
        super().reset()
        if isinstance(self.callback, _PositionalController):
            self.callback.reset()

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
            "memory": deepcopy(self._initial_memory),
            "name": self.name,
            "enabled": self._initial_enabled,
            "disable_on": encode(disable_on),
            "enable_on": encode(enable_on),
            "outputs": self.output_defaults,
            "time_overshootable": self.time_overshootable,
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
        # Saved before controllers took the context
        function = _legacy_controller(function, sampling_rate)
        gates = []
        for key in ("disable_on", "enable_on"):
            gate = data.get(key)
            # A preset name, a time or None is kept as saved
            if isinstance(gate, str) and gate not in PRESETS:
                try:
                    gate = from_hex_decode(gate)
                except _LOAD_ERRORS as error:
                    warnings.warn(
                        f"The `{key}` function of the controller '{name}' could "
                        f"not be loaded ({type(error).__name__}: {error}), so "
                        "the controller is loaded disabled. Build it again and "
                        "add it to the rocket to fly it.",
                        UserWarning,
                    )
                    gate, enabled = None, False
            gates.append(gate)
        return cls(
            controller_function=function,
            sampling_rate=sampling_rate,
            memory=data.get("memory"),
            name=name,
            enabled=enabled,
            disable_on=gates[0],
            enable_on=gates[1],
            outputs=data.get("outputs") or None,
            time_overshootable=data.get("time_overshootable", True),
        )
