from rocketpy.control._helpers import check_hook
from rocketpy.control.controllable import Controllable


class ContinuousController(Controllable):
    """Adds a continuous controller to the rocket: a control law that acts at
    every instant of the flight, with no sampling rate. Add it with
    :meth:`rocketpy.Rocket.add_controller`.

    It models what responds continuously to the rocket's motion: a passive
    damper, an analog loop, or an ideal law studied before a sampling rate
    is chosen. A :class:`rocketpy.Controller` is a flight computer that
    reads and acts at discrete samples; a continuous controller is part of
    the rocket's dynamics, and the flight integrates the closed loop as one.

    The law is given the controller's own states and the flight's
    conditions, and sets the control inputs within their limits. A law with
    dynamics of its own, such as an integral term or a filter, keeps them as
    states that the flight integrates with the rocket's, starting from 0. A
    controller with states and no law writes no controls: an observer, whose
    estimate the other controllers read as a state.

    Because it acts at every instant, a continuous controller has no memory
    between instants, no records, no commands and no outputs: its law must
    depend on its arguments alone, and it cannot read the outputs of the
    flight's events. After the flight, its states are read from
    ``flight.state_history`` and what it set from ``flight.controls``.

    See :ref:`continuous_controller` for examples.

    Attributes
    ----------
    law_function : callable or None
        The control law, or ``None`` for a controller with states and no law.
    derivative_function : callable or None
        The derivative of the controller's states with respect to time, or
        ``None`` for a controller without states.
    states : tuple of str
        Names of the controller's own states.
    """

    def __init__(
        self,
        law=None,
        states=None,
        state_derivative=None,
        name="Continuous Controller",
    ):
        """Build a continuous controller.

        Parameters
        ----------
        law : callable, optional
            ``law(states, controls, conditions)``, the control law: it reads
            its own states and the flight's conditions (see
            :class:`rocketpy.control.FlightConditions`), and writes the
            control inputs as ``controls.canards.fin_1``. Required, except
            for a controller with ``states``, which may have none.
        states : sequence of str, optional
            Names of the controller's own states, such as ``("integral",)``.
            Each starts the flight at 0. Default: none.
        state_derivative : callable, optional
            ``state_derivative(states, controls, conditions)``: returns a list
            with the derivative with respect to time of each state, one value
            per state in the order of ``states``, even for a single state
            (``[conditions.state.w3]``, not ``conditions.state.w3``).
            Required with ``states``.
        name : str, optional
            Name of the controller, which names its states in the flight.
            Default ``"Continuous Controller"``.
        """
        # A controllable with states and no controls of its own
        super().__init__(name, ())
        if states is not None:
            # One name given alone is one state
            self.states = (states,) if isinstance(states, str) else tuple(states)
        self._check_state_names()
        if state_derivative is not None and not self.states:
            raise ValueError(
                f"'{name}' has a state_derivative but no states: name them "
                'with `states=`, like `states=("integral",)`.'
            )
        arguments = ("states", "controls", "conditions")
        # A controller that only has states writes no controls and needs no law
        self.law_function = check_hook(
            self, "law", law, ContinuousController.law, arguments, not self.states
        )
        self.derivative_function = check_hook(
            self,
            "state_derivative",
            state_derivative,
            ContinuousController.state_derivative,
            arguments,
            needed=bool(self.states),
        )

    def law(self, states, controls, conditions):
        """Write the control inputs for this instant, by calling the control
        law.

        Parameters
        ----------
        states : sequence of float
            The controller's own states, in the order of ``states``.
        controls : ControlsView
            The control inputs, to read and write as
            ``controls.canards.fin_1``.
        conditions : FlightConditions
            The flight at this moment.
        """
        if self.law_function is not None:
            self.law_function(states, controls, conditions)

    def state_derivative(self, states, commands, conditions):
        """Return the derivative with respect to time of each state, in the
        order of ``states``.

        Parameters
        ----------
        states : sequence of float
            The controller's own states, in the order of ``states``.
        commands : ControlsView
            The control inputs, read as ``commands.canards.fin_1``, as the
            laws left them, inside their limits: not the values a
            controllable with states of its own, such as a servo, acts with.
        conditions : FlightConditions
            The flight at this moment.

        Returns
        -------
        sequence of float
        """
        return self.derivative_function(states, commands, conditions)

    def __repr__(self):
        return f"ContinuousController(name={self.name!r}, states={self.states!r})"
