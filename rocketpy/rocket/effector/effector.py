class Effector:
    """Abstract control effector: injects a body-frame force and/or moment
    **directly** into the equations of motion.

    Unlike an aerodynamic surface -- whose force is produced by the aerodynamic
    model from dynamic pressure -- an effector contributes an arbitrary body
    force/moment computed from its control state (and, optionally, the flight
    state). This is the non-aerodynamic control path: reaction-control thrusters,
    reaction wheels, and (later) thrust vector control. It works regardless of
    airspeed or atmosphere.

    An effector is a :class:`rocketpy.control.controlled_object.ControlledObject`:
    a :class:`rocketpy.Controller` sets its control state each step (via
    :meth:`set_control`), and the effector maps that state to a force/moment in
    :meth:`evaluate`. Register it on a rocket with
    :meth:`rocketpy.Rocket.add_effector`; the equations of motion then sum its
    contribution alongside the aerodynamic surfaces.

    Naming note: this maps ``control_state -> force/moment`` (the *effector*
    role). A future ``Actuator`` layer -- modelling actuator dynamics such as
    rate/position limits and lag between commanded and realized control -- would
    sit between the :class:`Controller` and the effector; ``_clamp_control`` is
    the saturation seam it would build on.

    Attributes
    ----------
    Effector.name : str
        Human-readable name (used for controller rewiring on load).
    Effector.position : float or tuple
        Application station along the rocket axis (float), or a full ``(x, y, z)``
        point in the user coordinate system. Sets the moment arm for a pure force.
    Effector.control_variables : list of str
        Ordered names of the control axes.
    Effector.control_state : dict
        Current value of each control variable.
    Effector.initial_control_state : dict
        Control values restored at the start of each simulation.
    """

    def __init__(self, position=0.0, controls=("command",), name="Effector"):
        """Initialize the effector.

        Parameters
        ----------
        position : float or tuple, optional
            Application station (float, axial) or ``(x, y, z)`` point in the user
            coordinate system. Defaults to ``0.0``.
        controls : iterable of str, optional
            Names of the control axes. Each becomes a key in
            :attr:`control_state`. Defaults to ``("command",)``.
        name : str, optional
            Effector name. Defaults to ``"Effector"``.
        """
        self.name = name
        self.position = position
        self.control_variables = list(controls)
        self.control_state = {name: 0.0 for name in self.control_variables}
        self.initial_control_state = dict(self.control_state)

    # --- ControlledObject protocol -------------------------------------- #
    def _clamp_control(self, name, value):  # pylint: disable=unused-argument
        """Hook to constrain a control value before storing. No clamping by
        default; subclasses may override (e.g. thruster saturation)."""
        return value

    def set_control(self, name, value):
        """Set the current value of a control variable (applying any clamping)."""
        if name not in self.control_state:
            raise KeyError(
                f"Unknown control variable '{name}'. "
                f"Valid controls are: {self.control_variables}."
            )
        self.control_state[name] = self._clamp_control(name, value)

    def get_control(self, name):
        """Return the current value of a control variable."""
        return self.control_state[name]

    def _reset(self):
        """Restore all control variables to their initial values. Run by the
        controller at the start of each simulation so control state does not
        leak across flights."""
        for name, value in self.initial_control_state.items():
            self.set_control(name, value)

    # --- Equations-of-motion contribution ------------------------------- #
    def evaluate(self, force_arm, time, velocity_body, omega, mach, environment):
        """Return this effector's body-frame force and moment contribution.

        Parameters
        ----------
        force_arm : Vector
            Position of the effector's application point relative to the center
            of dry mass, in the body frame (its moment arm).
        time : float
            Simulation time, in seconds.
        velocity_body : Vector
            Rocket velocity in the body frame.
        omega : Vector
            Body angular velocity ``(w1, w2, w3)``.
        mach : float
            Free-stream Mach number.
        environment : Environment
            The flight environment.

        Returns
        -------
        tuple of Vector
            ``(force_body, moment_body)`` -- the force and the moment **about the
            center of dry mass**, both in the body frame.
        """
        raise NotImplementedError("Effector subclasses must implement evaluate().")

    def to_dict(self, include_outputs=False, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError
