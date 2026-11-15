import keyword

from rocketpy.control._helpers import (
    _LOAD_ERRORS,
    UnloadedControllable,
    check_hook,
    limited_value,
)
from rocketpy.prints.controller_prints import _ControllablePrints
from rocketpy.tools import from_hex_decode, snake_case_key, to_hex_encode


class Controllable:
    """A part of the rocket with controls, which a controller can move.

    A controllable never holds the live value of a control: during a flight
    the values live in the flight's control inputs, and the controllable's
    physics receives them as arguments. Its controls are named
    ``"<name in snake_case>.<control>"`` in the flight, such as
    ``canards.fin_1``.

    A surface with no controls is still a :class:`Controllable`: every
    aerodynamic surface is one, with ``controls`` empty.

    Attributes
    ----------
    name : str
        Name of the controllable. Its control inputs are named after it.
    controls : tuple of str
        Names of the controls, in the order the physics expects their values.
    control_defaults : dict
        Value of each control at the start of every flight, and the value used
        outside a flight.
    control_limits : dict
        ``(low, high)`` of each control, or ``None`` for no limits. A value
        written outside them is moved to the nearest limit.
    states : tuple of str
        Names of the controllable's own states, which the flight integrates
        along with the rocket's: the position an actuator has reached, the
        gas left in a thruster's tank, the integral of a continuous law.
        Empty (the default) for a controllable that acts at once with the
        values written to its controls. In the flight, a state ``angle`` of
        the controllable ``"Canards"`` is named ``canards_angle``.
    """

    states = ()

    def __init__(
        self,
        name,
        controls=(),
        control_defaults=None,
        control_limits=None,
    ):
        """Set the controllable's name and controls.

        Parameters
        ----------
        name : str
            Name of the controllable. Its control inputs are named after it in
            snake_case, lower case with words joined by underscores:
            ``"Air Brakes"`` gives ``air_brakes.<control>``, ``"TVCGimbal"``
            gives ``tvc_gimbal.<control>`` and ``"Fin #2 (left)"`` gives
            ``fin_2_left.<control>``.
        controls : tuple of str, optional
            Names of the controls a controller can set, such as
            ``("firing",)``. Each must be made of letters, digits and
            underscores and not start with a digit or an underscore. Default
            is no controls.
        control_defaults : dict, optional
            Value of each control at the start of every flight, by name. A
            control left out starts at 0. Default ``None``.
        control_limits : dict, optional
            ``(low, high)`` of each control that has limits, by name: the
            range the mechanism can reach. A value written outside it is
            moved to the nearest limit. Default ``None``, no limits.

        Raises
        ------
        ValueError
            If a control name is not a valid name or is repeated, if
            ``control_defaults`` or ``control_limits`` name something that is
            not a control, or if limits are not ``(low, high)`` with
            ``low <= high``.
        TypeError
            If the default of a control is not a number.
        """
        self.name = name
        self._set_controls(controls, control_defaults, control_limits)
        self.prints = _ControllablePrints(self)

    def info(self):
        """Print the controllable's name, the functions it was built from,
        its controls and its states.

        Returns
        -------
        None
        """
        self.prints.all()

    def all_info(self):
        """Print all the information about the controllable.

        Returns
        -------
        None
        """
        self.info()

    def _set_controls(self, controls, defaults, limits):
        """Check and store the controls. Needs ``self.name``."""
        # One name given alone is one control
        self.controls = (controls,) if isinstance(controls, str) else tuple(controls)
        # A control left out of the defaults starts at 0
        self.control_defaults = {
            **dict.fromkeys(self.controls, 0.0),
            **dict(defaults or {}),
        }
        self.control_limits = dict(limits or {})
        self._check_controls()
        for control in self.controls:
            given = self.control_limits.get(control)
            self.control_limits[control] = None if given is None else tuple(given)
            self.control_defaults[control] = limited_value(
                self.control_defaults[control], self.control_limits[control]
            )

    def _check_controls(self):
        """Raise if the controls cannot be flown: a control named twice or not
        as a plain name, a control without a number as its default, limits
        that are not ``(low, high)`` with ``low <= high``, or defaults or
        limits for something that is not a control.

        Called when the controllable is built and again when it is added to a rocket,
        since the defaults and limits can be changed in between."""
        name, controls = self.name, tuple(self.controls)
        defaults, limits = self.control_defaults, self.control_limits
        if len(set(controls)) != len(controls):
            raise ValueError(f"'{name}' lists a control more than once: {controls}.")
        for control in controls:
            if (
                not isinstance(control, str)
                or not control.isidentifier()
                or keyword.iskeyword(control)
                or control.startswith("_")
            ):
                raise ValueError(
                    f"'{name}' has a control named {control!r}. A control name "
                    "must be made of letters, digits and underscores, must not "
                    "start with a digit or an underscore, and must not be a "
                    "Python keyword."
                )
            if control not in defaults:
                raise ValueError(f"The control '{control}' of '{name}' has no default.")
            try:
                float(defaults[control])
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"The default of the control '{control}' of '{name}' must be "
                    f"a number, got {defaults[control]!r}."
                ) from error
            bounds = limits.get(control)
            if bounds is None:
                continue
            what = f"The limits of the control '{control}' of '{name}'"
            try:
                low, high = (float(value) for value in bounds)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{what} must be (low, high) or None, got {bounds!r}."
                ) from error
            if low > high:
                raise ValueError(
                    f"{what} are ({low}, {high}): the first must not be larger "
                    "than the second."
                )
        for setting, given in (
            ("control_defaults", defaults),
            ("control_limits", limits),
        ):
            unknown = sorted(set(given) - set(controls))
            if unknown:
                raise ValueError(
                    f"'{name}' has {setting} for {unknown}, which are not among "
                    f"its controls {controls}."
                )

    def _check_state_names(self):
        """Raise if a state is not named as a plain name. The flight names
        each state ``<name in snake_case>_<state>`` and reads it by that name,
        which must not look like a control input, ``<key>.<control>``."""
        for state in self.states:
            if not isinstance(state, str) or not state.isidentifier():
                raise ValueError(
                    f"The states of '{self.name}' must be names made of letters, "
                    f"digits and underscores, got {state!r}."
                )

    def _check_states(self):
        """Raise if the states cannot be flown: a state not named as a plain
        name, or an ``initial_states`` that does not give one value per state,
        since the flight seeds the states from it. Called when the
        controllable is added to a rocket."""
        self._check_state_names()
        starts = len(list(self.initial_states()))
        if starts != len(self.states):
            raise ValueError(
                f"'{self.name}' has {len(self.states)} states, but its "
                f"initial_states gives {starts} values."
            )

    def initial_states(self):
        """Return the value of each of the controllable's states at the start of a
        flight, in the order of ``states``. Defaults to 0 for every state.

        Returns
        -------
        sequence of float
        """
        return [0.0] * len(self.states)

    def state_derivative(self, states, commands, conditions):  # pylint: disable=unused-argument
        """Return the derivative with respect to time of each of the controllable's
        states, in the order of ``states``. It must depend on its arguments
        alone: no memory, no side effects.

        Parameters
        ----------
        states : sequence of float
            The values of the controllable's states, in the order of ``states``.
        commands : list of float
            The values written to the controllable's controls, in the order of
            ``controls``.
        conditions : FlightConditions
            The flight at this moment. See
            :class:`rocketpy.control.FlightConditions`.

        Returns
        -------
        sequence of float
            One value per state. Empty for a controllable without states.
        """
        return ()

    def actual_controls(self, states, commands, conditions):  # pylint: disable=unused-argument
        """Return the values the controllable acts with, one per control, from what
        was written to its controls and the controllable's states.

        The controllable's physics receives these values, not the ones
        written: an actuator returns the position it has reached, one of its
        states. Defaults to the values written, unchanged. It must depend on
        its arguments alone.

        Parameters
        ----------
        states : sequence of float
            The values of the controllable's states, in the order of ``states``.
        commands : list of float
            The values written to the controllable's controls, in the order of
            ``controls``.
        conditions : FlightConditions
            The flight at this moment. See
            :class:`rocketpy.control.FlightConditions`.

        Returns
        -------
        sequence of float
            One value per control, in the order of ``controls``.
        """
        return commands

    def _acts_on_its_own(self):
        """Whether the controllable has anything to add to the equations of
        motion beyond its physics: states of its own, or an ``actual_controls``
        of its own."""
        return bool(self.states) or (
            type(self).actual_controls is not Controllable.actual_controls
        )

    def _state_names(self):
        """The names of the controllable's states in the flight: ``<name in
        snake_case>_<state>``, such as ``canards_angle``."""
        key = snake_case_key(self.name)
        return tuple(f"{key}_{state}" for state in self.states)

    def _default_control_values(self):
        """The defaults of the controls, in order: the values used outside a
        flight."""
        return tuple(self.control_defaults[name] for name in self.controls)

    def _control_settings(self):
        """The controls, defaults and limits, as saved by ``to_dict``."""
        return {
            "controls": list(self.controls),
            "control_defaults": dict(self.control_defaults),
            "control_limits": {
                name: None if limits is None else list(limits)
                for name, limits in self.control_limits.items()
            },
        }

    def to_dict(self, **kwargs):
        """Return the controllable as a dictionary, to be saved.

        A controllable written by the user holds its physics in code, so the controllable
        itself is saved pickled, next to its control settings for reading.
        RocketPy's own controllables save their parameters instead.

        Parameters
        ----------
        **kwargs : dict
            ``allow_pickle`` (default ``True``): save the controllable itself. With
            ``False`` only its name and control settings are saved, and the
            controllable is left out of the rocket when the file is loaded.

        Returns
        -------
        dict
        """
        data = {"name": self.name, **self._control_settings()}
        if kwargs.get("allow_pickle", True):
            data["controllable"] = to_hex_encode(self)
        return data

    @classmethod
    def from_dict(cls, data):
        """Build the controllable from a dictionary made by :meth:`to_dict`.

        Parameters
        ----------
        data : dict

        Returns
        -------
        Controllable or UnloadedControllable
            The controllable, or an ``UnloadedControllable``, which only
            keeps its name, when it was saved without pickling or its class
            is not available here.
        """
        try:
            return from_hex_decode(data["controllable"])
        except _LOAD_ERRORS:
            return UnloadedControllable(data.get("name"))


class ThrustEffector(Controllable):
    """A controllable that turns the motor's thrust, such as a gimbaled nozzle
    or jet vanes. Add it with :meth:`rocketpy.Rocket.add_effector`.

    The thrust is the motor's, as the thrust curve of the rocket's
    :class:`rocketpy.Motor` gives it at that instant, along the rocket's
    axis at the nozzle exit. It is a force along a line: its vector and the
    nozzle exit it passes through. The effector receives both and returns
    both, and the moment of the thrust on the rocket is taken about the
    returned exit. Several thrust effectors act in turn, in the order they
    were added.

    Built from a function, or subclassed with :meth:`thrust_and_nozzle` as
    a method. See :ref:`controllers` for examples.

    Attributes
    ----------
    thrust_function : callable or None
        The function the effector was built from, or ``None`` when
        :meth:`thrust_and_nozzle` is defined as a method.
    """

    def __init__(
        self,
        thrust_and_nozzle=None,
        controls=(),
        control_defaults=None,
        control_limits=None,
        name="Thrust Effector",
    ):
        """Build a thrust effector.

        Parameters
        ----------
        thrust_and_nozzle : callable, optional
            ``thrust_and_nozzle(thrust, nozzle, *controls, conditions)``: the
            thrust so far, in N, where the nozzle exit is so far, in m from the
            center of dry mass, one value per control in the order of
            ``controls``, and the flight at this moment, a
            :class:`rocketpy.control.FlightConditions`. Returns ``(thrust,
            nozzle)`` after the effector, each three numbers in the body frame.
            Leave it out in a subclass that defines :meth:`thrust_and_nozzle`.
        controls : tuple of str, optional
            Names of the controls a controller can set, such as
            ``("pitch", "yaw")``. Default: none.
        control_defaults : dict, optional
            Value of each control at the start of a flight. Controls not
            listed start at 0.
        control_limits : dict, optional
            ``(low, high)`` for each control that has limits. A value written
            outside them is moved to the nearest limit.
        name : str, optional
            Name of the effector. Its controls are named after it, in
            snake_case: ``"TVC Gimbal"`` gives ``tvc_gimbal.<control>``.
            Default ``"Thrust Effector"``.
        """
        super().__init__(
            name=name,
            controls=controls,
            control_defaults=control_defaults,
            control_limits=control_limits,
        )
        self.thrust_function = check_hook(
            self,
            "thrust_and_nozzle",
            thrust_and_nozzle,
            ThrustEffector.thrust_and_nozzle,
            ("thrust", "nozzle", *self.controls, "conditions"),
        )

    def thrust_and_nozzle(self, thrust, nozzle, controls, conditions):
        """Return the thrust force and the nozzle exit after this effector has
        acted on them.

        Parameters
        ----------
        thrust : Vector
            The thrust force so far, in N, in the rocket's body frame:
            ``[0, 0, T]`` for the motor alone, with z along the rocket's axis
            toward the nose.
        nozzle : Vector
            Where the nozzle exit is so far, in m, in the body frame from the
            center of dry mass. The thrust acts along a line through it.
        controls : list of float
            The values the effector acts with, in the order of ``controls``.
        conditions : FlightConditions
            The flight at this moment. See
            :class:`rocketpy.control.FlightConditions`.

        Returns
        -------
        tuple
            ``(thrust, nozzle)``: the new thrust force, in N, and where the
            nozzle exit now is, in m, each three numbers in the body frame.
            An effector that does not move the nozzle returns ``nozzle`` as
            it was given.
        """
        return self.thrust_function(thrust, nozzle, *controls, conditions)

    def __repr__(self):
        return f"ThrustEffector(name={self.name!r}, controls={self.controls!r})"


class ForceEffector(Controllable):
    """A controllable that pushes or turns the rocket directly, other than
    through the air or the motor: a cold-gas thruster, a reaction wheel. Add
    it with :meth:`rocketpy.Rocket.add_effector`.

    It returns a force and a moment in the rocket's body frame: ``z`` along
    the rocket toward the nose, ``x`` and ``y`` the axes the fins are placed
    in (see :ref:`rocket_axes`). The moment is about the center of dry mass,
    unless the effector was placed with the ``position`` of
    :meth:`rocketpy.Rocket.add_effector`: the flight then adds the moment of
    the force about that point, and the returned moment is only a pure
    couple, such as a reaction wheel's torque. An effector is not
    aerodynamic: it takes no part in the static margin or the center of
    pressure.

    Built from a function, or subclassed with :meth:`force_and_moment` as a
    method. See :ref:`controllers` for examples and the frame conventions
    in full.

    Attributes
    ----------
    force_function : callable or None
        The function the effector was built from, or ``None`` when
        :meth:`force_and_moment` is defined as a method.
    """

    def __init__(
        self,
        force_and_moment=None,
        controls=(),
        control_defaults=None,
        control_limits=None,
        name="Force Effector",
    ):
        """Build a force effector.

        Parameters
        ----------
        force_and_moment : callable, optional
            ``force_and_moment(*controls, conditions)``: one value per control
            in the order of ``controls``, and the flight at this moment, a
            :class:`rocketpy.control.FlightConditions`. Returns ``(force,
            moment)``, each three numbers in the body frame: the force in N
            and the moment in N m, about the center of dry mass or, for a
            placed effector, the pure couple. Leave it out in a subclass that
            defines :meth:`force_and_moment`.
        controls : tuple of str, optional
            Names of the controls a controller can set, such as ``("firing",)``.
            Default: none.
        control_defaults : dict, optional
            Value of each control at the start of a flight. Controls not
            listed start at 0.
        control_limits : dict, optional
            ``(low, high)`` for each control that has limits. A value written
            outside them is moved to the nearest limit.
        name : str, optional
            Name of the effector. Its controls are named after it, in
            snake_case: ``"RCS"`` gives ``rcs.<control>``. Default
            ``"Force Effector"``.
        """
        super().__init__(
            name=name,
            controls=controls,
            control_defaults=control_defaults,
            control_limits=control_limits,
        )
        self.force_function = check_hook(
            self,
            "force_and_moment",
            force_and_moment,
            ForceEffector.force_and_moment,
            (*self.controls, "conditions"),
        )

    def force_and_moment(self, controls, conditions):
        """Return the force and moment the effector applies to the rocket, in
        the body frame.

        Parameters
        ----------
        controls : list of float
            The values the effector acts with, in the order of ``controls``.
        conditions : FlightConditions
            The flight at this moment. See
            :class:`rocketpy.control.FlightConditions`.

        Returns
        -------
        tuple
            ``(force, moment)``, each three numbers in the body frame: the
            force in N and the moment in N m, about the rocket's center of dry
            mass or, for a placed effector, the pure couple added to the
            moment of the force where it acts.
        """
        return self.force_function(*controls, conditions)

    def __repr__(self):
        return f"ForceEffector(name={self.name!r}, controls={self.controls!r})"
