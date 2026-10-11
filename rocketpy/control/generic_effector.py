"""A part that pushes or turns the rocket directly, such as a cold-gas
thruster or a reaction wheel."""

import inspect

from rocketpy.control.control_signal import (
    _check_control_name,
    _check_limits,
    limited_value,
)
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.tools import from_hex_decode, to_hex_encode


class GenericEffector:
    """A part that pushes or turns the rocket directly, other than through
    the air or the motor: a cold-gas thruster, a reaction wheel, or any force
    and moment you can write as a function.

    The force and moment come from a function of the time, the state and the
    effector's controls::

        def thruster(time, state, firing):
            # 5 N sideways, 0.4 m ahead of the center of dry mass
            force = [5.0 * firing, 0, 0]
            moment = [0, 0.4 * 5.0 * firing, 0]
            return force, moment

        rcs = GenericEffector(
            thruster,
            controls=("firing",),
            control_limits={"firing": (0, 1)},
            name="RCS",
        )
        rocket.add_actuator(rcs)

    A controller then sets ``context.controls["rcs.firing"]``.

    An effector is not aerodynamic: it takes no part in the static margin,
    the center of pressure or ``rocket.to_surface()``. A reaction wheel is
    modeled as the moment it applies, the usual approximation.
    """

    def __init__(
        self,
        force_and_moment,
        controls=(),
        control_defaults=None,
        control_limits=None,
        clamp=True,
        name="Effector",
    ):
        """Build an effector.

        Parameters
        ----------
        force_and_moment : callable
            ``force_and_moment(time, state, *controls)``: called with the time
            in seconds, the state ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1,
            w2, w3]`` and then one value per control, in the order of
            ``controls``. It returns ``(force, moment)``, each three numbers
            in the rocket's body frame (z along the rocket's axis, toward the
            nose): the force in N and the moment in N m about the rocket's
            center of dry mass.
        controls : tuple of str, optional
            Names of the controls a controller can set, such as
            ``("firing",)``. Defaults to none.
        control_defaults : dict, optional
            Value of each control at the start of a flight. Controls not
            listed start at 0.
        control_limits : dict, optional
            ``(low, high)`` for each control that has limits.
        clamp : bool, optional
            What a value outside the limits does: ``True`` (default) moves it
            to the nearest limit, ``False`` keeps it and warns.
        name : str, optional
            Name of the effector. Its controls are named after it, in
            snake_case: ``"RCS"`` gives ``rcs.<control>``. Defaults to
            ``"Effector"``.
        """
        self.name = name
        self.controls = tuple(controls)
        for control in self.controls:
            _check_control_name(control, name)
        if len(set(self.controls)) != len(self.controls):
            raise ValueError(f"'{name}' lists a control twice: {self.controls}.")
        defaults = dict(control_defaults or {})
        limits = dict(control_limits or {})
        for setting, given in (
            ("control_defaults", defaults),
            ("control_limits", limits),
        ):
            unknown = sorted(set(given) - set(self.controls))
            if unknown:
                raise ValueError(
                    f"'{name}' has {setting} for {unknown}, which are not among "
                    f"its controls {self.controls}."
                )
        self.clamp = bool(clamp)
        self.control_limits = {}
        self.control_defaults = {}
        for control in self.controls:
            _check_limits(limits.get(control), control, name)
            given = limits.get(control)
            self.control_limits[control] = None if given is None else tuple(given)
            self.control_defaults[control] = limited_value(
                defaults.get(control, 0.0),
                self.control_limits[control],
                self.clamp,
                f"{name}.{control}",
            )
        self.function = self.__check_function(force_and_moment)

    def __check_function(self, function):
        if not callable(function):
            raise TypeError(f"force_and_moment must be callable, got {function!r}.")
        try:
            inspect.signature(function).bind(0.0, [0.0] * 13, *self.controls)
        except TypeError as error:
            raise ValueError(
                f"The force_and_moment function of '{self.name}' must take the "
                f"time, the state and then one value per control "
                f"({', '.join(self.controls) or 'none'})."
            ) from error
        return function

    def force_and_moment(self, t, state, controls):
        """Return the force and moment on the rocket, in the body frame.

        Parameters
        ----------
        t : float
            Time, in s.
        state : list
            ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        controls : sequence of float
            The values of the controls, in the order of ``controls``.

        Returns
        -------
        tuple of Vector
            The force, in N, and the moment about the center of dry mass, in
            N m.
        """
        force, moment = self.function(t, state, *controls)
        return Vector(force), Vector(moment)

    def __repr__(self):
        return f"GenericEffector(name={self.name!r}, controls={self.controls!r})"

    def to_dict(self, **kwargs):
        """Return the effector as a dictionary, to be saved."""
        function = self.function
        if kwargs.get("allow_pickle", True):
            function = to_hex_encode(function)
        else:
            function = getattr(function, "__name__", type(function).__name__)
        return {
            "force_and_moment": function,
            "controls": list(self.controls),
            "control_defaults": dict(self.control_defaults),
            "control_limits": {
                name: None if limits is None else list(limits)
                for name, limits in self.control_limits.items()
            },
            "clamp": self.clamp,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        """Build an effector from a dictionary made by :meth:`to_dict`."""
        return cls(
            from_hex_decode(data["force_and_moment"]),
            controls=data.get("controls", ()),
            control_defaults=data.get("control_defaults"),
            control_limits={
                name: limits
                for name, limits in data.get("control_limits", {}).items()
                if limits is not None
            },
            clamp=data.get("clamp", True),
            name=data.get("name", "Effector"),
        )
