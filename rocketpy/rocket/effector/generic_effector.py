from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.tools import from_hex_decode, to_hex_encode

from .effector import Effector

_ZERO = Vector([0.0, 0.0, 0.0])


class GenericEffector(Effector):
    """A control effector defined by user-supplied force/moment functions.

    The non-aerodynamic analog of :class:`rocketpy.GenericSurface`: instead of
    aerodynamic coefficients, you provide the body-frame **force** and/or
    **moment** directly as functions of the control state and flight state. The
    moment produced by a force at an offset application point is added
    automatically (``force_arm x force``), so a pure lateral thruster only needs
    a ``force`` function.

    Both functions take keyword arguments only and receive the current control
    values (by control-variable name) plus ``time``, ``omega`` (body angular
    velocity), ``velocity_body``, ``mach`` and ``environment``.

    Examples
    --------
    A roll-torque effector driven by a controller::

        rcs = GenericEffector(
            moment=lambda **k: (0.0, 0.0, k["roll_torque"]),
            controls=("roll_torque",),
            name="Roll RCS",
        )

    A lateral (side-force) thruster at a nozzle station::

        side = GenericEffector(
            force=lambda **k: (0.0, k["side_force"], 0.0),
            position=-1.2,
            controls=("side_force",),
            name="Side RCS",
        )
    """

    def __init__(
        self,
        force=None,
        moment=None,
        position=0.0,
        controls=("command",),
        name="Generic Effector",
    ):
        """Initialize the generic effector.

        Parameters
        ----------
        force : callable, optional
            ``force(**kwargs) -> (fx, fy, fz)`` body-frame force. Defaults to
            ``None`` (zero force).
        moment : callable, optional
            ``moment(**kwargs) -> (mx, my, mz)`` body-frame moment about the
            application point. Defaults to ``None`` (zero moment).
        position : float or tuple, optional
            Application station (axial float) or ``(x, y, z)`` point; sets the
            moment arm for the force term. Defaults to ``0.0``.
        controls : iterable of str, optional
            Names of the control axes. Defaults to ``("command",)``.
        name : str, optional
            Effector name. Defaults to ``"Generic Effector"``.
        """
        super().__init__(position=position, controls=controls, name=name)
        self.force = force
        self.moment = moment

    def evaluate(self, force_arm, time, velocity_body, omega, mach, environment):
        """See :meth:`Effector.evaluate`. Returns the body-frame force and the
        moment about the center of dry mass (own moment plus ``force_arm x
        force``)."""
        kwargs = dict(self.control_state)
        kwargs.update(
            time=time,
            omega=omega,
            velocity_body=velocity_body,
            mach=mach,
            environment=environment,
        )
        force = Vector(list(self.force(**kwargs))) if self.force else _ZERO
        moment = Vector(list(self.moment(**kwargs))) if self.moment else _ZERO
        return force, moment + (force_arm ^ force)

    def to_dict(self, include_outputs=False, **kwargs):  # pylint: disable=unused-argument
        allow_pickle = kwargs.get("allow_pickle", True)
        force = self.force
        moment = self.moment
        if allow_pickle:
            force = to_hex_encode(force) if force is not None else None
            moment = to_hex_encode(moment) if moment is not None else None
        else:
            force = getattr(force, "__name__", None)
            moment = getattr(moment, "__name__", None)
        return {
            "force": force,
            "moment": moment,
            "position": self.position,
            "controls": self.control_variables,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        force = data.get("force")
        moment = data.get("moment")
        try:
            force = from_hex_decode(force)
        except (TypeError, ValueError):
            pass
        try:
            moment = from_hex_decode(moment)
        except (TypeError, ValueError):
            pass
        return cls(
            force=force if callable(force) else None,
            moment=moment if callable(moment) else None,
            position=data.get("position", 0.0),
            controls=data.get("controls", ("command",)),
            name=data.get("name", "Generic Effector"),
        )
