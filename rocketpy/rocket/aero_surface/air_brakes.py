import warnings

import numpy as np

from rocketpy.control._helpers import limited_value
from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _AirBrakesPlots
from rocketpy.prints.aero_surface_prints import _AirBrakesPrints

from .generic_surface import GenericSurface


class AirBrakes(GenericSurface):
    """Air brakes: flaps that open to add drag and slow the rocket down,
    usually to hit a target apogee.

    The air brakes have one control, ``deployment_level``, from 0 (retracted)
    to 1 (fully open), and a drag coefficient that depends on it and on the
    Mach number. In a flight the deployment level is a control input named
    after the air brakes, ``air_brakes.deployment_level`` for the default
    name, which a controller writes as
    ``context.controls.air_brakes.deployment_level``. The air brakes object
    itself is not changed by a flight: :attr:`initial_deployment_level` is
    where every flight starts, and ``flight.controls`` holds what was
    flown.

    The drag acts along the rocket's axis, backward, at the point the air
    brakes are added at. Retracted air brakes make no force at all: the
    rocket then flies with its own drag curves. The
    model has no sideways force or moment of its own: a deployed flap at an
    angle of attack is treated as pure drag.

    Attributes
    ----------
    AirBrakes.drag_coefficient : Function
        The drag coefficient as a function of the deployment level and the
        Mach number, in that order, as given.
    AirBrakes.drag_coefficient_curve : int, float, callable, array, str, Function
        The curve as given, the source of :attr:`drag_coefficient`.
    AirBrakes.initial_deployment_level : float
        The deployment level every flight starts at, from 0 to 1.
    AirBrakes.reference_area : float
        The area the drag coefficient is multiplied by, in m².
    AirBrakes.override_rocket_drag : bool
        Whether the curve is the drag of the whole rocket at each deployment
        level, and not of the air brakes alone.
    AirBrakes.clamp : bool
        Whether the deployment level is kept between 0 and 1.
    AirBrakes.name : str
        The name of the air brakes, which also names their control input.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        drag_coefficient_curve,
        reference_area,
        clamp=True,
        override_rocket_drag=False,
        initial_deployment_level=0,
        name="AirBrakes",
        active_during="always",
        active=True,
        deployment_level=None,
    ):
        """Build the air brakes.

        Parameters
        ----------
        drag_coefficient_curve : int, float, callable, array, str, Function
            The drag coefficient as a function of the deployment level (0 to
            1) and the Mach number, in that order. A number is a constant
            coefficient at every level above 0. A function is called as
            ``curve(deployment_level, mach)``. An array has three columns,
            deployment level, Mach number and drag coefficient, one row per
            point; a string is the path to a ``.csv`` or ``.txt`` file with
            the same three columns. Points are interpolated linearly. The
            coefficient is of the air brakes alone, unless
            ``override_rocket_drag`` is ``True``. Whatever the curve says,
            retracted air brakes (deployment level 0) add no drag.
        reference_area : float
            The area the drag coefficient is multiplied by to get the drag
            force, in m²: the one the curve was made with, usually the
            rocket's cross section or the flaps' area.
        clamp : bool, optional
            Whether the deployment level is kept between 0 (retracted) and 1
            (fully open): ``True`` (the default) moves a value outside that
            range to the nearest end, ``False`` puts no limit on it and the
            drag curve is extrapolated.
        override_rocket_drag : bool, optional
            What the curve is the drag coefficient of.

            - ``False`` (the default): the air brakes alone. Their drag is
              added to the rocket's.
            - ``True``: the whole rocket at each deployment level, as measured
              in a wind tunnel or computed for the complete rocket. The air
              brakes then add the difference between the curve and its value
              at deployment level 0, at the same Mach number, to the rocket's
              own drag. The curve is of the rocket with its motor off: while
              the motor burns, the same difference is added to the rocket's
              ``power_on_drag``. The curve must include deployment level 0, and
              the rocket's ``power_off_drag`` should be that retracted drag: the
              total drag is then the curve. If they differ, the rocket's drag curves
              are the drag with the air brakes retracted and the curve only
              gives what opening them adds; a warning is shown when the air
              brakes are added to a rocket and the two differ by more than
              5%. A constant curve adds nothing.
        initial_deployment_level : float, optional
            The deployment level every flight starts at, from 0 (retracted)
            to 1 (fully open), the fraction of the flaps' travel. Default 0.
        name : str, optional
            The name of the air brakes. It also names their control input, in
            snake_case: the default ``"AirBrakes"`` gives
            ``air_brakes.deployment_level``. Two air brakes on one rocket need
            different names.
        active_during : str, optional
            The motor phase the air brakes make force in: ``"always"`` (the
            default), ``"power_on"`` or ``"power_off"``.
        active : bool, optional
            Whether the air brakes start each flight switched on. Default
            ``True``.
        deployment_level : float, optional
            Deprecated name of ``initial_deployment_level``. Support for it
            will be removed in v1.16.

        Raises
        ------
        ValueError
            If ``reference_area`` is not positive, the curve does not depend on
            the deployment level and the Mach number, or ``active_during`` is
            not one of the three phases.
        """
        if deployment_level is not None:
            warnings.warn(
                "`AirBrakes(deployment_level=...)` is deprecated: it is the "
                "deployment level every flight starts at, now named "
                "`initial_deployment_level`. Support for `deployment_level` will "
                "be removed in v1.16.",
                DeprecationWarning,
                stacklevel=2,
            )
            initial_deployment_level = deployment_level
        if not reference_area > 0:
            raise ValueError(
                f"The reference area must be positive, in m², got {reference_area!r}."
            )
        self.override_rocket_drag = override_rocket_drag
        self.clamp = bool(clamp)
        self.drag_coefficient_curve = drag_coefficient_curve
        # The curve over (deployment level, Mach), as given: shown, saved and
        # the source of the axial coefficient below
        source = drag_coefficient_curve
        if isinstance(source, (int, float)):
            # A number: the same coefficient at every level and Mach number
            def constant(deployment_level, mach):  # pylint: disable=unused-argument
                return drag_coefficient_curve

            source = constant

        try:
            self.drag_coefficient = Function(
                source,
                inputs=["Deployment Level", "Mach"],
                outputs="Drag Coefficient",
                interpolation="linear",
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                "The drag coefficient curve must depend on the deployment level "
                "and the Mach number, in that order: a number, a function of the "
                "two, or an array or file with three columns (deployment level, "
                f"Mach, drag coefficient), got {drag_coefficient_curve!r}."
            ) from error

        super().__init__(
            reference_area=reference_area,
            reference_length=2 * (reference_area / np.pi) ** 0.5,
            coefficients={"cA": self._axial_coefficient()},
            center_of_pressure=(0, 0, 0),
            name=name,
            active_during=active_during,
            active=active,
            controls=("deployment_level",),
            control_defaults={"deployment_level": initial_deployment_level},
            control_limits={"deployment_level": (0, 1) if clamp else None},
        )

        self.prints = _AirBrakesPrints(self)
        self.plots = _AirBrakesPlots(self)

    def _axial_coefficient(self):
        """The surface's axial coefficient, over only the two variables it
        uses: the force acts along the rocket's axis. Retracted air brakes
        make no force, whatever the curve says. The curve is read from the
        attribute at each call, so that replacing it after the air brakes are
        built changes the drag flown."""
        if self.override_rocket_drag:
            # A curve of the whole rocket: the air brakes are what it adds to
            # its retracted column, and the rocket keeps its own drag curves

            def axial_coefficient(mach, deployment_level):
                if deployment_level == 0:
                    return 0.0
                curve = self.drag_coefficient.get_value_opt
                return curve(deployment_level, mach) - curve(0, mach)

        else:

            def axial_coefficient(mach, deployment_level):
                if deployment_level == 0:
                    return 0.0
                return self.drag_coefficient.get_value_opt(deployment_level, mach)

        return axial_coefficient

    def _bind_coefficients(self):
        """Rebuild the axial coefficient, which reads the drag curve of the
        air brakes it was built for, so that a deep copy reads its own."""
        coefficient = self._as_coefficient(self._axial_coefficient(), "cA")
        self._input_coefficients["cA"] = self.cA = coefficient

    @property
    def initial_deployment_level(self):
        """The deployment level the air brakes start every flight at, from 0
        to 1. Setting it outside that range moves it to the nearest end, unless
        ``clamp`` is ``False``. During a flight the deployment level is a control input
        of the flight, ``flight.controls`` gives it after the flight,
        and this attribute does not change."""
        return self.control_defaults["deployment_level"]

    @initial_deployment_level.setter
    def initial_deployment_level(self, value):
        self.control_defaults["deployment_level"] = limited_value(
            value, self.control_limits["deployment_level"], "deployment_level"
        )

    @property
    def deployment_level(self):
        """Deprecated: the same as :attr:`initial_deployment_level`. A flight
        no longer moves the air brakes object: read the deployment level flown
        in ``flight.controls``."""
        _warn_deployment_level()
        return self.initial_deployment_level

    @deployment_level.setter
    def deployment_level(self, value):
        _warn_deployment_level()
        self.initial_deployment_level = value

    def evaluate_geometrical_parameters(self):
        """Nothing to compute: the air brakes' geometry is their reference
        area, given."""

    def _drag_in(self, flight):
        """The drag force of the air brakes over ``flight``, in N, as a
        Function of time, from the air at each stored row and the deployment
        level then in force. Zero while retracted or switched off.

        Private for now: where post-flight analyses of a part live is still
        open, see ``docs/development/post_events_issues/issue_flight_analysis.md``.
        ``plots.drag(flight)`` draws it.

        Raises
        ------
        ValueError
            If the air brakes are not on the flight's rocket.
        """
        on_rocket = self._on_rocket_of(flight)
        level = flight.controls[on_rocket.name]["deployment_level"]
        times = flight.time
        drag = []
        for t in times:
            if not on_rocket.is_active(t, flight):
                drag.append(0.0)
                continue
            speed = flight.free_stream_speed.get_value_opt(t)
            mach = flight.mach_number.get_value_opt(t)
            rho = flight.density.get_value_opt(t)
            coefficient = on_rocket.cA(0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0, level(t))
            drag.append(0.5 * rho * speed**2 * on_rocket.reference_area * coefficient)
        return Function(
            np.column_stack((times, drag)),
            "Time (s)",
            f"Drag of {on_rocket.name} (N)",
            "linear",
            "zero",
        )

    def info(self):
        """Print the air brakes' settings and their drag coefficient at a few
        deployment levels and Mach numbers.

        Returns
        -------
        None
        """
        self.prints.geometry()
        self.prints.drag_coefficient()

    def all_info(self):
        """Print everything :meth:`info` prints and plot the drag coefficient
        curve.

        Returns
        -------
        None
        """
        self.prints.all()
        self.plots.all()

    def to_dict(self, include_outputs=False, **kwargs):  # pylint: disable=unused-argument
        return {
            "drag_coefficient_curve": self.drag_coefficient,
            "reference_area": self.reference_area,
            "clamp": self.clamp,
            "override_rocket_drag": self.override_rocket_drag,
            "initial_deployment_level": self.initial_deployment_level,
            "name": self.name,
            "active_during": self.active_during,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data):
        # "deployment_level" is the key of files saved before v1.15
        level = data.get("initial_deployment_level", data.get("deployment_level", 0))
        return cls(
            drag_coefficient_curve=data["drag_coefficient_curve"],
            reference_area=data["reference_area"],
            clamp=data.get("clamp", True),
            override_rocket_drag=data.get("override_rocket_drag", False),
            initial_deployment_level=level,
            name=data.get("name", "AirBrakes"),
            active_during=data.get("active_during", "always"),
            active=data.get("active", True),
        )


def _warn_deployment_level():
    warnings.warn(
        "`AirBrakes.deployment_level` is deprecated: it is the deployment level "
        "the air brakes start every flight at, now named "
        "`initial_deployment_level`. A flight no longer moves the air brakes "
        "object: read the deployment level flown in `flight.controls`. "
        "Support for `deployment_level` will be removed in v1.16.",
        DeprecationWarning,
        stacklevel=3,
    )
