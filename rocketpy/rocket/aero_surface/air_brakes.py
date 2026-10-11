import numpy as np

from rocketpy.control.control_signal import limited_value
from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _AirBrakesPlots
from rocketpy.prints.aero_surface_prints import _AirBrakesPrints

from .generic_surface import GenericSurface


class AirBrakes(GenericSurface):
    """AirBrakes class. A :class:`GenericSurface` with one control,
    ``deployment_level``, and a drag coefficient that depends on it.

    During a flight the deployment level is the ``air_brakes.deployment_level``
    channel of the flight's control inputs, written by a controller as
    ``context.controls["air_brakes.deployment_level"]``. The air brakes object
    itself is not changed by a flight.

    Attributes
    ----------
    AirBrakes.drag_coefficient : Function
        Drag coefficient as a function of deployment level and Mach number.
    AirBrakes.drag_coefficient_curve : int, float, callable, array, string, Function
        Curve that defines the drag coefficient as a function of deployment level
        and Mach number.  Used as the source of `AirBrakes.drag_coefficient`.
        It is stored as the surface's axial force coefficient ``cA``, read at
        the current deployment level in flight; the wind-frame ``cD`` is a view
        derived from it.
    AirBrakes.deployment_level : float
        Deployment level at the start of every flight, and outside a flight,
        ranging from 0 to 1. Deployment level is the fraction of the total
        airbrake area that is deployed. It is the default of the
        ``deployment_level`` control.
    AirBrakes.reference_area : int, float
        Reference area used to calculate the drag force of the air brakes
        from the drag coefficient curve. Units of m^2.
    AirBrakes.clamp : bool, optional
        If True, the simulation will clamp the deployment level to 0 or 1 if
        the deployment level is out of bounds. If False, the simulation will
        not clamp the deployment level and will instead raise a warning if
        the deployment level is out of bounds. Default is True.
    AirBrakes.name : str
        Name of the air brakes.
    """

    def __init__(
        self,
        drag_coefficient_curve,
        reference_area,
        clamp=True,
        override_rocket_drag=False,
        deployment_level=0,
        name="AirBrakes",
    ):
        """Initializes the AirBrakes class.

        Parameters
        ----------
        drag_coefficient_curve : int, float, callable, array, string, Function
            This parameter represents the drag coefficient associated with the
            air brakes and/or the entire rocket, depending on the value of
            ``override_rocket_drag``.

            - If a constant, it should be an integer or a float representing a
              fixed drag coefficient value.
            - If a function, it must take two parameters: deployment level and
              Mach number, and return the drag coefficient. This function allows
              for dynamic computation based on deployment and Mach number.
            - If an array, it should be a 2D array with three columns: the first
              column for deployment level, the second for Mach number, and the
              third for the corresponding drag coefficient.
            - If a string, it should be the path to a .csv or .txt file. The
              file must contain three columns: the first for deployment level,
              the second for Mach number, and the third for the drag
              coefficient.
            - If a Function, it must take two parameters: deployment level and
              Mach number, and return the drag coefficient.

            .. note:: At deployment level 0 the drag coefficient is 0,
                whatever the curve says: retracted air brakes add no drag.

        reference_area : int, float
            Reference area used to calculate the drag force of the air brakes
            from the drag coefficient curve. Units of m^2.
        clamp : bool, optional
            If True, the simulation will clamp the deployment level to 0 or 1 if
            the deployment level is out of bounds. If False, the simulation will
            not clamp the deployment level and will instead raise a warning if
            the deployment level is out of bounds. Default is True.
        override_rocket_drag : bool, optional
            If False, the drag of the air brakes is added to the drag of the
            rocket. If True, while the air brakes are deployed (deployment
            level above 0) their drag coefficient curve is the drag of the
            whole rocket: the rocket's own drag curve is left out. Retracted,
            the rocket flies with its own drag. Default is False.
        deployment_level : float, optional
            Initial deployment level, ranging from 0 to 1. Deployment level is
            the fraction of the total airbrake area that is deployed. Default
            is 0.
        name : str, optional
            Name of the air brakes. Default is "AirBrakes". It also names their
            control channels: ``"AirBrakes"`` gives
            ``air_brakes.deployment_level``. Two air brakes on one rocket need
            different names.

        Returns
        -------
        None
        """
        self.override_rocket_drag = override_rocket_drag
        self.drag_coefficient_curve = drag_coefficient_curve
        # Back-compatible 2-input (deployment level, Mach) drag curve, kept for
        # display/serialization and as the source of the multivariable drag
        # coefficient below.
        self.drag_coefficient = Function(
            drag_coefficient_curve,
            inputs=["Deployment Level", "Mach"],
            outputs="Drag Coefficient",
            interpolation="linear",
        )

        # The curve is given to the surface as its axial coefficient, over
        # only the two variables it uses: the force acts along the rocket's
        # axis. Retracted air brakes make no force, whatever the curve says.
        def drag_coefficient_function(mach, deployment_level):
            if deployment_level == 0:
                return 0.0
            return self.drag_coefficient.get_value_opt(deployment_level, mach)

        super().__init__(
            reference_area=reference_area,
            reference_length=2 * (reference_area / np.pi) ** 0.5,
            coefficients={"cA": drag_coefficient_function},
            center_of_pressure=(0, 0, 0),
            name=name,
            controls=("deployment_level",),
            control_defaults={"deployment_level": deployment_level},
            control_limits={"deployment_level": (0, 1)},
            clamp=clamp,
        )

        # Set by Rocket.add_air_brakes when no position is given: the force
        # then acts at the center of dry mass, wherever it is
        self._at_center_of_dry_mass = False

        self.prints = _AirBrakesPrints(self)
        self.plots = _AirBrakesPlots(self)

    @property
    def deployment_level(self):
        """The deployment level the air brakes start every flight at, from 0
        to 1. Setting it outside that range clamps it, or warns when ``clamp``
        is ``False``. During a flight the deployment level is a channel of the
        flight's control inputs, not this attribute."""
        return self.control_defaults["deployment_level"]

    @deployment_level.setter
    def deployment_level(self, value):
        self.control_defaults["deployment_level"] = limited_value(
            value,
            self.control_limits["deployment_level"],
            self.clamp,
            f"{self.name}.deployment_level",
        )

    @property
    def initial_deployment_level(self):
        """The deployment level at the start of every flight. The same as
        :attr:`deployment_level`."""
        return self.deployment_level

    @initial_deployment_level.setter
    def initial_deployment_level(self, value):
        self.deployment_level = value

    def evaluate_geometrical_parameters(self):
        """Evaluates the geometrical parameters of the aerodynamic surface.

        Returns
        -------
        None
        """

    def info(self):
        """Prints and plots summarized information of the aerodynamic surface.

        Returns
        -------
        None
        """
        self.prints.geometry()

    def all_info(self):
        """Prints and plots all information of the aerodynamic surface.

        Returns
        -------
        None
        """
        self.info()
        self.plots.drag_coefficient_curve()

    def to_dict(self, include_outputs=False, **kwargs):  # pylint: disable=unused-argument
        return {
            "drag_coefficient_curve": self.drag_coefficient,
            "reference_area": self.reference_area,
            "clamp": self.clamp,
            "override_rocket_drag": self.override_rocket_drag,
            "deployment_level": self.initial_deployment_level,
            "name": self.name,
            "at_center_of_dry_mass": self._at_center_of_dry_mass,
        }

    @classmethod
    def from_dict(cls, data):
        air_brakes = cls(
            drag_coefficient_curve=data.get("drag_coefficient_curve"),
            reference_area=data.get("reference_area"),
            clamp=data.get("clamp"),
            override_rocket_drag=data.get("override_rocket_drag"),
            deployment_level=data.get("deployment_level"),
            name=data.get("name"),
        )
        # Saved before air brakes were surfaces: they acted at the center of
        # dry mass
        air_brakes._at_center_of_dry_mass = data.get("at_center_of_dry_mass", True)
        return air_brakes
