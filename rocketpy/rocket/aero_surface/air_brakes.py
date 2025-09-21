import warnings

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _AirBrakesPlots
from rocketpy.prints.aero_surface_prints import _AirBrakesPrints

from .aero_surface import AeroSurface


class AirBrakes(AeroSurface):
    """AirBrakes class. Inherits from AeroSurface.

    Attributes
    ----------
    AirBrakes.drag_coefficient : Function
        Drag coefficient as a function of deployment level and Mach number.
    AirBrakes.drag_coefficient_curve : int, float, callable, array, string, Function
        Curve that defines the drag coefficient as a function of deployment level
        and Mach number.  Used as the source of `AirBrakes.drag_coefficient`.
    AirBrakes.deployment_level : float
        Current deployment level, ranging from 0 to 1. Deployment level is the
        fraction of the total airbrake area that is deployed.
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

            .. note:: For ``override_rocket_drag = False``, at
                deployment level 0, the drag coefficient is assumed to be 0,
                independent of the input drag coefficient curve. This means that
                the simulation always considers that at a deployment level of 0,
                the air brakes are completely retracted and do not contribute to
                the drag of the rocket.

        reference_area : int, float
            Reference area used to calculate the drag force of the air brakes
            from the drag coefficient curve. Units of m^2.
        clamp : bool, optional
            If True, the simulation will clamp the deployment level to 0 or 1 if
            the deployment level is out of bounds. If False, the simulation will
            not clamp the deployment level and will instead raise a warning if
            the deployment level is out of bounds. Default is True.
        override_rocket_drag : bool, optional
            If False, the air brakes drag coefficient will be added to the
            rocket's power off drag coefficient curve. If True, during the
            simulation, the rocket's power off drag will be ignored and the air
            brakes drag coefficient will be used for the entire rocket instead.
            Default is False.
        deployment_level : float, optional
            Initial deployment level, ranging from 0 to 1. Deployment level is
            the fraction of the total airbrake area that is Deployment. Default
            is 0.
        name : str, optional
            Name of the air brakes. Default is "AirBrakes".

        Returns
        -------
        None
        """
        super().__init__(name, reference_area, None)
        self.drag_coefficient_curve = drag_coefficient_curve
        self.drag_coefficient = Function(
            drag_coefficient_curve,
            inputs=["Deployment Level", "Mach"],
            outputs="Drag Coefficient",
        )
        self.clamp = clamp
        self.override_rocket_drag = override_rocket_drag
        self.initial_deployment_level = deployment_level
        self.deployment_level = deployment_level
        self.prints = _AirBrakesPrints(self)
        self.plots = _AirBrakesPlots(self)

    @property
    def deployment_level(self):
        """Returns the deployment level of the air brakes."""
        return self._deployment_level

    @deployment_level.setter
    def deployment_level(self, value):
        # Check if deployment level is within bounds and warn user if not
        if value < 0 or value > 1:
            # Clamp deployment level if clamp is True
            if self.clamp:
                # Make sure deployment level is between 0 and 1
                value = np.clip(value, 0, 1)
            else:
                # Raise warning if clamp is False
                warnings.warn(
                    f"Deployment level of {self.name} is smaller than 0 or "
                    + "larger than 1. Extrapolation for the drag coefficient "
                    + "curve will be used.",
                    UserWarning,
                )
        self._deployment_level = value

    def _reset(self):
        """Resets the air brakes to their initial state. This is ran at the
        beginning of each simulation to ensure the air brakes are in the correct
        state."""
        self.deployment_level = self.initial_deployment_level

    def evaluate_center_of_pressure(self):
        """Evaluates the center of pressure of the aerodynamic surface in local
        coordinates.

        For air brakes, all components of the center of pressure position are
        0.

        Returns
        -------
        None
        """
        self.cpx = 0
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)

    def evaluate_lift_coefficient(self):
        """Evaluates the lift coefficient curve of the aerodynamic surface.

        For air brakes, the current model assumes no lift is generated.
        Therefore, the lift coefficient (C_L) and its derivative relative to the
        angle of attack (C_L_alpha), is 0.

        Returns
        -------
        None
        """
        self.clalpha = Function(
            lambda mach: 0,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: 0,
            ["Alpha (rad)", "Mach"],
            "Lift Coefficient",
        )

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

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "drag_coefficient_curve": self.drag_coefficient,
            "reference_area": self.reference_area,
            "clamp": self.clamp,
            "override_rocket_drag": self.override_rocket_drag,
            "deployment_level": self.initial_deployment_level,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            drag_coefficient_curve=data.get("drag_coefficient_curve"),
            reference_area=data.get("reference_area"),
            clamp=data.get("clamp"),
            override_rocket_drag=data.get("override_rocket_drag"),
            deployment_level=data.get("deployment_level"),
            name=data.get("name"),
        )
