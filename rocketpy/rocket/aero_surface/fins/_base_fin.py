import math
from abc import abstractmethod

import numpy as np

from rocketpy.mathutils.function import Function

from ..aero_surface import AeroSurface


class _BaseFin(AeroSurface):
    """
    Base class for fins, shared by both Fin and Fins classes.
    Inherits from AeroSurface.

    Handles shared initialization logic and common properties.
    """

    def __init__(
        self, name, rocket_radius, root_chord, span, airfoil=None, cant_angle=0
    ):
        """
        Initialize the base fin class.

        Parameters
        ----------
        name : str
            Name of the fin or fin set.
        rocket_radius : float
            Rocket radius in meters.
        root_chord : float
            Root chord of the fin in meters.
        span : float
            Span of the fin in meters.
        airfoil : tuple, optional
            Tuple containing airfoil data and unit ('degrees' or 'radians').
        cant_angle : float, optional
            Cant angle in degrees.
        """
        self.name = name
        self._rocket_radius = rocket_radius
        self._root_chord = root_chord
        self._span = span
        self._airfoil = airfoil
        self._cant_angle = cant_angle
        self._cant_angle_rad = math.radians(cant_angle)

        self.d = 2 * rocket_radius
        self.ref_area = np.pi * rocket_radius**2

        super().__init__(name, self.ref_area, self.d)

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def root_chord(self):
        return self._root_chord

    @root_chord.setter
    def root_chord(self, value):
        self._root_chord = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, value):
        self._span = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def cant_angle(self):
        return self._cant_angle

    @cant_angle.setter
    def cant_angle(self, value):
        self._cant_angle = value
        self.cant_angle_rad = math.radians(value)

    @property
    def cant_angle_rad(self):
        return self._cant_angle_rad

    @cant_angle_rad.setter
    def cant_angle_rad(self, value):
        self._cant_angle_rad = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def airfoil(self):
        return self._airfoil

    @airfoil.setter
    def airfoil(self, value):
        self._airfoil = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    def evaluate_single_fin_lift_coefficient(self):
        if not self.airfoil:
            # Defines clalpha2D as 2*pi for planar fins
            clalpha2D_incompressible = 2 * np.pi
        else:
            # Defines clalpha2D as the derivative of the lift coefficient curve
            # for the specific airfoil
            self.airfoil_cl = Function(
                self.airfoil[0],
                title="Airfoil lift coefficient",
                interpolation="linear",
            )

            # Differentiating at alpha = 0 to get cl_alpha
            clalpha2D_incompressible = self.airfoil_cl.differentiate_complex_step(
                x=1e-3, dx=1e-3
            )

            # Convert to radians if needed
            if self.airfoil[1] == "degrees":
                clalpha2D_incompressible *= 180 / np.pi

        # Correcting for compressible flow (apply Prandtl-Glauert correction)
        clalpha2D = Function(lambda mach: clalpha2D_incompressible / self._beta(mach))

        # Diederich's Planform Correlation Parameter
        planform_correlation_parameter = (
            2 * np.pi * self.AR / (clalpha2D * np.cos(self.gamma_c))
        )

        # Lift coefficient derivative for a single fin
        def lift_source(mach):
            return (
                clalpha2D(mach)
                * planform_correlation_parameter(mach)
                * (self.Af / self.ref_area)
                * np.cos(self.gamma_c)
            ) / (
                2
                + planform_correlation_parameter(mach)
                * np.sqrt(1 + (2 / planform_correlation_parameter(mach)) ** 2)
            )

        self.clalpha_single_fin = Function(
            lift_source,
            "Mach",
            "Lift coefficient derivative for a single fin",
        )

    @abstractmethod
    def evaluate_lift_coefficient(self):
        pass

    @abstractmethod
    def evaluate_roll_parameters(self):
        pass

    @abstractmethod
    def evaluate_center_of_pressure(self):
        pass

    @abstractmethod
    def evaluate_geometrical_parameters(self):
        pass

    @abstractmethod
    def evaluate_shape(self):
        pass

    @abstractmethod
    def draw(self):
        pass
