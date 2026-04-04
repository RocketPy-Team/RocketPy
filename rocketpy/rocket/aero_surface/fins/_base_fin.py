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
        self.geometry = None

        self.reference_area = np.pi * rocket_radius**2

        super().__init__(name, self.reference_area, self.rocket_diameter)

    def _update_reference_quantities(self):
        """Update quantities that depend on rocket radius."""
        self.reference_area = np.pi * self._rocket_radius**2
        self.reference_length = self.rocket_diameter

    def _update_geometry_chain(self):
        """Update geometry-dependent quantities in dependency order."""
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def rocket_radius(self):
        """Rocket radius in meters.

        Returns
        -------
        float
            Rocket radius in meters.
        """
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        """Set rocket radius and update dependent properties.

        Parameters
        ----------
        value : float
            Rocket radius in meters.
        """
        self._rocket_radius = value
        self._update_reference_quantities()
        self._update_geometry_chain()

    @property
    def rocket_diameter(self):
        """Reference rocket diameter in meters."""
        return 2 * self._rocket_radius

    @rocket_diameter.setter
    def rocket_diameter(self, value):
        """Set reference rocket diameter in meters."""
        self.rocket_radius = value / 2

    @property
    def diameter(self):
        """Backward-compatible alias for :attr:`rocket_diameter`."""
        return self.rocket_diameter

    @diameter.setter
    def diameter(self, value):
        """Backward-compatible alias setter for :attr:`rocket_diameter`."""
        self.rocket_diameter = value

    @property
    def d(self):
        """Backward-compatible alias for :attr:`rocket_diameter`."""
        return self.rocket_diameter

    @d.setter
    def d(self, value):
        """Backward-compatible alias setter for :attr:`rocket_diameter`."""
        self.rocket_diameter = value

    @property
    def ref_area(self):
        """Backward-compatible alias for :attr:`reference_area`."""
        return self.reference_area

    @ref_area.setter
    def ref_area(self, value):
        """Backward-compatible alias setter for :attr:`reference_area`."""
        self.reference_area = value

    @property
    def root_chord(self):
        """Root chord length in meters.

        Returns
        -------
        float
            Root chord length in meters.
        """
        return self._root_chord

    @root_chord.setter
    def root_chord(self, value):
        """Set root chord and update dependent properties.

        Parameters
        ----------
        value : float
            Root chord length in meters.
        """
        self._root_chord = value
        self._update_geometry_chain()
        self.evaluate_shape()

    @property
    def span(self):
        """Fin span in meters.

        Returns
        -------
        float
            Fin span in meters.
        """
        return self._span

    @span.setter
    def span(self, value):
        """Set fin span and update dependent properties.

        Parameters
        ----------
        value : float
            Fin span in meters.
        """
        self._span = value
        self._update_geometry_chain()
        self.evaluate_shape()

    @property
    def cant_angle(self):
        """Cant angle in degrees.

        Returns
        -------
        float
            Cant angle in degrees.
        """
        return self._cant_angle

    @cant_angle.setter
    def cant_angle(self, value):
        """Set cant angle and update radian representation.

        Parameters
        ----------
        value : float
            Cant angle in degrees.
        """
        self._cant_angle = value
        self.cant_angle_rad = math.radians(value)

    @property
    def cant_angle_rad(self):
        """Cant angle in radians.

        Returns
        -------
        float
            Cant angle in radians.
        """
        return self._cant_angle_rad

    @cant_angle_rad.setter
    def cant_angle_rad(self, value):
        """Set cant angle in radians and update dependent properties.

        Parameters
        ----------
        value : float
            Cant angle in radians.
        """
        self._cant_angle_rad = value
        self._update_geometry_chain()

    @property
    def airfoil(self):
        """Airfoil data for the fin.

        Returns
        -------
        tuple or None
            Tuple containing airfoil data and unit ('degrees' or 'radians'),
            or None if using planar fin.
        """
        return self._airfoil

    @airfoil.setter
    def airfoil(self, value):
        """Set airfoil data and update dependent properties.

        Parameters
        ----------
        value : tuple or None
            Tuple containing airfoil data and unit ('degrees' or 'radians'),
            or None for planar fin.
        """
        self._airfoil = value
        self._update_geometry_chain()

    def info(self):
        """Print fin geometry and lift information."""
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        """Print all available fin information and show all fin plots."""
        self.prints.all()
        self.plots.all()

    def evaluate_single_fin_lift_coefficient(self):
        """Evaluate the lift coefficient derivative for a single fin.

        Computes the lift coefficient derivative (clalpha) considering the
        fin's geometry, airfoil characteristics (if provided), and Mach number
        effects using Prandtl-Glauert compressibility correction and
        Diederich's planform correlation.

        Sets the `clalpha_single_fin` attribute as a Function of Mach number.
        """
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
            clalpha2D_incompressible = self.airfoil_cl.differentiate(x=1e-3, dx=1e-3)

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
                * (self.Af / self.reference_area)
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
        """Evaluate the lift coefficient for the fin."""

    @abstractmethod
    def evaluate_roll_parameters(self):
        """Evaluate roll-related parameters for the fin."""

    @abstractmethod
    def evaluate_center_of_pressure(self):
        """Evaluate the center of pressure for the fin."""

    def evaluate_geometrical_parameters(self):
        """Evaluate geometric parameters of the fin.

        This method delegates to the configured geometry strategy.
        """
        geometry_data = self.geometry.evaluate_geometrical_parameters()
        for key, value in geometry_data.items():
            setattr(self, key, value)

    def evaluate_shape(self):
        """Evaluate the shape representation of the fin.

        This method delegates to the configured geometry strategy.
        """
        self.shape_vec = self.geometry.evaluate_shape()

    @abstractmethod
    def draw(self):
        """Draw or render the fin."""
