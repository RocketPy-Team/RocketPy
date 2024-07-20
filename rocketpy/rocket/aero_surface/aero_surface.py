from abc import ABC, abstractmethod

import numpy as np


class AeroSurface(ABC):
    """Abstract class used to define aerodynamic surfaces."""

    def __init__(self, name, reference_area, reference_length):
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.name = name

        self.cpx = 0
        self.cpy = 0
        self.cpz = 0

    @staticmethod
    def _beta(mach):
        """Defines a parameter that is often used in aerodynamic
        equations. It is commonly used in the Prandtl factor which
        corrects subsonic force coefficients for compressible flow.
        This is applied to the lift coefficient of the nose cone,
        fins and tails/transitions as in [1].

        Parameters
        ----------
        mach : int, float
            Number of mach.

        Returns
        -------
        beta : int, float
            Value that characterizes flow speed based on the mach number.

        References
        ----------
        [1] Barrowman, James S. https://arc.aiaa.org/doi/10.2514/6.1979-504
        """

        if mach < 0.8:
            return np.sqrt(1 - mach**2)
        elif mach < 1.1:
            return np.sqrt(1 - 0.8**2)
        else:
            return np.sqrt(mach**2 - 1)

    @abstractmethod
    def evaluate_center_of_pressure(self):
        """Evaluates the center of pressure of the aerodynamic surface in local
        coordinates.

        Returns
        -------
        None
        """

    @abstractmethod
    def evaluate_lift_coefficient(self):
        """Evaluates the lift coefficient curve of the aerodynamic surface.

        Returns
        -------
        None
        """

    @abstractmethod
    def evaluate_geometrical_parameters(self):
        """Evaluates the geometrical parameters of the aerodynamic surface.

        Returns
        -------
        None
        """

    @abstractmethod
    def info(self):
        """Prints and plots summarized information of the aerodynamic surface.

        Returns
        -------
        None
        """

    @abstractmethod
    def all_info(self):
        """Prints and plots all the available information of the aero surface.

        Returns
        -------
        None
        """
