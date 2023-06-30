# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Mateus Stano and Pedro Henrique Marinho Bressan"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from dataclasses import dataclass

from rocketpy.plots.fluid_plots import _FluidPlots
from rocketpy.prints.fluid_prints import _FluidPrints


@dataclass
class Fluid:
    """Class that represents a fluid.

    Attributes
    ----------
    name : str
        Name of the fluid.
    density : float
        Density of the fluid in kg/m³.
    quality : float
        Quality of the fluid, between 0 and 1.
    """

    name: str
    density: float
    quality: float

    def __post_init__(self):
        """Post initialization method.

        Raises
        ------
        ValueError
            If the name is not a string.
        ValueError
            If the density is not a positive number.
        ValueError
            If the quality is not a number between 0 and 1.
        """

        if not isinstance(self.name, str):
            raise ValueError("The name must be a string.")
        if self.density < 0:
            raise ValueError("The density must be a positive number.")
        if self.quality < 0 or self.quality > 1:
            raise ValueError("The quality must be a number between 0 and 1.")

        # Initialize plots and prints object
        self.prints = _FluidPrints(self)
        self.plots = _FluidPlots(self)
        return None

    def __repr__(self):
        """Representation method.

        Returns
        -------
        str
            String representation of the class.
        """

        return f"Fluid(name={self.name}, density={self.density})"

    def __str__(self):
        """String method.

        Returns
        -------
        str
            String representation of the class.
        """

        return f"Fluid: {self.name}"
