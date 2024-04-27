import numpy as np

from dataclasses import dataclass

from ..mathutils.function import Function
from ..plots.fluid_plots import _FluidPlots
from ..prints.fluid_prints import _FluidPrints


@dataclass
class Fluid:
    """Class that represents a fluid.

    Attributes
    ----------
    name : str
        Name of the fluid.
    density : float, Function
        Density of the fluid in kg/m³.
    """

    name: str
    density: float | Function

    def __post_init__(self):
        """Post initialization method.

        Raises
        ------
        ValueError
            If the name is not a string.
        ValueError
            If the density is not a positive number.
        """

        if not isinstance(self.name, str):
            raise ValueError("The name must be a string.")

        if isinstance(self.density, (int, float, np.number)):
            if self.density < 0:
                raise ValueError("The density must be a positive number.")

            self.density = Function(
                self.density,
                interpolation="linear",
                extrapolation="constant",
                inputs="Time (s)",
                outputs="Density (kg/m³)",
            )
        elif isinstance(self.density, Function):
            if (self.density < 0).any():
                raise ValueError("The density must be a positive number.")

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
