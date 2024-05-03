from dataclasses import dataclass
from typing import Union

import numpy as np

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
    density : int, float, callable, string, array, Function
        Density of the fluid in kg/m³. If a int or float is given,
        it is considered a constant in time. A callable, csv file
        or an list of points can be given to express the density
        as a function of time. The parameter is used as a ``Function``
        source.

        See more on :class:`rocketpy.Function` source types.
    """

    name: str
    density: Union[int, float, str, list, Function]

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
