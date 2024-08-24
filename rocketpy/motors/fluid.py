import numpy as np

from ..mathutils.function import Function
from ..plots.fluid_plots import _FluidPlots
from ..prints.fluid_prints import _FluidPrints


class Fluid:
    """Class that represents a fluid object and its simple quantities,
    such as density.

    Attributes
    ----------
    name : str
        Name of the fluid.
    density : int, float, callable, string, array, Function
        Input density of the fluid in kg/m³.
    density_function : Function
        Density of the fluid as a function of temperature and pressure.
    """

    def __init__(self, name, density):
        """Initializes a Fluid object.

        Parameters
        ----------
        name : str
            Name of the fluid.
        density : int, float, callable, string, array, Function
            Density of the fluid in kg/m³ as a function of temperature
            and pressure. If a int or float is given, it is considered
            a constant function. A callable, csv file or an list of points
            can be given to express the density values.
            The parameter is used as a ``Function`` source.

            See more on :class:`rocketpy.Function` source types.
        """

        self.name = name
        self.density = density
        self.density_function = density

        self.prints = _FluidPrints(self)
        self.plots = _FluidPlots(self)

    @property
    def density_function(self):
        """Density of the fluid as a function of temperature and pressure."""
        return self._density_function

    @density_function.setter
    def density_function(self, value):
        """Setter for the density function of the fluid. Numeric values
        are converted to constant functions.

        Parameters
        ----------
        value : int, float, callable, string, array, Function
            Density of the fluid in kg/m³ as a function of temperature
            and pressure. The value is used as a ``Function`` source.
        """
        if isinstance(value, (int, float, np.number)):
            # pylint: disable=unused-argument
            def density_function(temperature, pressure):
                return value

        else:
            density_function = value

        self._density_function = Function(
            density_function,
            interpolation="shepard",
            extrapolation="natural",
            inputs=["Temperature (K)", "Pressure (Pa)"],
            outputs="Density (kg/m³)",
        )

    def get_time_variable_density(self, temperature, pressure):
        """Get the density of the fluid as a function of time.

        Parameters
        ----------
        temperature : Function
            Temperature of the fluid in Kelvin.
        pressure : Function
            Pressure of the fluid in Pascals.

        Returns
        -------
        Function
            Density of the fluid in kg/m³ as function of time.
        """
        if callable(temperature.source):
            return Function(
                lambda time: self.density_function.get_value(
                    temperature.source(time), pressure.source(time)
                ),
                inputs="Time (s)",
                outputs="Density (kg/m³)",
            )
        else:
            density_time = self.density_function.get_value(
                temperature.y_array, pressure.y_array
            )
            return Function(
                np.column_stack(
                    (
                        temperature.x_array,
                        density_time,
                    )
                ),
                interpolation="linear",
                extrapolation="constant",
                inputs="Time (s)",
                outputs="Density (kg/m³)",
            )

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

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {"name": self.name, "density": self.density}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["density"])
