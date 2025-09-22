import numpy as np
from scipy.constants import atm, zero_Celsius

from ..mathutils.function import NUMERICAL_TYPES, Function
from ..plots.fluid_plots import _FluidPlots
from ..prints.fluid_prints import _FluidPrints


class Fluid:
    """Class that represents a fluid object and its attributes,
    such as density.

    Attributes
    ----------
    name : str
        Name of the fluid.
    density : int, float, callable, string, array, Function
        Input density of the fluid in kg/m³.
        Used internally to compute the density_function.
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
    def density(self):
        """Density of the fluid from the class parameter input."""
        return self._density

    @density.setter
    def density(self, value):
        """Setter of the density class parameter."""
        if isinstance(value, NUMERICAL_TYPES):
            # Numerical value kept for retro-compatibility
            self._density = value
        else:
            self._density = Function(
                value,
                interpolation="linear",
                extrapolation="natural",
                inputs=["Temperature (K)", "Pressure (Pa)"],
                outputs="Density (kg/m³)",
            )

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
        if isinstance(value, NUMERICAL_TYPES):
            # pylint: disable=unused-argument
            def density_function(temperature, pressure):
                return value

        else:
            density_function = value

        self._density_function = Function(
            density_function,
            interpolation="linear",
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
        is_temperature_callable = callable(temperature.source)
        is_pressure_callable = callable(pressure.source)
        if is_temperature_callable and is_pressure_callable:
            return Function(
                lambda time: self.density_function.get_value(
                    temperature.source(time), pressure.source(time)
                ),
                inputs="Time (s)",
                outputs="Density (kg/m³)",
            )

        if is_temperature_callable or is_pressure_callable:
            time_scale = (
                temperature.x_array if not is_temperature_callable else pressure.x_array
            )
        else:
            time_scale = np.unique(
                np.concatenate((temperature.x_array, pressure.x_array))
            )
        density_time = self.density_function.get_value(
            temperature.get_value(time_scale), pressure.get_value(time_scale)
        )
        return Function(
            np.column_stack((time_scale, density_time)),
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
        discretize = kwargs.get("discretize", False)

        density = self.density
        if discretize and isinstance(density, Function):
            density = density.set_discrete(
                lower=(100, atm * 0.9),
                upper=(zero_Celsius + 30, atm * 50),
                samples=(25, 25),
                mutate_self=False,
            )

        return {"name": self.name, "density": density}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["density"])
