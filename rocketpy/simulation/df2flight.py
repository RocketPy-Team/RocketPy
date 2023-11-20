"""Starts with a pandas DataFrame containing the rocket's collected flight data
and build a rocketpy.Flight object from it.
"""

from dataclasses import dataclass
from functools import cached_property, wraps

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.tools import import_optional_dependency

pandas = import_optional_dependency("pandas")
pd = pandas


def df2property(
    col_name, output_name, interpolation="spline", extrapolation="constant"
):
    """Decorator to create a property from a data frame. The property works as a
    rocketpy.Function object. Check the DataFrameToFlight class for additional
    details

    Parameters
    ----------
    col_name : str
        The name of the column in the data frame. This is case sensitive.
    output_name : str
        The label of the output to be used in the rocketpy.Function object.
    interpolation : str, optional
        The interpolation method to be used in the rocketpy.Function object.
        The default is "spline".
    extrapolation : str, optional
        The extrapolation method to be used in the rocketpy.Function object.
        The default is "constant".

    Returns
    -------
    decorator
        The decorator to be used in the DataFrameToFlight class. This only works
        in the DataFrameToFlight class.
    """

    class Decorator:
        """A decorator class that wraps a function and provides create a
        property from a data frame. This is made via class because it's the only
        way to use the __set_name__ method. This method is used to get the name
        of the property in the class.
        """

        func = None

        def __init__(self, func):
            self.attrname = None
            self.func = func

        def __set_name__(self, owner, name):
            self.attrname = name

        @wraps(func)
        def __get__(self, instance, owner=None):
            """Get method that retrieves the value from the data frame and
            returns a Function object.
            """
            try:
                name = instance.columns_map[col_name]
            except KeyError as e:
                raise KeyError(
                    f"Cannot get the '{col_name}' variable from the data frame."
                    + " Please check your columns_map."
                ) from e
            x_values = instance.df[name].values
            source = np.column_stack((instance.time, x_values))
            return Function(
                source, "Time (s)", output_name, interpolation, extrapolation
            )

    return Decorator


@dataclass
class DataFrameToFlight:
    """A class to create a rocketpy.Flight object from a pandas data frame. Each
    property of a Flight object will be created using values from the data frame

    There are different ways of calculating a property from the data frame.
        1. The property is already in data frame and you want to use it as is
        2. The property is not in the data frame but you can calculate it from
            other properties in the data frame.

    Initially, this class is designed to work with the option '1.'

    Parameters
    ----------
    name : str
        The name of the flight. This is useful to identify the flight in the
        simulation. Example: "Rocket Calisto, Spaceport America, 2019"
    time_col : str
        The name of the column containing the time values. This is case
        sensitive. This column is crucial because it is used to create the
        other rocketpy.Function objects.
    df : pd.DataFrame
        The data frame containing the flight data. The data frame must contain
        the time column.
    columns_map : dict, optional
        A dictionary containing the mapping between the column names in the
        data frame and the names to be used in the Flight object. This is useful
        when the column names in the data frame are not the same as the ones
        used in the Flight object. The default is None.
    interpolation : str, optional
        The interpolation method to be used in the rocketpy.Function objects.
        The default is "linear".
    extrapolation : str, optional
        The extrapolation method to be used in the rocketpy.Function objects.
        The default is "zero".
    """

    name: str
    time_col: str
    df: pd.DataFrame
    columns_map: dict = None
    interpolation: str = "linear"
    extrapolation: str = "zero"

    def __repr__(self):
        return (
            f"'PostFlight object: name = '{self.name}', time_col = '{self.time_col})''"
        )

    @cached_property
    def time(self):
        """An np.ndarray containing the time values of the flight. The time
        values are taken from the data frame so they have the same units as the
        time column in the data frame. It is recommended to use time in seconds.
        """
        return self.df[self.time_col].values

    @df2property("x", "X position (m)", interpolation, extrapolation)
    def x(self):
        pass

    @df2property("y", "Y position (m)", interpolation, extrapolation)
    def y(self):
        pass

    @df2property("z", "Z position (m)", interpolation, extrapolation)
    def z(self):
        pass

    @df2property("vx", "X velocity (m/s)", interpolation, extrapolation)
    def vz(self):
        pass

    @df2property("altitude", "Altitude (m)", interpolation, extrapolation)
    def altitude(self):
        pass

    @df2property("vy", "Y velocity (m/s)", interpolation, extrapolation)
    def vy(self):
        pass

    @df2property("vz", "Z velocity (m/s)", interpolation, extrapolation)
    def vz(self):
        pass

    @df2property("ax", "X acceleration (m/s^2)", interpolation, extrapolation)
    def ax(self):
        pass

    @df2property("ay", "Y acceleration (m/s^2)", interpolation, extrapolation)
    def ay(self):
        pass

    @df2property("az", "Z acceleration (m/s^2)", interpolation, extrapolation)
    def az(self):
        pass

    @df2property("w1", "Angular velocity ω1 (rad/s)", interpolation, extrapolation)
    def w1(self):
        pass

    @df2property("w2", "Angular velocity ω2 (rad/s)", interpolation, extrapolation)
    def w2(self):
        pass

    @df2property("w3", "Angular velocity ω3 (rad/s)", interpolation, extrapolation)
    def w3(self):
        pass

    @df2property("latitude", "Latitude (deg)", interpolation, extrapolation)
    def latitude(self):
        pass

    @df2property("longitude", "Longitude (deg)", interpolation, extrapolation)
    def longitude(self):
        pass

    @property
    def apogee(self):
        """Returns the apogee of the flight. This is calculated using the
        maximum altitude of the flight.

        Returns
        -------
        float
            The apogee of the flight in meters.
        """
        return self.z.max
