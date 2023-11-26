"""Starts with .csv or .txt file containing the rocket's collected flight data
and build a rocketpy.Flight object from it.
"""
import warnings

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.units import UNITS_CONVERSION_DICT

FLIGHT_LABEL_MAP = {
    # "name of Flight Attribute": "Label to be displayed"
    "time": "Time (s)",
    "x": "Position x (m)",
    "y": "Position y (m)",
    "z": "Position z (m)",
    "vx": "Velocity x (m/s)",
    "vy": "Velocity y (m/s)",
    "vz": "Velocity z (m/s)",
    "ax": "Acceleration x (m/s^2)",
    "ay": "Acceleration y (m/s^2)",
    "az": "Acceleration z (m/s^2)",
    "altitude": "Altitude AGL (m)",
    "latitude": "Latitude (deg)",
    "longitude": "Longitude (deg)",
}


class FlightDataImporter:
    """A class to create a rocketpy.Flight object from a .csv file. The data
    frame must contain a time column in seconds.

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
    filepath : str, Path
        The path
    columns_map : dict, optional
        A dictionary containing the mapping between the column names in the
        data frame and the names to be used in the Flight object. This is useful
        when the column names in the data frame are not the same as the ones
        used in the Flight object. The default is None.
    units : dict, optional
        A dictionary containing the units of the columns present in the file.
        This is important when the units in the file are not SI. The default is
        None, meaning that no unit conversion will be performed. If a dictionary
        is passed, the columns will be converted to SI units if follows the
        format: {"column_name1": "unit1", "column_name2": "unit2", ...}.
    interpolation : str, optional
        The interpolation method to be used in the rocketpy.Function objects.
        The default is "linear".
    extrapolation : str, optional
        The extrapolation method to be used in the rocketpy.Function objects.
        The default is "zero".
    separator : str, optional
        The separator used in the data frame. The default is ",".
    encoding : str, optional
        The file's encoding, used to avoid errors when reading the file. The
        default is "utf-8".

    Notes
    -----
    Try to avoid using forbidden characters in the column names, such as
    parenthesis, brackets, etc. These characters may cause errors when
    accessing the attributes of the Flight object.
    """

    def __init__(
        self,
        name,
        filepath,
        columns_map=None,
        units=None,
        interpolation="linear",
        extrapolation="zero",
        separator=",",
        encoding="utf-8",
    ):
        self.name = name
        self.filepath = filepath
        self.columns_map = columns_map
        self.units = units
        self.interpolation = interpolation
        self.extrapolation = extrapolation
        self.separator = separator
        self.encoding = encoding

        self.columns = None
        self.data = None
        self.time_col = None

        self.__handle_data()
        self.__determine_time_column()
        self.__create_attributes()

    def __repr__(self):
        return f"FlightDataImporter object: '{self.name}'"

    def __handle_data(self):
        raw_data = np.genfromtxt(
            self.filepath,
            delimiter=self.separator,
            encoding=self.encoding,
            dtype=np.float64,
            names=True,
            deletechars="",
        )

        # Store the column names
        self.original_columns = raw_data.dtype.names
        self.data = raw_data.view((np.float64, len(self.original_columns)))

        # Create the columns map if necessary
        if not self.columns_map:
            self.columns_map = {col: col for col in self.original_columns}

        # Map original columns to their new names (if mapped), otherwise keep as is
        self.columns = [self.columns_map.get(col, col) for col in self.original_columns]

    def __determine_time_column(self):
        if self.columns_map:
            for col, atr in self.columns_map.items():
                if atr == "time":
                    self.time_col = col
                    return None
        if "time" in self.original_columns:
            self.time_col = "time"
            return None
        raise ValueError("Unable to determine the time column...")

    def __create_attributes(self):
        # Extract time values
        try:
            times = self.data[:, self.original_columns.index(self.time_col)]
        except ValueError as e:
            raise ValueError(
                f"Unable to find column '{self.time_col}' in the header of the file."
                + "The available columns are:"
                + str(self.original_columns)
            ) from e

        created = []
        for col, name in self.columns_map.items():
            if name == "time":  # Handle time separately
                setattr(self, name, times)
                created.append(name)
                continue

            # Find the index of the current column in the filtered data
            try:
                col_idx = self.original_columns.index(col)
            except ValueError:
                warnings.warn(
                    f"Unable to find column '{col}' in the header of the file. "
                    + f"The attribute '{name}' won't be set. The available "
                    + "columns are:"
                    + str(self.original_columns),
                    UserWarning,
                )
                continue

            # Extract values for the current column
            values = self.data[:, col_idx]

            # Convert units if necessary
            if self.units and col in self.units:
                values /= UNITS_CONVERSION_DICT[self.units[col]]
                print(f"Attribute '{name}' converted from {self.units[col]} to SI")

            # Create Function object and set as attribute
            setattr(
                self,
                name,
                Function(
                    np.column_stack((times, values)),
                    "Time (s)",
                    FLIGHT_LABEL_MAP.get(name, name),
                    self.interpolation,
                    self.extrapolation,
                ),
            )
            created.append(name)

        print(
            "The following attributes were create and are now available to be used: ",
            created,
        )
