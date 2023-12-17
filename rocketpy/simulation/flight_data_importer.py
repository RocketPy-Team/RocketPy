"""Starts with .csv or .txt file containing the rocket's collected flight data
and build a rocketpy.Flight object from it.
"""
import warnings
from os import listdir
from os.path import isfile, join

import numpy as np

from rocketpy.mathutils import Function
from rocketpy.units import UNITS_CONVERSION_DICT

FLIGHT_LABEL_MAP = {
    # "name of Flight Attribute": "Label to be displayed"
    "acceleration": "Acceleration (m/s^2)",
    "alpha1": "Angular acceleration x (rad/s^2)",
    "alpha2": "Angular acceleration y (rad/s^2)",
    "alpha3": "Angular acceleration z (rad/s^2)",
    "altitude": "Altitude AGL (m)",
    "ax": "Acceleration x (m/s^2)",
    "ay": "Acceleration y (m/s^2)",
    "az": "Acceleration z (m/s^2)",
    "bearing": "Bearing (deg)",
    "density": "Density (kg/m^3)",
    "drift": "Drift (deg)",
    "e0": "Quaternion e0",
    "e1": "Quaternion e1",
    "e2": "Quaternion e2",
    "e3": "Quaternion e3",
    "latitude": "Latitude (deg)",
    "longitude": "Longitude (deg)",
    "mach_number": "Mach number",
    "name": "Time (s)",
    "speed": "Speed (m/s)",
    "stability_margin": "Stability margin",
    "static_margin": "Static margin",
    "vz": "Velocity z (m/s)",
    "vx": "Velocity x (m/s)",
    "vy": "Velocity y (m/s)",
    "w1": "Angular velocity x (rad/s)",
    "w2": "Angular velocity y (rad/s)",
    "w3": "Angular velocity z (rad/s)",
    "x": "Position x (m)",
    "y": "Position y (m)",
    "z": "Position z (m)",
}


class FlightDataImporter:
    """A class to create a rocketpy.Flight object from a .csv file. The data
    frame must contain a time column in seconds.

    There are different ways of calculating a property from the data frame.
        1. The property is already in data frame and you want to use it as is
        2. The property is not in the data frame but you can calculate it from
            other properties in the data frame.

    Initially, this class is designed to work with the option '1.'
    """

    def __init__(
        self,
        name,
        path,
        columns_map=None,
        units=None,
        interpolation="linear",
        extrapolation="zero",
        separator=",",
        encoding="utf-8",
    ):
        """_summary_

        Parameters
        ----------

        Notes
        -----
        Try to avoid using forbidden characters in the column names, such as
        parenthesis, brackets, etc. These characters may cause errors when
        accessing the attributes of the Flight object.
        """
        self.name = name
        self.path = path

        # Initialize debuggers
        self._columns_map = {}
        self._units = {}
        self._columns = {}
        self._time_cols = {}
        self._original_columns = {}
        self._data = {}
        self._separators = {}
        self._encodings = {}
        self._files = None

        # So now we are going to loop through the files and read them
        self.read_data(
            path,
            columns_map,
            units,
            interpolation,
            extrapolation,
            separator,
            encoding,
        )

    def __repr__(self):
        """Representation method for the FlightDataImporter class.

        Returns
        -------
        str
            A string representation of the FlightDataImporter class.
        """
        return f"FlightDataImporter(name='{self.name}', dataset='{self.path}')"

    def __reveal_files(self, path):
        """Get a list of all the .csv or .txt files in the given path, or simply
        return the path of the file if it is a file.

        Parameters
        ----------
        path : str
            The path to the folder or file.

        Returns
        -------
        list
            A list of all the .csv or .txt files in the given path.
        """
        if path.endswith(".csv") or path.endswith(".txt"):
            return [path]

        return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    def __handle_dataset(self, filepath, separator, encoding):
        """Reads the data from the specified file and stores it in the
        appropriate attributes.

        Parameters
        ----------
        filepath : str
            The path to the file we are reading from. This should be either a
            .csv or .txt file.
        separator : str
            The separator used in the data file.
        encoding : str
            The encoding of the data file.
        """
        raw_data = np.genfromtxt(
            filepath,
            delimiter=separator,
            encoding=encoding,
            dtype=np.float64,
            names=True,
            deletechars="",
        )

        # Store the original columns and the data
        self._original_columns[filepath] = raw_data.dtype.names
        self._data[filepath] = raw_data.view(
            (np.float64, len(self._original_columns[filepath]))
        )

        # Create the columns map if necessary
        if self._columns_map[filepath] is None:
            self._columns_map[filepath] = {
                col: col for col in self._original_columns[filepath]
            }

        # Map original columns to their new names (if mapped), otherwise keep as is
        self._columns[filepath] = [
            self._columns_map[filepath].get(col, col)
            for col in self._original_columns[filepath]
        ]

    def __define_time_column(self, filepath):
        """Defines the time column of the data frame. The time column must be in
        seconds and must be named 'time'. Alternatively, you can specify the
        time column using the 'columns_map' argument.

        Parameters
        ----------
        filepath : str
            The path to the file we are reading from.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the time column is not found in the header of the file.
        """
        if self._columns_map[filepath]:
            for col, atr in self._columns_map[filepath].items():
                if atr.lower() == "time":
                    self._time_cols[filepath] = col
                    return None
        if "time" in self._columns[filepath]:
            self._time_cols[filepath] = "time"
            return None
        raise ValueError(
            "Unable to determine the time column, please specify it. The time column "
            + "must be in seconds and must be named 'time'. Alternatively, you "
            + "can specify the time column using the 'columns_map' argument. "
            + "Available columns are: "
            + str(self._columns[filepath])
        )

    def __create_attributes(self, filepath, interpolation, extrapolation):
        """Creates the attributes to emulate a Flight object.

        Parameters
        ----------
        filepath : str
            The path to the file we are reading from.
        interpolation : str
            The interpolation method to use when setting the rocketpy.Function
            objects.
        extrapolation : str
            The extrapolation method to use when setting the rocketpy.Function
            objects.

        Raises
        ------
        ValueError
            If the time column is not found in the header of the file.
        """
        time_col = self._time_cols[filepath]
        units = self._units[filepath]

        # Extract time values
        try:
            times = self._data[filepath][
                :, self._original_columns[filepath].index(time_col)
            ]
        except ValueError as e:
            raise ValueError(
                f"Unable to find column '{time_col}' in the header of the file."
                + "The available columns are:"
                + str(self._original_columns[filepath])
            ) from e

        created = []
        for col, name in self._columns_map[filepath].items():
            if name == "time":  # Handle time separately
                setattr(self, name, times)
                created.append(name)
                continue

            # Find the index of the current column in the filtered data
            try:
                col_idx = self._original_columns[filepath].index(col)
            except ValueError:
                warnings.warn(
                    f"Unable to find column '{col}' in the header of the file. "
                    + f"The attribute '{name}' won't be set. The available "
                    + "columns are:"
                    + str(self._original_columns[filepath]),
                    UserWarning,
                )
                continue

            # Extract values for the current column
            values = self._data[filepath][:, col_idx]

            # Convert units if necessary
            if units and col in units:
                values /= UNITS_CONVERSION_DICT[units[col]]
                print(f"Attribute '{name}' converted from {units[col]} to SI")

            # Create Function object and set as attribute
            setattr(
                self,
                name,
                Function(
                    np.column_stack((times, values)),
                    "Time (s)",
                    FLIGHT_LABEL_MAP.get(name, name),
                    interpolation,
                    extrapolation,
                ),
            )
            created.append(name)

        print(
            "The following attributes were create and are now available to be used: ",
            created,
        )

    def read_data(
        self,
        path,
        columns_map=None,
        units=None,
        interpolation="linear",
        extrapolation="zero",
        separator=",",
        encoding="utf-8",
    ):
        """Reads flight data from the specified path.

        Parameters
        ----------
        path : str
            The path to the flight data file or directory. Only .csv and .txt
            files are supported.
        columns_map : dict, optional
            A dictionary mapping column names to desired column names.
            Defaults to None, which will keep the original column names.
        units : dict, optional
            A dictionary mapping column names to desired units.
            Defaults to None, which will consider that all the data is in SI.
        interpolation : str, optional
            The interpolation method to use for missing data.
            Defaults to "linear", see rocketpy.mathutils.Function for more
            information.
        extrapolation : str, optional
            The extrapolation method to use for data outside the range.
            Defaults to "zero", see rocketpy.mathutils.Function for more
            information.
        separator : str, optional
            The separator used in the data file. Defaults to ",".
        encoding : str, optional
            The encoding of the data file. Defaults to "utf-8".

        Returns
        -------
        None

        Notes
        -----
        This method handles multiple files within the same path and processes
        each of them. It reads the data, applies the specified interpolation and
        extrapolation methods, and sets the appropriate encoding and separator.
        """
        # We need to handle multiple files within the same path
        list_of_files = self.__reveal_files(path)
        if not self._files:
            self._files = list_of_files
        else:
            self._files.extend(list_of_files)

        # Loop through the files and read each of them
        for filepath in self._files:
            self._units[filepath] = units
            self._columns_map[filepath] = columns_map
            self._encodings[filepath] = encoding
            self._separators[filepath] = separator
            self.__handle_dataset(filepath, separator, encoding)
            self.__define_time_column(filepath)
            self.__create_attributes(filepath, interpolation, extrapolation)

    @property
    def flight_attributes(self):
        """A list of flight attributes associated with the class.

        Returns
        -------
        list
            A list of flight attributes excluding private and weakly private
            attributes.
        """
        to_exclude = ["name", "path", "read_data", "flight_attributes"]
        return [
            attr
            for attr in dir(self)
            if not attr.startswith("_") and attr not in to_exclude
        ]
