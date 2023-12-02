import bisect
import copy
import datetime
import json
from collections import defaultdict

import netCDF4
import numpy as np
import pytz

from ..mathutils.function import Function
from ..plots.environment_analysis_plots import _EnvironmentAnalysisPlots
from ..prints.environment_analysis_prints import _EnvironmentAnalysisPrints
from ..tools import (
    bilinear_interpolation,
    check_requirement_version,
    geopotential_to_height_agl,
    geopotential_to_height_asl,
    import_optional_dependency,
    time_num_to_date_string,
)
from ..units import convert_units
from .environment import Environment

try:
    from functools import cached_property
except ImportError:
    from ..tools import cached_property

# TODO: the average_wind_speed_profile_by_hour and similar methods could be more abstract than currently are


class EnvironmentAnalysis:
    """Class for analyzing the environment.

    List of properties currently implemented:
        - average max/min temperature at surface level
        - record max/min temperature at surface level
        - temperature progression throughout the day
        - temperature profile over an average day
        - average max wind gust at surface level
        - record max wind gust at surface level
        - average, 1, 2, 3 sigma wind profile
        - average day wind rose
        - animation of how average wind rose evolves throughout an average day
        - animation of how wind profile evolves throughout an average day
        - pressure profile over an average day
        - wind velocity x profile over average day
        - wind velocity y profile over average day
        - wind speed profile over an average day
        - average max surface 100m wind speed
        - average max surface 10m wind speed
        - average min surface 100m wind speed
        - average min surface 10m wind speed
        - average sustained surface100m wind along day
        - average sustained surface10m wind along day
        - maximum surface 10m wind speed
        - average cloud base height
        - percentage of days with no cloud coverage
        - percentage of days with precipitation

    You can also visualize all those attributes by exploring the methods:
        - plot of wind gust distribution (should be Weibull)
        - plot wind profile over average day
        - plot sustained surface wind speed distribution over average day
        - plot wind gust distribution over average day
        - plot average day wind rose all hours
        - plot average day wind rose specific hour
        - plot average pressure profile
        - plot average surface10m wind speed along day
        - plot average sustained surface100m wind speed along day
        - plot average temperature along day
        - plot average wind speed profile
        - plot surface10m wind speed distribution
        - animate wind profile over average day
        - animate sustained surface wind speed distribution over average day
        - animate wind gust distribution over average day
        - animate average wind rose
        - animation of how the wind gust distribution evolves over average day
        - all_info

    All items listed are relevant to either
        1. participant safety
        2. launch operations (range closure decision)
        3. rocket performance

    How does this class work?
        - The class is initialized with a start_date, end_date, start_hour and
          end_hour.
        - The class then parses the weather data from the start date to the end
          date.
          Always parsing the data from start_hour to end_hour.
        - The class then calculates the average max/min temperature, average max
          wind gust, and average day wind rose.
        - The class then allows for plotting the average max/min temperature,
          average max wind gust, and average day wind rose.
    """

    def __init__(
        self,
        start_date,
        end_date,
        latitude,
        longitude,
        start_hour=0,
        end_hour=24,
        surface_data_file=None,
        pressure_level_data_file=None,
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        max_expected_altitude=None,
    ):
        """Constructor for the EnvironmentAnalysis class.

        Parameters
        ----------
        start_date : datetime.datetime
            Start date and time of the analysis. When parsing the weather data
            from the source file, only data after this date will be parsed.
        end_date : datetime.datetime
            End date and time of the analysis. When parsing the weather data
            from the source file, only data before this date will be parsed.
        latitude : float
            Latitude coordinate of the location where the analysis will be
            carried out.
        longitude : float
            Longitude coordinate of the location where the analysis will be
            carried out.
        start_hour : int, optional
            Starting hour of the analysis. When parsing the weather data
            from the source file, only data after this hour will be parsed.
        end_hour : int, optional
            End hour of the analysis. When parsing the weather data
            from the source file, only data before this hour will be parsed.
        surface_data_file : str, optional
            Path to the netCDF file containing the surface data.
        pressure_level_data_file : str, optional
            Path to the netCDF file containing the pressure level data.
        timezone : str, optional
            Name of the timezone to be used when displaying results. To see all
            available time zones, import pytz and run print(pytz.all_timezones).
            Default time zone is the local time zone at the latitude and
            longitude specified.
        unit_system : str, optional
            Unit system to be used when displaying results.
            Options are: SI, metric, imperial. Default is metric.
        forecast_date : datetime.date, optional
            Date for the forecast models. It will be requested the environment
            forecast for multiple hours within that specified date.
        forecast_args : dictionary, optional
            Arguments for setting the forecast on the Environment class. With this argument
            it is possible to change the forecast model being used.
        max_expected_altitude : float, optional
            Maximum expected altitude for your analysis. This is used to calculate
            plot limits from pressure level data profiles. If None is set, the
            maximum altitude will be calculated from the pressure level data.
            Default is None.

        Returns
        -------
        None
        """

        # Save inputs
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.latitude = latitude
        self.longitude = longitude
        self.surface_data_file = surface_data_file
        self.pressure_level_data_file = pressure_level_data_file
        self.preferred_timezone = timezone
        self.unit_system = unit_system
        self.max_expected_altitude = max_expected_altitude

        # Check if extra requirements are installed
        self.__check_requirements()

        # Manage units and timezones
        self.__init_data_parsing_units()
        self.__find_preferred_timezone()
        self.__localize_input_dates()

        # Convert units
        self.__set_unit_system(unit_system)

        # Initialize plots and prints object
        self.plots = _EnvironmentAnalysisPlots(self)
        self.prints = _EnvironmentAnalysisPrints(self)

        # Processing forecast
        self.forecast = None
        if forecast_date:
            self.forecast = {}
            hours = list(self.original_pressure_level_data.values())[0].keys()
            for hour in hours:
                hour_date_time = datetime.datetime(
                    year=forecast_date.year,
                    month=forecast_date.month,
                    day=forecast_date.day,
                    hour=int(hour),
                )

                env = Environment(
                    date=hour_date_time,
                    latitude=self.latitude,
                    longitude=self.longitude,
                    elevation=self.converted_elevation,
                )
                forecast_args = forecast_args or {"type": "Forecast", "file": "GFS"}
                env.set_atmospheric_model(**forecast_args)
                self.forecast[hour] = env
        return None

    # Private, auxiliary methods

    def __check_requirements(self):
        """Check if extra requirements are installed. If not, print a message
        informing the user that some methods may not work and how to install
        the extra requirements for environment analysis.

        Returns
        -------
        None
        """
        env_analysis_require = {  # The same as in the setup.py file
            "timezonefinder": "",
            "windrose": ">=1.6.8",
            "IPython": "",
            "ipywidgets": ">=7.6.3",
            "jsonpickle": "",
        }
        has_error = False
        for module_name, version in env_analysis_require.items():
            version = ">=0" if not version else version
            try:
                check_requirement_version(module_name, version)
            except (ValueError, ImportError) as e:
                has_error = True
                print(
                    f"The following error occurred while importing {module_name}: {e}"
                )
        if has_error:
            print(
                "Given the above errors, some methods may not work. Please run "
                + "'pip install rocketpy[env_analysis]' to install extra requirements."
            )
        return None

    def __init_surface_dictionary(self):
        # Create dictionary of file variable names to process surface data
        return {
            "surface100m_wind_velocity_x": "u100",
            "surface100m_wind_velocity_y": "v100",
            "surface10m_wind_velocity_x": "u10",
            "surface10m_wind_velocity_y": "v10",
            "surface_temperature": "t2m",
            "cloud_base_height": "cbh",
            "surface_wind_gust": "i10fg",
            "surface_pressure": "sp",
            "total_precipitation": "tp",
        }

    def __init_pressure_level_dictionary(self):
        # Create dictionary of file variable names to process pressure level data
        return {
            "geopotential": "z",
            "wind_velocity_x": "u",
            "wind_velocity_y": "v",
            "temperature": "t",
        }

    def __get_nearest_index(self, array, value):
        """Find nearest index of the given value in the array.
        Made for latitudes and longitudes, supporting arrays that range from
        -180 to 180 or from 0 to 360.

        Parameters
        ----------
        array : array
            Array of values.
        value : float
            Value to be found in the array.

        Returns
        -------
        index : int
            Index of the nearest value in the array.
        """
        # Create value convention
        if np.min(array) < 0:
            # File uses range from -180 to 180, make sure value follows convention
            value = value if value < 180 else value % 180 - 180  # Example: 190 => -170
        else:
            # File probably uses range from 0 to 360, make sure value follows convention
            value = value % 360  # Example: -10 becomes 350

        # Find index
        if array[0] < array[-1]:
            # Array is sorted correctly, find index
            # Deal with sorted array
            index = bisect.bisect(array, value)
        else:
            # Array is reversed, no big deal, just bisect reversed one and subtract length
            index = len(array) - bisect.bisect_left(array[::-1], value)

        # Apply fix
        if index == len(array) and array[index - 1] == value:
            # If value equal the last array entry, fix to avoid being considered out of grid
            index = index - 1

        return index

    def __extract_surface_data_value(
        self, surface_data, variable, indices, lon_array, lat_array
    ):
        """Extract value from surface data netCDF4 file. Performs bilinear
        interpolation along longitude and latitude.

        Parameters
        ----------
        surface_data : netCDF4.Dataset
            Surface data netCDF4 file.
        variable : str
            Variable to be extracted from the file. Must be an existing variable
            in the surface_data.
        indices : tuple
            Indices of the variable in the file. Must be given as a tuple
            (time_index, lon_index, lat_index).
        lon_array : array
            Array of longitudes.
        lat_array : array
            Array of latitudes.

        Returns
        -------
        value : float
            Value of the variable at the given indices.
        """

        time_index, lon_index, lat_index = indices
        variable_data = surface_data[variable]

        # Get values for variable on the four nearest points
        z11 = variable_data[time_index, lon_index - 1, lat_index - 1]
        z12 = variable_data[time_index, lon_index - 1, lat_index]
        z21 = variable_data[time_index, lon_index, lat_index - 1]
        z22 = variable_data[time_index, lon_index, lat_index]

        # Compute interpolated value on desired lat lon pair
        value = bilinear_interpolation(
            x=self.longitude,
            y=self.latitude,
            x1=lon_array[lon_index - 1],
            x2=lon_array[lon_index],
            y1=lat_array[lat_index - 1],
            y2=lat_array[lat_index],
            z11=z11,
            z12=z12,
            z21=z21,
            z22=z22,
        )

        return value

    def __extract_pressure_level_data_value(
        self, pressure_level_data, variable, indices, lon_array, lat_array
    ):
        """Extract value from surface data netCDF4 file. Performs bilinear
        interpolation along longitude and latitude.

        Parameters
        ----------
        pressure_level_data : netCDF4.Dataset
            Pressure level data netCDF4 file.
        variable : str
            Variable to be extracted from the file. Must be an existing variable
            in the pressure_level_data.
        indices : tuple
            Indices of the variable in the file. Must be given as a tuple
            (time_index, lon_index, lat_index).
        lon_array : array
            Array of longitudes.
        lat_array : array
            Array of latitudes.

        Returns
        -------
        value : float
            Value of the variable at the given indices.
        """

        time_index, lon_index, lat_index = indices
        variable_data = pressure_level_data[variable]

        # Get values for variable on the four nearest points
        z11 = variable_data[time_index, :, lon_index - 1, lat_index - 1]
        z12 = variable_data[time_index, :, lon_index - 1, lat_index]
        z21 = variable_data[time_index, :, lon_index, lat_index - 1]
        z22 = variable_data[time_index, :, lon_index, lat_index]

        # Compute interpolated value on desired lat lon pair
        value_list_as_a_function_of_pressure_level = bilinear_interpolation(
            x=self.longitude,
            y=self.latitude,
            x1=lon_array[lon_index - 1],
            x2=lon_array[lon_index],
            y1=lat_array[lat_index - 1],
            y2=lat_array[lat_index],
            z11=z11,
            z12=z12,
            z21=z21,
            z22=z22,
        )

        return value_list_as_a_function_of_pressure_level

    def __check_coordinates_inside_grid(
        self, lon_index, lat_index, lon_array, lat_array
    ):
        if (
            lon_index == 0
            or lon_index > len(lon_array) - 1
            or lat_index == 0
            or lat_index > len(lat_array) - 1
        ):
            raise ValueError(
                f"Latitude and longitude pair {(self.latitude, self.longitude)} is outside the grid available in the given file, which is defined by {(lat_array[0], lon_array[0])} and {(lat_array[-1], lon_array[-1])}."
            )
        else:
            return None

    def __localize_input_dates(self):
        if self.start_date.tzinfo is None:
            self.start_date = self.preferred_timezone.localize(self.start_date)
        if self.end_date.tzinfo is None:
            self.end_date = self.preferred_timezone.localize(self.end_date)

    def __find_preferred_timezone(self):
        if self.preferred_timezone is None:
            # Use local timezone based on lat lon pair
            try:
                timezonefinder = import_optional_dependency("timezonefinder")
                tf = timezonefinder.TimezoneFinder()
                self.preferred_timezone = pytz.timezone(
                    tf.timezone_at(lng=self.longitude, lat=self.latitude)
                )
            except ImportError:
                print(
                    "'timezonefinder' not installed, defaulting to UTC."
                    + " Install timezonefinder to get local time zone."
                    + " To do so, run 'pip install timezonefinder'"
                )
                self.preferred_timezone = pytz.timezone("UTC")
        elif isinstance(self.preferred_timezone, str):
            self.preferred_timezone = pytz.timezone(self.preferred_timezone)

    def __init_data_parsing_units(self):
        """Define units for pressure level and surface data parsing"""
        self.current_units = {
            "height_ASL": "m",
            "pressure": "hPa",
            "temperature": "K",
            "wind_direction": "deg",
            "wind_heading": "deg",
            "wind_speed": "m/s",
            "wind_velocity_x": "m/s",
            "wind_velocity_y": "m/s",
            "surface100m_wind_velocity_x": "m/s",
            "surface100m_wind_velocity_y": "m/s",
            "surface10m_wind_velocity_x": "m/s",
            "surface10m_wind_velocity_y": "m/s",
            "surface_temperature": "K",
            "cloud_base_height": "m",
            "surface_wind_gust": "m/s",
            "surface_pressure": "Pa",
            "total_precipitation": "m",
        }
        # Create a variable to store updated units when units are being updated
        self.updated_units = self.current_units.copy()

        return None

    def __init_unit_system(self):
        """Initialize preferred units for output (SI, metric or imperial)."""
        if self.unit_system_string == "metric":
            self.unit_system = {
                "length": "m",
                "velocity": "m/s",
                "acceleration": "g",
                "mass": "kg",
                "time": "s",
                "pressure": "hPa",
                "temperature": "degC",
                "angle": "deg",
                "precipitation": "mm",
                "wind_speed": "m/s",
            }
        elif self.unit_system_string == "imperial":
            self.unit_system = {
                "length": "ft",
                "velocity": "mph",
                "acceleration": "ft/s^2",
                "mass": "lb",
                "time": "s",
                "pressure": "inHg",
                "temperature": "degF",
                "angle": "deg",
                "precipitation": "in",
                "wind_speed": "mph",
            }
        else:
            # Default to SI
            print(
                f"Defaulting to SI unit system, the {self.unit_system_string} was not found."
            )
            self.unit_system = {
                "length": "m",
                "velocity": "m/s",
                "acceleration": "m/s^2",
                "mass": "kg",
                "time": "s",
                "pressure": "Pa",
                "temperature": "K",
                "angle": "rad",
                "precipitation": "m",
                "wind_speed": "m/s",
            }

    def __set_unit_system(self, unit_system="metric"):
        """Set preferred unit system for output (SI, metric or imperial). The
        data with new values will be stored in ``converted_pressure_level_data``
        and ``converted_surface_data`` dictionaries, while the original parsed
        data will be kept in ``original_pressure_level_data`` and
        ``original_surface_data``. The performance of this method is not optimal
        since it will loop through all the data (dates, hours and variables) and
        convert the units of each variable, one by one. However, this method is
        only called once.

        Parameters
        ----------
        unit_system : str, optional
            The unit system to be used, by default "metric".
            The options are "metric", "imperial" or "SI".

        Returns
        -------
        None
        """
        # Check if unit system is valid and define units mapping
        self.unit_system_string = unit_system
        self.__init_unit_system()
        # Update current units
        self.current_units = self.updated_units.copy()

        return None

    # General properties

    @cached_property
    def __parse_pressure_level_data(self):
        """
        Parse pressure level data from a weather file.

        Sources of information:
        - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

        Must get the following variables from a ERA5 file:
        - Geopotential
        - U-component of wind
        - V-component of wind
        - Temperature

        Must compute the following for each date and hour available in the dataset:
        - pressure = Function(..., inputs="Height Above Ground Level (m)", outputs="Pressure (Pa)")
        - temperature = Function(..., inputs="Height Above Ground Level (m)", outputs="Temperature (K)")
        - wind_direction = Function(..., inputs="Height Above Ground Level (m)", outputs="Wind Direction (Deg True)")
        - wind_heading = Function(..., inputs="Height Above Ground Level (m)", outputs="Wind Heading (Deg True)")
        - wind_speed = Function(..., inputs="Height Above Ground Level (m)", outputs="Wind Speed (m/s)")
        - wind_velocity_x = Function(..., inputs="Height Above Ground Level (m)", outputs="Wind Velocity X (m/s)")
        - wind_velocity_y = Function(..., inputs="Height Above Ground Level (m)", outputs="Wind Velocity Y (m/s)")

        Return a dictionary with all the computed data with the following structure:

        .. code-block:: python

            pressure_level_data_dict = {
                "date" : {
                    "hour": {
                        "data": ...,
                        "data": ...
                    },
                    "hour": {
                        "data": ...,
                        "data": ...
                    }
                },
                "date" : {
                    "hour": {
                        "data": ...,
                        "data": ...
                    },
                    "hour": {
                        "data": ...,
                        "data": ...
                    }
                }
            }

        The results will be cached, so that the parsing is only done once.
        """
        dictionary = {}
        # Setup dictionary used to read weather file
        pressure_level_file_dict = self.__init_pressure_level_dictionary()
        # Read weather file
        pressure_level_data = netCDF4.Dataset(self.pressure_level_data_file)

        # Get time, pressure levels, latitude and longitude data from file
        time_num_array = pressure_level_data.variables["time"]
        pressure_level_array = pressure_level_data.variables["level"]
        lon_array = pressure_level_data.variables["longitude"]
        lat_array = pressure_level_data.variables["latitude"]
        # Determine latitude and longitude range for pressure level file
        lat0 = lat_array[0]
        lat1 = lat_array[-1]
        lon0 = lon_array[0]
        lon1 = lon_array[-1]

        # Find index needed for latitude and longitude for specified location
        lon_index = self.__get_nearest_index(lon_array, self.longitude)
        lat_index = self.__get_nearest_index(lat_array, self.latitude)

        # Can't handle lat and lon out of grid
        self.__check_coordinates_inside_grid(lon_index, lat_index, lon_array, lat_array)

        # Loop through time and save all values
        for time_index, time_num in enumerate(time_num_array):
            date_string, hour_string, date_time = time_num_to_date_string(
                time_num,
                time_num_array.units,
                self.preferred_timezone,
                calendar="gregorian",
            )

            # Check if date is within analysis range
            if not (self.start_date <= date_time < self.end_date):
                continue
            if not (self.start_hour <= date_time.hour < self.end_hour):
                continue
            # Make sure keys exist
            if date_string not in dictionary:
                dictionary[date_string] = {}
            if hour_string not in dictionary[date_string]:
                dictionary[date_string][hour_string] = {}

            # Extract data from weather file
            indices = (time_index, lon_index, lat_index)

            # Retrieve geopotential first and compute altitudes
            geopotential_array = self.__extract_pressure_level_data_value(
                pressure_level_data,
                pressure_level_file_dict["geopotential"],
                indices,
                lon_array,
                lat_array,
            )
            height_above_ground_level_array = geopotential_to_height_agl(
                geopotential_array, self.original_elevation
            )

            # Loop through wind components and temperature, get value and convert to Function
            for key, value in pressure_level_file_dict.items():
                value_array = self.__extract_pressure_level_data_value(
                    pressure_level_data, value, indices, lon_array, lat_array
                )
                variable_points_array = np.array(
                    [height_above_ground_level_array, value_array]
                ).T
                variable_function = Function(
                    variable_points_array,
                    inputs="Height Above Ground Level (m)",
                    outputs=key,
                    extrapolation="constant",
                )
                dictionary[date_string][hour_string][key] = variable_function

            # Create function for pressure levels
            pressure_points_array = np.array(
                [height_above_ground_level_array, pressure_level_array]
            ).T
            pressure_function = Function(
                pressure_points_array,
                inputs="Height Above Ground Level (m)",
                outputs="Pressure (Pa)",
                extrapolation="constant",
            )
            dictionary[date_string][hour_string]["pressure"] = pressure_function

            # Create function for wind speed levels
            wind_velocity_x_array = self.__extract_pressure_level_data_value(
                pressure_level_data,
                pressure_level_file_dict["wind_velocity_x"],
                indices,
                lon_array,
                lat_array,
            )
            wind_velocity_y_array = self.__extract_pressure_level_data_value(
                pressure_level_data,
                pressure_level_file_dict["wind_velocity_y"],
                indices,
                lon_array,
                lat_array,
            )
            wind_speed_array = np.sqrt(
                np.square(wind_velocity_x_array) + np.square(wind_velocity_y_array)
            )

            wind_speed_points_array = np.array(
                [height_above_ground_level_array, wind_speed_array]
            ).T
            wind_speed_function = Function(
                wind_speed_points_array,
                inputs="Height Above Ground Level (m)",
                outputs="Wind Speed (m/s)",
                extrapolation="constant",
            )
            dictionary[date_string][hour_string]["wind_speed"] = wind_speed_function

            # Create function for wind heading levels
            wind_heading_array = (
                np.arctan2(wind_velocity_x_array, wind_velocity_y_array)
                * (180 / np.pi)
                % 360
            )

            wind_heading_points_array = np.array(
                [height_above_ground_level_array, wind_heading_array]
            ).T
            wind_heading_function = Function(
                wind_heading_points_array,
                inputs="Height Above Ground Level (m)",
                outputs="Wind Heading (Deg True)",
                extrapolation="constant",
            )
            dictionary[date_string][hour_string]["wind_heading"] = wind_heading_function

            # Create function for wind direction levels
            wind_direction_array = (wind_heading_array - 180) % 360
            wind_direction_points_array = np.array(
                [height_above_ground_level_array, wind_direction_array]
            ).T
            wind_direction_function = Function(
                wind_direction_points_array,
                inputs="Height Above Ground Level (m)",
                outputs="Wind Direction (Deg True)",
                extrapolation="constant",
            )
            dictionary[date_string][hour_string][
                "wind_direction"
            ] = wind_direction_function

        return (dictionary, lat0, lat1, lon0, lon1)

    @property
    def original_pressure_level_data(self):
        """Return the original pressure level data dictionary. Units are
        defined by the units in the file.

        Returns
        -------
        dictionary
            Dictionary with the original pressure level data. This dictionary
            has the following structure:

            .. code-block:: python

                original_pressure_level_data = {
                    "date" : {
                        "hour": {
                            "data": ...,
                            "data": ...
                        },
                        "hour": {
                            "data": ...,
                            "data": ...
                        }
                    },
                    "date" : {
                        "hour": {
                        ...
                        }
                    }
                }
        """
        return self.__parse_pressure_level_data[0]

    @property
    def pressure_level_lat0(self):
        """Return the initial latitude of the pressure level data."""
        return self.__parse_pressure_level_data[1]

    @property
    def pressure_level_lat1(self):
        """Return the final latitude of the pressure level data."""
        return self.__parse_pressure_level_data[2]

    @property
    def pressure_level_lon0(self):
        """Return the initial longitude of the pressure level data."""
        return self.__parse_pressure_level_data[3]

    @property
    def pressure_level_lon1(self):
        """Return the final longitude of the pressure level data."""
        return self.__parse_pressure_level_data[4]

    @cached_property
    def __parse_surface_data(self):
        """
        Parse surface data from a weather file.
        Currently only supports files from ECMWF.
        You can download a file from the following website:
        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

        Must get the following variables:
        - surface elevation: float  # Select 'Geopotential'
        - 2m temperature: float
        - Surface pressure: float
        - 10m u-component of wind: float
        - 10m v-component of wind: float
        - 100m u-component of wind: float
        - 100m V-component of wind: float
        - Instantaneous 10m wind gust: float
        - Total precipitation: float
        - Cloud base height: float

        Return a dictionary with all the computed data with the following
        structure:

        .. code-block:: python

            surface_data_dict = {
                "date" : {
                    "hour": {
                        "data": ...,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        # Setup dictionary used to read weather file
        dictionary = {}
        surface_file_dict = self.__init_surface_dictionary()

        # Read weather file
        surface_data = netCDF4.Dataset(self.surface_data_file)

        # Get time, latitude and longitude data from file
        time_num_array = surface_data.variables["time"]
        lon_array = surface_data.variables["longitude"]
        lat_array = surface_data.variables["latitude"]
        # Determine latitude and longitude range for surface level file
        lat0 = lat_array[0]
        lat1 = lat_array[-1]
        lon0 = lon_array[0]
        lon1 = lon_array[-1]

        # Find index needed for latitude and longitude for specified location
        lon_index = self.__get_nearest_index(lon_array, self.longitude)
        lat_index = self.__get_nearest_index(lat_array, self.latitude)

        # Can't handle lat and lon out of grid
        self.__check_coordinates_inside_grid(lon_index, lat_index, lon_array, lat_array)

        # Loop through time and save all values
        for time_index, time_num in enumerate(time_num_array):
            date_string, hour_string, date_time = time_num_to_date_string(
                time_num,
                time_num_array.units,
                self.preferred_timezone,
                calendar="gregorian",
            )

            # Check if date is within analysis range
            if not (self.start_date <= date_time < self.end_date):
                continue
            if not (self.start_hour <= date_time.hour < self.end_hour):
                continue

            # Make sure keys exist
            if date_string not in dictionary:
                dictionary[date_string] = {}
            if hour_string not in dictionary[date_string]:
                dictionary[date_string][hour_string] = {}

            # Extract data from weather file
            indices = (time_index, lon_index, lat_index)
            for key, value in surface_file_dict.items():
                dictionary[date_string][hour_string][
                    key
                ] = self.__extract_surface_data_value(
                    surface_data, value, indices, lon_array, lat_array
                )

        # Get elevation, time index does not matter, use last one
        surface_geopotential = self.__extract_surface_data_value(
            surface_data, "z", indices, lon_array, lat_array
        )
        elevation = geopotential_to_height_asl(surface_geopotential)

        return (dictionary, lat0, lat1, lon0, lon1, elevation)

    @property
    def original_surface_data(self):
        """Returns the surface data dictionary. Units are defined by the units
        in the file.

        Returns
        -------
        dictionary:
            Dictionary with the original surface data. This dictionary has the
            following structure:

            .. code-block:: python

                original_surface_data: {
                    "date" : {
                        "hour": {
                            "data": ...,
                            "data": ...
                        },
                        "hour": {
                            "data": ...,
                            "data": ...
                        }
                    },
                    "date" : {
                        "hour": {
                        ...
                        }
                    }
                }
        """
        return self.__parse_surface_data[0]

    @property
    def original_elevation(self):
        """Return the elevation of the surface data."""
        return self.__parse_surface_data[5]

    @property
    def single_level_lat0(self):
        """Return the initial latitude of the surface data."""
        return self.__parse_surface_data[1]

    @property
    def single_level_lat1(self):
        """Return the final latitude of the surface data."""
        return self.__parse_surface_data[2]

    @property
    def single_level_lon0(self):
        """Return the initial longitude of the surface data."""
        return self.__parse_surface_data[3]

    @property
    def single_level_lon1(self):
        """Return the final longitude of the surface data."""
        return self.__parse_surface_data[4]

    @cached_property
    def converted_pressure_level_data(self):
        """Convert pressure level data to desired unit system. This method will
        loop through all the data (dates, hours and variables) and convert
        the units of each variable. therefore, the performance of this method is
        not optimal. However, this method is only called once and the results
        are cached, so that the conversion is only done once.

        Returns
        -------
        dictionary
            Dictionary with the converted pressure level data. This dictionary
            has the same structure as the ``original_pressure_level_data``
            dictionary.
        """
        # Create conversion dict (key: to_unit)
        conversion_dict = {
            "pressure": self.unit_system["pressure"],
            "temperature": self.unit_system["temperature"],
            "wind_direction": self.unit_system["angle"],
            "wind_heading": self.unit_system["angle"],
            "wind_speed": self.unit_system["wind_speed"],
            "wind_velocity_x": self.unit_system["wind_speed"],
            "wind_velocity_y": self.unit_system["wind_speed"],
        }

        # Make a deep copy of the dictionary
        converted_dict = copy.deepcopy(self.original_pressure_level_data)

        # Loop through dates
        for date in self.original_pressure_level_data:
            # Loop through hours
            for hour in self.original_pressure_level_data[date]:
                # Loop through variables
                for key, value in self.original_pressure_level_data[date][hour].items():
                    # Skip geopotential x asl
                    if key not in conversion_dict:
                        continue
                    # Convert x axis
                    variable = convert_units(
                        variable=value,
                        from_unit=self.current_units["height_ASL"],
                        to_unit=self.unit_system["length"],
                        axis=0,
                    )
                    # Update current units
                    self.updated_units["height_ASL"] = self.unit_system["length"]
                    # Convert y axis
                    variable = convert_units(
                        variable=value,
                        from_unit=self.current_units[key],
                        to_unit=conversion_dict[key],
                        axis=1,
                    )
                    # Update current units
                    self.updated_units[key] = conversion_dict[key]
                    # Save converted Function
                    converted_dict[date][hour][key] = variable

        return converted_dict

    @cached_property
    def converted_surface_data(self):
        """Convert surface data to desired unit system. This method will loop
        through all the data (dates, hours and variables) and convert the units
        of each variable. Therefore, the performance of this method is not
        optimal. However, this method is only called once and the results are
        cached, so that the conversion is only done once.

        Returns
        -------
        dictionary
            Dictionary with the converted surface data. This dictionary has the
            same structure as the original_surface_data dictionary.
        """
        # Create conversion dict (key: from_unit, to_unit)
        conversion_dict = {
            "surface100m_wind_velocity_x": self.unit_system["wind_speed"],
            "surface100m_wind_velocity_y": self.unit_system["wind_speed"],
            "surface10m_wind_velocity_x": self.unit_system["wind_speed"],
            "surface10m_wind_velocity_y": self.unit_system["wind_speed"],
            "surface_temperature": self.unit_system["temperature"],
            "cloud_base_height": self.unit_system["length"],
            "surface_wind_gust": self.unit_system["wind_speed"],
            "surface_pressure": self.unit_system["pressure"],
            "total_precipitation": self.unit_system["precipitation"],
        }

        # Make a deep copy of the dictionary
        converted_dict = copy.deepcopy(self.original_surface_data)

        # Loop through dates
        for date in self.original_surface_data:
            # Loop through hours
            for hour in self.original_surface_data[date]:
                # Loop through variables
                for key, value in self.original_surface_data[date][hour].items():
                    variable = convert_units(
                        variable=value,
                        from_unit=self.current_units[key],
                        to_unit=conversion_dict[key],
                    )
                    converted_dict[date][hour][key] = variable
                    # Update current units
                    self.updated_units[key] = conversion_dict[key]

        self.updated_units["height_ASL"] = self.unit_system["length"]
        return converted_dict

    @cached_property
    def hours(self):
        """A list containing all the hours available in the dataset. The list
        is flattened, so that it is a 1D list with all the values. The result
        is cached so that the computation is only done once.

        Returns
        -------
        list
            List with all the hours available in the dataset.
        """
        hours = list(
            set(
                [
                    int(hour)
                    for day_dict in self.converted_surface_data.values()
                    for hour in day_dict.keys()
                ]
            )
        )
        hours.sort()
        return hours

    @cached_property
    def days(self):
        """A list containing all the days available in the dataset. The list
        is flattened, so that it is a 1D list with all the values. The result
        is cached so that the computation is only done once.

        Returns
        -------
        list
            List with all the days available in the dataset.
        """
        return list(self.converted_surface_data.keys())

    # Surface level data

    @cached_property
    def converted_elevation(self):
        """The surface elevation converted to the preferred unit system. The
        result is cached so that the computation is only done once.

        Returns
        -------
        float
            Surface elevation converted to the preferred unit system.
        """
        return convert_units(
            self.original_elevation,
            self.current_units["height_ASL"],
            self.unit_system["length"],
        )

    # Surface level data - Flattened lists

    @cached_property
    def cloud_base_height(self):
        """A np.ma.array containing the cloud base height for each hour and day
        in the dataset. The array is masked where no cloud base height is
        available. The array is flattened, so that it is a 1D array with all the
        values. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        np.ma.array
            Array with cloud base height for each hour and day in the dataset.
        """
        cloud_base_height = [
            day_dict[hour]["cloud_base_height"]
            for day_dict in self.converted_surface_data.values()
            for hour in day_dict.keys()
        ]

        masked_elem = np.ma.core.MaskedConstant
        unmasked_cloud_base_height = [
            np.inf if isinstance(elem, masked_elem) else elem
            for elem in cloud_base_height
        ]
        mask = [isinstance(elem, masked_elem) for elem in cloud_base_height]
        return np.ma.array(unmasked_cloud_base_height, mask=mask)

    @cached_property
    def pressure_at_surface_list(self):
        """A list containing the pressure at surface for each hour and day
        in the dataset. The list is flattened, so that it is a 1D list with
        all the values. The result is cached so that the computation is only
        done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with pressure at surface for each hour and day in the dataset.
        """
        return [
            day_dict[hour]["surface_pressure"]
            for day_dict in self.converted_surface_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def temperature_list(self):
        """A list containing the temperature for each hour and day in the
        dataset. The list is flattened, so that it is a 1D list with all the
        values. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with temperature for each hour and day in the dataset.
        """
        return [
            day_dict[hour]["surface_temperature"]
            for day_dict in self.converted_surface_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def max_temperature_list(self):
        """A list containing the maximum temperature for each day in the
        dataset. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with maximum temperature for each day in the dataset.
        """
        return [
            np.max([day_dict[hour]["surface_temperature"] for hour in day_dict.keys()])
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def min_temperature_list(self):
        """A list containing the minimum temperature for each day in the
        dataset. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with minimum temperature for each day in the dataset.
        """
        return [
            np.min([day_dict[hour]["surface_temperature"] for hour in day_dict.keys()])
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def wind_gust_list(self):
        """A list containing the wind gust for each hour and day in the dataset.
        The list is flattened, so that it is a 1D list with all the values. The
        result is cached so that the computation is only done once. The units
        are converted to the preferred unit system.

        Returns
        -------
        list
            List with wind gust for each hour and day in the dataset.
        """
        return [
            day_dict[hour]["surface_wind_gust"]
            for day_dict in self.converted_surface_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def max_wind_gust_list(self):
        """A list containing the maximum wind gust for each day in the dataset.
        The result is cached so that the computation is only done once. The
        units are converted to the preferred unit system.

        Returns
        -------
        list
            List with maximum wind gust for each day in the dataset.
        """
        return [
            np.max([day_dict[hour]["surface_wind_gust"] for hour in day_dict.keys()])
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def precipitation_per_day(self):
        """A list containing the total precipitation for each day in the
        dataset. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with total precipitation for each day in the dataset.
        """
        return [
            sum([day_dict[hour]["total_precipitation"] for hour in day_dict.keys()])
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def surface_10m_wind_speed_list(self):
        """A list containing the wind speed at surface+10m level for each hour
        and day in the dataset. The list is flattened, so that it is a 1D list
        with all the values. The result is cached so that the computation is
        only done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with surface 10m wind speed for each hour and day in the
            dataset.
        """
        return [
            (
                day_dict[hour]["surface10m_wind_velocity_x"] ** 2
                + day_dict[hour]["surface10m_wind_velocity_y"] ** 2
            )
            ** 0.5
            for day_dict in self.converted_surface_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def max_surface_10m_wind_speed_list(self):
        """A list containing the maximum wind speed at surface+10m level for
        each day in the dataset. The result is cached so that the computation
        is only done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with maximum wind speed at surface+10m level for each day in
            the dataset.
        """
        return [
            np.max(
                [
                    (
                        day_dict[hour]["surface10m_wind_velocity_x"] ** 2
                        + day_dict[hour]["surface10m_wind_velocity_y"] ** 2
                    )
                    ** 0.5
                    for hour in day_dict.keys()
                ]
            )
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def min_surface_10m_wind_speed_list(self):
        """A list containing the minimum wind speed at surface+10m level for
        each day in the dataset. The result is cached so that the computation
        is only done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with minimum wind speed at surface+10m level for each day.
        """
        return [
            np.min(
                [
                    (
                        day_dict[hour]["surface10m_wind_velocity_x"] ** 2
                        + day_dict[hour]["surface10m_wind_velocity_y"] ** 2
                    )
                    ** 0.5
                    for hour in day_dict.keys()
                ]
            )
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def surface_100m_wind_speed_list(self):
        """A list containing the wind speed at surface+100m level for each hour
        and day in the dataset. The list is flattened, so that it is a 1D list
        with all the values. The result is cached so that the computation is
        only done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with surface 100m wind speed for each hour and day in the
            dataset.
        """
        return [
            (
                day_dict[hour]["surface100m_wind_velocity_x"] ** 2
                + day_dict[hour]["surface100m_wind_velocity_y"] ** 2
            )
            ** 0.5
            for day_dict in self.converted_surface_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def max_surface_100m_wind_speed_list(self):
        """A list containing the maximum wind speed at surface+100m level for
        each day in the dataset. The result is cached so that the computation
        is only done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with maximum wind speed at surface+100m level for each day.
        """
        return [
            np.max(
                [
                    (
                        day_dict[hour]["surface100m_wind_velocity_x"] ** 2
                        + day_dict[hour]["surface100m_wind_velocity_y"] ** 2
                    )
                    ** 0.5
                    for hour in day_dict.keys()
                ]
            )
            for day_dict in self.converted_surface_data.values()
        ]

    @cached_property
    def min_surface_100m_wind_speed_list(self):
        """A list containing the minimum wind speed at surface+100m level for
        each day in the dataset. The result is cached so that the computation
        is only done once. The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with minimum wind speed at surface+100m level for each day.
        """
        return [
            np.min(
                [
                    (
                        day_dict[hour]["surface100m_wind_velocity_x"] ** 2
                        + day_dict[hour]["surface100m_wind_velocity_y"] ** 2
                    )
                    ** 0.5
                    for hour in day_dict.keys()
                ]
            )
            for day_dict in self.converted_surface_data.values()
        ]

    # Surface level data - Maximum and minimum values

    @property
    def record_max_surface_100m_wind_speed(self):
        """The overall maximum wind speed at surface+100m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Record maximum wind speed at surface+100m level.
        """
        return np.max(self.surface_100m_wind_speed_list)

    @property
    def record_min_surface_100m_wind_speed(self):
        """The overall minimum wind speed at surface+100m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Record minimum wind speed at surface+100m level.
        """
        return np.min(self.surface_100m_wind_speed_list)

    @property
    def record_min_cloud_base_height(self):
        """The overall minimum cloud base height considering all the days
        available in the surface level dataset. It uses the converted surface
        level data.

        Returns
        -------
        float
            Record minimum cloud base height.
        """
        return np.ma.min(self.cloud_base_height, fill_value=np.inf)

    @property
    def record_max_temperature(self):
        """The overall maximum temperature considering all the days available
        in the surface level dataset. It uses the converted surface level data.

        Returns
        -------
        float
            Record maximum temperature.
        """
        return np.max(self.temperature_list)

    @property
    def record_min_temperature(self):
        """The overall minimum temperature considering all the days available
        in the surface level dataset. It uses the converted surface level data.

        Returns
        -------
        float
            Record minimum temperature.
        """
        return np.min(self.temperature_list)

    @property
    def record_max_wind_gust(self):
        """The overall maximum wind gust considering all the days available

        Returns
        -------
        float
            Record maximum wind gust.
        """
        return np.max(self.wind_gust_list)

    @cached_property
    def record_max_surface_wind_speed(self):
        """The overall maximum wind speed at surface level considering all the
        days available in the surface level dataset. Units are converted to the
        preferred unit system.

        Returns
        -------
        float
            Record maximum wind speed at surface level.
        """
        max_speed = float("-inf")
        for hour in self.surface_wind_speed_by_hour.keys():
            speed = max(self.surface_wind_speed_by_hour[hour])
            if speed > max_speed:
                max_speed = speed
        return max_speed

    @cached_property
    def record_min_surface_wind_speed(self):
        """The overall minimum wind speed at surface level considering all the
        days available in the surface level dataset. Units are converted to the
        preferred unit system.

        Returns
        -------
        float
            Record minimum wind speed at surface level.
        """
        min_speed = float("inf")
        for hour in self.surface_wind_speed_by_hour.keys():
            speed = max(self.surface_wind_speed_by_hour[hour])
            if speed < min_speed:
                min_speed = speed
        return min_speed

    @property
    def record_max_surface_10m_wind_speed(self):
        """The overall maximum wind speed at surface+10m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Record maximum wind speed at surface+10m level.
        """
        return np.max(self.surface_10m_wind_speed_list)

    @property
    def record_min_surface_10m_wind_speed(self):
        """The overall minimum wind speed at surface+10m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Record minimum wind speed at surface+10m level.
        """
        return np.min(self.surface_10m_wind_speed_list)

    # Surface level data - Average values

    @property
    def average_surface_pressure(self):
        """The average surface pressure for all the days and hours available
        in the surface level dataset. Units are converted to the preferred
        unit system.

        Returns
        -------
        float
            Average surface pressure."""
        return np.average(self.pressure_at_surface_list)

    @property
    def std_surface_pressure(self):
        """The standard deviation of the surface pressure for all the days
        and hours available in the surface level dataset. Units are converted
        to the preferred unit system.
        """
        return np.std(self.pressure_at_surface_list)

    @property
    def average_cloud_base_height(self):
        """The average cloud base height considering all the days available
        in the surface level dataset. It uses the converted surface level data.
        If information is not available for a certain day, the day will be
        ignored.

        Returns
        -------
        float
            Average cloud base height.
        """
        return np.ma.mean(self.cloud_base_height)

    @property
    def average_max_temperature(self):
        """The average maximum temperature considering all the days available
        in the surface level dataset. It uses the converted surface level data.

        Returns
        -------
        float
            Average maximum temperature.
        """
        return np.average(self.max_temperature_list)

    @property
    def average_min_temperature(self):
        """The average minimum temperature considering all the days available
        in the surface level dataset. It uses the converted surface level data.

        Returns
        -------
        float
            Average minimum temperature.
        """
        return np.average(self.min_temperature_list)

    @property
    def average_max_wind_gust(self):
        """The average maximum wind gust considering all the days available
        in the surface level dataset. It uses the converted surface level data.

        Returns
        -------
        float
            Average maximum wind gust.
        """
        return np.average(self.max_wind_gust_list)

    @property
    def average_max_surface_10m_wind_speed(self):
        """The average maximum wind speed at surface+10m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Average maximum wind speed at surface+10m level.
        """
        return np.average(self.max_surface_10m_wind_speed_list)

    @property
    def average_min_surface_10m_wind_speed(self):
        """The average minimum wind speed at surface+10m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Average minimum wind speed at surface+10m level.
        """
        return np.average(self.min_surface_10m_wind_speed_list)

    @property
    def average_max_surface_100m_wind_speed(self):
        """The average maximum wind speed at surface+100m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Average maximum wind speed at surface+100m level.
        """
        return np.average(self.max_surface_100m_wind_speed_list)

    @property
    def average_min_surface_100m_wind_speed(self):
        """The average minimum wind speed at surface+100m level considering all
        the days available in the surface level dataset. It uses the converted
        surface level data. Units are converted to the preferred unit system.

        Returns
        -------
        float
            Average minimum wind speed at surface+100m level.
        """
        return np.average(self.min_surface_100m_wind_speed_list)

    # Surface level data - Other important values

    @property
    def percentage_of_days_with_no_cloud_coverage(self):
        """Calculate percentage of days with cloud coverage.

        Returns
        -------
        float
            Percentage of days with no cloud coverage."""
        return np.ma.count(self.cloud_base_height) / len(self.cloud_base_height)

    @cached_property
    def percentage_of_days_with_precipitation(self):
        """Computes the ratio between days with precipitation (> 10 mm) and
        total days. The result is cached so that the computation is only done
        once.

        Returns
        -------
        float
            Percentage of days with precipitation.
        """
        days_with_precipitation_count = 0
        for precipitation in self.precipitation_per_day:
            if precipitation > convert_units(
                10, "mm", self.unit_system["precipitation"]
            ):
                days_with_precipitation_count += 1

        return days_with_precipitation_count / len(self.precipitation_per_day)

    # Surface level data - Dictionaries by hour

    @cached_property
    def temperature_by_hour(self):
        """A dictionary containing the temperature for each hour and day in the
        dataset. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system. It flips the
        data dictionary to get the hour as key instead of the date.

        Returns
        -------
        dictionary
            Dictionary with temperature for each hour and day. The dictionary
            has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: {
                        date1: temperature1,
                        date2: temperature2,
                        ...
                        dateN: temperatureN,
                    },
                    ...
                    hourN: {
                        date1: temperature1,
                        date2: temperature2,
                        ...
                        dateN: temperatureN,
                    },
                }

        """
        history = defaultdict(dict)
        for date, val in self.converted_surface_data.items():
            for hour, sub_val in val.items():
                history[hour][date] = sub_val["surface_temperature"]
        return history

    @cached_property
    def average_temperature_by_hour(self):
        """The average temperature for each hour of the day. The result is
        cached so that the computation is only done once. The units are
        converted to the preferred unit system.

        Returns
        -------
        dictionary
            Dictionary with average temperature for each hour of the day. The
            dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: average_temperature1,
                    hour2: average_temperature2,
                    ...
                    hourN: average_temperatureN
                }

        """
        return {
            hour: np.average(list(dates.values()))
            for hour, dates in self.temperature_by_hour.items()
        }

    @cached_property
    def std_temperature_by_hour(self):
        """The standard deviation of the temperature for each hour of the day.
        The result is cached so that the computation is only done once. The units
        are converted to the preferred unit system.

        Returns
        -------
        dictionary
            Dictionary with standard deviation of the temperature for each hour
            of the day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: std_temperature1,
                    hour2: std_temperature2,
                    ...
                    hourN: std_temperatureN
                }

        """
        return {
            hour: np.std(list(dates.values()))
            for hour, dates in self.temperature_by_hour.items()
        }

    @cached_property
    def surface_10m_wind_speed_by_hour(self):
        """A dictionary containing the wind speed at surface+10m level for each
        hour and day in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system. It flips the data dictionary to get the hour as key instead
        of the date.

        Returns
        -------
        dictionary
            Dictionary with surface 10m wind speed for each hour and day. The
            dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: {
                        date1: wind_speed1,
                        date2: wind_speed2,
                        ...
                        dateN: wind_speedN
                    },
                    ...
                    hourN: {
                        date1: wind_speed1,
                        date2: wind_speed2,
                        ...
                        dateN: wind_speedN
                    }
                }

        """
        dictionary = defaultdict(dict)
        for date, val in self.converted_surface_data.items():
            for hour, sub_val in val.items():
                dictionary[hour][date] = (
                    sub_val["surface10m_wind_velocity_x"] ** 2
                    + sub_val["surface10m_wind_velocity_y"] ** 2
                ) ** 0.5
        return dictionary

    @cached_property
    def average_surface_10m_wind_speed_by_hour(self):
        """The average wind speed at surface+10m level for each hour of the day.
        The result is cached so that the computation is only done once. The
        units are converted to the preferred unit system.

        Returns
        -------
        dictionary
            Dictionary with average surface 10m wind speed for each hour of the
            day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: average_surface_10m_wind_speed1,
                    hour2: average_surface_10m_wind_speed2,
                    ...
                    hourN: average_surface_10m_wind_speedN
                }

        """
        return {
            hour: np.average(list(dates.values()))
            for hour, dates in self.surface_10m_wind_speed_by_hour.items()
        }

    @cached_property
    def std_surface_10m_wind_speed_by_hour(self):
        """The standard deviation of the wind speed at surface+10m level for
        each hour of the day. The result is cached so that the computation is
        only done once. The units are converted to the preferred unit system.

        Returns
        -------
        dictionary
            Dictionary with standard deviation of the surface 10m wind speed for
            each hour of the day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: std_surface_10m_wind_speed1,
                    hour2: std_surface_10m_wind_speed2,
                    ...
                    hourN: std_surface_10m_wind_speedN
                }

        """
        return {
            hour: np.std(list(dates.values()))
            for hour, dates in self.surface_10m_wind_speed_by_hour.items()
        }

    @cached_property
    def surface_100m_wind_speed_by_hour(self):
        """A dictionary containing the wind speed at surface+100m level for each
        hour and day in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system. It flips the data dictionary to get the hour as key instead
        of the date.

        Returns
        -------
        dictionary
            Dictionary with surface 100m wind speed for each hour and day. The
            dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: {
                        date1: wind_speed1,
                        date2: wind_speed2,
                        ...
                        dateN: wind_speedN
                    },
                    ...
                    hourN: {
                        date1: wind_speed1,
                        date2: wind_speed2,
                        ...
                        dateN: wind_speedN
                    }
                }

        """
        dictionary = defaultdict(dict)
        for date, val in self.converted_surface_data.items():
            for hour, sub_val in val.items():
                dictionary[hour][date] = (
                    sub_val["surface100m_wind_velocity_x"] ** 2
                    + sub_val["surface100m_wind_velocity_y"] ** 2
                ) ** 0.5
        return dictionary

    @cached_property
    def average_surface_100m_wind_speed_by_hour(self):
        """The average wind speed at surface+100m level for each hour of the
        day. The result is cached so that the computation is only done once. The
        units are converted to the preferred unit system.

        Returns
        -------
        dictionary
            Dictionary with average surface 100m wind speed for each hour of the
            day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: average_surface_100m_wind_speed1,
                    hour2: average_surface_100m_wind_speed2,
                    ...
                    hourN: average_surface_100m_wind_speedN
                }

        """
        return {
            hour: np.average(list(dates.values()))
            for hour, dates in self.surface_100m_wind_speed_by_hour.items()
        }

    @cached_property
    def std_surface_100m_wind_speed_by_hour(self):
        """The standard deviation of the wind speed at surface+100m level for
        each hour of the day. The result is cached so that the computation is
        only done once. The units are converted to the preferred unit system.

        Returns
        -------
        dictionary
            Dictionary with standard deviation of the surface 100m wind speed
            for each hour of the day. The dictionary has the following
            structure:

            .. code-block:: python

                dictionary = {
                    hour1: std_surface_100m_wind_speed1,
                    hour2: std_surface_100m_wind_speed2,
                    ...
                    hourN: std_surface_100m_wind_speedN
                }

        """
        return {
            hour: np.std(list(dates.values()))
            for hour, dates in self.surface_100m_wind_speed_by_hour.items()
        }

    @cached_property
    def __process_surface_wind_data(self):
        """Process the wind speed and wind direction data to generate lists of
        all the wind_speeds recorded for a following hour of the day and also
        the wind direction.

        Returns
        -------
        tuple
            Tuple containing the wind speed and wind direction lists. The
            structure of the tuple is the following:

            .. code-block:: python

                tuple = (surface_wind_speed_by_hour, surface_wind_direction_by_hour)

        """

        wind_speed = {}
        wind_dir = {}

        for hour in self.hours:
            # The following two lines avoid the use of append, which is slow
            wind_speed[hour] = ["" for _ in range(len(self.days))]
            wind_dir[hour] = ["" for _ in range(len(self.days))]
            for index, day in enumerate(self.days):
                try:
                    vx = self.converted_surface_data[day][str(hour)][
                        "surface10m_wind_velocity_x"
                    ]
                    vy = self.converted_surface_data[day][str(hour)][
                        "surface10m_wind_velocity_y"
                    ]
                    wind_speed[hour][index] = (vx**2 + vy**2) ** 0.5

                    # Wind direction means where the wind is blowing from, 180 deg opposite from wind heading
                    direction = (180 + (np.arctan2(vy, vx) * 180 / np.pi)) % 360
                    wind_dir[hour][index] = direction
                except KeyError:
                    # Not all days have all hours stored, that is fine
                    pass

        # Remove the undesired "" values
        for hour in self.hours:
            wind_speed[hour] = [x for x in wind_speed[hour] if x != ""]
            wind_dir[hour] = [x for x in wind_dir[hour] if x != ""]

        return wind_speed, wind_dir

    @property
    def surface_wind_speed_by_hour(self):
        """A dictionary containing the wind speed at surface level for each hour
        and day in the dataset. The result is cached so that the computation is
        only done once. The units are converted to the preferred unit system.
        It flips the data dictionary to get the hour as key instead of the date.

        Returns
        -------
        dictionary
            Dictionary with surface wind speed for each hour and day. The
            dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [wind_speed1, wind_speed2, ..., wind_speedN],
                    ...
                    hourN: [wind_speed1, wind_speed2, ..., wind_speedN]
                }

        """
        return self.__process_surface_wind_data[0]

    @property
    def surface_wind_direction_by_hour(self):
        """A dictionary containing the wind direction at surface level for each
        hour and day in the dataset. It flips the data dictionary to get the
        hour as key instead of the date.

        Returns
        -------
        dictionary
            Dictionary with surface wind direction for each hour and day. The
            dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: {
                        date1: wind_direction1,
                        date2: wind_direction2,
                        ...
                        dateN: wind_directionN
                    },
                    ...
                    hourN: {
                        date1: wind_direction1,
                        date2: wind_direction2,
                        ...
                        dateN: wind_directionN
                    }
                }

        """
        return self.__process_surface_wind_data[1]

    @cached_property
    def surface_wind_gust_by_hour(self):
        wind_gusts = {}
        # Iterate over all hours
        for hour in self.hours:
            values = []
            # Iterate over all days
            for day_dict in self.converted_surface_data.values():
                try:
                    # Get wind gust value for this hour
                    values += [day_dict[str(hour)]["surface_wind_gust"]]
                except KeyError:
                    # Some day does not have data for the desired hour (probably the last one)
                    # No need to worry, just average over the other days
                    pass
            wind_gusts[hour] = values
        return wind_gusts

    # Pressure level data

    @cached_property
    def altitude_AGL_range(self):
        """The altitude range for the pressure level data. The minimum altitude
        is always 0, and the maximum altitude is the maximum altitude of the
        pressure level data, or the maximum expected altitude if it is set.
        Units are kept as they are in the original data.

        Returns
        -------
        tuple
            Tuple containing the minimum and maximum altitude. The first element
            is the minimum altitude, and the second element is the maximum.
        """
        min_altitude = 0
        if self.max_expected_altitude == None:
            max_altitudes = [
                np.max(day_dict[hour]["wind_speed"].source[-1, 0])
                for day_dict in self.original_pressure_level_data.values()
                for hour in day_dict.keys()
            ]
            max_altitude = np.min(max_altitudes)
        else:
            max_altitude = self.max_expected_altitude
        return min_altitude, max_altitude

    @cached_property
    def altitude_list(self, points=200):
        """A list of altitudes, from 0 to the maximum altitude of the pressure
        level data, or the maximum expected altitude if it is set. The list is
        cached so that the computation is only done once. Units are kept as they
        are in the original data.

        Parameters
        ----------
        points : int, optional
            Number of points to use in the list. The default is 200.

        Returns
        -------
        numpy.ndarray
            List of altitudes.
        """
        return np.linspace(*self.altitude_AGL_range, points)

    # Pressure level data - Flattened lists

    @cached_property
    def pressure_at_1000ft_list(self):
        """A list containing the pressure at 1000 feet for each hour and day
        in the dataset. The list is flattened, so that it is a 1D list with
        all the values. The result is cached so that the computation is only
        done once. It uses the converted pressure level data.
        """
        return [
            day_dict[hour]["pressure"](
                convert_units(1000, "ft", self.current_units["height_ASL"])
            )
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def pressure_at_10000ft_list(self):
        """A list containing the pressure at 10000 feet for each hour and day
        in the dataset. The list is flattened, so that it is a 1D list with
        all the values. The result is cached so that the computation is only
        done once. It uses the converted pressure level data.

        Returns
        -------
        list
            List with pressure at 10000 feet for each hour and day in the
            dataset.
        """
        # Pressure at 10000 feet
        return [
            day_dict[hour]["pressure"](
                convert_units(10000, "ft", self.current_units["height_ASL"])
            )
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def pressure_at_30000ft_list(self):
        """A list containing the pressure at 30000 feet for each hour and day
        in the dataset. The list is flattened, so that it is a 1D list with
        all the values. The result is cached so that the computation is only
        done once. It uses the converted pressure level data.

        Returns
        -------
        list
            List with pressure at 30000 feet for each hour and day in the
            dataset.
        """
        # Pressure at 30000 feet
        return [
            day_dict[hour]["pressure"](
                convert_units(30000, "ft", self.current_units["height_ASL"])
            )
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    # Pressure level data - Average profiles by hour (dictionaries)

    @cached_property
    def average_temperature_profile_by_hour(self):
        """Compute the average temperature profile for each available hour of a
        day, over all days in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system.

        Returns
        -------
        dictionary
            Dictionary with average temperature profile for each hour of the day.
            The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [average_temperature_profile1, altitude_list1],
                    hour2: [average_temperature_profile2, altitude_list2],
                    ...
                    hourN: [average_temperature_profileN, altitude_listN]
                }

        """

        profiles_by_hour = {}

        for hour in self.hours:
            values = []
            for day_dict in self.converted_pressure_level_data.values():
                try:
                    values += [day_dict[str(hour)]["temperature"](self.altitude_list)]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            average = np.mean(values, axis=0)
            profiles_by_hour[hour] = [average, self.altitude_list]
        return profiles_by_hour

    @cached_property
    def average_pressure_profile_by_hour(self):
        """Compute the average pressure profile for each available hour of a
        day, over all days in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system.

        Returns
        -------
        dictionary
            Dictionary with average pressure profile for each hour of the day.
            The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [average_pressure_profile1, altitude_list1],
                    hour2: [average_pressure_profile2, altitude_list2],
                    ...
                    hourN: [average_pressure_profileN, altitude_listN]
                }

        """

        pressures = {}

        for hour in self.hours:
            values = []
            for day_dict in self.converted_pressure_level_data.values():
                try:
                    values += [day_dict[str(hour)]["pressure"](self.altitude_list)]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            average_pressure_list = np.mean(values, axis=0)
            pressures[hour] = [average_pressure_list, self.altitude_list]

        return pressures

    @cached_property
    def average_wind_speed_profile_by_hour(self):
        """Compute the average wind speed profile for each available hour of a
        day, over all days in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system.

        Returns
        -------
        dictionary
            Dictionary with average wind profile for each hour of the day. The
            dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [average_wind_profile1, altitude_list1],
                    hour2: [average_wind_profile2, altitude_list2],
                    ...
                    hourN: [average_wind_profileN, altitude_listN]
                }

        """

        wind_speed = {}

        for hour in self.hours:
            values = []
            for day_dict in self.converted_pressure_level_data.values():
                try:
                    values += [day_dict[str(hour)]["wind_speed"](self.altitude_list)]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            average_values = np.mean(values, axis=0)
            wind_speed[hour] = [average_values, self.altitude_list]

        return wind_speed

    @cached_property
    def average_wind_velocity_x_profile_by_hour(self):
        """Compute the average wind_velocity_x profile for each available hour
        of a day, over all days in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system.

        Returns
        -------
        dictionary
            Dictionary with average wind_velocity_x profile for each hour of the
            day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [average_windVelocityX_profile1, altitude_list1],
                    hour2: [average_windVelocityX_profile2, altitude_list2],
                    ...
                    hourN: [average_windVelocityX_profileN, altitude_listN]
                }

        """

        wind_x_values = {}

        for hour in self.hours:
            values = []
            for day_dict in self.converted_pressure_level_data.values():
                try:
                    values += [
                        day_dict[str(hour)]["wind_velocity_x"](self.altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            average_values = np.mean(values, axis=0)
            wind_x_values[hour] = [average_values, self.altitude_list]

        return wind_x_values

    @cached_property
    def average_wind_velocity_y_profile_by_hour(self):
        """Compute the average wind_velocity_y profile for each available hour
        of a day, over all days in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system.

        Returns
        -------
        dictionary
            Dictionary with average wind_velocity_y profile for each hour of the
            day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [average_windVelocityY_profile1, altitude_list1],
                    hour2: [average_windVelocityY_profile2, altitude_list2],
                    ...
                    hourN: [average_windVelocityY_profileN, altitude_listN]
                }

        """

        wind_y_speed = {}

        for hour in self.hours:
            values = []
            for day_dict in self.converted_pressure_level_data.values():
                try:
                    values += [
                        day_dict[str(hour)]["wind_velocity_y"](self.altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            average_values = np.mean(values, axis=0)
            wind_y_speed[hour] = [average_values, self.altitude_list]

        return wind_y_speed

    @cached_property
    def average_wind_heading_profile_by_hour(self):
        """Compute the average wind heading profile for each available hour of a
        day, over all days in the dataset. The result is cached so that the
        computation is only done once. The units are converted to the preferred
        unit system.

        Returns
        -------
        dictionary
            Dictionary with average wind heading profile for each hour of the
            day. The dictionary has the following structure:

            .. code-block:: python

                dictionary = {
                    hour1: [average_wind_heading_profile1, altitude_list1],
                    hour2: [average_wind_heading_profile2, altitude_list2],
                    ...
                    hourN: [average_wind_heading_profileN, altitude_listN]
                }

        """

        avg_profiles = {}

        for hour in self.hours:
            headings = [
                np.arctan2(
                    self.average_wind_velocity_x_profile_by_hour[hour][0],
                    self.average_wind_velocity_y_profile_by_hour[hour][0],
                )
                * (180 / np.pi)
                % 360,
                self.altitude_list,
            ]
            avg_profiles[hour] = headings

        return avg_profiles

    # Pressure level data - Average profiles of all hours (lists)

    @cached_property
    def wind_velocity_x_profiles_list(self):
        return [
            day_dict[hour]["wind_velocity_x"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def wind_velocity_y_profiles_list(self):
        return [
            day_dict[hour]["wind_velocity_y"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def wind_speed_profiles_list(self):
        """A list containing the wind speed profile for each hour and day in the
        dataset. The list is flattened, so that it is a 1D list with all the
        values. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with wind speed profile for each hour and day in the dataset.
        """
        return [
            day_dict[hour]["wind_speed"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def wind_heading_profiles_list(self):
        return [
            np.arctan2(
                day_dict[hour]["wind_velocity_x"](self.altitude_list),
                day_dict[hour]["wind_velocity_y"](self.altitude_list),
            )
            * (180 / np.pi)
            % 360
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def pressure_profiles_list(self):
        """A list containing the pressure profile for each hour and day in the
        dataset. The list is flattened, so that it is a 1D list with all the
        values. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with pressure profile for each hour and day in the dataset.
        """
        return [
            day_dict[hour]["pressure"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    @cached_property
    def temperature_profiles_list(self):
        """A list containing the temperature profile for each hour and day in
        the dataset. The list is flattened, so that it is a 1D list with all the
        values. The result is cached so that the computation is only done once.
        The units are converted to the preferred unit system.

        Returns
        -------
        list
            List with temperature profile for each hour and day in the dataset.
        """
        return [
            day_dict[hour]["temperature"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]

    # Pressure level data - Maximum and minimum values

    @cached_property
    def max_average_temperature_at_altitude(self):
        """The maximum average temperature considering all the hours of the day
        and all the days available in the pressure level dataset. It uses the
        converted pressure level data. Units are converted to the preferred
        unit system.

        Returns
        -------
        float
            Maximum average temperature.
        """
        max_temp = float("-inf")
        for hour in self.average_temperature_profile_by_hour.keys():
            max_temp = max(
                max_temp,
                np.max(self.average_temperature_profile_by_hour[hour][0]),
            )
        return max_temp

    @cached_property
    def min_average_temperature_at_altitude(self):
        """The minimum average temperature considering all the hours of the day
        and all the days available in the pressure level dataset. It uses the
        converted pressure level data. Units are converted to the preferred
        unit system.

        Returns
        -------
        float
            Minimum average temperature.
        """
        min_temp = float("inf")
        for hour in self.average_temperature_profile_by_hour.keys():
            min_temp = min(
                min_temp,
                np.min(self.average_temperature_profile_by_hour[hour][0]),
            )
        return min_temp

    @cached_property
    def max_average_wind_speed_at_altitude(self):
        """The maximum average wind speed considering all the hours of the day
        and all the days available in the pressure level dataset. It uses the
        converted pressure level data. Units are converted to the preferred
        unit system. The result is cached so that the computation is only done
        once.

        Returns
        -------
        float
            Maximum average wind speed.
        """
        max_wind_speed = float("-inf")
        for hour in self.average_wind_speed_profile_by_hour.keys():
            max_wind_speed = max(
                max_wind_speed,
                np.max(self.average_wind_speed_profile_by_hour[hour][0]),
            )
        return max_wind_speed

    # Pressure level data - Average values

    @property
    def average_pressure_at_1000ft(self):
        """The average pressure at 1000 feet for all the days and hours
        available in the pressure level dataset. It uses the converted pressure
        level data.
        """
        return np.average(self.pressure_at_1000ft_list)

    @property
    def std_pressure_at_1000ft(self):
        """The standard deviation of the pressure at 1000 feet for all the days
        and hours available in the pressure level dataset. It uses the converted
        pressure level data.

        Returns
        -------
        float
            Standard deviation of the pressure at 1000 feet.
        """
        return np.std(self.pressure_at_1000ft_list)

    @property
    def average_pressure_at_10000ft(self):
        """The average pressure at 10000 feet for all the days and hours
        available in the pressure level dataset. It uses the converted pressure
        level data.

        Returns
        -------
        float
            Average pressure at 10000 feet.
        """
        return np.average(self.pressure_at_10000ft_list)

    @property
    def std_pressure_at_10000ft(self):
        """The standard deviation of the pressure at 10000 feet for all the days
        and hours available in the pressure level dataset. It uses the converted
        pressure level data.

        Returns
        -------
        float
            Standard deviation of the pressure at 10000 feet.
        """
        return np.std(self.pressure_at_10000ft_list)

    @property
    def average_pressure_at_30000ft(self):
        """The average pressure at 30000 feet for all the days and hours
        available in the pressure level dataset. It uses the converted pressure
        level data.

        Returns
        -------
        float
            Average pressure at 30000 feet.
        """
        return np.average(self.pressure_at_30000ft_list)

    @property
    def std_pressure_at_30000ft(self):
        """The standard deviation of the pressure at 30000 feet for all the days
        and hours available in the pressure level dataset. It uses the converted
        pressure level data.

        Returns
        -------
        float
            Standard deviation of the pressure at 30000 feet.
        """
        return np.std(self.pressure_at_30000ft_list)

    # Pressure level data - Average profiles over all days and hours

    @cached_property
    def average_wind_velocity_x_profile(self):
        wind_x_values = [
            day_dict[hour]["wind_velocity_x"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]
        return np.mean(wind_x_values, axis=0)

    @cached_property
    def average_wind_velocity_y_profile(self):
        wind_y_values = [
            day_dict[hour]["wind_velocity_y"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]
        return np.mean(wind_y_values, axis=0)

    @cached_property
    def average_wind_speed_profile(self):
        return np.mean(self.wind_speed_profiles_list, axis=0)

    @cached_property
    def average_wind_heading_profile(self):
        return (
            np.arctan2(
                self.average_wind_velocity_x_profile,
                self.average_wind_velocity_y_profile,
            )
            * (180 / np.pi)
            % 360
        )

    @cached_property
    def average_pressure_profile(self):
        pressures = [
            day_dict[hour]["pressure"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]
        return np.mean(pressures, axis=0)

    @cached_property
    def average_temperature_profile(self):
        temperatures = [
            day_dict[hour]["temperature"](self.altitude_list)
            for day_dict in self.converted_pressure_level_data.values()
            for hour in day_dict.keys()
        ]
        return np.mean(temperatures, axis=0)

    # Plots

    def info(self):
        """Prints out the most important data and graphs available about the
        Environment Analysis.

        Returns
        -------
        None
        """

        self.prints.all()
        self.plots.info()
        return None

    def all_info(self):
        """Prints out all data and graphs available.

        Returns
        -------
        None
        """

        self.prints.all()
        self.plots.all()

        return None

    def export_mean_profiles(self, filename="export_env_analysis"):
        """
        Exports the mean profiles of the weather data to a file in order to it
        be used as inputs on Environment Class by using the custom_atmosphere
        model.

        Parameters
        ----------
        filename : str, optional
            Name of the file where to be saved, by default "env_analysis_dict"

        Returns
        -------
        None
        """

        flipped_temperature_dict = {}
        flipped_pressure_dict = {}
        flipped_wind_x_dict = {}
        flipped_wind_y_dict = {}

        for hour in self.average_temperature_profile_by_hour.keys():
            flipped_temperature_dict[hour] = np.column_stack(
                (
                    self.average_temperature_profile_by_hour[hour][1],
                    self.average_temperature_profile_by_hour[hour][0],
                )
            ).tolist()
            flipped_pressure_dict[hour] = np.column_stack(
                (
                    self.average_pressure_profile_by_hour[hour][1],
                    self.average_pressure_profile_by_hour[hour][0],
                )
            ).tolist()
            flipped_wind_x_dict[hour] = np.column_stack(
                (
                    self.average_wind_velocity_x_profile_by_hour[hour][1],
                    self.average_wind_velocity_x_profile_by_hour[hour][0],
                )
            ).tolist()
            flipped_wind_y_dict[hour] = np.column_stack(
                (
                    self.average_wind_velocity_y_profile_by_hour[hour][1],
                    self.average_wind_velocity_y_profile_by_hour[hour][0],
                )
            ).tolist()

        self.export_dictionary = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.converted_elevation,
            "timezone": self.preferred_timezone,
            "unit_system": self.unit_system,
            "surface_data_file": self.surface_data_file,
            "pressure_level_data_file": self.pressure_level_data_file,
            "atmospheric_model_pressure_profile": flipped_pressure_dict,
            "atmospheric_model_temperature_profile": flipped_temperature_dict,
            "atmospheric_model_wind_velocity_x_profile": flipped_wind_x_dict,
            "atmospheric_model_wind_velocity_y_profile": flipped_wind_y_dict,
        }

        # Convert to json
        f = open(filename + ".json", "w")

        # write json object to file
        f.write(
            json.dumps(self.export_dictionary, sort_keys=False, indent=4, default=str)
        )

        # close file
        f.close()
        print(
            "Your Environment Analysis file was saved, check it out: "
            + filename
            + ".json"
        )
        print(
            "You can use it in the future by using the customAtmosphere atmospheric model."
        )

        return None

    @classmethod
    def load(self, filename="env_analysis_dict"):
        """Load a previously saved Environment Analysis file.
        Example: EnvA = EnvironmentAnalysis.load("filename").

        Parameters
        ----------
        filename : str, optional
            Name of the previous saved file, by default "env_analysis_dict"

        Returns
        -------
        EnvironmentAnalysis object

        """
        jsonpickle = import_optional_dependency("jsonpickle")
        encoded_class = open(filename).read()
        return jsonpickle.decode(encoded_class)

    def save(self, filename="env_analysis_dict"):
        """Save the Environment Analysis object to a file so it can be used
        later.

        Parameters
        ----------
        filename : str, optional
            Name of the file where to be saved, by default "env_analysis_dict"

        Returns
        -------
        None
        """
        jsonpickle = import_optional_dependency("jsonpickle")
        encoded_class = jsonpickle.encode(self)
        file = open(filename, "w")
        file.write(encoded_class)
        file.close()
        print("Your Environment Analysis file was saved, check it out: " + filename)

        return None
