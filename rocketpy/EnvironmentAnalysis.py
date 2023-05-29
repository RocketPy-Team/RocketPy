# -*- coding: utf-8 -*-

__author__ = "Patrick Sampaio, Giovani Hidalgo Ceotto, Guilherme Fernandes Alves, Franz Masatoshi Yuri, Mateus Stano Junqueira,"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import bisect
import copy
import datetime
import json
import warnings
from collections import defaultdict

import jsonpickle
import netCDF4
import numpy as np
import pytz

from rocketpy.Environment import Environment
from rocketpy.Function import Function
from rocketpy.units import convert_units

from .plots.environment_analysis_plots import _EnvironmentAnalysisPlots
from .prints.environment_analysis_prints import _EnvironmentAnalysisPrints
from .tools import (
    bilinear_interpolation,
    geopotential_to_height_agl,
    geopotential_to_height_asl,
    time_num_to_date_string,
)

try:
    from functools import cached_property
except ImportError:
    from .tools import cached_property


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

    You can also visualize all those attributes by exploring some of the methods:
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
        - animation of who wind gust distribution evolves over average day
        - allInfo

    All items listed are relevant to either
        1. participant safety
        2. launch operations (range closure decision)
        3. rocket performance

    How does this class work?
    - The class is initialized with a start_date, end_date, start_hour and end_hour.
    - The class then parses the weather data from the start date to the end date.
    Always parsing the data from start_hour to end_hour.
    - The class then calculates the average max/min temperature, average max wind gust, and average day wind rose.
    - The class then allows for plotting the average max/min temperature, average max wind gust, and average day wind rose.

    """

    def __init__(
        self,
        start_date,
        end_date,
        latitude,
        longitude,
        start_hour=0,
        end_hour=24,
        surfaceDataFile=None,
        pressureLevelDataFile=None,
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        maxExpectedAltitude=None,
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
        surfaceDataFile : str, optional
            Path to the netCDF file containing the surface data.
        pressureLevelDataFile : str, optional
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
            Date for the forecast models. It will be requested the environment forecast
            for multiple hours within that specified date.
        forecast_args : dictionary, optional
            Arguments for setting the forecast on the Environment class. With this argument
            it is possible to change the forecast model being used.
        maxExpectedAltitude : float, optional
            Maximum expected altitude for your analysis. This is used to calculate
            plot limits from pressure level data profiles. If None is set, the
            maximum altitude will be calculated from the pressure level data.
            Default is None.
        Returns
        -------
        None
        """
        warnings.warn(
            "Please notice this class is still under development, and some features may not work as expected as they were not exhaustively tested yet. In case of having any trouble, please raise an issue at https://github.com/RocketPy-Team/RocketPy/issues"
        )

        # Save inputs
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.latitude = latitude
        self.longitude = longitude
        self.surfaceDataFile = surfaceDataFile
        self.pressureLevelDataFile = pressureLevelDataFile
        self.preferred_timezone = timezone
        self.unit_system = unit_system
        self.maxExpectedAltitude = maxExpectedAltitude

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
                hour_datetime = datetime.datetime(
                    year=forecast_date.year,
                    month=forecast_date.month,
                    day=forecast_date.day,
                    hour=int(hour),
                )

                Env = Environment(
                    railLength=5,
                    date=hour_datetime,
                    latitude=self.latitude,
                    longitude=self.longitude,
                    elevation=self.converted_elevation,
                )
                forecast_args = forecast_args or {"type": "Forecast", "file": "GFS"}
                Env.setAtmosphericModel(**forecast_args)
                self.forecast[hour] = Env
        return None

    def __init_surface_dictionary(self):
        # Create dictionary of file variable names to process surface data
        return {
            "surface100mWindVelocityX": "u100",
            "surface100mWindVelocityY": "v100",
            "surface10mWindVelocityX": "u10",
            "surface10mWindVelocityY": "v10",
            "surfaceTemperature": "t2m",
            "cloudBaseHeight": "cbh",
            "surfaceWindGust": "i10fg",
            "surfacePressure": "sp",
            "totalPrecipitation": "tp",
        }

    def __init_pressure_level_dictionary(self):
        # Create dictionary of file variable names to process pressure level data
        return {
            "geopotential": "z",
            "windVelocityX": "u",
            "windVelocityY": "v",
            "temperature": "t",
        }

    def __getNearestIndex(self, array, value):
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

        Examples
        --------
        >>> array = np.array([-180, -90, 0, 90, 180])
        >>> value = 0
        >>> index = self.__getNearestIndex(array, value)
        >>> index
        2

        >>> array = np.array([0, 90, 180, 270, 360])
        >>> value = 0
        >>> index = self.__getNearestIndex(array, value)
        >>> index
        0
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

    def __extractSurfaceDataValue(
        self, surfaceData, variable, indices, lonArray, latArray
    ):
        """Extract value from surface data netCDF4 file. Performs bilinear
        interpolation along longitude and latitude."""

        timeIndex, lonIndex, latIndex = indices
        variableData = surfaceData[variable]

        # Get values for variable on the four nearest points
        z11 = variableData[timeIndex, lonIndex - 1, latIndex - 1]
        z12 = variableData[timeIndex, lonIndex - 1, latIndex]
        z21 = variableData[timeIndex, lonIndex, latIndex - 1]
        z22 = variableData[timeIndex, lonIndex, latIndex]

        # Compute interpolated value on desired lat lon pair
        value = bilinear_interpolation(
            x=self.longitude,
            y=self.latitude,
            x1=lonArray[lonIndex - 1],
            x2=lonArray[lonIndex],
            y1=latArray[latIndex - 1],
            y2=latArray[latIndex],
            z11=z11,
            z12=z12,
            z21=z21,
            z22=z22,
        )

        return value

    def __extractPressureLevelDataValue(
        self, pressureLevelData, variable, indices, lonArray, latArray
    ):
        """Extract value from surface data netCDF4 file. Performs bilinear
        interpolation along longitude and latitude."""

        timeIndex, lonIndex, latIndex = indices
        variableData = pressureLevelData[variable]

        # Get values for variable on the four nearest points
        z11 = variableData[timeIndex, :, lonIndex - 1, latIndex - 1]
        z12 = variableData[timeIndex, :, lonIndex - 1, latIndex]
        z21 = variableData[timeIndex, :, lonIndex, latIndex - 1]
        z22 = variableData[timeIndex, :, lonIndex, latIndex]

        # Compute interpolated value on desired lat lon pair
        value_list_as_a_function_of_pressure_level = bilinear_interpolation(
            x=self.longitude,
            y=self.latitude,
            x1=lonArray[lonIndex - 1],
            x2=lonArray[lonIndex],
            y1=latArray[latIndex - 1],
            y2=latArray[latIndex],
            z11=z11,
            z12=z12,
            z21=z21,
            z22=z22,
        )

        return value_list_as_a_function_of_pressure_level

    def __check_coordinates_inside_grid(self, lonIndex, latIndex, lonArray, latArray):
        if (
            lonIndex == 0
            or lonIndex > len(lonArray) - 1
            or latIndex == 0
            or latIndex > len(latArray) - 1
        ):
            raise ValueError(
                f"Latitude and longitude pair {(self.latitude, self.longitude)} is outside the grid available in the given file, which is defined by {(latArray[0], lonArray[0])} and {(latArray[-1], lonArray[-1])}."
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
            try:
                from timezonefinder import TimezoneFinder
            except ImportError:
                raise ImportError(
                    "The timezonefinder package is required to automatically "
                    + "determine local timezone based on lat,lon coordinates. "
                    + "Please specify the desired timezone using the `timezone` "
                    + "argument when initializing the EnvironmentAnalysis class "
                    + "or install timezonefinder with `pip install timezonefinder`."
                )
            # Use local timezone based on lat lon pair
            tf = TimezoneFinder()
            self.preferred_timezone = pytz.timezone(
                tf.timezone_at(lng=self.longitude, lat=self.latitude)
            )
        elif isinstance(self.preferred_timezone, str):
            self.preferred_timezone = pytz.timezone(self.preferred_timezone)

    def __init_data_parsing_units(self):
        """Define units for pressure level and surface data parsing"""
        self.current_units = {
            "height_ASL": "m",
            "pressure": "hPa",
            "temperature": "K",
            "windDirection": "deg",
            "windHeading": "deg",
            "windSpeed": "m/s",
            "windVelocityX": "m/s",
            "windVelocityY": "m/s",
            "surface100mWindVelocityX": "m/s",
            "surface100mWindVelocityY": "m/s",
            "surface10mWindVelocityX": "m/s",
            "surface10mWindVelocityY": "m/s",
            "surfaceTemperature": "K",
            "cloudBaseHeight": "m",
            "surfaceWindGust": "m/s",
            "surfacePressure": "Pa",
            "totalPrecipitation": "m",
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
        """Set preferred unit system for output (SI, metric or imperial).
        The data with new values will be stored in `converted_pressure_level_data`
        and `converted_surface_data` dictionaries, while the original parsed
        data will be kept in `original_pressure_level_data` and `original_surface_data`.
        The performance of this method is not optimal since it will loop through
        all the data (dates, hours and variables) and convert the units of each
        variable, one by one. However, this method is only called once.

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
        # Convert units
        # self.converted_pressure_level_data = self.convertPressureLevelData()
        # self.converted_surface_data = self.convertSurfaceData()
        # Update current units
        self.current_units = self.updated_units.copy()

        return None

    @cached_property
    def __parsePressureLevelData(self):
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
        - pressure = Function(..., inputs="Height Above Sea Level (m)", outputs="Pressure (Pa)")
        - temperature = Function(..., inputs="Height Above Sea Level (m)", outputs="Temperature (K)")
        - windDirection = Function(..., inputs="Height Above Sea Level (m)", outputs="Wind Direction (Deg True)")
        - windHeading = Function(..., inputs="Height Above Sea Level (m)", outputs="Wind Heading (Deg True)")
        - windSpeed = Function(..., inputs="Height Above Sea Level (m)", outputs="Wind Speed (m/s)")
        - windVelocityX = Function(..., inputs="Height Above Sea Level (m)", outputs="Wind Velocity X (m/s)")
        - windVelocityY = Function(..., inputs="Height Above Sea Level (m)", outputs="Wind Velocity Y (m/s)")

        Return a dictionary with all the computed data with the following structure:
        pressureLevelDataDict: {
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
        pressureLevelFileDict = self.__init_pressure_level_dictionary()
        # Read weather file
        pressureLevelData = netCDF4.Dataset(self.pressureLevelDataFile)

        # Get time, pressure levels, latitude and longitude data from file
        timeNumArray = pressureLevelData.variables["time"]
        pressureLevelArray = pressureLevelData.variables["level"]
        lonArray = pressureLevelData.variables["longitude"]
        latArray = pressureLevelData.variables["latitude"]
        # Determine latitude and longitude range for pressure level file
        lat0 = latArray[0]
        lat1 = latArray[-1]
        lon0 = lonArray[0]
        lon1 = lonArray[-1]

        # Find index needed for latitude and longitude for specified location
        lonIndex = self.__getNearestIndex(lonArray, self.longitude)
        latIndex = self.__getNearestIndex(latArray, self.latitude)

        # Can't handle lat and lon out of grid
        self.__check_coordinates_inside_grid(lonIndex, latIndex, lonArray, latArray)

        # Loop through time and save all values
        for timeIndex, timeNum in enumerate(timeNumArray):
            dateString, hourString, dateTime = time_num_to_date_string(
                timeNum,
                timeNumArray.units,
                self.preferred_timezone,
                calendar="gregorian",
            )

            # Check if date is within analysis range
            if not (self.start_date <= dateTime < self.end_date):
                continue
            if not (self.start_hour <= dateTime.hour < self.end_hour):
                continue
            # Make sure keys exist
            if dateString not in dictionary:
                dictionary[dateString] = {}
            if hourString not in dictionary[dateString]:
                dictionary[dateString][hourString] = {}

            # Extract data from weather file
            indices = (timeIndex, lonIndex, latIndex)

            # Retrieve geopotential first and compute altitudes
            geopotentialArray = self.__extractPressureLevelDataValue(
                pressureLevelData,
                pressureLevelFileDict["geopotential"],
                indices,
                lonArray,
                latArray,
            )
            heightAboveSeaLevelArray = geopotential_to_height_agl(
                geopotentialArray, self.original_elevation
            )

            # Loop through wind components and temperature, get value and convert to Function
            for key, value in pressureLevelFileDict.items():
                valueArray = self.__extractPressureLevelDataValue(
                    pressureLevelData, value, indices, lonArray, latArray
                )
                variablePointsArray = np.array([heightAboveSeaLevelArray, valueArray]).T
                variableFunction = Function(
                    variablePointsArray,
                    inputs="Height Above Ground Level (m)",  # TODO: Check if it is really AGL or ASL here, see 3 lines above
                    outputs=key,
                    extrapolation="constant",
                )
                dictionary[dateString][hourString][key] = variableFunction

            # Create function for pressure levels
            pressurePointsArray = np.array(
                [heightAboveSeaLevelArray, pressureLevelArray]
            ).T
            pressureFunction = Function(
                pressurePointsArray,
                inputs="Height Above Sea Level (m)",
                outputs="Pressure (Pa)",
                extrapolation="constant",
            )
            dictionary[dateString][hourString]["pressure"] = pressureFunction

            # Create function for wind speed levels
            windVelocityXArray = self.__extractPressureLevelDataValue(
                pressureLevelData,
                pressureLevelFileDict["windVelocityX"],
                indices,
                lonArray,
                latArray,
            )
            windVelocityYArray = self.__extractPressureLevelDataValue(
                pressureLevelData,
                pressureLevelFileDict["windVelocityY"],
                indices,
                lonArray,
                latArray,
            )
            windSpeedArray = np.sqrt(
                np.square(windVelocityXArray) + np.square(windVelocityYArray)
            )

            windSpeedPointsArray = np.array(
                [heightAboveSeaLevelArray, windSpeedArray]
            ).T
            windSpeedFunction = Function(
                windSpeedPointsArray,
                inputs="Height Above Sea Level (m)",
                outputs="Wind Speed (m/s)",
                extrapolation="constant",
            )
            dictionary[dateString][hourString]["windSpeed"] = windSpeedFunction

            # Create function for wind heading levels
            windHeadingArray = (
                np.arctan2(windVelocityXArray, windVelocityYArray) * (180 / np.pi) % 360
            )

            windHeadingPointsArray = np.array(
                [heightAboveSeaLevelArray, windHeadingArray]
            ).T
            windHeadingFunction = Function(
                windHeadingPointsArray,
                inputs="Height Above Sea Level (m)",
                outputs="Wind Heading (Deg True)",
                extrapolation="constant",
            )
            dictionary[dateString][hourString]["windHeading"] = windHeadingFunction

            # Create function for wind direction levels
            windDirectionArray = (windHeadingArray - 180) % 360
            windDirectionPointsArray = np.array(
                [heightAboveSeaLevelArray, windDirectionArray]
            ).T
            windDirectionFunction = Function(
                windDirectionPointsArray,
                inputs="Height Above Sea Level (m)",
                outputs="Wind Direction (Deg True)",
                extrapolation="constant",
            )
            dictionary[dateString][hourString]["windDirection"] = windDirectionFunction

        return (dictionary, lat0, lat1, lon0, lon1)

    @cached_property
    def original_pressure_level_data(self):
        """Return the pressure level data dictionary."""
        return self.__parsePressureLevelData[0]

    @cached_property
    def pressureLevelInitLat(self):
        """Return the initial latitude of the pressure level data."""
        return self.__parsePressureLevelData[1]

    @cached_property
    def pressureLevelEndLat(self):
        """Return the final latitude of the pressure level data."""
        return self.__parsePressureLevelData[2]

    @cached_property
    def pressureLevelInitLon(self):
        """Return the initial longitude of the pressure level data."""
        return self.__parsePressureLevelData[3]

    @cached_property
    def pressureLevelEndLon(self):
        """Return the final longitude of the pressure level data."""
        return self.__parsePressureLevelData[4]

    @cached_property
    def __parseSurfaceData(self):
        """
        Parse surface data from a weather file.
        Currently only supports files from ECMWF.
        You can download a file from the following website: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

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

        Return a dictionary with all the computed data with the following structure:
        surfaceDataDict: {
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
        surfaceFileDict = self.__init_surface_dictionary()

        # Read weather file
        surfaceData = netCDF4.Dataset(self.surfaceDataFile)

        # Get time, latitude and longitude data from file
        timeNumArray = surfaceData.variables["time"]
        lonArray = surfaceData.variables["longitude"]
        latArray = surfaceData.variables["latitude"]
        # Determine latitude and longitude range for surface level file
        lat0 = latArray[0]
        lat1 = latArray[-1]
        lon0 = lonArray[0]
        lon1 = lonArray[-1]

        # Find index needed for latitude and longitude for specified location
        lonIndex = self.__getNearestIndex(lonArray, self.longitude)
        latIndex = self.__getNearestIndex(latArray, self.latitude)

        # Can't handle lat and lon out of grid
        self.__check_coordinates_inside_grid(lonIndex, latIndex, lonArray, latArray)

        # Loop through time and save all values
        for timeIndex, timeNum in enumerate(timeNumArray):
            dateString, hourString, dateTime = time_num_to_date_string(
                timeNum,
                timeNumArray.units,
                self.preferred_timezone,
                calendar="gregorian",
            )

            # Check if date is within analysis range
            if not (self.start_date <= dateTime < self.end_date):
                continue
            if not (self.start_hour <= dateTime.hour < self.end_hour):
                continue

            # Make sure keys exist
            if dateString not in dictionary:
                dictionary[dateString] = {}
            if hourString not in dictionary[dateString]:
                dictionary[dateString][hourString] = {}

            # Extract data from weather file
            indices = (timeIndex, lonIndex, latIndex)
            for key, value in surfaceFileDict.items():
                dictionary[dateString][hourString][
                    key
                ] = self.__extractSurfaceDataValue(
                    surfaceData, value, indices, lonArray, latArray
                )

        # Get elevation, time index does not matter, use last one
        surface_geopotential = self.__extractSurfaceDataValue(
            surfaceData, "z", indices, lonArray, latArray
        )
        elevation = geopotential_to_height_asl(surface_geopotential)

        return dictionary, lat0, lat1, lon0, lon1, elevation

    @cached_property
    def original_surface_data(self):
        """Return the surface data dictionary."""
        return self.__parseSurfaceData[0]

    @cached_property
    def original_elevation(self):
        """Return the elevation of the surface data."""
        return self.__parseSurfaceData[5]

    @cached_property
    def singleLevelInitLat(self):
        """Return the initial latitude of the surface data."""
        return self.__parseSurfaceData[1]

    @cached_property
    def singleLevelEndLat(self):
        """Return the final latitude of the surface data."""
        return self.__parseSurfaceData[2]

    @cached_property
    def singleLevelInitLon(self):
        """Return the initial longitude of the surface data."""
        return self.__parseSurfaceData[3]

    @cached_property
    def singleLevelEndLon(self):
        """Return the final longitude of the surface data."""
        return self.__parseSurfaceData[4]

    @cached_property
    def converted_pressure_level_data(self):
        """Convert pressure level data to desired unit system. This method will
        loop through all the data (dates, hours and variables) and convert
        the units of each variable. The performance of this method is not
        optimal.
        """
        # Create conversion dict (key: to_unit)
        conversion_dict = {
            "pressure": self.unit_system["pressure"],
            "temperature": self.unit_system["temperature"],
            "windDirection": self.unit_system["angle"],
            "windHeading": self.unit_system["angle"],
            "windSpeed": self.unit_system["wind_speed"],
            "windVelocityX": self.unit_system["wind_speed"],
            "windVelocityY": self.unit_system["wind_speed"],
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
        """Convert surface data to desired unit system."""
        # Create conversion dict (key: from_unit, to_unit)
        conversion_dict = {
            "surface100mWindVelocityX": self.unit_system["wind_speed"],
            "surface100mWindVelocityY": self.unit_system["wind_speed"],
            "surface10mWindVelocityX": self.unit_system["wind_speed"],
            "surface10mWindVelocityY": self.unit_system["wind_speed"],
            "surfaceTemperature": self.unit_system["temperature"],
            "cloudBaseHeight": self.unit_system["length"],
            "surfaceWindGust": self.unit_system["wind_speed"],
            "surfacePressure": self.unit_system["pressure"],
            "totalPrecipitation": self.unit_system["precipitation"],
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
    def converted_elevation(self):
        return convert_units(
            self.original_elevation,
            self.current_units["height_ASL"],
            self.unit_system["length"],
        )

    @cached_property
    def cloud_base_height(self):
        cloud_base_height = [
            dayDict[hour]["cloudBaseHeight"]
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
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
        return [
            dayDict[hour]["surfacePressure"]
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def average_surface_pressure(self):
        return np.average(self.pressure_at_surface_list)

    @cached_property
    def std_surface_pressure(self):
        return np.std(self.pressure_at_surface_list)

    @cached_property
    def pressure_at_1000ft_list(self):
        # Pressure at 1000 feet
        return [
            dayDict[hour]["pressure"](
                convert_units(1000, "ft", self.current_units["height_ASL"])
            )
            for dayDict in self.original_pressure_level_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def average_pressure_at_1000ft(self):
        return np.average(self.pressure_at_1000ft_list)

    @cached_property
    def std_pressure_at_1000ft(self):
        return np.std(self.pressure_at_1000ft_list)

    @cached_property
    def pressure_at_10000ft_list(self):
        # Pressure at 10000 feet
        return [
            dayDict[hour]["pressure"](
                convert_units(10000, "ft", self.current_units["height_ASL"])
            )
            for dayDict in self.original_pressure_level_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def average_pressure_at_10000ft(self):
        return np.average(self.pressure_at_10000ft_list)

    @cached_property
    def std_pressure_at_10000ft(self):
        return np.std(self.pressure_at_10000ft_list)

    @cached_property
    def pressure_at_30000ft_list(self):
        # Pressure at 30000 feet
        return [
            dayDict[hour]["pressure"](
                convert_units(30000, "ft", self.current_units["height_ASL"])
            )
            for dayDict in self.original_pressure_level_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def average_pressure_at_30000ft(self):
        return np.average(self.pressure_at_30000ft_list)

    @cached_property
    def std_pressure_at_30000ft(self):
        return np.std(self.pressure_at_30000ft_list)

    @cached_property
    def average_cloud_base_height(self):
        """Calculate average cloud base height."""
        return np.ma.mean(self.cloud_base_height)

    @cached_property
    def record_min_cloud_base_height(self):
        """Calculate minium cloud base height."""
        return np.ma.min(self.cloud_base_height, fill_value=np.inf)

    @cached_property
    def percentage_of_days_with_no_cloud_coverage(self):
        """Calculate percentage of days with cloud coverage."""
        return np.ma.count(self.cloud_base_height) / len(self.cloud_base_height)

    @cached_property
    def precipitation_per_day(self):
        return [
            sum([dayDict[hour]["totalPrecipitation"] for hour in dayDict.keys()])
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def percentage_of_days_with_precipitation(self):
        """Computes the ratio between days with precipitation (> 10 mm) and total days."""
        days_with_precipitation_count = 0
        for precipitation in self.precipitation_per_day:
            if precipitation > convert_units(
                10, "mm", self.unit_system["precipitation"]
            ):
                days_with_precipitation_count += 1

        return days_with_precipitation_count / len(self.precipitation_per_day)

    @cached_property
    def temperature_list(self):
        return [
            dayDict[hour]["surfaceTemperature"]
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def max_temperature_list(self):
        return [
            np.max([dayDict[hour]["surfaceTemperature"] for hour in dayDict.keys()])
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def min_temperature_list(self):
        return [
            np.min([dayDict[hour]["surfaceTemperature"] for hour in dayDict.keys()])
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def average_max_temperature(self):
        return np.average(self.max_temperature_list)

    @cached_property
    def average_min_temperature(self):
        return np.average(self.min_temperature_list)

    @cached_property
    def record_max_temperature(self):
        return np.max(self.temperature_list)

    @cached_property
    def record_min_temperature(self):
        return np.min(self.temperature_list)

    @cached_property
    def wind_gust_list(self):
        return [
            dayDict[hour]["surfaceWindGust"]
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def max_wind_gust_list(self):
        return [
            np.max([dayDict[hour]["surfaceWindGust"] for hour in dayDict.keys()])
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def average_max_wind_gust(self):
        return np.average(self.max_wind_gust_list)

    @cached_property
    def record_max_wind_gust(self):
        return np.max(self.wind_gust_list)

    @cached_property
    def surface_10m_wind_speed_list(self):
        surface_10m_wind_speed_list = [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
        ]
        return surface_10m_wind_speed_list

    @cached_property
    def record_max_surface_10m_wind_speed(self):
        return np.max(self.surface_10m_wind_speed_list)

    @cached_property
    def max_surface_10m_wind_speed_list(self):
        return [
            np.max(
                [
                    (
                        dayDict[hour]["surface10mWindVelocityX"] ** 2
                        + dayDict[hour]["surface10mWindVelocityY"] ** 2
                    )
                    ** 0.5
                    for hour in dayDict.keys()
                ]
            )
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def average_max_surface_10m_wind_speed(self):
        return np.average(self.max_surface_10m_wind_speed_list)

    @cached_property
    def min_surface_10m_wind_speed_list(self):
        return [
            np.min(
                [
                    (
                        dayDict[hour]["surface10mWindVelocityX"] ** 2
                        + dayDict[hour]["surface10mWindVelocityY"] ** 2
                    )
                    ** 0.5
                    for hour in dayDict.keys()
                ]
            )
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def average_min_surface_10m_wind_speed(self):
        return np.average(self.min_surface_10m_wind_speed_list)

    @cached_property
    def surface_10m_wind_speed(self):
        return [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
        ]

    @cached_property
    def record_max_surface_10m_wind_speed(self):
        return np.max(self.surface_10m_wind_speed)

    @cached_property
    def record_min_surface_10m_wind_speed(self):
        return np.min(self.surface_10m_wind_speed)

    @cached_property
    def max_surface_100m_wind_speed_list(self):
        return [
            np.max(
                [
                    (
                        dayDict[hour]["surface100mWindVelocityX"] ** 2
                        + dayDict[hour]["surface100mWindVelocityY"] ** 2
                    )
                    ** 0.5
                    for hour in dayDict.keys()
                ]
            )
            for dayDict in self.converted_surface_data.values()
        ]

    @cached_property
    def average_max_surface_100m_wind_speed(self):
        return np.average(self.max_surface_100m_wind_speed_list)

    @cached_property
    def min_surface_100m_wind_speed_list(self):
        min_surface_100m_wind_speed_list = [
            np.min(
                [
                    (
                        dayDict[hour]["surface100mWindVelocityX"] ** 2
                        + dayDict[hour]["surface100mWindVelocityY"] ** 2
                    )
                    ** 0.5
                    for hour in dayDict.keys()
                ]
            )
            for dayDict in self.converted_surface_data.values()
        ]
        return min_surface_100m_wind_speed_list

    @cached_property
    def average_min_surface_100m_wind_speed(self):
        return np.average(self.min_surface_100m_wind_speed_list)

    @cached_property
    def surface_100m_wind_speed(self):
        surface_100m_wind_speed = [
            (
                dayDict[hour]["surface100mWindVelocityX"] ** 2
                + dayDict[hour]["surface100mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.converted_surface_data.values()
            for hour in dayDict.keys()
        ]
        return surface_100m_wind_speed

    @cached_property
    def record_max_surface_100m_wind_speed(self):
        return np.max(self.surface_100m_wind_speed)

    @cached_property
    def record_min_surface_100m_wind_speed(self):
        return np.min(self.surface_100m_wind_speed)

    @cached_property
    def historical_temperatures_each_hour(self):
        history = defaultdict(dict)
        for date, val in self.converted_surface_data.items():
            for hour, sub_val in val.items():
                history[hour][date] = sub_val["surfaceTemperature"]
        return history

    @cached_property
    def average_temperature_at_given_hour(self):
        # Flip dictionary to get hour as key instead of date
        return {
            hour: np.average(list(dates.values()))
            for hour, dates in self.historical_temperatures_each_hour.items()
        }

    @cached_property
    def average_temperature_sigmas_at_given_hour(self):
        # Flip dictionary to get hour as key instead of date
        return {
            hour: np.std(list(dates.values()))
            for hour, dates in self.historical_temperatures_each_hour.items()
        }

    @cached_property
    def historical_surface10m_wind_speeds_each_hour(self):
        """Computes average sustained wind speed progression throughout the
        day, including sigma contours."""
        # Flip dictionary to get hour as key instead of date
        dictionary = defaultdict(dict)
        for date, val in self.converted_surface_data.items():
            for hour, sub_val in val.items():
                dictionary[hour][date] = (
                    sub_val["surface10mWindVelocityX"] ** 2
                    + sub_val["surface10mWindVelocityY"] ** 2
                ) ** 0.5
        return dictionary

    @cached_property
    def average_surface10m_wind_speed_at_given_hour(self):
        return {
            hour: np.average(list(dates.values()))
            for hour, dates in self.historical_surface10m_wind_speeds_each_hour.items()
        }

    @cached_property
    def average_surface10m_wind_speed_sigmas_at_given_hour(self):
        return {
            hour: np.std(list(dates.values()))
            for hour, dates in self.historical_surface10m_wind_speeds_each_hour.items()
        }

    @cached_property
    def historical_surface100m_wind_speeds_each_hour(self):
        """Computes average sustained wind speed progression throughout the
        day, including sigma contours."""

        # Flip dictionary to get hour as key instead of date
        dictionary = defaultdict(dict)
        for date, val in self.converted_surface_data.items():
            for hour, sub_val in val.items():
                dictionary[hour][date] = (
                    sub_val["surface100mWindVelocityX"] ** 2
                    + sub_val["surface100mWindVelocityY"] ** 2
                ) ** 0.5

        return dictionary

    @cached_property
    def average_surface100m_wind_speed_at_given_hour(self):
        return {
            hour: np.average(list(dates.values()))
            for hour, dates in self.historical_surface100m_wind_speeds_each_hour.items()
        }

    @cached_property
    def average_surface100m_wind_speed_sigmas_at_given_hour(self):
        return {
            hour: np.std(list(dates.values()))
            for hour, dates in self.historical_surface100m_wind_speeds_each_hour.items()
        }

    @cached_property
    def _wind_data_for_average_day(self):
        """Process the wind_speed and wind_direction data to generate lists of all the wind_speeds recorded
        for a following hour of the day and also the wind direction. Also calculates the greater and the smallest
        wind_speed recorded

        Returns
        -------
        None
        """
        max_wind_speed = float("-inf")
        min_wind_speed = float("inf")

        days = list(self.converted_surface_data.keys())
        hours = list(self.converted_surface_data[days[0]].keys())

        windSpeed = {}
        windDir = {}

        for hour in hours:
            windSpeed[hour] = []
            windDir[hour] = []
            for day in days:
                try:
                    hour_wind_speed = (
                        self.converted_surface_data[day][hour][
                            "surface10mWindVelocityX"
                        ]
                        ** 2
                        + self.converted_surface_data[day][hour][
                            "surface10mWindVelocityY"
                        ]
                        ** 2
                    ) ** 0.5

                    max_wind_speed = (
                        hour_wind_speed
                        if hour_wind_speed > max_wind_speed
                        else max_wind_speed
                    )
                    min_wind_speed = (
                        hour_wind_speed
                        if hour_wind_speed < min_wind_speed
                        else min_wind_speed
                    )

                    windSpeed[hour].append(hour_wind_speed)
                    # Wind direction means where the wind is blowing from, 180 deg opposite from wind heading
                    vx = self.converted_surface_data[day][hour][
                        "surface10mWindVelocityX"
                    ]
                    vy = self.converted_surface_data[day][hour][
                        "surface10mWindVelocityY"
                    ]
                    windDir[hour].append(
                        (180 + (np.arctan2(vy, vx) * 180 / np.pi)) % 360
                    )
                except KeyError:
                    # Not all days have all hours stored, that is fine
                    pass

        return max_wind_speed, min_wind_speed, windSpeed, windDir

    @cached_property
    def max_wind_speed(self):
        return self._wind_data_for_average_day[0]

    @cached_property
    def min_wind_speed(self):
        return self._wind_data_for_average_day[1]

    @cached_property
    def wind_speed(self):
        return self._wind_data_for_average_day[2]

    @cached_property
    def wind_direction(self):
        return self._wind_data_for_average_day[3]

    @property
    def altitude_AGL_range(self):
        min_altitude = 0
        if self.maxExpectedAltitude == None:
            max_altitudes = [
                np.max(dayDict[hour]["windSpeed"].source[-1, 0])
                for dayDict in self.original_pressure_level_data.values()
                for hour in dayDict.keys()
            ]
            max_altitude = np.min(max_altitudes)
        else:
            max_altitude = self.maxExpectedAltitude
        return min_altitude, max_altitude

    @cached_property
    def _temperature_profile_over_average_day(self):
        """Compute the average temperature profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_temperature_profile_at_given_hour = {}
        max_average_temperature_at_altitude = 0
        hours = list(self.original_pressure_level_data.values())[0].keys()
        for hour in hours:
            temperature_values_for_this_hour = []
            for dayDict in self.original_pressure_level_data.values():
                try:
                    temperature_values_for_this_hour += [
                        dayDict[hour]["temperature"](altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            mean_temperature_values_for_this_hour = np.mean(
                temperature_values_for_this_hour, axis=0
            )
            average_temperature_profile_at_given_hour[hour] = [
                mean_temperature_values_for_this_hour,
                altitude_list,
            ]
            max_temperature = np.max(mean_temperature_values_for_this_hour)
            if max_temperature >= max_average_temperature_at_altitude:
                max_average_temperature_at_altitude = max_temperature

        return (
            average_temperature_profile_at_given_hour,
            max_average_temperature_at_altitude,
        )

    @cached_property
    def average_temperature_profile_at_given_hour(self):
        return self._temperature_profile_over_average_day[0]

    @cached_property
    def max_average_temperature_at_altitude(self):
        return self._temperature_profile_over_average_day[1]

    @cached_property
    def _pressure_profile_over_average_day(self):
        """Compute the average pressure profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_pressure_profile_at_given_hour = {}
        max_average_pressure_at_altitude = 0
        hours = list(self.original_pressure_level_data.values())[0].keys()
        for hour in hours:
            pressure_values_for_this_hour = []
            for dayDict in self.original_pressure_level_data.values():
                try:
                    pressure_values_for_this_hour += [
                        dayDict[hour]["pressure"](altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            mean_pressure_values_for_this_hour = np.mean(
                pressure_values_for_this_hour, axis=0
            )
            average_pressure_profile_at_given_hour[hour] = [
                mean_pressure_values_for_this_hour,
                altitude_list,
            ]
            max_pressure = np.max(mean_pressure_values_for_this_hour)
            if max_pressure >= max_average_pressure_at_altitude:
                max_average_pressure_at_altitude = max_pressure

        return average_pressure_profile_at_given_hour, max_average_pressure_at_altitude

    @cached_property
    def average_pressure_profile_at_given_hour(self):
        return self._pressure_profile_over_average_day[0]

    @cached_property
    def max_average_pressure_at_altitude(self):
        return self._pressure_profile_over_average_day[1]

    @cached_property
    def _wind_speed_profile_over_average_day(self):
        """Compute the average wind profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_wind_profile_at_given_hour = {}
        max_average_wind_at_altitude = 0
        hours = list(self.converted_pressure_level_data.values())[0].keys()

        # days = list(self.converted_surface_data.keys())
        # hours = list(self.converted_surface_data[days[0]].keys())
        for hour in hours:
            wind_speed_values_for_this_hour = []
            for dayDict in self.converted_pressure_level_data.values():
                try:
                    wind_speed_values_for_this_hour += [
                        dayDict[hour]["windSpeed"](altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            mean_wind_speed_values_for_this_hour = np.mean(
                wind_speed_values_for_this_hour, axis=0
            )
            average_wind_profile_at_given_hour[hour] = [
                mean_wind_speed_values_for_this_hour,
                altitude_list,
            ]
            max_wind = np.max(mean_wind_speed_values_for_this_hour)
            if max_wind >= max_average_wind_at_altitude:
                max_average_wind_at_altitude = max_wind

        return average_wind_profile_at_given_hour, max_average_wind_at_altitude

    @cached_property
    def average_wind_profile_at_given_hour(self):
        return self._wind_speed_profile_over_average_day[0]

    @cached_property
    def max_average_wind_at_altitude(self):
        return self._wind_speed_profile_over_average_day[1]

    @cached_property
    def _wind_velocity_x_profile_over_average_day(self):
        """Compute the average windVelocityX profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_windVelocityX_profile_at_given_hour = {}
        max_average_windVelocityX_at_altitude = 0
        hours = list(self.original_pressure_level_data.values())[0].keys()
        for hour in hours:
            windVelocityX_values_for_this_hour = []
            for dayDict in self.original_pressure_level_data.values():
                try:
                    windVelocityX_values_for_this_hour += [
                        dayDict[hour]["windVelocityX"](altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            mean_windVelocityX_values_for_this_hour = np.mean(
                windVelocityX_values_for_this_hour, axis=0
            )
            average_windVelocityX_profile_at_given_hour[hour] = [
                mean_windVelocityX_values_for_this_hour,
                altitude_list,
            ]
            max_windVelocityX = np.max(mean_windVelocityX_values_for_this_hour)
            if max_windVelocityX >= max_average_windVelocityX_at_altitude:
                max_average_windVelocityX_at_altitude = max_windVelocityX

        return (
            average_windVelocityX_profile_at_given_hour,
            max_average_windVelocityX_at_altitude,
        )

    @cached_property
    def average_windVelocityX_profile_at_given_hour(self):
        return self._wind_velocity_x_profile_over_average_day[0]

    @cached_property
    def max_average_windVelocityX_at_altitude(self):
        return self._wind_velocity_x_profile_over_average_day[1]

    @cached_property
    def _wind_velocity_y_profile_over_average_day(self):
        """Compute the average windVelocityY profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_windVelocityY_profile_at_given_hour = {}
        max_average_windVelocityY_at_altitude = 0
        hours = list(self.original_pressure_level_data.values())[0].keys()
        for hour in hours:
            windVelocityY_values_for_this_hour = []
            for dayDict in self.original_pressure_level_data.values():
                try:
                    windVelocityY_values_for_this_hour += [
                        dayDict[hour]["windVelocityY"](altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            mean_windVelocityY_values_for_this_hour = np.mean(
                windVelocityY_values_for_this_hour, axis=0
            )
            average_windVelocityY_profile_at_given_hour[hour] = [
                mean_windVelocityY_values_for_this_hour,
                altitude_list,
            ]
            max_windVelocityY = np.max(mean_windVelocityY_values_for_this_hour)
            if max_windVelocityY >= max_average_windVelocityY_at_altitude:
                max_average_windVelocityY_at_altitude = max_windVelocityY

        return (
            average_windVelocityY_profile_at_given_hour,
            max_average_windVelocityY_at_altitude,
        )

    @cached_property
    def average_windVelocityY_profile_at_given_hour(self):
        return self._wind_velocity_y_profile_over_average_day[0]

    @cached_property
    def max_average_windVelocityY_at_altitude(self):
        return self._wind_velocity_y_profile_over_average_day[1]

    @cached_property
    def _wind_heading_profile_over_average_day(self):
        """Compute the average wind velocities (both X and Y components) profile
        for each available hour of a day, over all days in the dataset.
        """
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)
        average_wind_velocity_X_profile_at_given_hour = {}
        average_wind_velocity_Y_profile_at_given_hour = {}
        average_wind_heading_profile_at_given_hour = {}
        max_average_wind_velocity_X_at_altitude = 0
        max_average_wind_velocity_Y_at_altitude = 0

        hours = list(self.original_pressure_level_data.values())[0].keys()

        for hour in hours:
            wind_velocity_X_values_for_this_hour = []
            wind_velocity_Y_values_for_this_hour = []
            for dayDict in self.converted_pressure_level_data.values():
                try:
                    wind_velocity_X_values_for_this_hour += [
                        dayDict[hour]["windVelocityX"](altitude_list)
                    ]
                    wind_velocity_Y_values_for_this_hour += [
                        dayDict[hour]["windVelocityY"](altitude_list)
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour
                    # No need to worry, just average over the other days
                    pass
            # Compute the average wind velocity profile for this hour
            mean_wind_velocity_X_values_for_this_hour = np.mean(
                wind_velocity_X_values_for_this_hour, axis=0
            )
            mean_wind_velocity_Y_values_for_this_hour = np.mean(
                wind_velocity_Y_values_for_this_hour, axis=0
            )
            # Store the ... wind velocity at each altitude
            average_wind_velocity_X_profile_at_given_hour[hour] = [
                mean_wind_velocity_X_values_for_this_hour,
                altitude_list,
            ]
            average_wind_velocity_Y_profile_at_given_hour[hour] = [
                mean_wind_velocity_Y_values_for_this_hour,
                altitude_list,
            ]
            average_wind_heading_profile_at_given_hour[hour] = [
                np.arctan2(
                    mean_wind_velocity_X_values_for_this_hour,
                    mean_wind_velocity_Y_values_for_this_hour,
                )
                * (180 / np.pi)
                % 360,
                altitude_list,
            ]
            # Store the maximum wind velocity at each altitude
            max_wind_X = np.max(mean_wind_velocity_X_values_for_this_hour)
            if max_wind_X >= max_average_wind_velocity_X_at_altitude:
                self.max_average_wind_X_at_altitude = max_wind_X
            max_wind_Y = np.max(mean_wind_velocity_Y_values_for_this_hour)
            if max_wind_Y >= max_average_wind_velocity_Y_at_altitude:
                self.max_average_wind_Y_at_altitude = max_wind_Y

        return (
            average_wind_velocity_X_profile_at_given_hour,
            average_wind_velocity_Y_profile_at_given_hour,
            average_wind_heading_profile_at_given_hour,
            max_average_wind_velocity_X_at_altitude,
        )

    @cached_property
    def average_wind_velocity_X_profile_at_given_hour(self):
        return self._wind_heading_profile_over_average_day[0]

    @cached_property
    def average_wind_velocity_Y_profile_at_given_hour(self):
        return self._wind_heading_profile_over_average_day[1]

    @cached_property
    def average_wind_heading_profile_at_given_hour(self):
        return self._wind_heading_profile_over_average_day[2]

    @cached_property
    def max_average_wind_velocity_X_at_altitude(self):
        return self._wind_heading_profile_over_average_day[3]

    @cached_property
    def max_average_wind_velocity_Y_at_altitude(self):
        return self._wind_heading_profile_over_average_day[4]

    def info(self):
        """Prints out the most important data and graphs available about the
        Environment Analysis.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.prints.all()
        self.plots.info()
        return None

    def allInfo(self):
        """Prints out all data and graphs available.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.prints.all()
        self.plots.all()

        return None

    def exportMeanProfiles(self, filename="export_env_analysis"):
        """
        Exports the mean profiles of the weather data to a file in order to it
        be used as inputs on Environment Class by using the CustomAtmosphere
        model.

        Parameters
        ----------
        filename : str, optional
            Name of the file where to be saved, by default "EnvAnalysisDict"

        Returns
        -------
        None
        """

        organized_temperature_dict = {}
        organized_pressure_dict = {}
        organized_windX_dict = {}
        organized_windY_dict = {}

        for hour in self.average_temperature_profile_at_given_hour.keys():
            organized_temperature_dict[hour] = np.column_stack(
                (
                    self.average_temperature_profile_at_given_hour[hour][1],
                    self.average_temperature_profile_at_given_hour[hour][0],
                )
            ).tolist()
            organized_pressure_dict[hour] = np.column_stack(
                (
                    self.average_pressure_profile_at_given_hour[hour][1],
                    self.average_pressure_profile_at_given_hour[hour][0],
                )
            ).tolist()
            organized_windX_dict[hour] = np.column_stack(
                (
                    self.average_windVelocityX_profile_at_given_hour[hour][1],
                    self.average_windVelocityX_profile_at_given_hour[hour][0],
                )
            ).tolist()
            organized_windY_dict[hour] = np.column_stack(
                (
                    self.average_windVelocityY_profile_at_given_hour[hour][1],
                    self.average_windVelocityY_profile_at_given_hour[hour][0],
                )
            ).tolist()

        self.exportEnvAnalDict = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.converted_elevation,
            "timeZone": self.preferred_timezone,
            "unit_system": self.unit_system,
            "surfaceDataFile": self.surfaceDataFile,
            "pressureLevelDataFile": self.pressureLevelDataFile,
            "atmosphericModelPressureProfile": organized_pressure_dict,
            "atmosphericModelTemperatureProfile": organized_temperature_dict,
            "atmosphericModelWindVelocityXProfile": organized_windX_dict,
            "atmosphericModelWindVelocityYProfile": organized_windY_dict,
        }

        # Convert to json
        f = open(filename + ".json", "w")

        # write json object to file
        f.write(
            json.dumps(self.exportEnvAnalDict, sort_keys=False, indent=4, default=str)
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
    def load(self, filename="EnvAnalysisDict"):
        """Load a previously saved Environment Analysis file.
        Example: EnvA = EnvironmentAnalysis.load("filename").

        Parameters
        ----------
        filename : str, optional
            Name of the previous saved file, by default "EnvAnalysisDict"

        Returns
        -------
        EnvironmentAnalysis object

        """
        encoded_class = open(filename).read()
        return jsonpickle.decode(encoded_class)

    def save(self, filename="EnvAnalysisDict"):
        """Save the Environment Analysis object to a file so it can be used later.

        Parameters
        ----------
        filename : str, optional
            Name of the file where to be saved, by default "EnvAnalysisDict"

        Returns
        -------
        None
        """
        encoded_class = jsonpickle.encode(self)
        file = open(filename, "w")
        file.write(encoded_class)
        file.close()
        print("Your Environment Analysis file was saved, check it out: " + filename)

        return None
