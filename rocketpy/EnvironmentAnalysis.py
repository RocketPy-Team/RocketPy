# -*- coding: utf-8 -*-

__author__ = "Patrick Sampaio, Giovani Hidalgo Ceotto, Guilherme Fernandes Alves, Franz Masatoshi Yuri, Mateus Stano Junqueira,"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import bisect
import datetime
import json
import warnings
from collections import defaultdict

import jsonpickle
import netCDF4
import numpy as np
import pytz
from cftime import num2pydate

from rocketpy.Environment import Environment
from rocketpy.Function import Function
from rocketpy.units import convert_units

from .plots.environment_analysis_plots import _EnvironmentAnalysisPlots
from .prints.environment_analysis_prints import _EnvironmentAnalysisPrints


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

    Remaining TODOs:
    - Make 'windSpeedLimit' a dynamic/flexible variable
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

        # Parse data files, surface goes first to calculate elevation
        self.surfaceDataDict = {}
        self.parseSurfaceData()
        self.pressureLevelDataDict = {}
        self.parsePressureLevelData()

        # Convert units
        self.set_unit_system(unit_system)

        # Initialize result variables
        self.average_max_temperature = 0
        self.average_min_temperature = 0
        self.record_max_temperature = 0
        self.record_min_temperature = 0
        self.average_max_wind_gust = 0
        self.maximum_wind_gust = 0
        self.wind_gust_distribution = None
        self.average_temperature_along_day = Function(0)
        self.average_wind_profile = Function(0)
        self.average_wind_profile_at_given_hour = None
        self.average_wind_heading_profile = Function(0)
        self.average_wind_heading_profile_at_given_hour = Function(0)

        self.max_wind_speed = None
        self.min_wind_speed = None
        self.wind_speed_per_hour = None
        self.wind_direction_per_hour = None

        # Initialize plots and prints object
        self.plots = _EnvironmentAnalysisPlots(self)
        self.prints = _EnvironmentAnalysisPrints(self)

        # Run calculations
        self.process_data()

        # Processing forecast
        self.forecast = None
        if forecast_date:
            self.forecast = {}
            hours = list(self.pressureLevelDataDict.values())[0].keys()
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
                    elevation=self.elevation,
                )
                forecast_args = forecast_args or {"type": "Forecast", "file": "GFS"}
                Env.setAtmosphericModel(**forecast_args)
                self.forecast[hour] = Env
        return None

    def __bilinear_interpolation(self, x, y, x1, x2, y1, y2, z11, z12, z21, z22):
        """
        Bilinear interpolation.

        Source: GitHub Copilot
        """
        return (
            z11 * (x2 - x) * (y2 - y)
            + z21 * (x - x1) * (y2 - y)
            + z12 * (x2 - x) * (y - y1)
            + z22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

    def __init_surface_dictionary(self):
        # Create dictionary of file variable names to process surface data
        self.surfaceFileDict = {
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
        self.pressureLevelFileDict = {
            "geopotential": "z",
            "windVelocityX": "u",
            "windVelocityY": "v",
            "temperature": "t",
        }

    def __getNearestIndex(self, array, value):
        """Find nearest index of the given value in the array.
        Made for latitudes and longitudes, supporting arrays that range from
        -180 to 180 or from 0 to 360.

        TODO: improve docs by giving one example

        Parameters
        ----------
        array : array
        value : float

        Returns
        -------
        index : int
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

    def __timeNumToDateString(self, timeNum, units, calendar="gregorian"):
        """Convert time number (usually hours before a certain date) into two
        strings: one for the date (example: 2022.04.31) and one for the hour
        (example: 14). See cftime.num2date for details on units and calendar.
        Automatically converts time number from UTC to local timezone based on
        lat,lon coordinates.
        """
        dateTimeUTC = num2pydate(timeNum, units, calendar=calendar)
        dateTimeUTC = dateTimeUTC.replace(tzinfo=pytz.UTC)
        dateTime = dateTimeUTC.astimezone(self.preferred_timezone)
        dateString = f"{dateTime.year}.{dateTime.month}.{dateTime.day}"
        hourString = f"{dateTime.hour}"
        return dateString, hourString, dateTime

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
        value = self.__bilinear_interpolation(
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
        value_list_as_a_function_of_pressure_level = self.__bilinear_interpolation(
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

    def __compute_height_above_sea_level(self, geopotential):
        """Compute height above sea level from geopotential.

        Source: https://en.wikipedia.org/wiki/Geopotential
        """
        R = 63781370  # Earth radius in m
        g = 9.80665  # Gravity acceleration in m/s^2
        geopotential_height = geopotential / g
        return R * geopotential_height / (R - geopotential_height)

    def __compute_height_above_ground_level(self, geopotential, elevation):
        """Compute height above ground level from geopotential and elevation."""
        return self.__compute_height_above_sea_level(geopotential) - elevation

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

    def set_unit_system(self, unit_system="metric"):
        self.unit_system_string = unit_system
        self.__init_unit_system()
        self.convertPressureLevelData()
        self.convertSurfaceData()
        self.current_units = self.updated_units.copy()

    @staticmethod
    def _find_two_closest_integer_factors(number):
        """Find the two closest integer factors of a number.

        Parameters
        ----------
        number: int

        Returns
        -------
        list[int]
        """
        number_sqrt = number**0.5
        if isinstance(number_sqrt, int):
            return number_sqrt, number_sqrt
        else:
            guess = int(number_sqrt)
            while True:
                if number % guess == 0:
                    return guess, number // guess
                else:
                    guess -= 1

    def _beaufort_wind_scale(self, units, max_wind_speed=None):
        """Returns a list of bins equivalent to the Beaufort wind scale in the
        desired unit system.

        Parameters
        ----------
        units: str
            Desired units for wind speed.
            Options are: "knot", "mph", "m/s", "ft/s: and "km/h".
        max_wind_speed: float
            Maximum wind speed to be included in the scale. Should be expressed
            in the same unit as the units parameter.

        Returns
        -------
        list[float]
        """
        beaufort_wind_scale_knots = np.array(
            [0, 1, 3, 6, 10, 16, 21, 27, 33, 40, 47, 55, 63, 71]
        )
        beaufort_wind_scale = beaufort_wind_scale_knots * convert_units(
            1, "knot", units
        )
        beaufort_wind_scale_truncated = beaufort_wind_scale[
            np.where(beaufort_wind_scale <= max_wind_speed)
        ]
        if beaufort_wind_scale[1] < 1:
            return np.round(beaufort_wind_scale_truncated, 1)
        else:
            return np.round(beaufort_wind_scale_truncated, 0)

    def parsePressureLevelData(self):
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
        """
        # Setup dictionary used to read weather file
        self.__init_pressure_level_dictionary()
        # Read weather file
        pressureLevelData = netCDF4.Dataset(self.pressureLevelDataFile)

        # Get time, pressure levels, latitude and longitude data from file
        timeNumArray = pressureLevelData.variables["time"]
        pressureLevelArray = pressureLevelData.variables["level"]
        lonArray = pressureLevelData.variables["longitude"]
        latArray = pressureLevelData.variables["latitude"]
        # Determine latitude and longitude range for pressure level file
        self.pressureLevelInitLat = latArray[0]
        self.pressureLevelEndLat = latArray[-1]
        self.pressureLevelInitLon = lonArray[0]
        self.pressureLevelEndLon = lonArray[-1]

        # Find index needed for latitude and longitude for specified location
        lonIndex = self.__getNearestIndex(lonArray, self.longitude)
        latIndex = self.__getNearestIndex(latArray, self.latitude)

        # Can't handle lat and lon out of grid
        self.__check_coordinates_inside_grid(lonIndex, latIndex, lonArray, latArray)

        # Loop through time and save all values
        for timeIndex, timeNum in enumerate(timeNumArray):
            dateString, hourString, dateTime = self.__timeNumToDateString(
                timeNum, timeNumArray.units, calendar="gregorian"
            )

            # Check if date is within analysis range
            if not (self.start_date <= dateTime < self.end_date):
                continue
            if not (self.start_hour <= dateTime.hour < self.end_hour):
                continue
            # Make sure keys exist
            if dateString not in self.pressureLevelDataDict:
                self.pressureLevelDataDict[dateString] = {}
            if hourString not in self.pressureLevelDataDict[dateString]:
                self.pressureLevelDataDict[dateString][hourString] = {}

            # Extract data from weather file
            indices = (timeIndex, lonIndex, latIndex)

            # Retrieve geopotential first and compute altitudes
            geopotentialArray = self.__extractPressureLevelDataValue(
                pressureLevelData,
                self.pressureLevelFileDict["geopotential"],
                indices,
                lonArray,
                latArray,
            )
            heightAboveSeaLevelArray = self.__compute_height_above_ground_level(
                geopotentialArray, self.elevation
            )

            # Loop through wind components and temperature, get value and convert to Function
            for key, value in self.pressureLevelFileDict.items():
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
                self.pressureLevelDataDict[dateString][hourString][
                    key
                ] = variableFunction

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
            self.pressureLevelDataDict[dateString][hourString][
                "pressure"
            ] = pressureFunction

            # Create function for wind speed levels
            windVelocityXArray = self.__extractPressureLevelDataValue(
                pressureLevelData,
                self.pressureLevelFileDict["windVelocityX"],
                indices,
                lonArray,
                latArray,
            )
            windVelocityYArray = self.__extractPressureLevelDataValue(
                pressureLevelData,
                self.pressureLevelFileDict["windVelocityY"],
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
            self.pressureLevelDataDict[dateString][hourString][
                "windSpeed"
            ] = windSpeedFunction

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
            self.pressureLevelDataDict[dateString][hourString][
                "windHeading"
            ] = windHeadingFunction

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
            self.pressureLevelDataDict[dateString][hourString][
                "windDirection"
            ] = windDirectionFunction

        return self.pressureLevelDataDict

    def parseSurfaceData(self):
        """
        Parse surface data from a weather file.
        Currently only supports files from ECMWF.
        You can download a file from the following website: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

        Must get the following variables:
        - surface elevation: self.elevation = float  # Select 'Geopotential'
        - 2m temperature: surfaceTemperature = float
        - Surface pressure: surfacePressure = float
        - 10m u-component of wind: surface10mWindVelocityX = float
        - 10m v-component of wind: surface10mWindVelocityY = float
        - 100m u-component of wind: surface100mWindVelocityX = float
        - 100m V-component of wind: surface100mWindVelocityY = float
        - Instantaneous 10m wind gust: surfaceWindGust = float
        - Total precipitation: totalPrecipitation = float
        - Cloud base height: cloudBaseHeight = float

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
        self.__init_surface_dictionary()

        # Read weather file
        surfaceData = netCDF4.Dataset(self.surfaceDataFile)

        # Get time, latitude and longitude data from file
        timeNumArray = surfaceData.variables["time"]
        lonArray = surfaceData.variables["longitude"]
        latArray = surfaceData.variables["latitude"]
        # Determine latitude and longitude range for surface level file
        self.singleLevelInitLat = latArray[0]
        self.singleLevelEndLat = latArray[-1]
        self.singleLevelInitLon = lonArray[0]
        self.singleLevelEndLon = lonArray[-1]

        # Find index needed for latitude and longitude for specified location
        lonIndex = self.__getNearestIndex(lonArray, self.longitude)
        latIndex = self.__getNearestIndex(latArray, self.latitude)

        # Can't handle lat and lon out of grid
        self.__check_coordinates_inside_grid(lonIndex, latIndex, lonArray, latArray)

        # Loop through time and save all values
        for timeIndex, timeNum in enumerate(timeNumArray):
            dateString, hourString, dateTime = self.__timeNumToDateString(
                timeNum, timeNumArray.units, calendar="gregorian"
            )

            # Check if date is within analysis range
            if not (self.start_date <= dateTime < self.end_date):
                continue
            if not (self.start_hour <= dateTime.hour < self.end_hour):
                continue

            # Make sure keys exist
            if dateString not in self.surfaceDataDict:
                self.surfaceDataDict[dateString] = {}
            if hourString not in self.surfaceDataDict[dateString]:
                self.surfaceDataDict[dateString][hourString] = {}

            # Extract data from weather file
            indices = (timeIndex, lonIndex, latIndex)
            for key, value in self.surfaceFileDict.items():
                self.surfaceDataDict[dateString][hourString][
                    key
                ] = self.__extractSurfaceDataValue(
                    surfaceData, value, indices, lonArray, latArray
                )

        # Get elevation, time index does not matter, use last one
        self.surface_geopotential = self.__extractSurfaceDataValue(
            surfaceData, "z", indices, lonArray, latArray
        )
        self.elevation = self.__compute_height_above_sea_level(
            self.surface_geopotential
        )

        return self.surfaceDataDict

    def convertPressureLevelData(self):
        """Convert pressure level data to desired unit system."""
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
        # Loop through dates
        for date in self.pressureLevelDataDict:
            for hour in self.pressureLevelDataDict[date]:
                for key, value in self.pressureLevelDataDict[date][hour].items():
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
                    self.pressureLevelDataDict[date][hour][key] = variable

    def convertSurfaceData(self):
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
        # Loop through dates
        for date in self.surfaceDataDict:
            for hour in self.surfaceDataDict[date]:
                for key, value in self.surfaceDataDict[date][hour].items():
                    variable = convert_units(
                        variable=value,
                        from_unit=self.current_units[key],
                        to_unit=conversion_dict[key],
                    )
                    self.surfaceDataDict[date][hour][key] = variable
                    # Update current units
                    self.updated_units[key] = conversion_dict[key]

        # Convert surface elevation
        self.elevation = convert_units(
            self.elevation, self.current_units["height_ASL"], self.unit_system["length"]
        )
        self.updated_units["height_ASL"] = self.unit_system["length"]

    # Calculations
    def process_data(self):
        """Process data that is shown in the allInfo method."""
        self.calculate_pressure_stats()
        self.calculate_average_max_temperature()
        self.calculate_average_min_temperature()
        self.calculate_record_max_temperature()
        self.calculate_record_min_temperature()
        self.calculate_average_max_wind_gust()
        self.calculate_maximum_wind_gust()
        self.calculate_maximum_surface_10m_wind_speed()
        self.calculate_average_max_surface_10m_wind_speed()
        self.calculate_average_min_surface_10m_wind_speed()
        self.calculate_record_max_surface_10m_wind_speed()
        self.calculate_record_min_surface_10m_wind_speed()
        self.calculate_average_max_surface_100m_wind_speed()
        self.calculate_average_min_surface_100m_wind_speed()
        self.calculate_record_max_surface_100m_wind_speed()
        self.calculate_record_min_surface_100m_wind_speed()
        self.calculate_percentage_of_days_with_precipitation()
        self.calculate_average_cloud_base_height()
        self.calculate_min_cloud_base_height()
        self.calculate_percentage_of_days_with_no_cloud_coverage()

    @property
    def cloud_base_height(self):
        cloud_base_height = [
            dayDict[hour]["cloudBaseHeight"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]

        masked_elem = np.ma.core.MaskedConstant
        unmasked_cloud_base_height = [
            np.inf if isinstance(elem, masked_elem) else elem
            for elem in cloud_base_height
        ]
        mask = [isinstance(elem, masked_elem) for elem in cloud_base_height]
        return np.ma.array(unmasked_cloud_base_height, mask=mask)

    def calculate_pressure_stats(self):
        """Calculate pressure level statistics."""
        # Surface pressure
        self.surface_pressure_list = [
            dayDict[hour]["surfacePressure"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.average_surface_pressure = np.average(self.surface_pressure_list)
        self.std_surface_pressure = np.std(self.surface_pressure_list)

        # Pressure at 1000 feet
        self.pressure_at_1000ft_list = [
            dayDict[hour]["pressure"](
                convert_units(1000, "ft", self.current_units["height_ASL"])
            )
            for dayDict in self.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.average_pressure_at_1000ft = np.average(self.pressure_at_1000ft_list)
        self.std_pressure_at_1000ft = np.std(self.pressure_at_1000ft_list)

        # Pressure at 10000 feet
        self.pressure_at_10000ft_list = [
            dayDict[hour]["pressure"](
                convert_units(10000, "ft", self.current_units["height_ASL"])
            )
            for dayDict in self.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.average_pressure_at_10000ft = np.average(self.pressure_at_10000ft_list)
        self.std_pressure_at_10000ft = np.std(self.pressure_at_10000ft_list)

        # Pressure at 30000 feet
        self.pressure_at_30000ft_list = [
            dayDict[hour]["pressure"](
                convert_units(30000, "ft", self.current_units["height_ASL"])
            )
            for dayDict in self.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.average_pressure_at_30000ft = np.average(self.pressure_at_30000ft_list)
        self.std_pressure_at_30000ft = np.std(self.pressure_at_30000ft_list)

        return self.average_surface_pressure, self.std_surface_pressure

    def calculate_average_cloud_base_height(self):
        """Calculate average cloud base height."""
        self.mean_cloud_base_height = np.ma.mean(self.cloud_base_height)
        return self.mean_cloud_base_height

    def calculate_min_cloud_base_height(self):
        """Calculate average cloud base height."""
        self.min_cloud_base_height = np.ma.min(
            self.cloud_base_height, fill_value=np.inf
        )
        return self.min_cloud_base_height

    def calculate_percentage_of_days_with_no_cloud_coverage(self):
        """Calculate percentage of days with cloud coverage."""
        self.percentage_of_days_with_no_cloud_coverage = np.ma.count(
            self.cloud_base_height
        ) / len(self.cloud_base_height)

        return self.percentage_of_days_with_no_cloud_coverage

    def calculate_percentage_of_days_with_precipitation(self):
        """Computes the ratio between days with precipitation (> 10 mm) and total days."""
        self.precipitation_per_day = [
            sum([dayDict[hour]["totalPrecipitation"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        days_with_precipitation_count = 0
        for precipitation in self.precipitation_per_day:
            if precipitation > convert_units(
                10, "mm", self.unit_system["precipitation"]
            ):
                days_with_precipitation_count += 1

        self.percentage_of_days_with_precipitation = (
            days_with_precipitation_count / len(self.precipitation_per_day)
        )

        return self.percentage_of_days_with_precipitation

    def calculate_average_max_temperature(self):
        self.max_temperature_list = [
            np.max([dayDict[hour]["surfaceTemperature"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_max_temperature = np.average(self.max_temperature_list)
        return self.average_max_temperature

    def calculate_average_min_temperature(self):
        self.min_temperature_list = [
            np.min([dayDict[hour]["surfaceTemperature"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_min_temperature = np.average(self.min_temperature_list)
        return self.average_min_temperature

    def calculate_record_max_temperature(self):
        self.temperature_list = [
            dayDict[hour]["surfaceTemperature"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_max_temperature = np.max(self.temperature_list)
        return self.record_max_temperature

    def calculate_record_min_temperature(self):
        self.temperature_list = [
            dayDict[hour]["surfaceTemperature"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_min_temperature = np.min(self.temperature_list)
        return self.record_min_temperature

    def calculate_average_max_wind_gust(self):
        self.max_wind_gust_list = [
            np.max([dayDict[hour]["surfaceWindGust"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_max_wind_gust = np.average(self.max_wind_gust_list)
        return self.average_max_wind_gust

    def calculate_maximum_wind_gust(self):
        self.wind_gust_list = [
            dayDict[hour]["surfaceWindGust"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.max_wind_gust = np.max(self.wind_gust_list)
        return self.max_wind_gust

    def calculate_maximum_surface_10m_wind_speed(self):
        self.surface_10m_wind_speed_list = [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.max_surface_10m_wind_speed = np.max(self.surface_10m_wind_speed_list)
        return self.max_surface_10m_wind_speed

    def calculate_average_max_surface_10m_wind_speed(self):
        self.max_surface_10m_wind_speed_list = [
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
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_max_surface_10m_wind_speed = np.average(
            self.max_surface_10m_wind_speed_list
        )
        return self.average_max_surface_10m_wind_speed

    def calculate_average_min_surface_10m_wind_speed(self):
        self.min_surface_10m_wind_speed_list = [
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
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_min_surface_10m_wind_speed = np.average(
            self.min_surface_10m_wind_speed_list
        )
        return self.average_min_surface_10m_wind_speed

    def calculate_record_max_surface_10m_wind_speed(self):
        self.surface_10m_wind_speed = [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_max_surface_10m_wind_speed = np.max(self.surface_10m_wind_speed)
        return self.record_max_surface_10m_wind_speed

    def calculate_record_min_surface_10m_wind_speed(self):
        self.surface_10m_wind_speed = [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_min_surface_10m_wind_speed = np.min(self.surface_10m_wind_speed)
        return self.record_min_surface_10m_wind_speed

    def calculate_average_max_surface_100m_wind_speed(self):
        self.max_surface_100m_wind_speed_list = [
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
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_max_surface_100m_wind_speed = np.average(
            self.max_surface_100m_wind_speed_list
        )
        return self.average_max_surface_100m_wind_speed

    def calculate_average_min_surface_100m_wind_speed(self):
        self.min_surface_100m_wind_speed_list = [
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
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_min_surface_100m_wind_speed = np.average(
            self.min_surface_100m_wind_speed_list
        )
        return self.average_min_surface_100m_wind_speed

    def calculate_record_max_surface_100m_wind_speed(self):
        self.surface_100m_wind_speed = [
            (
                dayDict[hour]["surface100mWindVelocityX"] ** 2
                + dayDict[hour]["surface100mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_max_surface_100m_wind_speed = np.max(self.surface_100m_wind_speed)
        return self.record_max_surface_100m_wind_speed

    def calculate_record_min_surface_100m_wind_speed(self):
        self.surface_100m_wind_speed = [
            (
                dayDict[hour]["surface100mWindVelocityX"] ** 2
                + dayDict[hour]["surface100mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_min_surface_100m_wind_speed = np.min(self.surface_100m_wind_speed)
        return self.record_min_surface_100m_wind_speed

    def calculate_average_temperature_along_day(self):
        """Computes average temperature progression throughout the
        day, including sigma contours."""

        # Flip dictionary to get hour as key instead of date
        historical_temperatures_each_hour = defaultdict(dict)
        for date, val in self.surfaceDataDict.items():
            for hour, sub_val in val.items():
                historical_temperatures_each_hour[hour][date] = sub_val[
                    "surfaceTemperature"
                ]

        self.average_temperature_at_given_hour = {
            hour: np.average(list(dates.values()))
            for hour, dates in historical_temperatures_each_hour.items()
        }

        self.average_temperature_sigmas_at_given_hour = {
            hour: np.std(list(dates.values()))
            for hour, dates in historical_temperatures_each_hour.items()
        }

        return (
            self.average_temperature_at_given_hour,
            self.average_temperature_sigmas_at_given_hour,
        )

    def calculate_average_sustained_surface10m_wind_along_day(self):
        """Computes average sustained wind speed progression throughout the
        day, including sigma contours."""

        # Flip dictionary to get hour as key instead of date
        historical_surface10m_wind_speeds_each_hour = defaultdict(dict)
        for date, val in self.surfaceDataDict.items():
            for hour, sub_val in val.items():
                historical_surface10m_wind_speeds_each_hour[hour][date] = (
                    sub_val["surface10mWindVelocityX"] ** 2
                    + sub_val["surface10mWindVelocityY"] ** 2
                ) ** 0.5

        self.average_surface10m_wind_speed_at_given_hour = {
            hour: np.average(list(dates.values()))
            for hour, dates in historical_surface10m_wind_speeds_each_hour.items()
        }

        self.average_surface10m_wind_speed_sigmas_at_given_hour = {
            hour: np.std(list(dates.values()))
            for hour, dates in historical_surface10m_wind_speeds_each_hour.items()
        }

        return (
            self.average_surface10m_wind_speed_at_given_hour,
            self.average_surface10m_wind_speed_sigmas_at_given_hour,
        )

    def calculate_average_sustained_surface100m_wind_along_day(self):
        """Computes average sustained wind speed progression throughout the
        day, including sigma contours."""

        # Flip dictionary to get hour as key instead of date
        historical_surface100m_wind_speeds_each_hour = defaultdict(dict)
        for date, val in self.surfaceDataDict.items():
            for hour, sub_val in val.items():
                historical_surface100m_wind_speeds_each_hour[hour][date] = (
                    sub_val["surface100mWindVelocityX"] ** 2
                    + sub_val["surface100mWindVelocityY"] ** 2
                ) ** 0.5

        self.average_surface100m_wind_speed_at_given_hour = {
            hour: np.average(list(dates.values()))
            for hour, dates in historical_surface100m_wind_speeds_each_hour.items()
        }

        self.average_surface100m_wind_speed_sigmas_at_given_hour = {
            hour: np.std(list(dates.values()))
            for hour, dates in historical_surface100m_wind_speeds_each_hour.items()
        }

        return (
            self.average_surface100m_wind_speed_at_given_hour,
            self.average_surface100m_wind_speed_sigmas_at_given_hour,
        )

    def process_wind_speed_and_direction_data_for_average_day(self):
        """Process the wind_speed and wind_direction data to generate lists of all the wind_speeds recorded
        for a following hour of the day and also the wind direction. Also calculates the greater and the smallest
        wind_speed recorded

        Returns
        -------
        None
        """
        max_wind_speed = float("-inf")
        min_wind_speed = float("inf")

        days = list(self.surfaceDataDict.keys())
        hours = list(self.surfaceDataDict[days[0]].keys())

        windSpeed = {}
        windDir = {}

        for hour in hours:
            windSpeed[hour] = []
            windDir[hour] = []
            for day in days:
                try:
                    hour_wind_speed = (
                        self.surfaceDataDict[day][hour]["surface10mWindVelocityX"] ** 2
                        + self.surfaceDataDict[day][hour]["surface10mWindVelocityY"]
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
                    vx = self.surfaceDataDict[day][hour]["surface10mWindVelocityX"]
                    vy = self.surfaceDataDict[day][hour]["surface10mWindVelocityY"]
                    windDir[hour].append(
                        (180 + (np.arctan2(vy, vx) * 180 / np.pi)) % 360
                    )
                except KeyError:
                    # Not all days have all hours stored, that is fine
                    pass

        self.max_wind_speed = max_wind_speed
        self.min_wind_speed = min_wind_speed
        self.wind_speed_per_hour = windSpeed
        self.wind_direction_per_hour = windDir

    @property
    def altitude_AGL_range(self):
        min_altitude = 0
        if self.maxExpectedAltitude == None:
            max_altitudes = [
                np.max(dayDict[hour]["windSpeed"].source[-1, 0])
                for dayDict in self.pressureLevelDataDict.values()
                for hour in dayDict.keys()
            ]
            max_altitude = np.min(max_altitudes)
        else:
            max_altitude = self.maxExpectedAltitude
        return min_altitude, max_altitude

    def process_temperature_profile_over_average_day(self):
        """Compute the average temperature profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_temperature_profile_at_given_hour = {}
        self.max_average_temperature_at_altitude = 0
        hours = list(self.pressureLevelDataDict.values())[0].keys()
        for hour in hours:
            temperature_values_for_this_hour = []
            for dayDict in self.pressureLevelDataDict.values():
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
            if max_temperature >= self.max_average_temperature_at_altitude:
                self.max_average_temperature_at_altitude = max_temperature
        self.average_temperature_profile_at_given_hour = (
            average_temperature_profile_at_given_hour
        )

        return None

    def process_pressure_profile_over_average_day(self):
        """Compute the average pressure profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_pressure_profile_at_given_hour = {}
        self.max_average_pressure_at_altitude = 0
        hours = list(self.pressureLevelDataDict.values())[0].keys()
        for hour in hours:
            pressure_values_for_this_hour = []
            for dayDict in self.pressureLevelDataDict.values():
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
            if max_pressure >= self.max_average_pressure_at_altitude:
                self.max_average_pressure_at_altitude = max_pressure
        self.average_pressure_profile_at_given_hour = (
            average_pressure_profile_at_given_hour
        )

        return None

    def process_wind_speed_profile_over_average_day(self):
        """Compute the average wind profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_wind_profile_at_given_hour = {}
        self.max_average_wind_at_altitude = 0
        hours = list(self.pressureLevelDataDict.values())[0].keys()

        # days = list(self.surfaceDataDict.keys())
        # hours = list(self.surfaceDataDict[days[0]].keys())
        for hour in hours:
            wind_speed_values_for_this_hour = []
            for dayDict in self.pressureLevelDataDict.values():
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
            if max_wind >= self.max_average_wind_at_altitude:
                self.max_average_wind_at_altitude = max_wind
        self.average_wind_profile_at_given_hour = average_wind_profile_at_given_hour

        return None

    def process_wind_velocity_x_profile_over_average_day(self):
        """Compute the average windVelocityX profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_windVelocityX_profile_at_given_hour = {}
        self.max_average_windVelocityX_at_altitude = 0
        hours = list(self.pressureLevelDataDict.values())[0].keys()
        for hour in hours:
            windVelocityX_values_for_this_hour = []
            for dayDict in self.pressureLevelDataDict.values():
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
            if max_windVelocityX >= self.max_average_windVelocityX_at_altitude:
                self.max_average_windVelocityX_at_altitude = max_windVelocityX
        self.average_windVelocityX_profile_at_given_hour = (
            average_windVelocityX_profile_at_given_hour
        )
        return None

    def process_wind_velocity_y_profile_over_average_day(self):
        """Compute the average windVelocityY profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_windVelocityY_profile_at_given_hour = {}
        self.max_average_windVelocityY_at_altitude = 0
        hours = list(self.pressureLevelDataDict.values())[0].keys()
        for hour in hours:
            windVelocityY_values_for_this_hour = []
            for dayDict in self.pressureLevelDataDict.values():
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
            if max_windVelocityY >= self.max_average_windVelocityY_at_altitude:
                self.max_average_windVelocityY_at_altitude = max_windVelocityY
        self.average_windVelocityY_profile_at_given_hour = (
            average_windVelocityY_profile_at_given_hour
        )
        return None

    def process_wind_heading_profile_over_average_day(self):
        """Compute the average wind velocities (both X and Y components) profile
        for each available hour of a day, over all days in the dataset.
        """
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)
        average_wind_velocity_X_profile_at_given_hour = {}
        average_wind_velocity_Y_profile_at_given_hour = {}
        average_wind_heading_profile_at_given_hour = {}
        self.max_average_wind_velocity_X_at_altitude = 0
        self.max_average_wind_velocity_Y_at_altitude = 0

        hours = list(self.pressureLevelDataDict.values())[0].keys()
        for hour in hours:
            wind_velocity_X_values_for_this_hour = []
            wind_velocity_Y_values_for_this_hour = []
            for dayDict in self.pressureLevelDataDict.values():
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
            if max_wind_X >= self.max_average_wind_velocity_X_at_altitude:
                self.max_average_wind_X_at_altitude = max_wind_X
            max_wind_Y = np.max(mean_wind_velocity_Y_values_for_this_hour)
            if max_wind_Y >= self.max_average_wind_velocity_Y_at_altitude:
                self.max_average_wind_Y_at_altitude = max_wind_Y
        # Store the average wind velocity profiles for each hour
        self.average_wind_velocity_X_profile_at_given_hour = (
            average_wind_velocity_X_profile_at_given_hour
        )
        self.average_wind_velocity_Y_profile_at_given_hour = (
            average_wind_velocity_Y_profile_at_given_hour
        )
        self.average_wind_heading_profile_at_given_hour = (
            average_wind_heading_profile_at_given_hour
        )

        return None

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
        OBS: Not all units are allowed as inputs of Environment Class.

        Parameters
        ----------
        filename : str, optional
            Name of the file where to be saved, by default "EnvAnalysisDict"

        Returns
        -------
        None
        """

        self.process_temperature_profile_over_average_day()
        self.process_pressure_profile_over_average_day()
        self.process_wind_velocity_x_profile_over_average_day()
        self.process_wind_velocity_y_profile_over_average_day()

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
            "elevation": self.elevation,
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
