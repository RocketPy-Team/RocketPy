# -*- coding: utf-8 -*-

__author__ = "Patrick Sampaio, Giovani Hidalgo Ceotto, Guilherme Fernandes Alves, Franz Masatoshi Yuri, Mateus Stano Junqueira,"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import bisect
import warnings
from collections import defaultdict

import ipywidgets as widgets
import matplotlib.ticker as mtick
import netCDF4
import numpy as np
import pytz
from cftime import num2pydate
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter as ImageWriter
from scipy import stats
from timezonefinder import TimezoneFinder
from windrose import WindroseAxes

from rocketpy.Function import Function
from rocketpy.units import convert_units


class EnvironmentAnalysis:
    """Class for analyzing the environment.

    List of ideas suggested by Logan:
        - average max/min temperature
        - record max/min temperature
        - average max wind gust
        - record max wind gust
        - plot of wind gust distribution (should be Weibull)
        - animation of who wind gust distribution evolves over average day
        - temperature progression throughout the day at some fine interval (ex: 10 min) with 1, 2, 3, sigma contours (sketch below)
        - average, 1, 2, 3 sigma wind profile from 0 - 35,000 ft AGL
        - average day wind rose
        - animation of how average wind rose evolves throughout an average day
        - animation of how wind profile evolves throughout an average day

    All items listed are relevant to either
        1. participant safety
        2. launch operations (range closure decision)
        3. rocket performance

    How does this class work?
    - The class is initialized with a start date and end date.
    - The class then parses the weather data from the start date to the end date.
    - The class then calculates the average max/min temperature, average max wind gust, and average day wind rose.
    - The class then plots the average max/min temperature, average max wind gust, and average day wind rose.
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
        Returns
        -------
        None
        """
        warnings.warn("Please notice this class is still under development")

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
        self.average_temperature_along_day_1_sigma = Function(0)
        self.average_temperature_along_day_2_sigma = Function(0)
        self.average_temperature_along_day_3_sigma = Function(0)
        self.average_wind_profile = Function(0)
        self.average_wind_profile_1_sigma = Function(0)
        self.average_wind_profile_2_sigma = Function(0)
        self.average_wind_profile_3_sigma = Function(0)
        self.average_wind_profile_at_given_hour = None

        self.max_wind_speed = None
        self.min_wind_speed = None
        self.wind_speed_per_hour = None
        self.wind_direction_per_hour = None

        # Run calculations
        self.process_data()

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
        - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-preliminary-back-extension?tab=overview
        -

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
                    inputs="Height Above Ground Level (m)",
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

        Must get the following variables:
        - surface elevation: self.elevation = float
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

    def plot_wind_gust_distribution(self):
        """Get all values of wind gust speed (for every date and hour available)
        and plot a single distribution. Expected result is a Weibull distribution.
        """
        self.wind_gust_list = [
            dayDict[hour]["surfaceWindGust"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        plt.figure()
        # Plot histogram
        plt.hist(
            self.wind_gust_list,
            bins=int(len(self.wind_gust_list) ** 0.5),
            density=True,
            histtype="stepfilled",
            alpha=0.2,
            label="Wind Gust Speed Distribution",
        )

        # Plot weibull distribution
        c, loc, scale = stats.weibull_min.fit(self.wind_gust_list, loc=0, scale=1)
        x = np.linspace(0, np.max(self.wind_gust_list), 100)
        plt.plot(
            x,
            stats.weibull_min.pdf(x, c, loc, scale),
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )

        # Label plot
        plt.ylabel("Probability")
        plt.xlabel(f"Wind gust speed ({self.unit_system['wind_speed']})")
        plt.title("Wind Gust Speed Distribution")
        plt.legend()
        plt.show()

        return None

    def plot_surface10m_wind_speed_distribution(self, SAcup_wind_constraints=False):
        """Get all values of sustained surface wind speed (for every date and hour available)
        and plot a single distribution. Expected result is a Weibull distribution.
        """
        self.wind_speed_list = [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        plt.figure()
        # Plot histogram
        plt.hist(
            self.wind_speed_list,
            bins=int(len(self.wind_speed_list) ** 0.5),
            density=True,
            histtype="stepfilled",
            alpha=0.2,
            label="Wind Gust Speed Distribution",
        )

        # Plot weibull distribution
        c, loc, scale = stats.weibull_min.fit(self.wind_speed_list, loc=0, scale=1)
        x = np.linspace(0, np.max(self.wind_speed_list), 100)
        plt.plot(
            x,
            stats.weibull_min.pdf(x, c, loc, scale),
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )

        if SAcup_wind_constraints:
            plt.vlines(
                convert_units(20, "mph", self.unit_system["wind_speed"]),
                0,
                0.3,
                "g",
                (0, (15, 5, 2, 5)),
                label="SAcup wind speed constraints",
            )  # Plot SAcup wind speed constraints

        # Label plot
        plt.ylabel("Probability")
        plt.xlabel(f"Sustained surface wind speed ({self.unit_system['wind_speed']})")
        plt.title("Sustained Surface Wind Speed Distribution")
        plt.legend()
        plt.show()

        return None

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

    def plot_average_temperature_along_day(self):
        """Plots average temperature progression throughout the day, including
        sigma contours."""

        # Compute values
        self.calculate_average_temperature_along_day()

        # Get handy arrays
        hours = np.fromiter(self.average_temperature_at_given_hour.keys(), np.float)
        temperature_mean = self.average_temperature_at_given_hour.values()
        temperature_mean = np.array(list(temperature_mean))
        temperature_std = np.array(
            list(self.average_temperature_sigmas_at_given_hour.values())
        )
        temperatures_p1sigma = temperature_mean + temperature_std
        temperatures_m1sigma = temperature_mean - temperature_std
        temperatures_p2sigma = temperature_mean + 2 * temperature_std
        temperatures_m2sigma = temperature_mean - 2 * temperature_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.surfaceDataDict.values():
            plt.plot(
                [int(hour) for hour in hour_entries.keys()],
                [val["surfaceTemperature"] for val in hour_entries.values()],
                "gray",
                alpha=0.1,
            )

        # Plot average temperature along day
        plt.plot(hours, temperature_mean, "r", label="$\\mu$")

        # Plot standard deviations temperature along day
        plt.plot(hours, temperatures_m1sigma, "b--", label=r"$\mu \pm \sigma$")
        plt.plot(hours, temperatures_p1sigma, "b--")
        plt.plot(hours, temperatures_p2sigma, "b--", alpha=0.5)
        plt.plot(
            hours, temperatures_m2sigma, "b--", label=r"$\mu \pm 2\sigma $", alpha=0.5
        )

        # Format plot
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_formatter(
            lambda x, pos: "{0:02.0f}:{1:02.0f}".format(*divmod(x * 60, 60))
        )
        plt.autoscale(enable=True, axis="x", tight=True)
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Temperature ({self.unit_system['temperature']})")
        plt.title("Average Temperature Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()

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

    def plot_average_surface10m_wind_speed_along_day(
        self, SAcup_wind_constraints=False
    ):
        """Plots average surface wind speed progression throughout the day, including
        sigma contours."""

        # Compute values
        self.calculate_average_sustained_surface10m_wind_along_day()

        # Get handy arrays
        hours = np.fromiter(
            self.average_surface10m_wind_speed_at_given_hour.keys(), np.float
        )
        wind_speed_mean = self.average_surface10m_wind_speed_at_given_hour.values()
        wind_speed_mean = np.array(list(wind_speed_mean))
        wind_speed_std = np.array(
            list(self.average_surface10m_wind_speed_sigmas_at_given_hour.values())
        )
        wind_speeds_p1sigma = wind_speed_mean + wind_speed_std
        wind_speeds_m1sigma = wind_speed_mean - wind_speed_std
        wind_speeds_p2sigma = wind_speed_mean + 2 * wind_speed_std
        wind_speeds_m2sigma = wind_speed_mean - 2 * wind_speed_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.surfaceDataDict.values():
            plt.plot(
                [int(hour) for hour in hour_entries.keys()],
                [
                    (
                        val["surface10mWindVelocityX"] ** 2
                        + val["surface10mWindVelocityY"] ** 2
                    )
                    ** 0.5
                    for val in hour_entries.values()
                ],
                "gray",
                alpha=0.1,
            )

        # Plot average temperature along day
        plt.plot(hours, wind_speed_mean, "r", label="$\\mu$")

        # Plot standard deviations temperature along day
        plt.plot(hours, wind_speeds_m1sigma, "b--", label=r"$\mu \pm \sigma$")
        plt.plot(hours, wind_speeds_p1sigma, "b--")
        plt.plot(hours, wind_speeds_p2sigma, "b--", alpha=0.5)
        plt.plot(
            hours, wind_speeds_m2sigma, "b--", label=r"$\mu \pm 2\sigma $", alpha=0.5
        )

        # Format plot
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_formatter(
            lambda x, pos: "{0:02.0f}:{1:02.0f}".format(*divmod(x * 60, 60))
        )
        plt.autoscale(enable=True, axis="x", tight=True)
        if SAcup_wind_constraints:
            plt.hlines(
                convert_units(20, "mph", self.unit_system["wind_speed"]),
                min(hours),
                max(hours),
                "g",
                (0, (15, 5, 2, 5)),
                label="SAcup wind speed constraints",
            )  # Plot SAcup wind speed constraints
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Surface Wind Speed ({self.unit_system['wind_speed']})")
        plt.title("Average Sustained Surface Wind Speed Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()

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

    def plot_average_sustained_surface100m_wind_speed_along_day(self):
        """Plots average surface wind speed progression throughout the day, including
        sigma contours."""

        # Compute values
        self.calculate_average_sustained_surface100m_wind_along_day()

        # Get handy arrays
        hours = np.fromiter(
            self.average_surface100m_wind_speed_at_given_hour.keys(), np.float
        )
        wind_speed_mean = self.average_surface100m_wind_speed_at_given_hour.values()
        wind_speed_mean = np.array(list(wind_speed_mean))
        wind_speed_std = np.array(
            list(self.average_surface100m_wind_speed_sigmas_at_given_hour.values())
        )
        wind_speeds_p1sigma = wind_speed_mean + wind_speed_std
        wind_speeds_m1sigma = wind_speed_mean - wind_speed_std
        wind_speeds_p2sigma = wind_speed_mean + 2 * wind_speed_std
        wind_speeds_m2sigma = wind_speed_mean - 2 * wind_speed_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.surfaceDataDict.values():
            plt.plot(
                [int(hour) for hour in hour_entries.keys()],
                [
                    (
                        val["surface100mWindVelocityX"] ** 2
                        + val["surface100mWindVelocityY"] ** 2
                    )
                    ** 0.5
                    for val in hour_entries.values()
                ],
                "gray",
                alpha=0.1,
            )

        # Plot average temperature along day
        plt.plot(hours, wind_speed_mean, "r", label="$\\mu$")

        # Plot standard deviations temperature along day
        plt.plot(hours, wind_speeds_m1sigma, "b--", label=r"$\mu \pm \sigma$")
        plt.plot(hours, wind_speeds_p1sigma, "b--")
        plt.plot(hours, wind_speeds_p2sigma, "b--", alpha=0.5)
        plt.plot(
            hours, wind_speeds_m2sigma, "b--", label=r"$\mu \pm 2\sigma $", alpha=0.5
        )

        # Format plot
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_formatter(
            lambda x, pos: "{0:02.0f}:{1:02.0f}".format(*divmod(x * 60, 60))
        )
        plt.autoscale(enable=True, axis="x", tight=True)
        plt.xlabel("Time (hours)")
        plt.ylabel(f"100m Wind Speed ({self.unit_system['wind_speed']})")
        plt.title("Average 100m Wind Speed Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()

    def plot_average_wind_speed_profile(self, SAcup_altitude_constraints=False):
        """Average wind speed for all datetimes available."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)
        wind_speed_profiles = [
            dayDict[hour]["windSpeed"](altitude_list)
            for dayDict in self.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.average_wind_speed_profile = np.mean(wind_speed_profiles, axis=0)
        # Plot
        plt.figure()
        plt.plot(self.average_wind_speed_profile, altitude_list, "r", label="$\\mu$")
        plt.plot(
            np.percentile(wind_speed_profiles, 50 - 34.1, axis=0),
            altitude_list,
            "b--",
            alpha=1,
            label="$\\mu \\pm \\sigma$",
        )
        plt.plot(
            np.percentile(wind_speed_profiles, 50 + 34.1, axis=0),
            altitude_list,
            "b--",
            alpha=1,
        )
        plt.plot(
            np.percentile(wind_speed_profiles, 50 - 47.4, axis=0),
            altitude_list,
            "b--",
            alpha=0.5,
            label="$\\mu \\pm 2\\sigma$",
        )
        plt.plot(
            np.percentile(wind_speed_profiles, 50 + 47.7, axis=0),
            altitude_list,
            "b--",
            alpha=0.5,
        )
        # plt.plot(np.percentile(wind_speed_profiles, 50-49.8, axis=0, method='weibull'), altitude_list, 'b--', alpha=0.25)
        # plt.plot(np.percentile(wind_speed_profiles, 50+49.8, axis=0, method='weibull'), altitude_list, 'b--', alpha=0.25)
        for wind_speed_profile in wind_speed_profiles:
            plt.plot(wind_speed_profile, altitude_list, "gray", alpha=0.01)

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if SAcup_altitude_constraints:
            # SA Cup altitude constraints region
            print(plt)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.fill_between(
                [xmin, xmax],
                0.7 * convert_units(10000, "ft", self.unit_system["length"]),
                1.3 * convert_units(10000, "ft", self.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [xmin, xmax],
                0.7 * convert_units(30000, "ft", self.unit_system["length"]),
                1.3 * convert_units(30000, "ft", self.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.unit_system['length']} ± 30%",
            )

        plt.xlabel(f"Wind speed ({self.unit_system['wind_speed']})")
        plt.ylabel(f"Altitude AGL ({self.unit_system['length']})")
        plt.title("Average Wind Speed Profile")
        plt.legend()
        plt.show()

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

    def plot_average_pressure_profile(self, SAcup_altitude_constraints=False):
        """Average wind speed for all datetimes available."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)
        pressure_profiles = [
            dayDict[hour]["pressure"](altitude_list)
            for dayDict in self.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.average_pressure_profile = np.mean(pressure_profiles, axis=0)
        # Plot
        plt.figure()
        plt.plot(self.average_pressure_profile, altitude_list, "r", label="$\\mu$")
        plt.plot(
            np.percentile(pressure_profiles, 50 - 34.1, axis=0),
            altitude_list,
            "b--",
            alpha=1,
            label="$\\mu \\pm \\sigma$",
        )
        plt.plot(
            np.percentile(pressure_profiles, 50 + 34.1, axis=0),
            altitude_list,
            "b--",
            alpha=1,
        )
        plt.plot(
            np.percentile(pressure_profiles, 50 - 47.4, axis=0),
            altitude_list,
            "b--",
            alpha=0.5,
            label="$\\mu \\pm 2\\sigma$",
        )
        plt.plot(
            np.percentile(pressure_profiles, 50 + 47.7, axis=0),
            altitude_list,
            "b--",
            alpha=0.5,
        )
        # plt.plot(np.percentile(pressure_profiles, 50-49.8, axis=0, method='weibull'), altitude_list, 'b--', alpha=0.25)
        # plt.plot(np.percentile(pressure_profiles, 50+49.8, axis=0, method='weibull'), altitude_list, 'b--', alpha=0.25)
        for pressure_profile in pressure_profiles:
            plt.plot(pressure_profile, altitude_list, "gray", alpha=0.01)

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if SAcup_altitude_constraints:
            # SA Cup altitude constraints region
            print(plt)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.fill_between(
                [xmin, xmax],
                0.7 * convert_units(10000, "ft", self.unit_system["length"]),
                1.3 * convert_units(10000, "ft", self.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [xmin, xmax],
                0.7 * convert_units(30000, "ft", self.unit_system["length"]),
                1.3 * convert_units(30000, "ft", self.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.unit_system['length']} ± 30%",
            )

        plt.xlabel(f"Pressure ({self.unit_system['pressure']})")
        plt.ylabel(f"Altitude AGL ({self.unit_system['length']})")
        plt.title("Average Pressure Profile")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_wind_rose(
        wind_direction, wind_speed, bins=None, title=None, fig=None, rect=None
    ):
        """Plot a windrose given the data.

        Parameters
        ----------
        wind_direction: list[float]
        wind_speed: list[float]
        bins: 1D array or integer, optional
            number of bins, or a sequence of bins variable. If not set, bins=6,
            then bins=linspace(min(var), max(var), 6)
        title: str, optional
            Title of the plot
        fig: matplotlib.pyplot.figure, optional

        Returns
        -------
        WindroseAxes
        """
        ax = WindroseAxes.from_ax(fig=fig, rect=rect)
        ax.bar(
            wind_direction,
            wind_speed,
            bins=bins,
            normed=True,
            opening=0.8,
            edgecolor="white",
        )
        ax.set_title(title)
        ax.set_legend()
        # Format the ticks (only integers, as percentage, at most 3 intervals)
        ax.yaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=3, prune="lower")
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        return ax

    def plot_average_day_wind_rose_specific_hour(self, hour, fig=None):
        """Plot a specific hour of the average windrose

        Parameters
        ----------
        hour: int
        fig: matplotlib.pyplot.figure

        Returns
        -------
        None
        """
        hour = str(hour)
        self.plot_wind_rose(
            self.wind_direction_per_hour[hour],
            self.wind_speed_per_hour[hour],
            bins=self._beaufort_wind_scale(
                self.unit_system["wind_speed"], max_wind_speed=self.max_wind_speed
            ),
            title=f"Wind Rose of an Average Day ({self.unit_system['wind_speed']}) - Hour {float(hour):05.2f}".replace(
                ".", ":"
            ),
            fig=fig,
        )
        plt.show()

    def plot_average_day_wind_rose_all_hours(self):
        """Plot windroses for all hours of a day, in a grid like plot."""
        # Get days and hours
        days = list(self.surfaceDataDict.keys())
        hours = list(self.surfaceDataDict[days[0]].keys())

        # Make sure necessary data has been calculated
        if not all(
            [
                self.max_wind_speed,
                self.min_wind_speed,
                self.wind_speed_per_hour,
                self.wind_direction_per_hour,
            ]
        ):
            self.process_wind_speed_and_direction_data_for_average_day()

        # Figure settings
        windrose_side = 2.5  # inches
        vertical_padding_top = 1.5  # inches
        plot_padding = 0.18  # percentage
        ncols, nrows = self._find_two_closest_integer_factors(len(hours))
        vertical_plot_area_percentage = (
            nrows * windrose_side / (nrows * windrose_side + vertical_padding_top)
        )

        # Create figure
        fig = plt.figure()
        fig.set_size_inches(
            ncols * windrose_side, nrows * windrose_side + vertical_padding_top
        )
        bins = self._beaufort_wind_scale(
            self.unit_system["wind_speed"], max_wind_speed=self.max_wind_speed
        )
        width = (1 - 2 * plot_padding) * 1 / ncols
        height = vertical_plot_area_percentage * (1 - 2 * plot_padding) * 1 / nrows
        # print(ncols, nrows)
        # print(ncols * windrose_side, nrows * windrose_side + vertical_padding_top)
        # print(vertical_plot_area_percentage)
        # print(width, height)
        for k, hour in enumerate(hours):
            i, j = len(hours) // nrows - k // ncols, k % ncols  # Row count bottom up
            left = j * 1 / ncols + plot_padding / ncols
            bottom = (
                vertical_plot_area_percentage * ((i - 2) / nrows + plot_padding / nrows)
                + 0.5
            )
            # print(left, bottom)

            ax = self.plot_wind_rose(
                self.wind_direction_per_hour[hour],
                self.wind_speed_per_hour[hour],
                bins=bins,
                title=f"{float(hour):05.2f}".replace(".", ":"),
                fig=fig,
                rect=[left, bottom, width, height],
            )
            if k == 0:
                ax.legend(
                    loc="upper center",
                    # 0.8 is a magic number
                    bbox_to_anchor=(ncols / 2 + 0.8, 1.55),
                    fancybox=True,
                    shadow=True,
                    ncol=6,
                )
            else:
                ax.legend().set_visible(False)
            fig.add_axes(ax)

        fig.suptitle(
            f"Wind Roses ({self.unit_system['wind_speed']})", fontsize=20, x=0.5, y=1
        )
        plt.show()

    def animate_average_wind_rose(self, figsize=(8, 8), filename="wind_rose.gif"):
        """Animates the wind_rose of an average day. The inputs of a wind_rose are the location of the
        place where we want to analyze, (x,y,z). The data is assembled by hour, which means, the windrose
        of a specific hour is generated by bringing together the data of all of the days available for that
        specific hour. It's possible to change the size of the gif using the parameter figsize, which is the
        height and width in inches.

        Parameters
        ----------
        figsize : array

        Returns
        -------
        Image : ipywidgets.widgets.widget_media.Image
        """
        days = list(self.surfaceDataDict.keys())
        hours = list(self.surfaceDataDict[days[0]].keys())

        if not all(
            [
                self.max_wind_speed,
                self.min_wind_speed,
                self.wind_speed_per_hour,
                self.wind_direction_per_hour,
            ]
        ):
            self.process_wind_speed_and_direction_data_for_average_day()

        metadata = dict(
            title="windrose",
            artist="windrose",
            comment="""Made with windrose
                http://www.github.com/scls19fr/windrose""",
        )
        writer = ImageWriter(fps=1, metadata=metadata)
        fig = plt.figure(facecolor="w", edgecolor="w", figsize=figsize)
        with writer.saving(fig, filename, 100):
            for hour in hours:
                self.plot_wind_rose(
                    self.wind_direction_per_hour[hour],
                    self.wind_speed_per_hour[hour],
                    bins=self._beaufort_wind_scale(
                        self.unit_system["wind_speed"],
                        max_wind_speed=self.max_wind_speed,
                    ),
                    title=f"Wind Rose of an Average Day ({self.unit_system['wind_speed']}). Hour {float(hour):05.2f}".replace(
                        ".", ":"
                    ),
                    fig=fig,
                )
                writer.grab_frame()
                plt.clf()

        with open(filename, "rb") as file:
            image = file.read()

        fig_width, fig_height = plt.gcf().get_size_inches() * fig.dpi
        plt.close(fig)
        return widgets.Image(
            value=image,
            format="gif",
            width=fig_width,
            height=fig_height,
        )

    def plot_wind_gust_distribution_over_average_day(self):
        """Plots shown in the animation of how the wind gust distribution varies throughout the day."""
        # Gather animation data
        average_wind_gust_at_given_hour = {}
        for hour in list(self.surfaceDataDict.values())[0].keys():
            wind_gust_values_for_this_hour = []
            for dayDict in self.surfaceDataDict.values():
                try:
                    wind_gust_values_for_this_hour += [dayDict[hour]["surfaceWindGust"]]
                except KeyError:
                    # Some day does not have data for the desired hour (probably the last one)
                    # No need to worry, just average over the other days
                    pass
            average_wind_gust_at_given_hour[hour] = wind_gust_values_for_this_hour

        # Create grid of plots for each hour
        hours = list(list(self.pressureLevelDataDict.values())[0].keys())
        nrows, ncols = self._find_two_closest_integer_factors(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for (i, j) in [(i, j) for i in range(nrows) for j in range(ncols)]:
            hour = hours[i * ncols + j]
            ax = axs[i, j]
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.hist(
                average_wind_gust_at_given_hour[hour],
                bins=int(len(average_wind_gust_at_given_hour[hour]) ** 0.5),
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                label="Wind Gust Speed Distribution",
            )
            ax.autoscale(enable=True, axis="y", tight=True)
            # Plot weibull distribution
            c, loc, scale = stats.weibull_min.fit(
                average_wind_gust_at_given_hour[hour], loc=0, scale=1
            )
            x = np.linspace(0, np.ceil(self.max_wind_gust), 100)
            ax.plot(
                x,
                stats.weibull_min.pdf(x, c, loc, scale),
                "r-",
                linewidth=2,
                label="Weibull Distribution",
            )
            current_x_max = ax.get_xlim()[1]
            current_y_max = ax.get_ylim()[1]
            x_max = current_x_max if current_x_max > x_max else x_max
            y_max = current_y_max if current_y_max > y_max else y_max
            ax.label_outer()
            ax.grid()
        # Set x and y limits for the last axis. Since axes are shared, set to all
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=5, prune="lower")
        )
        ax.yaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=4, prune="lower")
        )
        # Set title and axis labels for entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Average Wind Profile")
        fig.supxlabel(f"Wind Gust Speed ({self.unit_system['wind_speed']})")
        fig.supylabel("Probability")
        plt.show()

    def animate_wind_gust_distribution_over_average_day(self):
        """Animation of how the wind gust distribution varies throughout the day."""
        # Gather animation data
        wind_gusts_at_given_hour = {}
        for hour in list(self.surfaceDataDict.values())[0].keys():
            wind_gust_values_for_this_hour = []
            for dayDict in self.surfaceDataDict.values():
                try:
                    wind_gust_values_for_this_hour += [dayDict[hour]["surfaceWindGust"]]
                except KeyError:
                    # Some day does not have data for the desired hour (probably the last one)
                    # No need to worry, just average over the other days
                    pass
            wind_gusts_at_given_hour[hour] = wind_gust_values_for_this_hour

        # Create animation
        fig, ax = plt.subplots(dpi=200)
        # Initialize animation artists: histogram and hour text
        hist_bins = np.linspace(0, np.ceil(self.max_wind_gust), 25)  # Fix bins edges
        _, _, bar_container = plt.hist(
            [],
            bins=hist_bins,
            alpha=0.2,
            label="Wind Gust Speed Distribution",
        )
        (ln,) = plt.plot(
            [],
            [],
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )
        tx = plt.text(
            x=0.95,
            y=0.95,
            s="",
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=24,
        )

        # Define function to initialize animation
        def init():
            ax.set_xlim(0, np.ceil(self.max_wind_gust))
            ax.set_ylim(0, 0.3)  # TODO: parametrize
            ax.set_xlabel(f"Wind Gust Speed ({self.unit_system['wind_speed']})")
            ax.set_ylabel("Probability")
            ax.set_title("Wind Gust Distribution")
            # ax.grid(True)
            return ln, *bar_container.patches, tx

        # Define function which sets each animation frame
        def update(frame):
            # Update histogram
            data = frame[1]
            hist, _ = np.histogram(data, hist_bins, density=True)
            for count, rect in zip(hist, bar_container.patches):
                rect.set_height(count)
            # Update weibull distribution
            c, loc, scale = stats.weibull_min.fit(data, loc=0, scale=1)
            xdata = np.linspace(0, np.ceil(self.max_wind_gust), 100)
            ydata = stats.weibull_min.pdf(xdata, c, loc, scale)
            ln.set_data(xdata, ydata)
            # Update hour text
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return ln, *bar_container.patches, tx

        for frame in wind_gusts_at_given_hour.items():
            update(frame)

        animation = FuncAnimation(
            fig,
            update,
            frames=wind_gusts_at_given_hour.items(),
            interval=750,
            init_func=init,
            blit=True,
        )
        plt.close(fig)
        return HTML(animation.to_jshtml())

    def plot_sustained_surface_wind_speed_distribution_over_average_day(
        self, SAcup_wind_constraints=False
    ):
        """Plots shown in the animation of how the sustained surface wind speed distribution varies throughout the day."""
        # Gather animation data
        average_wind_speed_at_given_hour = {}
        for hour in list(self.surfaceDataDict.values())[0].keys():
            wind_speed_values_for_this_hour = []
            for dayDict in self.surfaceDataDict.values():
                try:
                    wind_speed_values_for_this_hour += [
                        (
                            dayDict[hour]["surface10mWindVelocityX"] ** 2
                            + dayDict[hour]["surface10mWindVelocityY"] ** 2
                        )
                        ** 0.5
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour (probably the last one)
                    # No need to worry, just average over the other days
                    pass
            average_wind_speed_at_given_hour[hour] = wind_speed_values_for_this_hour

        # Create grid of plots for each hour
        hours = list(list(self.pressureLevelDataDict.values())[0].keys())
        ncols, nrows = self._find_two_closest_integer_factors(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for (i, j) in [(i, j) for i in range(nrows) for j in range(ncols)]:
            hour = hours[i * ncols + j]
            ax = axs[i, j]
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.hist(
                average_wind_speed_at_given_hour[hour],
                bins=int(len(average_wind_speed_at_given_hour[hour]) ** 0.5),
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                label="Wind speed Speed Distribution",
            )
            ax.autoscale(enable=True, axis="y", tight=True)
            # Plot weibull distribution
            c, loc, scale = stats.weibull_min.fit(
                average_wind_speed_at_given_hour[hour], loc=0, scale=1
            )
            x = np.linspace(
                0, np.ceil(self.calculate_maximum_surface_10m_wind_speed()), 100
            )
            ax.plot(
                x,
                stats.weibull_min.pdf(x, c, loc, scale),
                "r-",
                linewidth=2,
                label="Weibull Distribution",
            )
            current_x_max = ax.get_xlim()[1]
            current_y_max = ax.get_ylim()[1]
            x_max = current_x_max if current_x_max > x_max else x_max
            y_max = current_y_max if current_y_max > y_max else y_max
            ax.label_outer()
            ax.grid()
        # Set x and y limits for the last axis. Since axes are shared, set to all
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=5, prune="lower")
        )
        ax.yaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=4, prune="lower")
        )

        if SAcup_wind_constraints:
            for (i, j) in [(i, j) for i in range(nrows) for j in range(ncols)]:
                # SA Cup altitude constraints region
                ax = axs[i, j]
                ax.vlines(
                    convert_units(20, "mph", self.unit_system["wind_speed"]),
                    0,
                    ax.get_ylim()[1],
                    "g",
                    (0, (15, 5, 2, 5)),
                    label="SA Cup Wind Constraints",
                )

        # Set title and axis labels for entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Average Wind Profile")
        fig.supxlabel(
            f"Sustained Surface Wind Speed ({self.unit_system['wind_speed']})"
        )
        fig.supylabel("Probability")
        plt.show()

    def animate_sustained_surface_wind_speed_distribution_over_average_day(
        self, SAcup_wind_constraints=False
    ):  # TODO: getting weird results
        """Animation of how the sustained surface wind speed distribution varies throughout the day."""
        # Gather animation data
        surface_wind_speeds_at_given_hour = {}
        for hour in list(self.surfaceDataDict.values())[0].keys():
            surface_wind_speed_values_for_this_hour = []
            for dayDict in self.surfaceDataDict.values():
                try:
                    surface_wind_speed_values_for_this_hour += [
                        (
                            dayDict[hour]["surface10mWindVelocityX"] ** 2
                            + dayDict[hour]["surface10mWindVelocityY"] ** 2
                        )
                        ** 0.5
                    ]
                except KeyError:
                    # Some day does not have data for the desired hour (probably the last one)
                    # No need to worry, just average over the other days
                    pass
            surface_wind_speeds_at_given_hour[
                hour
            ] = surface_wind_speed_values_for_this_hour

        # Create animation
        fig, ax = plt.subplots(dpi=200)
        # Initialize animation artists: histogram and hour text
        hist_bins = np.linspace(
            0, np.ceil(self.calculate_maximum_surface_10m_wind_speed()), 25
        )  # Fix bins edges
        _, _, bar_container = plt.hist(
            [],
            bins=hist_bins,
            alpha=0.2,
            label="Sustained Surface Wind Speed Distribution",
        )
        (ln,) = plt.plot(
            [],
            [],
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )
        tx = plt.text(
            x=0.95,
            y=0.95,
            s="",
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=24,
        )

        # Define function to initialize animation
        def init():
            ax.set_xlim(0, np.ceil(self.calculate_maximum_surface_10m_wind_speed()))
            ax.set_ylim(0, 0.3)  # TODO: parametrize
            ax.set_xlabel(
                f"Sustained Surface Wind Speed ({self.unit_system['wind_speed']})"
            )
            ax.set_ylabel("Probability")
            ax.set_title("Sustained Surface Wind Distribution")
            # ax.grid(True)

            if SAcup_wind_constraints:
                ax.vlines(
                    convert_units(20, "mph", self.unit_system["wind_speed"]),
                    0,
                    0.3,  # TODO: parametrize
                    "g",
                    (0, (15, 5, 2, 5)),
                    label="SAcup wind speed constraints",
                )  # Plot SAcup wind speed constraints

            return ln, *bar_container.patches, tx

        # Define function which sets each animation frame
        def update(frame):
            # Update histogram
            data = frame[1]
            hist, _ = np.histogram(data, hist_bins, density=True)
            for count, rect in zip(hist, bar_container.patches):
                rect.set_height(count)
            # Update weibull distribution
            c, loc, scale = stats.weibull_min.fit(data, loc=0, scale=1)
            xdata = np.linspace(
                0, np.ceil(self.calculate_maximum_surface_10m_wind_speed()), 100
            )
            ydata = stats.weibull_min.pdf(xdata, c, loc, scale)
            ln.set_data(xdata, ydata)
            # Update hour text
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return ln, *bar_container.patches, tx

        for frame in surface_wind_speeds_at_given_hour.items():
            update(frame)

        animation = FuncAnimation(
            fig,
            update,
            frames=surface_wind_speeds_at_given_hour.items(),
            interval=750,
            init_func=init,
            blit=True,
        )
        plt.close(fig)
        return HTML(animation.to_jshtml())

    @property
    def altitude_AGL_range(self):
        min_altitude = 0
        max_altitudes = [
            np.max(dayDict[hour]["windSpeed"].source[-1, 0])
            for dayDict in self.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        max_altitude = np.min(max_altitudes)
        return min_altitude, max_altitude

    def process_wind_profile_over_average_day(self):
        """Compute the average wind profile for each available hour of a day, over all
        days in the dataset."""
        altitude_list = np.linspace(*self.altitude_AGL_range, 100)

        average_wind_profile_at_given_hour = {}
        self.max_average_wind_at_altitude = 0
        hours = list(self.pressureLevelDataDict.values())[0].keys()
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

    def plot_wind_profile_over_average_day(self, SAcup_altitude_constraints=False):
        """Creates a grid of plots with the wind profile over the average day."""
        self.process_wind_profile_over_average_day()

        # Create grid of plots for each hour
        hours = list(list(self.pressureLevelDataDict.values())[0].keys())
        ncols, nrows = self._find_two_closest_integer_factors(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, np.inf, 0
        for (i, j) in [(i, j) for i in range(nrows) for j in range(ncols)]:
            hour = hours[i * ncols + j]
            ax = axs[i, j]
            ax.plot(*self.average_wind_profile_at_given_hour[hour], "r-")
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.autoscale(enable=True, axis="y", tight=True)
            current_x_max = ax.get_xlim()[1]
            current_y_min, current_y_max = ax.get_ylim()
            x_max = current_x_max if current_x_max > x_max else x_max
            y_max = current_y_max if current_y_max > y_max else y_max
            y_min = current_y_min if current_y_min < y_min else y_min
            ax.label_outer()
            ax.grid()
        # Set x and y limits for the last axis. Since axes are shared, set to all
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=5, prune="lower")
        )
        ax.yaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=4, prune="lower")
        )

        if SAcup_altitude_constraints:
            for (i, j) in [(i, j) for i in range(nrows) for j in range(ncols)]:
                # SA Cup altitude constraints region
                ax = axs[i, j]
                ax.fill_between(
                    [x_min, x_max],
                    0.7 * convert_units(10000, "ft", self.unit_system["length"]),
                    1.3 * convert_units(10000, "ft", self.unit_system["length"]),
                    color="g",
                    alpha=0.2,
                    label=f"10,000 {self.unit_system['length']} ± 30%",
                )
                ax.fill_between(
                    [x_min, x_max],
                    0.7 * convert_units(30000, "ft", self.unit_system["length"]),
                    1.3 * convert_units(30000, "ft", self.unit_system["length"]),
                    color="g",
                    alpha=0.2,
                    label=f"30,000 {self.unit_system['length']} ± 30%",
                )

        # Set title and axis labels for entire figure
        fig.suptitle("Average Wind Profile")
        fig.supxlabel(f"Wind speed ({self.unit_system['wind_speed']})")
        fig.supylabel(f"Altitude AGL ({self.unit_system['length']})")
        plt.show()

    def animate_wind_profile_over_average_day(self, SAcup_altitude_constraints=False):
        """Animation of how wind profile evolves throughout an average day."""
        self.process_wind_profile_over_average_day()

        # Create animation
        fig, ax = plt.subplots(dpi=200)
        # Initialize animation artists: curve and hour text
        (ln,) = plt.plot([], [], "r-")
        tx = plt.text(
            x=0.95,
            y=0.95,
            s="",
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=24,
        )
        # Define function to initialize animation

        def init():
            altitude_list = np.linspace(*self.altitude_AGL_range, 100)
            ax.set_xlim(0, self.max_average_wind_at_altitude + 5)
            ax.set_ylim(*self.altitude_AGL_range)
            ax.set_xlabel(f"Wind Speed ({self.unit_system['wind_speed']})")
            ax.set_ylabel(f"Altitude AGL ({self.unit_system['length']})")
            ax.set_title("Average Wind Profile")
            ax.grid(True)
            return ln, tx

        # Define function which sets each animation frame
        def update(frame):
            xdata = frame[1][0]
            ydata = frame[1][1]
            ln.set_data(xdata, ydata)
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return ln, tx

        animation = FuncAnimation(
            fig,
            update,
            frames=self.average_wind_profile_at_given_hour.items(),
            interval=1000,
            init_func=init,
            blit=True,
        )

        if SAcup_altitude_constraints:
            # SA Cup altitude constraints region
            ax.fill_between(
                [0, self.max_average_wind_at_altitude + 5],
                0.7 * convert_units(10000, "ft", self.unit_system["length"]),
                1.3 * convert_units(10000, "ft", self.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.unit_system['length']} ± 30%",
            )
            ax.fill_between(
                [0, self.max_average_wind_at_altitude + 5],
                0.7 * convert_units(30000, "ft", self.unit_system["length"]),
                1.3 * convert_units(30000, "ft", self.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.unit_system['length']} ± 30%",
            )
            fig.legend(loc="upper right")

        plt.close(fig)
        return HTML(animation.to_jshtml())

    def allInfo(self):
        print("Pressure Information")
        print(
            f"Average Surface Pressure: {self.average_surface_pressure:.2f} ± {self.std_surface_pressure:.2f} {self.unit_system['pressure']}"
        )
        print(
            f"Average Pressure at {convert_units(1000, 'ft', self.current_units['height_ASL']):.0f} {self.current_units['height_ASL']}: {self.average_pressure_at_1000ft:.2f} ± {self.std_pressure_at_1000ft:.2f} {self.unit_system['pressure']}"
        )
        print(
            f"Average Pressure at {convert_units(10000, 'ft', self.current_units['height_ASL']):.0f} {self.current_units['height_ASL']}: {self.average_pressure_at_10000ft:.2f} ± {self.std_pressure_at_1000ft:.2f} {self.unit_system['pressure']}"
        )
        print(
            f"Average Pressure at {convert_units(30000, 'ft', self.current_units['height_ASL']):.0f} {self.current_units['height_ASL']}: {self.average_pressure_at_30000ft:.2f} ± {self.std_pressure_at_1000ft:.2f} {self.unit_system['pressure']}"
        )
        print()

        print(
            f"Sustained Surface Wind Speed Information ({convert_units(10, 'm', self.unit_system['length']):.0f} {self.unit_system['length']} above ground)"
        )
        print(
            f"Historical Maximum Wind Speed: {self.record_max_surface_10m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Historical Minimum Wind Speed: {self.record_min_surface_10m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Maximum Wind Speed: {self.average_max_surface_10m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Minimum Wind Speed: {self.average_min_surface_10m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print()

        print(
            f"Elevated Wind Speed Information ({convert_units(100, 'm', self.unit_system['length']):.0f} {self.unit_system['length']} above ground)"
        )
        print(
            f"Historical Maximum Wind Speed: {self.record_max_surface_100m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Historical Minimum Wind Speed: {self.record_min_surface_100m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Maximum Wind Speed: {self.average_max_surface_100m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Minimum Wind Speed: {self.average_min_surface_100m_wind_speed:.2f} {self.unit_system['wind_speed']}"
        )
        print()

        print("Wind Gust Information")
        print(
            f"Historical Maximum Wind Gust: {self.max_wind_gust:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Maximum Wind Gust: {self.average_max_wind_gust:.2f} {self.unit_system['wind_speed']}"
        )
        print()

        print("Temperature Information")
        print(
            f"Historical Maximum Temperature: {self.record_max_temperature:.2f} {self.unit_system['temperature']}"
        )
        print(
            f"Historical Minimum Temperature: {self.record_min_temperature:.2f} {self.unit_system['temperature']}"
        )
        print(
            f"Average Daily Maximum Temperature: {self.average_max_temperature:.2f} {self.unit_system['temperature']}"
        )
        print(
            f"Average Daily Minimum Temperature: {self.average_min_temperature:.2f} {self.unit_system['temperature']}"
        )
        print()

        print("Precipitation Information")
        print(
            f"Percentage of Days with Precipitation: {100*self.percentage_of_days_with_precipitation:.1f}%"
        )
        print(
            f"Maximum Precipitation: {max(self.precipitation_per_day):.1f} {self.unit_system['precipitation']}"
        )
        print(
            f"Average Precipitation: {np.mean(self.precipitation_per_day):.1f} {self.unit_system['precipitation']}"
        )
        print()

        print("Cloud Base Height Information")
        print(
            f"Average Cloud Base Height: {self.mean_cloud_base_height:.2f} {self.unit_system['length']}"
        )
        print(
            f"Minimum Cloud Base Height: {self.min_cloud_base_height:.2f} {self.unit_system['length']}"
        )
        print(
            f"Percentage of Days Without Clouds: {100*self.percentage_of_days_with_no_cloud_coverage:.1f} %"
        )
