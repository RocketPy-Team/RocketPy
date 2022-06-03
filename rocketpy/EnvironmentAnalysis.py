from datetime import datetime, timedelta
import bisect

import ipywidgets as widgets
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


from matplotlib.animation import FuncAnimation, PillowWriter as ImageWriter
import matplotlib.ticker as mtick

from windrose import WindAxes, WindroseAxes
import netCDF4
from cftime import num2pydate
import pytz
from timezonefinder import TimezoneFinder

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
        elevation,
        surfaceDataFile=None,
        pressureLevelDataFile=None,
        timezone=None,
    ):
        # Save inputs
        self.start_date = start_date
        self.end_date = end_date
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.surfaceDataFile = surfaceDataFile
        self.pressureLevelDataFile = pressureLevelDataFile
        self.prefered_timezone = timezone

        # Manage units and timezones
        self.__init_data_parsing_units()
        self.__find_prefered_timezone()
        self.__localize_input_dates()

        # Parse data files
        self.pressureLevelDataDict = {}
        self.surfaceDataDict = {}
        self.parsePressureLevelData()
        self.parseSurfaceData()

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
        self.average_wind_profile_13_sigma = Function(0)
        self.average_wind_profile_at_given_hour = None

        self.max_wind_speed = None
        self.min_wind_speed = None
        self.wind_speed_per_hour = None
        self.wind_direction_per_hour = None

        # Run calculations
        # self.process_data()

    # TODO: Check implementation and use when parsing files
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
            "surface100mWindVelocityX": "v100",  # TODO: fix this to u100 when you have a good file
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
        Made for latitudes and longitudes, suporting arrays that range from
        -180 to 180 or from 0 to 360.

        TODO: improve docs

        Parameters
        ----------
        array : array
        value : float

        Returns
        -------
        index : int
        """
        # Create value convetion
        if np.min(array) < 0:
            # File uses range from -180 to 180, make sure value follows convetion
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
        dateTime = dateTimeUTC.astimezone(self.prefered_timezone)
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

        # Get values for variable on the four nearest poins
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

        # Get values for variable on the four nearest poins
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
            self.start_date = self.prefered_timezone.localize(self.start_date)
        if self.end_date.tzinfo is None:
            self.end_date = self.prefered_timezone.localize(self.end_date)

    def __find_prefered_timezone(self):
        if self.prefered_timezone is None:
            # Use local timezone based on lat lon pair
            tf = TimezoneFinder()
            self.prefered_timezone = pytz.timezone(
                tf.timezone_at(lng=self.longitude, lat=self.latitude)
            )
        elif isinstance(self.prefered_timezone, str):
            self.prefered_timezone = pytz.timezone(self.prefered_timezone)

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
        """Initialize prefered units for output (SI, metric or imperial)."""
        if self.unit_system_string == "metric":
            self.unit_system = {
                "length": "km",
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
                "wind_speed": "knot",
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

    # TODO: Needs tests
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

    # TODO: Needs tests
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
            heightAboveSeaLevelArray = self.__compute_height_above_sea_level(
                geopotentialArray
            )

            # Loop through wind components and temperature, get value and convert to Function
            for key, value in self.pressureLevelFileDict.items():
                valueArray = self.__extractPressureLevelDataValue(
                    pressureLevelData, value, indices, lonArray, latArray
                )
                variablePointsArray = np.array([heightAboveSeaLevelArray, valueArray]).T
                variableFunction = Function(
                    variablePointsArray,
                    inputs="Height Above Sea Level (m)",
                    outputs=key,
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
            )
            self.pressureLevelDataDict[dateString][hourString][
                "windDirection"
            ] = windDirectionFunction

        return self.pressureLevelDataDict

    # TODO: Needs tests
    def parseSurfaceData(self):
        """
        Parse surface data from a weather file.
        Currently only supports files from ECMWF.

        Must get the following variables:
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

        return self.surfaceDataDict

    # TODO: Needs tests
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

    # TODO: Needs tests
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

    # Calculations
    def process_data(self):
        self.calculate_average_max_temperature()
        self.calculate_average_min_temperature()
        self.calculate_record_max_temperature()
        self.calculate_record_min_temperature()
        self.calculate_average_max_wind_gust()
        self.calculate_maximum_wind_gust()
        self.calculate_wind_gust_distribution()
        self.calculate_average_temperature_along_day()
        self.calculate_average_wind_profile()
        self.calculate_average_day_wind_rose()

    # TODO: Create tests
    def calculate_average_max_temperature(self):
        self.max_temperature_list = [
            np.max([dayDict[hour]["surfaceTemperature"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_max_temperature = np.average(self.max_temperature_list)
        return self.average_max_temperature

    # TODO: Create tests
    def calculate_average_min_temperature(self):
        self.min_temperature_list = [
            np.min([dayDict[hour]["surfaceTemperature"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_min_temperature = np.average(self.min_temperature_list)
        return self.average_min_temperature

    # TODO: Create tests
    def calculate_record_max_temperature(self):
        self.temperature_list = [
            dayDict[hour]["surfaceTemperature"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_max_temperature = np.max(self.temperature_list)
        return self.record_max_temperature

    # TODO: Create tests
    def calculate_record_min_temperature(self):
        self.temperature_list = [
            dayDict[hour]["surfaceTemperature"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.record_min_temperature = np.min(self.temperature_list)
        return self.record_min_temperature

    # TODO: Create tests
    def calculate_average_max_wind_gust(self):
        self.max_wind_gust_list = [
            np.max([dayDict[hour]["surfaceWindGust"] for hour in dayDict.keys()])
            for dayDict in self.surfaceDataDict.values()
        ]
        self.average_max_wind_gust = np.average(self.max_wind_gust_list)
        return self.average_max_wind_gust

    # TODO: Create tests
    def calculate_maximum_wind_gust(self):
        self.wind_gust_list = [
            dayDict[hour]["surfaceWindGust"]
            for dayDict in self.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        self.max_wind_gust = np.max(self.wind_gust_list)
        return self.max_wind_gust

    # TODO: Create tests
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
        c, loc, scale = stats.weibull_min.fit(self.wind_gust_list, method="MM")
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
        plt.xlabel("Wind gust speed (m/s)")
        plt.title("Wind Gust Speed Distribution")
        plt.legend()
        plt.show()

        return None

    # TODO: Implement
    def calculate_average_temperature_along_day(self):
        """temperature progression throughout the day at some fine interval (ex: 2 hours) with 1, 2, 3, sigma contours"""
        ...

    # TODO: Create tests
    def plot_average_wind_speed_profile(self):
        """Average wind speed for all datetimes available."""
        altitude_list = list(list(self.pressureLevelDataDict.values())[0].values())[0][
            "windSpeed"
        ].source[:, 0]

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
        plt.ylim(altitude_list[0], altitude_list[-1])
        plt.xlabel(f"Wind speed ({self.unit_system['wind_speed']})")
        plt.ylabel(f"Altitude ({self.unit_system['length']})")
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
                    hour_wind_speed = self.pressureLevelDataDict[day][hour][
                        "windSpeed"
                    ](self.elevation)

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
                    windDir[hour].append(
                        self.pressureLevelDataDict[day][hour]["windDirection"](
                            self.elevation
                        )
                    )
                except KeyError:
                    # Not all days have all hours stored, that is fine
                    pass

        self.max_wind_speed = max_wind_speed
        self.min_wind_speed = min_wind_speed
        self.wind_speed_per_hour = windSpeed
        self.wind_direction_per_hour = windDir

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
            bins=np.linspace(self.min_wind_speed, self.max_wind_speed, 6),
            title=f"Windrose of an average day. Hour {float(hour):05.2f}".replace(
                ".", ":"
            ),
            fig=fig,
        )
        plt.show()

    # TODO: Create tests
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
        windrose_side = 3  # inches
        vertical_padding_top = 1  # inches
        plot_padding = 0.18  # percentage
        nrows, ncols = self._find_two_closest_integer_factors(len(hours))
        vertical_plot_area_percentage = (
            nrows * windrose_side / (nrows * windrose_side + vertical_padding_top)
        )

        # Create figure
        fig = plt.figure()
        fig.set_size_inches(
            ncols * windrose_side, nrows * windrose_side + vertical_padding_top
        )
        bins = np.linspace(self.min_wind_speed, self.max_wind_speed, 6)
        width = (1 - 2 * plot_padding) * 1 / ncols
        height = vertical_plot_area_percentage * (1 - 2 * plot_padding) * 1 / nrows

        for k, hour in enumerate(hours):
            i, j = len(hours) // nrows - k // ncols, k % ncols  # Row count bottom up
            left = j * 1 / ncols + plot_padding / ncols
            bottom = vertical_plot_area_percentage * (
                (i - 2) / nrows + plot_padding / nrows
            )

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
                    bbox_to_anchor=(ncols / 2 + 0.8, 1.5),  # 0.8 i a magic number
                    fancybox=True,
                    shadow=True,
                    ncol=6,
                )
            else:
                ax.legend().set_visible(False)
            fig.add_axes(ax)

        fig.suptitle("Wind Roses", fontsize=20, x=0.5, y=1)
        plt.show()

    def animate_average_wind_rose(self, figsize=(8, 8), filename="wind_rose.gif"):
        """Animates the wind_rose of an average day. The inputs of a wind_rose are the location of the
        place where we want to analyse, (x,y,z). The data is ensembled by hour, which means, the windrose
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
                    bins=np.linspace(self.min_wind_speed, self.max_wind_speed, 6),
                    title=f"Windrose of an average day. Hour {float(hour):05.2f}".replace(
                        ".", ":"
                    ),
                    fig=fig,
                )
                writer.grab_frame()
                plt.clf()

        with open(filename, "rb") as file:
            image = file.read()

        fig_width, fig_height = plt.gcf().get_size_inches() * fig.dpi
        return widgets.Image(
            value=image,
            format="gif",
            width=fig_width,
            height=fig_height,
        )

    def plot_wind_gust_distribution_over_average_day(self):
        """Plots shown in the animation of how the wind gust distribution varies throughout the day."""
        ...
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

        # Generate plots

        num_of_plots = len(list(self.surfaceDataDict.values())[0].keys())

        fig = plt.figure(figsize=(9, num_of_plots * 5))
        # plt.subplots_adjust(wspace=0.2,hspace=0.3)

        current_plot = 0
        for hour in list(self.surfaceDataDict.values())[0].keys():
            current_plot += 1
            ax = plt.subplot(num_of_plots, 2, current_plot)
            ax.hist(
                average_wind_gust_at_given_hour[hour],
                bins=int(len(average_wind_gust_at_given_hour[hour]) ** 0.5),
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                label="Wind Gust Speed Distribution",
            )

            # Plot weibull distribution
            c, loc, scale = stats.weibull_min.fit(average_wind_gust_at_given_hour[hour])
            x = np.linspace(0, np.max(average_wind_gust_at_given_hour[hour]), 100)
            ax.plot(
                x,
                stats.weibull_min.pdf(x, c, loc, scale),
                "r-",
                linewidth=2,
                label="Weibull Distribution",
            )

            # Label plot
            ax.set_ylim(0, 0.3)
            if current_plot % 2 != 0:
                ax.set_ylabel("Probability")
            ax.set_xlabel("Wind gust speed (m/s)")
            ax.set_title("Hour " + str(hour) + ":00")

        # set legend and title
        # TODO: fix legend and title position
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Wind Gust Speed Distribution", fontsize=16)

    # Animations

    # TODO: Implement
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
        fig, ax = plt.subplots()
        # Initialize animation artists: histogram and hour text
        hist_bins = np.linspace(0, 24, 25)  # Fix bins edges TODO: parametrize
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
            ax.set_xlim(0, 25)  # TODO: parametrize
            ax.set_ylim(0, 0.3)  # TODO: parametrize
            ax.set_xlabel("Wind Gust Speed (m/s)")
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
            c, loc, scale = stats.weibull_min.fit(data, method="MM")
            xdata = np.linspace(0, 25, 100)  # TODO: parametrize
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
            interval=1000,
            init_func=init,
            blit=True,
        )
        plt.show()

    # TODO: Create test
    def process_wind_profile_over_average_day(self):
        """Compute the average wind profile for each avaliable hour of a day, over all
        days in the dataset."""
        altitude_list = list(list(self.pressureLevelDataDict.values())[0].values())[0][
            "windSpeed"
        ].source[:, 0]
        average_wind_profile_at_given_hour = {}
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
            average_wind_profile_at_given_hour[hour] = [
                np.mean(wind_speed_values_for_this_hour, axis=0),
                altitude_list,
            ]
        self.average_wind_profile_at_given_hour = average_wind_profile_at_given_hour

    # TODO: Create test
    def plot_wind_profile_over_average_day(self):
        """Creates a grid of plots with the wind profile over the average day."""
        self.process_wind_profile_over_average_day()

        # Create grid of plots for each hour
        hours = list(list(self.pressureLevelDataDict.values())[0].keys())
        nrows, ncols = self._find_two_closest_integer_factors(len(hours))
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
        # Set title and axis labels for entire figure
        fig.suptitle("Average Wind Profile")
        fig.supxlabel(f"Wind speed ({self.unit_system['wind_speed']})")
        fig.supylabel(f"Altitude ASL ({self.unit_system['length']})")
        plt.show()

    # TODO: Create tests
    def animate_wind_profile_over_average_day(self):
        """Animation of how wind profile evolves throughout an average day."""
        self.process_wind_profile_over_average_day()

        # Create animation
        fig, ax = plt.subplots()
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
            altitude_list = list(list(self.pressureLevelDataDict.values())[0].values())[
                0
            ]["windSpeed"].source[:, 0]
            ax.set_xlim(0, 25)
            ax.set_ylim(self.elevation, max_altitude)
            ax.set_xlabel("Wind Speed (m/s)")
            ax.set_ylabel("Altitude (m)")
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
        plt.show()

    # Others
    # TODO: Addapt to new data format
    def wind_profile(self):
        windSpeed = []
        for idx in range(0, len(self.environments)):
            windSpeed.extend(self.environments[idx].windSpeed.source[:, 1])
        ax = WindAxes.from_ax()
        ax.pdf(windSpeed, Nbins=20)
        plt.show()

    def allInfo(self):
        print("Gust Information")
        print(
            f"Global Maximum wind gust: {self.max_wind_gust:.2f} {self.unit_system['wind_speed']}"
        )
        print(
            f"Average maximum wind gust: {self.average_max_wind_gust:.2f} {self.unit_system['wind_speed']}"
        )
        print("Temeprature Information")
        print(
            f"Global Maximum temperature: {self.record_max_temperature:.2f} {self.unit_system['temperature']}"
        )
        print(
            f"Global Minimum temperature: {self.record_min_temperature:.2f} {self.unit_system['temperature']}"
        )
        print(
            f"Average minimum temperture: {self.average_min_temperature:.2f} {self.unit_system['temperature']}"
        )
        print(
            f"Average maximum temperature: {self.average_max_temperature:.2f} {self.unit_system['temperature']}"
        )
