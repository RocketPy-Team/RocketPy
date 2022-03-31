from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from windrose import WindAxes, WindroseAxes

from rocketpy.Environment import Environment
from rocketpy.Function import Function


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
    ):
        # Save inputs
        self.start_date = start_date
        self.end_date = end_date
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.surfaceDataFile = surfaceDataFile
        self.pressureLevelDataFile = pressureLevelDataFile

        # Parse data files
        self.pressureLevelDataDict = {}
        self.surfaceDataDict = {}
        self.parsePressureLevelData()
        self.parseSurfaceData()

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

        # Run calculations
        self.process_data()

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

    # Parsing Files
    # TODO: Implement
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
        # TODO: Implement funciton
        # Inspiration: see Environment.processForecastReanalysis
        print("parsePressureLevelData not yet implemented.")
        self.pressureLevelDataDict = {}
        ...

    # TODO: Implement
    def parseSurfaceData(self):
        """
        Parse surface data from a weather file.

        Sources of information:
        - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-preliminary-back-extension?tab=overview
        -

        Must get the following variables:
        - 2m temperature
        - Surface pressure
        - 10m u-component of wind
        - 10m v-component of wind
        - 100m u-component of wind
        - 100m V-component of wind
        - Instantaneous 10m wind gust
        - Total precipitation
        - Cloud base height

        Must compute the following for each date and hour available in the dataset:
        - surfaceTemperature = float
        - surfacePressure = float
        - surface10mWindVelocityX = float
        - surface10mWindVelocityY = float
        - surface100mWindVelocityX = float
        - surface100mWindVelocityY = float
        - surfaceWindGust = float
        - totalPrecipitation = float
        - cloudBaseHeight = float

        Return a dictionary with all the computed data with the following structure:
        surfaceDataDict: {
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
        # TODO: Implement
        print("parseSurfaceData not yet implemented.")
        self.surfaceDataDict = {}
        ...

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

    # TODO: Adapt to new data format
    def calculate_average_max_temperature(self):
        self.average_max_temperature = np.average(
            [np.max(env.temperature.source[:, 1]) for env in self.environments]
        )

    # TODO: Adapt to new data format
    def calculate_average_min_temperature(self):
        self.average_min_temperature = np.average(
            [np.min(env.temperature.source[:, 1]) for env in self.environments]
        )

    # TODO: Adapt to new data format
    def calculate_record_max_temperature(self):
        self.record_max_temperature = np.max(
            [np.max(env.temperature.source[:, 1]) for env in self.environments]
        )

    # TODO: Adapt to new data format
    def calculate_record_min_temperature(self):
        self.record_min_temperature = np.min(
            [np.min(env.temperature.source[:, 1]) for env in self.environments]
        )

    # TODO: Implement
    def calculate_average_max_wind_gust(self):
        ...

    # TODO: Implement
    def calculate_maximum_wind_gust(self):
        ...

    # TODO: Implement
    def calculate_wind_gust_distribution(self):
        """Get all values of wind gust speed (for every date and hour available)
        and plot a single distribution. Expectedr result is a Weibull distribution.
        """
        ...

    # TODO: Implement
    def calculate_average_temperature_along_day(self):
        """temperature progression throughout the day at some fine interval (ex: 2 hours) with 1, 2, 3, sigma contours"""
        ...

    # TODO: Implement
    def calculate_average_wind_profile(self):
        """average, 1, 2, 3 sigma wind profile from 0 35,000 ft AGL"""
        ...

    # TODO: Implement
    def calculate_average_day_wind_rose(self):
        """average day wind rose"""
        ...

    # Animations
    # TODO: Implement
    def animate_wind_gust_distribution_over_average_day(self):
        """Animation of how the wind gust distribution varies throughout the day."""
        ...

    # TODO: Implement
    def animate_wind_profile_over_average_day(self):
        """Animation of how wind profile evolves throughout an average day."""
        ...

    # TODO: Adapt to new data format
    def animate_wind_rose(self):
        """Animation of how average wind rose evolves throughout an average day."""

        def get_data(i):
            windDirection = []
            windSpeed = []
            for idx in range(i, len(self.environments), 8):
                windDirection.extend(self.environments[idx].windDirection.source[:, 1])
                windSpeed.extend(self.environments[idx].windSpeed.source[:, 1])
            return windSpeed, windDirection

        ax = WindroseAxes.from_ax()
        for i in range(8):
            windSpeed, windDir = get_data(i)
            ax.bar(windSpeed, windDir, normed=True, opening=0.8, edgecolor="white")
            plt.pause(0.3)

        plt.show()

    # Others
    def wind_profile(self):
        windSpeed = []
        for idx in range(0, len(self.environments)):
            windSpeed.extend(self.environments[idx].windSpeed.source[:, 1])
        ax = WindAxes.from_ax()
        ax.pdf(windSpeed, Nbins=20)
        plt.show()

    def allInfo(self):
        print("Gust Information")
        print(f"Global Maximum wind gust: {self.maximum_wind_gust} m/s")
        print(f"Average maximum wind gust: {self.average_max_wind_gust} m/s")
        print("Temeprature Information")
        print(f"Global Maximum temperature: {self.record_max_temperature} ºC")
        print(f"Global Minimum temperature: {self.record_min_temperature} ºC")
        print(f"Average minimum temperture: {self.average_min_temperature} ºC")
        print(f"Average maximum temperature: {self.average_max_temperature} ºC")
