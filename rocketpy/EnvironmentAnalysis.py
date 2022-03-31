from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from windrose import WindAxes, WindroseAxes

from rocketpy.Environment import Environment


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
        latitude=0,
        longitude=0,
        elevation=0,
        timeZone="UTC",
        surfaceData=None,
        pressureLevelData=None,
    ):

        # not the bast fashion
        hours = (datetime(*end_date) - datetime(*start_date)).days * 24
        self.environments = []

        for hour in range(0, hours, 3):
            # error handling for days that there isn't data
            date = datetime(*start_date) + timedelta(hours=hour)

            # may be a solution more slow, but less intrusive on the environment class
            environment = Environment(
                railLength,
                gravity=gravity,
                date=date,
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
                datum=datum,
                timeZone=timeZone,
            )
            try:
                environment.setAtmosphericModel(type="Forecast", file="GFS")
            except ValueError as exc:
                print(str(exc))
                continue
            self.environments.append(environment)

        self.average_max_temperature = 0
        self.average_min_temperature = 0
        self.max_temperature = 0
        self.min_temperature = 0

        self.average_max_wind_gust = 0
        self.maximum_wind_gust = 0

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

    def parsePressureLevelData(self):
        """
        Parse pressure level data from a weather file.

        Sources of information:
        - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-preliminary-back-extension?tab=overview
        -

        Must get the following variables:
        - Geopotential
        - U-component of wind
        - V-component of wind
        - Temperature
        """
        pass

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

        """
        pass

    def process_data(self):
        self.calculate_average_max_temperature()
        self.calculate_average_min_temperature()
        self.calculate_max_temperature()
        self.calculate_min_temperature()

        self.calculate_average_max_wind_gust()
        self.calculate_maximum_wind_gust()

    def calculate_average_max_temperature(self):
        self.average_max_temperature = np.average(
            [np.max(env.temperature.source[:, 1]) for env in self.environments]
        )

    def calculate_average_min_temperature(self):
        self.average_min_temperature = np.average(
            [np.min(env.temperature.source[:, 1]) for env in self.environments]
        )

    def calculate_max_temperature(self):
        self.max_temperature = np.max(
            [np.max(env.temperature.source[:, 1]) for env in self.environments]
        )

    def calculate_min_temperature(self):
        self.min_temperature = np.min(
            [np.min(env.temperature.source[:, 1]) for env in self.environments]
        )

    def calculate_average_max_wind_gust(self):
        self.average_max_wind_gust = np.average(
            [np.max(env.windSpeed.source[:, 1]) for env in self.environments]
        )

    def calculate_maximum_wind_gust(self):
        self.maximum_wind_gust = np.max(
            [np.max(env.windSpeed.source[:, 1]) for env in self.environments]
        )

    def animate_wind_rose(self):
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
        print(f"Global Maximum temperature: {self.max_temperature} ºC")
        print(f"Global Minimum temperature: {self.min_temperature} ºC")
        print(f"Average minimum temperture: {self.average_min_temperature} ºC")
        print(f"Average maximum temperature: {self.average_max_temperature} ºC")
