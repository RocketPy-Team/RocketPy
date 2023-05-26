__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np

from rocketpy.units import convert_units


class _EnvironmentAnalysisPrints:
    """Class to print environment analysis results

    Parameters
    ----------
    envAnalysis : EnvironmentAnalysis
        EnvironmentAnalysis object to be printed

    """

    def __init__(self, envAnalysis):
        self.envAnalysis = envAnalysis
        return None

    def dataset(self):
        print("Dataset Information: ")
        print(
            f"Time Period: From {self.envAnalysis.start_date} to {self.envAnalysis.end_date}"
        )  # TODO: Improve Timezone
        print(
            f"Available hours: From {self.envAnalysis.start_hour} to {self.envAnalysis.end_hour}"
        )  # TODO: Improve Timezone
        print("Surface Data File Path: ", self.envAnalysis.surfaceDataFile)
        print(
            "Latitude Range: From ",
            self.envAnalysis.singleLevelInitLat,
            "° To ",
            self.envAnalysis.singleLevelEndLat,
            "°",
        )
        print(
            "Longitude Range: From ",
            self.envAnalysis.singleLevelInitLon,
            "° To ",
            self.envAnalysis.singleLevelEndLon,
            "°",
        )
        print("Pressure Data File Path: ", self.envAnalysis.pressureLevelDataFile)
        print(
            "Latitude Range: From ",
            self.envAnalysis.pressureLevelInitLat,
            "° To ",
            self.envAnalysis.pressureLevelEndLat,
            "°",
        )
        print(
            "Longitude Range: From ",
            self.envAnalysis.pressureLevelInitLon,
            "° To ",
            self.envAnalysis.pressureLevelEndLon,
            "°",
        )
        return None

    def launch_site(self):
        # Print launch site details
        print("\nLaunch Site Details")
        print("Launch Site Latitude: {:.5f}°".format(self.envAnalysis.latitude))
        print("Launch Site Longitude: {:.5f}°".format(self.envAnalysis.longitude))
        print(
            "Surface Elevation (from surface data file): ", self.envAnalysis.elevation
        )  # TODO: Improve units
        print(
            "Max Expected Altitude: ",
            self.envAnalysis.maxExpectedAltitude,
            " ",
            self.envAnalysis.unit_system["length"],
        )
        return None

    def pressure(self):
        print("\nPressure Information")
        print(
            f"Average Surface Pressure: {self.envAnalysis.average_surface_pressure:.2f} ± {self.envAnalysis.std_surface_pressure:.2f} {self.envAnalysis.unit_system['pressure']}"
        )
        print(
            f"Average Pressure at {convert_units(1000, 'ft', self.envAnalysis.current_units['height_ASL']):.0f} {self.envAnalysis.current_units['height_ASL']}: {self.envAnalysis.average_pressure_at_1000ft:.2f} ± {self.envAnalysis.std_pressure_at_1000ft:.2f} {self.envAnalysis.unit_system['pressure']}"
        )
        print(
            f"Average Pressure at {convert_units(10000, 'ft', self.envAnalysis.current_units['height_ASL']):.0f} {self.envAnalysis.current_units['height_ASL']}: {self.envAnalysis.average_pressure_at_10000ft:.2f} ± {self.envAnalysis.std_pressure_at_1000ft:.2f} {self.envAnalysis.unit_system['pressure']}"
        )
        print(
            f"Average Pressure at {convert_units(30000, 'ft', self.envAnalysis.current_units['height_ASL']):.0f} {self.envAnalysis.current_units['height_ASL']}: {self.envAnalysis.average_pressure_at_30000ft:.2f} ± {self.envAnalysis.std_pressure_at_1000ft:.2f} {self.envAnalysis.unit_system['pressure']}"
        )
        return None

    def temperature(self):
        print("\nTemperature Information")
        print(
            f"Historical Maximum Temperature: {self.envAnalysis.record_max_temperature:.2f} {self.envAnalysis.unit_system['temperature']}"
        )
        print(
            f"Historical Minimum Temperature: {self.envAnalysis.record_min_temperature:.2f} {self.envAnalysis.unit_system['temperature']}"
        )
        print(
            f"Average Daily Maximum Temperature: {self.envAnalysis.average_max_temperature:.2f} {self.envAnalysis.unit_system['temperature']}"
        )
        print(
            f"Average Daily Minimum Temperature: {self.envAnalysis.average_min_temperature:.2f} {self.envAnalysis.unit_system['temperature']}"
        )
        return None

    def wind_speed(self):
        print(
            f"\nElevated Wind Speed Information ({convert_units(100, 'm', self.envAnalysis.unit_system['length']):.0f} {self.envAnalysis.unit_system['length']} above ground)"
        )
        print(
            f"\nSustained Surface Wind Speed Information ({convert_units(10, 'm', self.envAnalysis.unit_system['length']):.0f} {self.envAnalysis.unit_system['length']} above ground)"
        )
        print(
            f"Historical Maximum Wind Speed: {self.envAnalysis.record_max_surface_10m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Historical Minimum Wind Speed: {self.envAnalysis.record_min_surface_10m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Maximum Wind Speed: {self.envAnalysis.average_max_surface_10m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Minimum Wind Speed: {self.envAnalysis.average_min_surface_10m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Historical Maximum Wind Speed: {self.envAnalysis.record_max_surface_100m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Historical Minimum Wind Speed: {self.envAnalysis.record_min_surface_100m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Maximum Wind Speed: {self.envAnalysis.average_max_surface_100m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Minimum Wind Speed: {self.envAnalysis.average_min_surface_100m_wind_speed:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        return None

    def wind_gust(self):
        print("\nWind Gust Information")
        print(
            f"Historical Maximum Wind Gust: {self.envAnalysis.max_wind_gust:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        print(
            f"Average Daily Maximum Wind Gust: {self.envAnalysis.average_max_wind_gust:.2f} {self.envAnalysis.unit_system['wind_speed']}"
        )
        return None

    def precipitation(self):
        print("\nPrecipitation Information")
        print(
            f"Percentage of Days with Precipitation: {100*self.envAnalysis.percentage_of_days_with_precipitation:.1f}%"
        )
        print(
            f"Maximum Precipitation: {max(self.envAnalysis.precipitation_per_day):.1f} {self.envAnalysis.unit_system['precipitation']}"
        )
        print(
            f"Average Precipitation: {np.mean(self.envAnalysis.precipitation_per_day):.1f} {self.envAnalysis.unit_system['precipitation']}"
        )
        return None

    def cloud_coverage(self):
        print("\nCloud Base Height Information")
        print(
            f"Average Cloud Base Height: {self.envAnalysis.mean_cloud_base_height:.2f} {self.envAnalysis.unit_system['length']}"
        )
        print(
            f"Minimum Cloud Base Height: {self.envAnalysis.min_cloud_base_height:.2f} {self.envAnalysis.unit_system['length']}"
        )
        print(
            f"Percentage of Days Without Clouds: {100*self.envAnalysis.percentage_of_days_with_no_cloud_coverage:.1f} %"
        )
        return None

    def all(self):
        self.dataset()
        self.launch_site()
        self.pressure()
        self.temperature()
        self.wind_speed()
        self.wind_gust()
        self.precipitation()
        self.cloud_coverage()
        return None
