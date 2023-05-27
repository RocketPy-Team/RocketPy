__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter as ImageWriter
from scipy import stats
from windrose import WindroseAxes

from rocketpy.units import convert_units
from ..tools import find_two_closest_integers

class _EnvironmentAnalysisPlots:
    """Class that holds plot methods for EnvironmentAnalysis class.

    Attributes
    ----------
    _EnvironmentAnalysisPlots.environment : EnvironmentAnalysis
        EnvironmentAnalysis object that will be used for the plots.
    """

    def __init__(self, env_analysis):
        """Initializes the class.

        Parameters
        ----------
        env_analysis : rocketpy.EnvironmentAnalysis
            Instance of the rocketpy EnvironmentAnalysis class

        Returns
        -------
        None
        """
        # Create height grid

        self.env_analysis = env_analysis

        return None

    def wind_gust_distribution(self):
        """Get all values of wind gust speed (for every date and hour available)
        and plot a single distribution. Expected result is a Weibull distribution.
        """
        self.env_analysis.wind_gust_list = [
            dayDict[hour]["surfaceWindGust"]
            for dayDict in self.env_analysis.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        plt.figure()
        # Plot histogram
        plt.hist(
            self.env_analysis.wind_gust_list,
            bins=int(len(self.env_analysis.wind_gust_list) ** 0.5),
            density=True,
            histtype="stepfilled",
            alpha=0.2,
            label="Wind Gust Speed Distribution",
        )

        # Plot weibull distribution
        c, loc, scale = stats.weibull_min.fit(
            self.env_analysis.wind_gust_list, loc=0, scale=1
        )
        x = np.linspace(0, np.max(self.env_analysis.wind_gust_list), 100)
        plt.plot(
            x,
            stats.weibull_min.pdf(x, c, loc, scale),
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )

        # Label plot
        plt.ylabel("Probability")
        plt.xlabel(f"Wind gust speed ({self.env_analysis.unit_system['wind_speed']})")
        plt.title("Wind Gust Speed Distribution")
        plt.legend()
        plt.show()

        return None

    def surface10m_wind_speed_distribution(self, windSpeedLimit=False):
        """Get all values of sustained surface wind speed (for every date and hour available)
        and plot a single distribution. Expected result is a Weibull distribution.
        """
        self.env_analysis.wind_speed_list = [
            (
                dayDict[hour]["surface10mWindVelocityX"] ** 2
                + dayDict[hour]["surface10mWindVelocityY"] ** 2
            )
            ** 0.5
            for dayDict in self.env_analysis.surfaceDataDict.values()
            for hour in dayDict.keys()
        ]
        plt.figure()
        # Plot histogram
        plt.hist(
            self.env_analysis.wind_speed_list,
            bins=int(len(self.env_analysis.wind_speed_list) ** 0.5),
            density=True,
            histtype="stepfilled",
            alpha=0.2,
            label="Wind Gust Speed Distribution",
        )

        # Plot weibull distribution
        c, loc, scale = stats.weibull_min.fit(
            self.env_analysis.wind_speed_list, loc=0, scale=1
        )
        x = np.linspace(0, np.max(self.env_analysis.wind_speed_list), 100)
        plt.plot(
            x,
            stats.weibull_min.pdf(x, c, loc, scale),
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )

        if windSpeedLimit:
            plt.vlines(
                convert_units(20, "mph", self.env_analysis.unit_system["wind_speed"]),
                0,
                0.3,
                "g",
                (0, (15, 5, 2, 5)),
                label="Wind Speed Limit",
            )  # Plot Wind Speed Limit

        # Label plot
        plt.ylabel("Probability")
        plt.xlabel(
            f"Sustained surface wind speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        plt.title("Sustained Surface Wind Speed Distribution")
        plt.legend()
        plt.show()

        return None

    def average_temperature_along_day(self):
        """Plots average temperature progression throughout the day, including
        sigma contours."""

        # Compute values
        self.env_analysis.calculate_average_temperature_along_day()

        # Get handy arrays
        hours = np.fromiter(
            self.env_analysis.average_temperature_at_given_hour.keys(), float
        )
        temperature_mean = self.env_analysis.average_temperature_at_given_hour.values()
        temperature_mean = np.array(list(temperature_mean))
        temperature_std = np.array(
            list(self.env_analysis.average_temperature_sigmas_at_given_hour.values())
        )
        temperatures_p1sigma = temperature_mean + temperature_std
        temperatures_m1sigma = temperature_mean - temperature_std
        temperatures_p2sigma = temperature_mean + 2 * temperature_std
        temperatures_m2sigma = temperature_mean - 2 * temperature_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.env_analysis.surfaceDataDict.values():
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
        plt.ylabel(f"Temperature ({self.env_analysis.unit_system['temperature']})")
        plt.title("Average Temperature Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()
        return None

    def average_surface10m_wind_speed_along_day(self, windSpeedLimit=False):
        """Plots average surface wind speed progression throughout the day, including
        sigma contours."""

        # Compute values
        self.env_analysis.calculate_average_sustained_surface10m_wind_along_day()

        # Get handy arrays
        hours = np.fromiter(
            self.env_analysis.average_surface10m_wind_speed_at_given_hour.keys(),
            float,
        )
        wind_speed_mean = (
            self.env_analysis.average_surface10m_wind_speed_at_given_hour.values()
        )
        wind_speed_mean = np.array(list(wind_speed_mean))
        wind_speed_std = np.array(
            list(
                self.env_analysis.average_surface10m_wind_speed_sigmas_at_given_hour.values()
            )
        )
        wind_speeds_p1sigma = wind_speed_mean + wind_speed_std
        wind_speeds_m1sigma = wind_speed_mean - wind_speed_std
        wind_speeds_p2sigma = wind_speed_mean + 2 * wind_speed_std
        wind_speeds_m2sigma = wind_speed_mean - 2 * wind_speed_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.env_analysis.surfaceDataDict.values():
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
        if windSpeedLimit:
            plt.hlines(
                convert_units(20, "mph", self.env_analysis.unit_system["wind_speed"]),
                min(hours),
                max(hours),
                "g",
                (0, (15, 5, 2, 5)),
                label="Wind Speed Limit",
            )  # Plot Wind Speed Limit
        plt.xlabel("Time (hours)")
        plt.ylabel(
            f"Surface Wind Speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        plt.title("Average Sustained Surface Wind Speed Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()
        return None

    def average_sustained_surface100m_wind_speed_along_day(self):
        """Plots average surface wind speed progression throughout the day, including
        sigma contours."""

        # Compute values
        self.env_analysis.calculate_average_sustained_surface100m_wind_along_day()

        # Get handy arrays
        hours = np.fromiter(
            self.env_analysis.average_surface100m_wind_speed_at_given_hour.keys(),
            float,
        )
        wind_speed_mean = (
            self.env_analysis.average_surface100m_wind_speed_at_given_hour.values()
        )
        wind_speed_mean = np.array(list(wind_speed_mean))
        wind_speed_std = np.array(
            list(
                self.env_analysis.average_surface100m_wind_speed_sigmas_at_given_hour.values()
            )
        )
        wind_speeds_p1sigma = wind_speed_mean + wind_speed_std
        wind_speeds_m1sigma = wind_speed_mean - wind_speed_std
        wind_speeds_p2sigma = wind_speed_mean + 2 * wind_speed_std
        wind_speeds_m2sigma = wind_speed_mean - 2 * wind_speed_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.env_analysis.surfaceDataDict.values():
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
        plt.ylabel(f"100m Wind Speed ({self.env_analysis.unit_system['wind_speed']})")
        plt.title("Average 100m Wind Speed Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()
        return None

    def average_wind_speed_profile(self, clear_range_limits=False):
        """Average wind speed for all datetimes available."""
        altitude_list = np.linspace(*self.env_analysis.altitude_AGL_range, 100)
        wind_speed_profiles = [
            dayDict[hour]["windSpeed"](altitude_list)
            for dayDict in self.env_analysis.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.env_analysis.average_wind_speed_profile = np.mean(
            wind_speed_profiles, axis=0
        )
        # Plot
        plt.figure()
        plt.plot(
            self.env_analysis.average_wind_speed_profile,
            altitude_list,
            "r",
            label="$\\mu$",
        )
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
        for wind_speed_profile in wind_speed_profiles:
            plt.plot(wind_speed_profile, altitude_list, "gray", alpha=0.01)

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            # Clear Sky Range Altitude Limits
            print(plt)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.fill_between(
                [xmin, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [xmin, xmax],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )

        plt.xlabel(f"Wind speed ({self.env_analysis.unit_system['wind_speed']})")
        plt.ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
        plt.xlim(0, 360)
        plt.title("Average Wind speed Profile")
        plt.legend()
        plt.xlim(0, max(np.percentile(wind_speed_profiles, 50 + 49.85, axis=0)))
        plt.show()
        return None

    def average_wind_heading_profile(self, clear_range_limits=False):
        """Average wind heading for all datetimes available."""
        altitude_list = np.linspace(*self.env_analysis.altitude_AGL_range, 100)

        wind_X_profiles = [
            dayDict[hour]["windVelocityX"](altitude_list)
            for dayDict in self.env_analysis.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.env_analysis.average_wind_X_profile = np.mean(wind_X_profiles, axis=0)

        wind_Y_profiles = [
            dayDict[hour]["windVelocityY"](altitude_list)
            for dayDict in self.env_analysis.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.env_analysis.average_wind_Y_profile = np.mean(wind_Y_profiles, axis=0)

        wind_heading_profiles = (
            np.arctan2(wind_X_profiles, wind_Y_profiles) * 180 / np.pi % 360
        )
        self.env_analysis.average_wind_heading_profile = (
            np.arctan2(
                self.env_analysis.average_wind_X_profile,
                self.env_analysis.average_wind_Y_profile,
            )
            * 180
            / np.pi
            % 360
        )

        # TODO: Add plot for wind X and wind Y profiles
        # Plot
        plt.figure()
        plt.plot(
            self.env_analysis.average_wind_heading_profile,
            altitude_list,
            "r",
            label="$\\mu$",
        )
        # plt.plot(
        #     np.percentile(wind_heading_profiles, 50 - 34.1, axis=0),
        #     altitude_list,
        #     "b--",
        #     alpha=1,
        #     label="$\\mu \\pm \\sigma$",
        # )
        # plt.plot(
        #     np.percentile(wind_heading_profiles, 50 + 34.1, axis=0),
        #     altitude_list,
        #     "b--",
        #     alpha=1,
        # )
        # plt.plot(
        #     np.percentile(wind_heading_profiles, 50 - 47.4, axis=0),
        #     altitude_list,
        #     "b--",
        #     alpha=0.5,
        #     label="$\\mu \\pm 2\\sigma$",
        # )
        # plt.plot(
        #     np.percentile(wind_heading_profiles, 50 + 47.7, axis=0),
        #     altitude_list,
        #     "b--",
        #     alpha=0.5,
        # )
        # for wind_heading_profile in wind_heading_profiles:
        #     plt.plot(wind_heading_profile, altitude_list, "gray", alpha=0.01)

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            # Clear Sky Range Altitude Limits
            print(plt)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.fill_between(
                [xmin, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [xmin, xmax],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )

        plt.xlabel(f"Wind heading ({self.env_analysis.unit_system['angle']})")
        plt.ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
        plt.xlim(0, 360)
        plt.title("Average Wind heading Profile")
        plt.legend()
        plt.show()
        return None

    def average_pressure_profile(self, clear_range_limits=False):
        """Average wind speed for all datetimes available."""
        altitude_list = np.linspace(*self.env_analysis.altitude_AGL_range, 100)
        pressure_profiles = [
            dayDict[hour]["pressure"](altitude_list)
            for dayDict in self.env_analysis.pressureLevelDataDict.values()
            for hour in dayDict.keys()
        ]
        self.env_analysis.average_pressure_profile = np.mean(pressure_profiles, axis=0)
        # Plot
        plt.figure()
        plt.plot(
            self.env_analysis.average_pressure_profile,
            altitude_list,
            "r",
            label="$\\mu$",
        )
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
        for pressure_profile in pressure_profiles:
            plt.plot(pressure_profile, altitude_list, "gray", alpha=0.01)

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            # Clear Sky Range Altitude Limits
            print(plt)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.fill_between(
                [xmin, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [xmin, xmax],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )

        plt.xlabel(f"Pressure ({self.env_analysis.unit_system['pressure']})")
        plt.ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
        plt.title("Average Pressure Profile")
        plt.legend()
        plt.xlim(0, max(np.percentile(pressure_profiles, 50 + 49.85, axis=0)))
        plt.show()
        return None

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

    def average_day_wind_rose_specific_hour(self, hour, fig=None):
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
            self.env_analysis.wind_direction_per_hour[hour],
            self.env_analysis.wind_speed_per_hour[hour],
            bins=self.env_analysis._beaufort_wind_scale(
                self.env_analysis.unit_system["wind_speed"],
                max_wind_speed=self.env_analysis.max_wind_speed,
            ),
            title=f"Wind Rose of an Average Day ({self.env_analysis.unit_system['wind_speed']}) - Hour {float(hour):05.2f}".replace(
                ".", ":"
            ),
            fig=fig,
        )
        plt.show()

        return None

    def average_day_wind_rose_all_hours(self):
        """Plot wind roses for all hours of a day, in a grid like plot."""
        # Get days and hours
        days = list(self.env_analysis.surfaceDataDict.keys())
        hours = list(self.env_analysis.surfaceDataDict[days[0]].keys())

        # Make sure necessary data has been calculated
        if not all(
            [
                self.env_analysis.max_wind_speed,
                self.env_analysis.min_wind_speed,
                self.env_analysis.wind_speed_per_hour,
                self.env_analysis.wind_direction_per_hour,
            ]
        ):
            self.env_analysis.process_wind_speed_and_direction_data_for_average_day()

        # Figure settings
        windrose_side = 2.5  # inches
        vertical_padding_top = 1.5  # inches
        plot_padding = 0.18  # percentage
        ncols, nrows = find_two_closest_integers(len(hours))
        vertical_plot_area_percentage = (
            nrows * windrose_side / (nrows * windrose_side + vertical_padding_top)
        )

        # Create figure
        fig = plt.figure()
        fig.set_size_inches(
            ncols * windrose_side, nrows * windrose_side + vertical_padding_top
        )
        bins = self.env_analysis._beaufort_wind_scale(
            self.env_analysis.unit_system["wind_speed"],
            max_wind_speed=self.env_analysis.max_wind_speed,
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
                self.env_analysis.wind_direction_per_hour[hour],
                self.env_analysis.wind_speed_per_hour[hour],
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
            f"Wind Roses ({self.env_analysis.unit_system['wind_speed']})",
            fontsize=20,
            x=0.5,
            y=1,
        )
        plt.show()
        return None

    def animate_average_wind_rose(self, figsize=(8, 8), filename="wind_rose.gif"):
        """Animates the wind_rose of an average day. The inputs of a wind_rose
        are the location of the place where we want to analyze, (x,y,z). The data
        is assembled by hour, which means, the windrose of a specific hour is
        generated by bringing together the data of all of the days available for
        that specific hour. It's possible to change the size of the gif using the
        parameter figsize, which is the height and width in inches.

        Parameters
        ----------
        figsize : array

        Returns
        -------
        Image : ipywidgets.widgets.widget_media.Image
        """
        days = list(self.env_analysis.surfaceDataDict.keys())
        hours = list(self.env_analysis.surfaceDataDict[days[0]].keys())

        if not all(
            [
                self.env_analysis.max_wind_speed,
                self.env_analysis.min_wind_speed,
                self.env_analysis.wind_speed_per_hour,
                self.env_analysis.wind_direction_per_hour,
            ]
        ):
            self.env_analysis.process_wind_speed_and_direction_data_for_average_day()

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
                    self.env_analysis.wind_direction_per_hour[hour],
                    self.env_analysis.wind_speed_per_hour[hour],
                    bins=self.env_analysis._beaufort_wind_scale(
                        self.env_analysis.unit_system["wind_speed"],
                        max_wind_speed=self.env_analysis.max_wind_speed,
                    ),
                    title=f"Wind Rose of an Average Day ({self.env_analysis.unit_system['wind_speed']}). Hour {float(hour):05.2f}".replace(
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

    def wind_gust_distribution_over_average_day(self):
        """Plots shown in the animation of how the wind gust distribution varies throughout the day."""
        # Gather animation data
        average_wind_gust_at_given_hour = {}
        for hour in list(self.env_analysis.surfaceDataDict.values())[0].keys():
            wind_gust_values_for_this_hour = []
            for dayDict in self.env_analysis.surfaceDataDict.values():
                try:
                    wind_gust_values_for_this_hour += [dayDict[hour]["surfaceWindGust"]]
                except KeyError:
                    # Some day does not have data for the desired hour (probably the last one)
                    # No need to worry, just average over the other days
                    pass
            average_wind_gust_at_given_hour[hour] = wind_gust_values_for_this_hour

        # Create grid of plots for each hour
        hours = list(list(self.env_analysis.pressureLevelDataDict.values())[0].keys())
        nrows, ncols = find_two_closest_integers(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
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
            x = np.linspace(0, np.ceil(self.env_analysis.max_wind_gust), 100)
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
        fig.supxlabel(
            f"Wind Gust Speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        fig.supylabel("Probability")
        plt.show()

        return None

    def animate_wind_gust_distribution_over_average_day(self):
        """Animation of how the wind gust distribution varies throughout the day."""
        # Gather animation data
        wind_gusts_at_given_hour = {}
        for hour in list(self.env_analysis.surfaceDataDict.values())[0].keys():
            wind_gust_values_for_this_hour = []
            for dayDict in self.env_analysis.surfaceDataDict.values():
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
        hist_bins = np.linspace(
            0, np.ceil(self.env_analysis.max_wind_gust), 25
        )  # Fix bins edges
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
            ax.set_xlim(0, np.ceil(self.env_analysis.max_wind_gust))
            ax.set_ylim(0, 0.3)  # TODO: parametrize
            ax.set_xlabel(
                f"Wind Gust Speed ({self.env_analysis.unit_system['wind_speed']})"
            )
            ax.set_ylabel("Probability")
            ax.set_title("Wind Gust Distribution")
            # ax.grid(True)
            return (ln, *bar_container.patches, tx)

        # Define function which sets each animation frame
        def update(frame):
            # Update histogram
            data = frame[1]
            hist, _ = np.histogram(data, hist_bins, density=True)
            for count, rect in zip(hist, bar_container.patches):
                rect.set_height(count)
            # Update weibull distribution
            c, loc, scale = stats.weibull_min.fit(data, loc=0, scale=1)
            xdata = np.linspace(0, np.ceil(self.env_analysis.max_wind_gust), 100)
            ydata = stats.weibull_min.pdf(xdata, c, loc, scale)
            ln.set_data(xdata, ydata)
            # Update hour text
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return (ln, *bar_container.patches, tx)

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

    def sustained_surface_wind_speed_distribution_over_average_day(
        self, windSpeedLimit=False
    ):
        """Plots shown in the animation of how the sustained surface wind speed distribution varies throughout the day."""
        # Gather animation data
        average_wind_speed_at_given_hour = {}
        for hour in list(self.env_analysis.surfaceDataDict.values())[0].keys():
            wind_speed_values_for_this_hour = []
            for dayDict in self.env_analysis.surfaceDataDict.values():
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
        hours = list(list(self.env_analysis.pressureLevelDataDict.values())[0].keys())
        ncols, nrows = find_two_closest_integers(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
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
                0,
                np.ceil(self.env_analysis.calculate_maximum_surface_10m_wind_speed()),
                100,
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

        if windSpeedLimit:
            for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
                # Clear Sky Range Altitude Limits j]
                ax = axs[i, j]
                ax.vlines(
                    convert_units(
                        20, "mph", self.env_analysis.unit_system["wind_speed"]
                    ),
                    0,
                    ax.get_ylim()[1],
                    "g",
                    (0, (15, 5, 2, 5)),
                    label="Wind Speed Limits",
                )

        # Set title and axis labels for entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Average Wind Profile")
        fig.supxlabel(
            f"Sustained Surface Wind Speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        fig.supylabel("Probability")
        plt.show()

        return None

    def animate_sustained_surface_wind_speed_distribution_over_average_day(
        self, windSpeedLimit=False
    ):
        # TODO: getting weird results since the 0.3 on y axis is not parametrized
        """Animation of how the sustained surface wind speed distribution varies throughout the day."""
        # Gather animation data
        surface_wind_speeds_at_given_hour = {}
        for hour in list(self.env_analysis.surfaceDataDict.values())[0].keys():
            surface_wind_speed_values_for_this_hour = []
            for dayDict in self.env_analysis.surfaceDataDict.values():
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
            0, np.ceil(self.env_analysis.calculate_maximum_surface_10m_wind_speed()), 25
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
            ax.set_xlim(
                0, np.ceil(self.env_analysis.calculate_maximum_surface_10m_wind_speed())
            )
            ax.set_ylim(0, 0.3)  # TODO: parametrize
            ax.set_xlabel(
                f"Sustained Surface Wind Speed ({self.env_analysis.unit_system['wind_speed']})"
            )
            ax.set_ylabel("Probability")
            ax.set_title("Sustained Surface Wind Distribution")
            # ax.grid(True)

            if windSpeedLimit:
                ax.vlines(
                    convert_units(
                        20, "mph", self.env_analysis.unit_system["wind_speed"]
                    ),
                    0,
                    0.3,  # TODO: parametrize
                    "g",
                    (0, (15, 5, 2, 5)),
                    label="Wind Speed Limit",
                )  # Plot Wind Speed Limit

            return (ln, *bar_container.patches, tx)

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
                0,
                np.ceil(self.env_analysis.calculate_maximum_surface_10m_wind_speed()),
                100,
            )
            ydata = stats.weibull_min.pdf(xdata, c, loc, scale)
            ln.set_data(xdata, ydata)
            # Update hour text
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return (ln, *bar_container.patches, tx)

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

    def wind_profile_over_average_day(self, clear_range_limits=False):
        """Creates a grid of plots with the wind profile over the average day."""
        self.env_analysis.process_wind_speed_profile_over_average_day()

        # Create grid of plots for each hour
        hours = list(list(self.env_analysis.pressureLevelDataDict.values())[0].keys())
        ncols, nrows = find_two_closest_integers(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, np.inf, 0
        for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
            hour = hours[i * ncols + j]
            ax = axs[i, j]
            ax.plot(*self.env_analysis.average_wind_profile_at_given_hour[hour], "r-")
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.autoscale(enable=True, axis="y", tight=True)
            current_x_max = ax.get_xlim()[1]
            current_y_min, current_y_max = ax.get_ylim()
            x_max = current_x_max if current_x_max > x_max else x_max
            y_max = current_y_max if current_y_max > y_max else y_max
            y_min = current_y_min if current_y_min < y_min else y_min

            if self.env_analysis.forecast:
                forecast = self.env_analysis.forecast
                y = self.env_analysis.average_wind_profile_at_given_hour[hour][1]
                x = forecast[hour].windSpeed.getValue(y) * convert_units(
                    1, "m/s", self.env_analysis.unit_system["wind_speed"]
                )
                ax.plot(x, y, "b--")

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

        if clear_range_limits:
            for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
                # Clear Sky Range Altitude Limits
                ax = axs[i, j]
                ax.fill_between(
                    [x_min, x_max],
                    0.7
                    * convert_units(
                        10000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    1.3
                    * convert_units(
                        10000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    color="g",
                    alpha=0.2,
                    label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
                )
                ax.fill_between(
                    [x_min, x_max],
                    0.7
                    * convert_units(
                        30000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    1.3
                    * convert_units(
                        30000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    color="g",
                    alpha=0.2,
                    label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
                )

        # Set title and axis labels for entire figure
        fig.suptitle("Average Wind Profile")
        fig.supxlabel(f"Wind speed ({self.env_analysis.unit_system['wind_speed']})")
        fig.supylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
        plt.show()

        return None

    def wind_heading_profile_over_average_day(self, clear_range_limits=False):
        """Creates a grid of plots with the wind heading profile over the average day."""
        self.env_analysis.process_wind_heading_profile_over_average_day()

        # Create grid of plots for each hour
        hours = list(list(self.env_analysis.pressureLevelDataDict.values())[0].keys())
        ncols, nrows = find_two_closest_integers(len(hours))
        fig = plt.figure(figsize=(ncols * 2, nrows * 2.2))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, np.inf, 0
        for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
            hour = hours[i * ncols + j]
            ax = axs[i, j]
            ax.plot(
                *self.env_analysis.average_wind_heading_profile_at_given_hour[hour],
                "r-",
            )
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
        # ax.set_xlim(x_min, x_max)
        ax.set_xlim(0, 360)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=5, prune="lower")
        )
        ax.yaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=4, prune="lower")
        )

        if clear_range_limits:
            for i, j in [(i, j) for i in range(nrows) for j in range(ncols)]:
                # Clear Sky range limits
                ax = axs[i, j]
                ax.fill_between(
                    [x_min, x_max],
                    0.7
                    * convert_units(
                        10000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    1.3
                    * convert_units(
                        10000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    color="g",
                    alpha=0.2,
                    label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
                )
                ax.fill_between(
                    [x_min, x_max],
                    0.7
                    * convert_units(
                        30000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    1.3
                    * convert_units(
                        30000, "ft", self.env_analysis.unit_system["length"]
                    ),
                    color="g",
                    alpha=0.2,
                    label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
                )

        # Set title and axis labels for entire figure
        fig.suptitle("Average Wind Heading Profile")
        fig.supxlabel(f"Wind heading ({self.env_analysis.unit_system['angle']})")
        fig.supylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
        plt.show()

        return None

    def animate_wind_profile_over_average_day(self, clear_range_limits=False):
        """Animation of how wind profile evolves throughout an average day."""
        self.env_analysis.process_wind_speed_profile_over_average_day()

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
            altitude_list = np.linspace(*self.env_analysis.altitude_AGL_range, 100)
            ax.set_xlim(0, self.env_analysis.max_average_wind_at_altitude + 5)
            ax.set_ylim(*self.env_analysis.altitude_AGL_range)
            ax.set_xlabel(f"Wind Speed ({self.env_analysis.unit_system['wind_speed']})")
            ax.set_ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
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
            frames=self.env_analysis.average_wind_profile_at_given_hour.items(),
            interval=1000,
            init_func=init,
            blit=True,
        )

        if clear_range_limits:
            # Clear sky range limits
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_at_altitude + 5],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_at_altitude + 5],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            fig.legend(loc="upper right")

        plt.close(fig)
        return HTML(animation.to_jshtml())

    def animate_wind_heading_profile_over_average_day(self, clear_range_limits=False):
        """Animation of how wind heading profile evolves throughout an average day."""
        self.env_analysis.process_wind_heading_profile_over_average_day()

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
            altitude_list = np.linspace(*self.env_analysis.altitude_AGL_range, 100)
            ax.set_xlim(0, 360)
            ax.set_ylim(*self.env_analysis.altitude_AGL_range)
            ax.set_xlabel(f"Wind Heading ({self.env_analysis.unit_system['angle']})")
            ax.set_ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
            ax.set_title("Average Wind Heading Profile")
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
            frames=self.env_analysis.average_wind_heading_profile_at_given_hour.items(),
            interval=1000,
            init_func=init,
            blit=True,
        )

        if clear_range_limits:
            # Clear sjy range limits
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_at_altitude + 5],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_at_altitude + 5],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            fig.legend(loc="upper right")

        plt.close(fig)
        return HTML(animation.to_jshtml())

    def all_animations(self):
        "Plots all the available animations together"

        self.animate_average_wind_rose()
        self.animate_wind_gust_distribution_over_average_day()
        self.animate_sustained_surface_wind_speed_distribution_over_average_day()
        self.animate_wind_heading_profile_over_average_day(clear_range_limits=True)
        self.animate_wind_profile_over_average_day()

        return None

    def all_plots(self):
        """Plots all the available plots together, this avoids having animations

        Returns
        -------
        None
        """
        self.wind_gust_distribution()
        self.surface10m_wind_speed_distribution()
        self.average_temperature_along_day()
        self.average_surface10m_wind_speed_along_day()
        self.average_sustained_surface100m_wind_speed_along_day()
        self.average_wind_speed_profile()
        self.average_wind_heading_profile()
        self.average_pressure_profile()
        self.average_day_wind_rose_all_hours()
        self.wind_gust_distribution_over_average_day()
        self.sustained_surface_wind_speed_distribution_over_average_day()
        self.wind_profile_over_average_day()
        self.wind_heading_profile_over_average_day()

        return None

    def info(self):
        "Plots only the most important plots together"
        self.wind_gust_distribution()
        self.average_wind_speed_profile()
        self.wind_profile_over_average_day()
        self.wind_heading_profile_over_average_day()

        return None

    def all(self):
        """Plots all the available plots and animations together. This method
        simply invokes the `all_plots` and `all_animations` methods.

        Returns
        -------
        None
        """
        self.all_plots()
        self.all_animations()

        return None
