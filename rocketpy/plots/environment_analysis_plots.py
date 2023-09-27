import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter as ImageWriter
from scipy import stats

from rocketpy.units import convert_units

from ..tools import find_two_closest_integers, import_optional_dependency

# TODO: `wind_speed_limit` and `clear_range_limits` and should be numbers, not booleans


class _EnvironmentAnalysisPlots:
    """Class that holds plot methods for EnvironmentAnalysis class.

    Attributes
    ----------
    _EnvironmentAnalysisPlots.env_analysis : EnvironmentAnalysis
        EnvironmentAnalysis object that will be used for the plots.
    _EnvironmentAnalysisPlots.surface_level_dict : dict
        Dictionary with all surface level data.
    _EnvironmentAnalysisPlots.pressure_level_dict : dict
        Dictionary with all pressure level data.
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
        # Save attributes
        self.env_analysis = env_analysis

        # Save commonly used attributes
        self.surface_level_dict = self.env_analysis.converted_surface_data
        self.pressure_level_dict = self.env_analysis.converted_pressure_level_data

        return None

    def __beaufort_wind_scale(self, units, max_wind_speed=None):
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
        wind_scale_knots = np.array(
            [0, 1, 3, 6, 10, 16, 21, 27, 33, 40, 47, 55, 63, 71]
        )
        wind_scale = wind_scale_knots * convert_units(1, "knot", units)
        wind_scale_truncated = wind_scale[np.where(wind_scale <= max_wind_speed)]
        if wind_scale[1] < 1:
            return np.round(wind_scale_truncated, 1)
        else:
            return np.round(wind_scale_truncated, 0)

    # Surface level plots

    def wind_gust_distribution(self):
        """Get all values of wind gust speed (for every date and hour available)
        and plot a single distribution. Expected result is a Weibull distribution,
        however, the result is not always a perfect fit, and sometimes it may
        look like a normal distribution.

        Returns
        -------
        None
        """
        plt.figure()
        # Plot histogram
        plt.hist(
            self.env_analysis.wind_gust_list,
            bins=int(len(self.env_analysis.wind_gust_list) ** 0.5),
            density=True,
            histtype="stepfilled",
            alpha=0.2,
            label="Wind Gust",
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
        plt.title("Wind Gust Speed Distribution (at surface)")
        plt.xlim(0, max(self.env_analysis.wind_gust_list))
        plt.legend()
        plt.show()

        return None

    def surface10m_wind_speed_distribution(self, wind_speed_limit=False):
        """Get all values of sustained surface wind speed (for every date and
        hour available) and plot a single distribution. Expected result is a
        Weibull distribution. The wind speed limit is plotted as a vertical line.

        Parameters
        ----------
        wind_speed_limit : bool, optional
            If True, plots the wind speed limit as a vertical line. The default
            is False.

        Returns
        -------
        None
        """
        plt.figure()
        # Plot histogram
        plt.hist(
            self.env_analysis.surface_10m_wind_speed_list,
            bins=int(len(self.env_analysis.surface_10m_wind_speed_list) ** 0.5),
            density=True,
            histtype="stepfilled",
            alpha=0.2,
            label="Wind Speed",
        )

        # Plot weibull distribution
        c, loc, scale = stats.weibull_min.fit(
            self.env_analysis.surface_10m_wind_speed_list, loc=0, scale=1
        )
        x = np.linspace(0, np.max(self.env_analysis.surface_10m_wind_speed_list), 100)
        plt.plot(
            x,
            stats.weibull_min.pdf(x, c, loc, scale),
            "r-",
            linewidth=2,
            label="Weibull Distribution",
        )

        if wind_speed_limit:
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
        plt.title("Sustained Wind Speed Distribution (at surface+10m)")
        plt.xlim(0, max(self.env_analysis.surface_10m_wind_speed_list))
        plt.legend()
        plt.show()

        return None

    def average_surface_temperature_evolution(self):
        """Plots average temperature progression throughout the day, including
        sigma contours.

        Returns
        -------
        None
        """

        # Get handy arrays
        temperature_mean = np.array(
            list(self.env_analysis.average_temperature_by_hour.values())
        )
        temperature_std = np.array(
            list(self.env_analysis.std_temperature_by_hour.values())
        )
        temperatures_p1sigma = temperature_mean + temperature_std
        temperatures_m1sigma = temperature_mean - temperature_std
        temperatures_p2sigma = temperature_mean + 2 * temperature_std
        temperatures_m2sigma = temperature_mean - 2 * temperature_std

        plt.figure()

        # Plot temperature along day for each available date
        for hour_entries in self.surface_level_dict.values():
            plt.plot(
                [int(hour) for hour in hour_entries.keys()],
                [val["surface_temperature"] for val in hour_entries.values()],
                "gray",
                alpha=0.1,
            )

        # Plot average temperature along day
        plt.plot(self.env_analysis.hours, temperature_mean, "r", label="$\\mu$")

        # Plot standard deviations temperature along day
        plt.plot(
            self.env_analysis.hours,
            temperatures_m1sigma,
            "b--",
            label=r"$\mu \pm \sigma$",
        )
        plt.plot(self.env_analysis.hours, temperatures_p1sigma, "b--")
        plt.plot(self.env_analysis.hours, temperatures_p2sigma, "b--", alpha=0.5)
        plt.plot(
            self.env_analysis.hours,
            temperatures_m2sigma,
            "b--",
            label=r"$\mu \pm 2\sigma $",
            alpha=0.5,
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

    def average_surface10m_wind_speed_evolution(self, wind_speed_limit=False):
        """Plots average surface wind speed progression throughout the day,
        including sigma contours.

        Parameters
        ----------
        wind_speed_limit : bool, optional
            If True, plots the wind speed limit as a horizontal line. The default
            is False.

        Returns
        -------
        None
        """

        # Get handy arrays
        wind_speed_mean = np.array(
            list((self.env_analysis.average_surface_10m_wind_speed_by_hour.values()))
        )

        wind_speed_std = np.array(
            list(self.env_analysis.std_surface_10m_wind_speed_by_hour.values())
        )

        wind_speeds_p1sigma = wind_speed_mean + wind_speed_std
        wind_speeds_m1sigma = wind_speed_mean - wind_speed_std
        wind_speeds_p2sigma = wind_speed_mean + 2 * wind_speed_std
        wind_speeds_m2sigma = wind_speed_mean - 2 * wind_speed_std

        plt.figure()

        # Plot average wind speed along day
        for hour_entries in self.surface_level_dict.values():
            plt.plot(
                [x for x in self.env_analysis.hours],
                [
                    (
                        val["surface10m_wind_velocity_x"] ** 2
                        + val["surface10m_wind_velocity_y"] ** 2
                    )
                    ** 0.5
                    for val in hour_entries.values()
                ],
                "gray",
                alpha=0.1,
            )

        # Plot average temperature along day
        plt.plot(self.env_analysis.hours, wind_speed_mean, "r", label="$\\mu$")

        # Plot standard deviations temperature along day
        plt.plot(
            self.env_analysis.hours,
            wind_speeds_m1sigma,
            "b--",
            label=r"$\mu \pm \sigma$",
        )
        plt.plot(self.env_analysis.hours, wind_speeds_p1sigma, "b--")
        plt.plot(self.env_analysis.hours, wind_speeds_p2sigma, "b--", alpha=0.5)
        plt.plot(
            self.env_analysis.hours,
            wind_speeds_m2sigma,
            "b--",
            label=r"$\mu \pm 2\sigma $",
            alpha=0.5,
        )

        # Format plot
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_formatter(
            lambda x, pos: "{0:02.0f}:{1:02.0f}".format(*divmod(x * 60, 60))
        )
        plt.autoscale(enable=True, axis="x", tight=True)

        if wind_speed_limit:
            plt.hlines(
                convert_units(20, "mph", self.env_analysis.unit_system["wind_speed"]),
                min(self.env_analysis.hours),
                max(self.env_analysis.hours),
                "g",
                (0, (15, 5, 2, 5)),
                label="Wind Speed Limit",
            )

        plt.xlabel("Time (hours)")
        plt.ylabel(
            f"Surface Wind Speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        plt.title("Average Sustained Surface Wind Speed Along Day")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()

        return None

    def average_surface100m_wind_speed_evolution(self):
        """Plots average surface wind speed progression throughout the day, including
        sigma contours.

        Returns
        -------
        None
        """

        # Get handy arrays
        wind_speed_mean = (
            self.env_analysis.average_surface_100m_wind_speed_by_hour.values()
        )
        wind_speed_mean = np.array(list(wind_speed_mean))
        wind_speed_std = np.array(
            list(self.env_analysis.std_surface_100m_wind_speed_by_hour.values())
        )
        wind_speeds_p1sigma = wind_speed_mean + wind_speed_std
        wind_speeds_m1sigma = wind_speed_mean - wind_speed_std
        wind_speeds_p2sigma = wind_speed_mean + 2 * wind_speed_std
        wind_speeds_m2sigma = wind_speed_mean - 2 * wind_speed_std

        plt.figure()
        # Plot temperature along day for each available date
        for hour_entries in self.surface_level_dict.values():
            plt.plot(
                [int(hour) for hour in hour_entries.keys()],
                [
                    (
                        val["surface100m_wind_velocity_x"] ** 2
                        + val["surface100m_wind_velocity_y"] ** 2
                    )
                    ** 0.5
                    for val in hour_entries.values()
                ],
                "gray",
                alpha=0.1,
            )

        # Plot average temperature along day
        plt.plot(self.env_analysis.hours, wind_speed_mean, "r", label="$\\mu$")

        # Plot standard deviations temperature along day
        plt.plot(
            self.env_analysis.hours,
            wind_speeds_m1sigma,
            "b--",
            label=r"$\mu \pm \sigma$",
        )
        plt.plot(self.env_analysis.hours, wind_speeds_p1sigma, "b--")
        plt.plot(self.env_analysis.hours, wind_speeds_p2sigma, "b--", alpha=0.5)
        plt.plot(
            self.env_analysis.hours,
            wind_speeds_m2sigma,
            "b--",
            label=r"$\mu \pm 2\sigma $",
            alpha=0.5,
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

    # Average profiles plots (pressure level data)

    def average_wind_speed_profile(self, clear_range_limits=False):
        """Average wind speed for all datetimes available. The plot also includes
        sigma contours.

        Parameters
        ----------
        clear_range_limits : bool, optional
            If True, clears the range limits. The default is False.

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(
            self.env_analysis.average_wind_speed_profile,
            self.env_analysis.altitude_list,
            "r",
            label="$\\mu$ speed",
        )
        plt.plot(
            np.percentile(
                self.env_analysis.wind_speed_profiles_list, 50 - 34.1, axis=0
            ),
            self.env_analysis.altitude_list,
            "b--",
            alpha=1,
            label="$\\mu \\pm \\sigma$",
        )
        plt.plot(
            np.percentile(
                self.env_analysis.wind_speed_profiles_list, 50 + 34.1, axis=0
            ),
            self.env_analysis.altitude_list,
            "b--",
            alpha=1,
        )
        plt.plot(
            np.percentile(
                self.env_analysis.wind_speed_profiles_list, 50 - 47.4, axis=0
            ),
            self.env_analysis.altitude_list,
            "b--",
            alpha=0.5,
            label="$\\mu \\pm 2\\sigma$",
        )
        plt.plot(
            np.percentile(
                self.env_analysis.wind_speed_profiles_list, 50 + 47.7, axis=0
            ),
            self.env_analysis.altitude_list,
            "b--",
            alpha=0.5,
        )
        for wind_speed_profile in self.env_analysis.wind_speed_profiles_list:
            plt.plot(
                wind_speed_profile, self.env_analysis.altitude_list, "gray", alpha=0.01
            )

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            x_min, xmax, _, _ = plt.axis()
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [x_min, xmax],
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
        plt.xlim(
            0,
            max(
                np.percentile(
                    self.env_analysis.wind_speed_profiles_list, 50 + 49.85, axis=0
                )
            ),
        )
        plt.show()

        return None

    def average_wind_velocity_xy_profile(self, clear_range_limits=False):
        """Average wind X and wind Y for all datetimes available. The X component
        is the wind speed in the direction of East, and the Y component is the
        wind speed in the direction of North.

        Parameters
        ----------
        clear_range_limits : bool, optional
            If True, clears the range limits. The default is False.

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(
            self.env_analysis.average_wind_velocity_x_profile,
            self.env_analysis.altitude_list,
            "r",
            label="$\\mu$ X",
        )
        plt.plot(
            self.env_analysis.average_wind_velocity_y_profile,
            self.env_analysis.altitude_list,
            "b",
            label="$\\mu$ Y",
        )

        if clear_range_limits:
            x_min, xmax, _, _ = plt.axis()
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        plt.xlabel(f"Wind speed ({self.env_analysis.unit_system['wind_speed']})")
        plt.ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")

        plt.title("Average Wind X and Y Profile")
        plt.legend()
        plt.grid()
        plt.show()

        return None

    def average_wind_heading_profile(self, clear_range_limits=False):
        """Average wind heading for all datetimes available.

        Parameters
        ----------
        clear_range_limits : bool, optional
            If True, clears the range limits. The default is False.

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(
            self.env_analysis.average_wind_heading_profile,
            self.env_analysis.altitude_list,
            "r",
            label="$\\mu$",
        )

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            x_min, xmax = 0, 360
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [x_min, xmax],
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
        """Average pressure profile for all datetimes available. The plot also
        includes sigma contours.

        Parameters
        ----------
        clear_range_limits : bool, optional
            If True, clears the range limits. The default is False.

        Returns
        -------
        None
        """

        plt.figure()
        plt.plot(
            np.mean(self.env_analysis.pressure_profiles_list, axis=0),
            self.env_analysis.altitude_list,
            "r",
            label="$\\mu$",
        )
        plt.plot(
            np.percentile(self.env_analysis.pressure_profiles_list, 15.9, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=1,
            label="$\\mu \\pm \\sigma$",
        )
        plt.plot(
            np.percentile(self.env_analysis.pressure_profiles_list, 84.1, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=1,
        )
        plt.plot(
            np.percentile(self.env_analysis.pressure_profiles_list, 2.6, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=0.5,
            label="$\\mu \\pm 2\\sigma$",
        )
        plt.plot(
            np.percentile(self.env_analysis.pressure_profiles_list, 97.4, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=0.5,
        )
        for pressure_profile in self.env_analysis.pressure_profiles_list:
            plt.plot(
                pressure_profile, self.env_analysis.altitude_list, "gray", alpha=0.01
            )

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            x_min, xmax, _, _ = plt.axis()
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [x_min, xmax],
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
        plt.xlim(
            0,
            max(np.percentile(self.env_analysis.pressure_profiles_list, 99.85, axis=0)),
        )
        plt.show()
        return None

    def average_temperature_profile(self, clear_range_limits=False):
        """Average temperature profile for all datetimes available. The plot
        also includes sigma contours.

        Parameters
        ----------
        clear_range_limits : bool, optional
            If True, clears the range limits. The default is False.

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(
            np.mean(self.env_analysis.temperature_profiles_list, axis=0),
            self.env_analysis.altitude_list,
            "r",
            label="$\\mu$",
        )
        plt.plot(
            np.percentile(self.env_analysis.temperature_profiles_list, 15.9, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=1,
            label="$\\mu \\pm \\sigma$",
        )
        plt.plot(
            np.percentile(self.env_analysis.temperature_profiles_list, 84.1, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=1,
        )
        plt.plot(
            np.percentile(self.env_analysis.temperature_profiles_list, 2.6, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=0.5,
            label="$\\mu \\pm 2\\sigma$",
        )
        plt.plot(
            np.percentile(self.env_analysis.temperature_profiles_list, 97.4, axis=0),
            self.env_analysis.altitude_list,
            "b--",
            alpha=0.5,
        )
        for pressure_profile in self.env_analysis.temperature_profiles_list:
            plt.plot(
                pressure_profile, self.env_analysis.altitude_list, "gray", alpha=0.01
            )

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)

        if clear_range_limits:
            x_min, xmax, ymax, ymin = plt.axis()
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            plt.fill_between(
                [x_min, xmax],
                0.7
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(30000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"30,000 {self.env_analysis.unit_system['length']} ± 30%",
            )

        plt.xlabel(f"Temperature ({self.env_analysis.unit_system['temperature']})")
        plt.ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
        plt.title("Average Temperature Profile")
        plt.legend()
        plt.xlim(
            min(
                np.percentile(
                    self.env_analysis.temperature_profiles_list, 99.85, axis=0
                )
            ),
            max(
                np.percentile(
                    self.env_analysis.temperature_profiles_list, 99.85, axis=0
                )
            ),
        )
        plt.show()

        return None

    # Wind roses (surface level data)

    @staticmethod
    def plot_wind_rose(
        wind_direction, wind_speed, bins=None, title=None, fig=None, rect=None
    ):
        """Plot a windrose given the data.

        Parameters
        ----------
        wind_direction: list[float]
            The wind direction.
        wind_speed: list[float]
            The wind speed
        bins: 1D array or integer, optional
            number of bins, or a sequence of bins variable. If not set, bins=6,
            then bins=linspace(min(var), max(var), 6)
        title: str, optional
            Title of the plot
        fig: matplotlib.pyplot.figure, optional
            Figure to plot the windrose

        Returns
        -------
        WindroseAxes
        """
        windrose = import_optional_dependency("windrose")
        WindroseAxes = windrose.WindroseAxes
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

    def average_wind_rose_specific_hour(self, hour, fig=None):
        """Plot a specific hour of the average windrose

        Parameters
        ----------
        hour: int
            Hour to be plotted
        fig: matplotlib.pyplot.figure
            Figure to plot the windrose

        Returns
        -------
        None
        """
        self.plot_wind_rose(
            self.env_analysis.surface_wind_direction_by_hour[hour],
            self.env_analysis.surface_wind_speed_by_hour[hour],
            bins=self.__beaufort_wind_scale(
                units=self.env_analysis.unit_system["wind_speed"],
                max_wind_speed=self.env_analysis.record_max_surface_wind_speed,
            ),
            title=f"Wind Rose of an Average Day ({self.env_analysis.unit_system['wind_speed']}) - Hour {float(hour):05.2f}".replace(
                ".", ":"
            ),
            fig=fig,
        )
        plt.show()

        return None

    def average_wind_rose_grid(self):
        """Plot wind roses for all hours of a day, in a grid like plot.

        Returns
        -------
        None
        """

        # Figure settings
        windrose_side = 2.5  # inches
        vertical_padding_top = 2.5  # inches
        plot_padding = 0.18  # percentage
        n_cols, n_rows = find_two_closest_integers(len(self.env_analysis.hours))
        vertical_plot_area_percentage = (
            n_rows * windrose_side / (n_rows * windrose_side + vertical_padding_top)
        )

        # Create figure
        fig = plt.figure()
        fig.set_size_inches(
            n_cols * windrose_side, n_rows * windrose_side + vertical_padding_top
        )
        bins = self.__beaufort_wind_scale(
            units=self.env_analysis.unit_system["wind_speed"],
            max_wind_speed=self.env_analysis.record_max_surface_wind_speed,
        )
        width = (1 - 2 * plot_padding) * 1 / n_cols
        height = vertical_plot_area_percentage * (1 - 2 * plot_padding) * 1 / n_rows
        for k, hour in enumerate(self.env_analysis.hours):
            # Row count bottom up
            i, j = len(self.env_analysis.hours) // n_rows - k // n_cols, k % n_cols
            left = j * 1 / n_cols + plot_padding / n_cols
            bottom = (
                vertical_plot_area_percentage
                * ((i - 2) / n_rows + plot_padding / n_rows)
                + 0.5
            )

            ax = self.plot_wind_rose(
                self.env_analysis.surface_wind_direction_by_hour[hour],
                self.env_analysis.surface_wind_speed_by_hour[hour],
                bins=bins,
                title=f"{float(hour):05.2f}".replace(".", ":"),
                fig=fig,
                rect=[left, bottom, width, height],
            )
            if k == 0:
                ax.legend(
                    loc="upper center",
                    # 0.8 is a magic number
                    bbox_to_anchor=(n_cols / 2 + 0.8, 1.8),
                    fancybox=True,
                    shadow=True,
                    ncol=n_cols,
                )
            else:
                ax.legend().set_visible(False)
            fig.add_axes(ax)

        fig.suptitle(
            f"Wind Roses ({self.env_analysis.unit_system['wind_speed']})",
            fontsize=17,
            x=0.5,
            y=1,
        )
        plt.bbox_inches = "tight"
        plt.show()
        return None

    def animate_average_wind_rose(self, figsize=(5, 5), filename="wind_rose.gif"):
        """Animates the wind_rose of an average day. The inputs of a wind_rose
        are the location of the place where we want to analyze, (x,y,z). The data
        is assembled by hour, which means, the windrose of a specific hour is
        generated by bringing together the data of all of the days available for
        that specific hour. It's possible to change the size of the gif using the
        parameter figsize, which is the height and width in inches.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure in inches, (width, height). The default is (8, 8).
        filename : str
            Name of the file to save the gif. The default is "wind_rose.gif".

        Returns
        -------
        Image : ipywidgets.widget_media.Image
        """
        widgets = import_optional_dependency("ipywidgets")
        metadata = dict(
            title="windrose",
            artist="windrose",
            comment="""Made with windrose
                http://www.github.com/scls19fr/windrose""",
        )
        writer = ImageWriter(fps=1, metadata=metadata)
        fig = plt.figure(facecolor="w", edgecolor="w", figsize=figsize)
        with writer.saving(fig, filename, 100):
            for hour in self.env_analysis.hours:
                self.plot_wind_rose(
                    self.env_analysis.surface_wind_direction_by_hour[hour],
                    self.env_analysis.surface_wind_speed_by_hour[hour],
                    bins=self.__beaufort_wind_scale(
                        units=self.env_analysis.unit_system["wind_speed"],
                        max_wind_speed=self.env_analysis.record_max_surface_wind_speed,
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

    # More plots and animations

    def wind_gust_distribution_grid(self):
        """Plots shown in the animation of how the wind gust distribution varies
        throughout the day.

        Returns
        -------
        None
        """
        wind_gusts = self.env_analysis.surface_wind_gust_by_hour

        # Create grid of plots to show a distribution for each hour
        n_rows, n_cols = find_two_closest_integers(len(self.env_analysis.hours))
        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2.2))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0

        # Iterate over all hours and plot histograms
        for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
            hour = self.env_analysis.hours[i * n_cols + j]
            ax = axs[i, j]
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.hist(
                wind_gusts[hour],
                bins=int(len(wind_gusts[hour]) ** 0.5),
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                label="Wind Gust",
            )
            ax.autoscale(enable=True, axis="y", tight=True)
            # Plot weibull distribution
            c, loc, scale = stats.weibull_min.fit(wind_gusts[hour], loc=0, scale=1)
            x = np.linspace(0, np.ceil(self.env_analysis.max_wind_gust_list), 50)
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
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
        )
        fig.suptitle("Wind Gust Distribution")
        fig.supxlabel(
            f"Wind Gust Speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        fig.supylabel("Probability")
        plt.show()

        return None

    def animate_wind_gust_distribution(self):
        """Animation of how the wind gust distribution varies throughout the day.
        Each frame is a histogram of the wind gust distribution for a specific hour.

        Returns
        -------
        HTML : IPython.core.display.HTML
            The animation as an HTML object
        """
        module = import_optional_dependency("IPython.display")
        HTML = module.HTML  # this is a class

        # Gather animation data
        wind_gusts = self.env_analysis.surface_wind_gust_by_hour

        # Create animation
        fig, ax = plt.subplots(dpi=200)
        # Initialize animation artists: histogram and hour text
        hist_bins = np.linspace(
            0, np.ceil(self.env_analysis.record_max_wind_gust), 25
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
        max_probability_density = 1.2 * max(
            stats.weibull_min.pdf(
                np.linspace(0, np.ceil(self.env_analysis.record_max_wind_gust), 50),
                *stats.weibull_min.fit(
                    wind_gusts[self.env_analysis.hours[0]], loc=0, scale=1
                ),
            )
        )

        def init():
            ax.set_xlim(0, np.ceil(self.env_analysis.record_max_wind_gust))
            ax.set_ylim(0, max_probability_density)
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
            x = np.linspace(0, np.ceil(self.env_analysis.record_max_wind_gust), 50)
            y = stats.weibull_min.pdf(x, c, loc, scale)
            ln.set_data(x, y)
            # Update hour text
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return (ln, *bar_container.patches, tx)

        for frame in wind_gusts.items():
            update(frame)

        animation = FuncAnimation(
            fig,
            update,
            frames=wind_gusts.items(),
            interval=750,
            init_func=init,
            blit=True,
        )
        plt.close(fig)
        return HTML(animation.to_jshtml())

    def surface_wind_speed_distribution_grid(self, wind_speed_limit=False):
        """Plots shown in the animation of how the sustained surface wind speed
        distribution varies throughout the day. The plots are histograms of the
        wind speed distribution for a specific hour. The plots are arranged in a
        grid like plot.

        Parameters
        ----------
        wind_speed_limit : bool, optional
            Whether to plot the wind speed limit as a vertical line

        Returns
        -------
        None
        """
        # Gather animation data
        average_wind_speed_at_given_hour = self.env_analysis.surface_wind_speed_by_hour

        # Create grid of plots for each hour
        n_cols, n_rows = find_two_closest_integers(len(self.env_analysis.hours))
        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2.2))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
            hour = self.env_analysis.hours[i * n_cols + j]
            ax = axs[i, j]
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.hist(
                average_wind_speed_at_given_hour[hour],
                bins=int(len(average_wind_speed_at_given_hour[hour]) ** 0.5),
                density=True,
                histtype="stepfilled",
                alpha=0.2,
                label="Wind Speed",
            )
            ax.autoscale(enable=True, axis="y", tight=True)
            # Plot weibull distribution
            c, loc, scale = stats.weibull_min.fit(
                average_wind_speed_at_given_hour[hour], loc=0, scale=1
            )
            x = np.linspace(
                0,
                np.ceil(self.env_analysis.max_surface_10m_wind_speed_list),
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

        if wind_speed_limit:
            for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
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
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
        )
        fig.suptitle("Sustained Surface Wind Speed Distributions")
        fig.supxlabel(
            f"Sustained Surface Wind Speed ({self.env_analysis.unit_system['wind_speed']})"
        )
        fig.supylabel("Probability")
        plt.show()

        return None

    def animate_surface_wind_speed_distribution(self, wind_speed_limit=False):
        """Animation of how the sustained surface wind speed distribution varies
        throughout the day. Each frame is a histogram of the wind speed distribution
        for a specific hour.

        Parameters
        ----------
        wind_speed_limit : bool, optional
            Whether to plot the wind speed limit as a vertical line

        Returns
        -------
        HTML : IPython.core.display.HTML
        """
        module = import_optional_dependency("IPython.display")
        HTML = module.HTML  # this is a class

        # Gather animation data
        surface_wind_speeds_at_given_hour = self.env_analysis.surface_wind_speed_by_hour

        # Create animation
        fig, ax = plt.subplots(dpi=200)
        # Initialize animation artists: histogram and hour text
        hist_bins = np.linspace(
            0, np.ceil(self.env_analysis.record_max_surface_10m_wind_speed), 25
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

        maximum_probability_density = 1.2 * max(
            stats.weibull_min.pdf(
                np.linspace(
                    0,
                    np.ceil(self.env_analysis.record_max_surface_10m_wind_speed),
                    100,
                ),
                *stats.weibull_min.fit(
                    surface_wind_speeds_at_given_hour[self.env_analysis.hours[0]],
                    loc=0,
                    scale=1,
                ),
            )
        )

        # Define function to initialize animation
        def init():
            ax.set_xlim(0, np.ceil(self.env_analysis.record_max_surface_10m_wind_speed))
            ax.set_ylim(0, maximum_probability_density)
            ax.set_xlabel(
                f"Sustained Surface Wind Speed ({self.env_analysis.unit_system['wind_speed']})"
            )
            ax.set_ylabel("Probability")
            ax.set_title("Sustained Surface Wind Distribution")

            if wind_speed_limit:
                ax.vlines(
                    convert_units(
                        20, "mph", self.env_analysis.unit_system["wind_speed"]
                    ),
                    0,
                    0.3,
                    "g",
                    (0, (15, 5, 2, 5)),
                    label="Wind Speed Limit",
                )

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
            x = np.linspace(
                0,
                np.ceil(self.env_analysis.record_max_surface_10m_wind_speed),
                100,
            )
            y = stats.weibull_min.pdf(x, c, loc, scale)
            ln.set_data(x, y)
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

    def wind_speed_profile_grid(self, clear_range_limits=False):
        """Creates a grid of plots with the wind profile over the average day.
        Each subplot represents a different hour of the day.

        Parameters
        ----------
        clear_range_limits : bool, optional
            Whether to clear the sky range limits or not, by default False

        Returns
        -------
        None
        """

        # Create grid of plots for each hour
        n_cols, n_rows = find_two_closest_integers(len(self.env_analysis.hours))
        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2.2))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        x_min, x_max, y_min, y_max = 0, 0, np.inf, 0

        for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
            hour = self.env_analysis.hours[i * n_cols + j]
            ax = axs[i, j]
            ax.plot(*self.env_analysis.average_wind_speed_profile_by_hour[hour], "r-")
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.autoscale(enable=True, axis="y", tight=True)
            current_x_max = ax.get_xlim()[1]
            current_y_min, current_y_max = ax.get_ylim()
            x_max = current_x_max if current_x_max > x_max else x_max
            y_max = current_y_max if current_y_max > y_max else y_max
            y_min = current_y_min if current_y_min < y_min else y_min

            if self.env_analysis.forecast:
                forecast = self.env_analysis.forecast
                y = self.env_analysis.average_wind_speed_profile_by_hour[hour][1]
                x = forecast[hour].wind_speed.get_value(y) * convert_units(
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
            for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
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

    def wind_heading_profile_grid(self, clear_range_limits=False):
        """Creates a grid of plots with the wind heading profile over the
        average day. Each subplot represents a different hour of the day.

        Parameters
        ----------
        clear_range_limits : bool, optional
            Whether to clear the sky range limits or not, by default False. This
            is useful when the launch site is constrained in terms or altitude.

        Returns
        -------
        None
        """

        # Create grid of plots for each hour
        n_cols, n_rows = find_two_closest_integers(len(self.env_analysis.hours))

        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2.2))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0, wspace=0, left=0.12)
        axs = gs.subplots(sharex=True, sharey=True)
        _, y_min, y_max = 0, np.inf, 0

        for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
            hour = self.env_analysis.hours[i * n_cols + j]
            ax = axs[i, j]
            ax.plot(
                *self.env_analysis.average_wind_heading_profile_by_hour[hour],
                "r-",
            )
            ax.set_title(f"{float(hour):05.2f}".replace(".", ":"), y=0.8)
            ax.autoscale(enable=True, axis="y", tight=True)
            current_y_min, current_y_max = ax.get_ylim()
            y_max = current_y_max if current_y_max > y_max else y_max
            y_min = current_y_min if current_y_min < y_min else y_min
            ax.label_outer()
            ax.grid()

        # Set x and y limits for the last axis. Since axes are shared, set to all
        ax.set_xlim(0, 360)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=5, prune="lower")
        )
        ax.yaxis.set_major_locator(
            mtick.MaxNLocator(integer=True, nbins=4, prune="lower")
        )

        if clear_range_limits:
            for i, j in [(i, j) for i in range(n_rows) for j in range(n_cols)]:
                # Clear Sky range limits
                ax = axs[i, j]
                ax.fill_between(
                    [0, 360],
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
                    [0, 360],
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

    def animate_wind_speed_profile(self, clear_range_limits=False):
        """Animation of how wind profile evolves throughout an average day.

        Parameters
        ----------
        clear_range_limits : bool, optional
            Whether to clear the sky range limits or not, by default False. This
            is useful when the launch site is constrained in terms or altitude.
        """
        module = import_optional_dependency("IPython.display")
        HTML = module.HTML  # this is a class

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
            ax.set_xlim(0, self.env_analysis.max_average_wind_speed_at_altitude + 5)
            ax.set_ylim(*self.env_analysis.altitude_AGL_range)
            ax.set_xlabel(f"Wind Speed ({self.env_analysis.unit_system['wind_speed']})")
            ax.set_ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
            ax.set_title("Average Wind Profile")
            ax.grid(True)
            return ln, tx

        # Define function which sets each animation frame
        def update(frame):
            x = frame[1][0]
            y = frame[1][1]
            ln.set_data(x, y)
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return ln, tx

        animation = FuncAnimation(
            fig,
            update,
            frames=self.env_analysis.average_wind_speed_profile_by_hour.items(),
            interval=1000,
            init_func=init,
            blit=True,
        )

        if clear_range_limits:
            # Clear sky range limits
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_speed_at_altitude + 5],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_speed_at_altitude + 5],
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

    def animate_wind_heading_profile(self, clear_range_limits=False):
        """Animation of how the wind heading profile evolves throughout an
        average day. Each frame is a different hour of the day.

        Parameters
        ----------
        clear_range_limits : bool, optional
            Whether to clear the sky range limits or not, by default False. This
            is useful when the launch site is constrained in terms or altitude.
        """
        module = import_optional_dependency("IPython.display")
        HTML = module.HTML  # this is a class

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
            ax.set_xlim(0, 360)
            ax.set_ylim(*self.env_analysis.altitude_AGL_range)
            ax.set_xlabel(f"Wind Heading ({self.env_analysis.unit_system['angle']})")
            ax.set_ylabel(f"Altitude AGL ({self.env_analysis.unit_system['length']})")
            ax.set_title("Average Wind Heading Profile")
            ax.grid(True)
            return ln, tx

        # Define function which sets each animation frame
        def update(frame):
            x = frame[1][0]
            y = frame[1][1]
            ln.set_data(x, y)
            tx.set_text(f"{float(frame[0]):05.2f}".replace(".", ":"))
            return ln, tx

        animation = FuncAnimation(
            fig,
            update,
            frames=self.env_analysis.average_wind_heading_profile_by_hour.items(),
            interval=1000,
            init_func=init,
            blit=True,
        )

        if clear_range_limits:
            # Clear sky range limits
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_speed_at_altitude + 5],
                0.7
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                1.3
                * convert_units(10000, "ft", self.env_analysis.unit_system["length"]),
                color="g",
                alpha=0.2,
                label=f"10,000 {self.env_analysis.unit_system['length']} ± 30%",
            )
            ax.fill_between(
                [0, self.env_analysis.max_average_wind_speed_at_altitude + 5],
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
        """Plots all the available animations together

        Returns
        -------
        None"""

        self.animate_average_wind_rose()
        self.animate_wind_gust_distribution()
        self.animate_surface_wind_speed_distribution()
        self.animate_wind_heading_profile(clear_range_limits=True)
        self.animate_wind_speed_profile()

        return None

    def all_plots(self):
        """Plots all the available plots together, this avoids having animations

        Returns
        -------
        None
        """
        self.wind_gust_distribution()
        self.surface10m_wind_speed_distribution()
        self.average_surface_temperature_evolution()
        self.average_surface10m_wind_speed_evolution()
        self.average_surface100m_wind_speed_evolution()
        self.average_wind_speed_profile()
        self.average_wind_heading_profile()
        self.average_pressure_profile()
        self.average_wind_rose_grid()
        self.wind_gust_distribution_grid()
        self.surface_wind_speed_distribution_grid()
        self.wind_speed_profile_grid()
        self.wind_heading_profile_grid()

        return None

    def info(self):
        """Plots only the most important plots together. This method simply
        invokes the `wind_gust_distribution`, `average_wind_speed_profile`,
        `wind_speed_profile_grid` and `wind_heading_profile_grid`
        methods.

        Returns
        -------
        None
        """
        self.wind_gust_distribution()
        self.average_wind_speed_profile()
        self.wind_speed_profile_grid()
        self.wind_heading_profile_grid()

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
