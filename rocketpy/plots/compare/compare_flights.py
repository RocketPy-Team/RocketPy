__author__ = "Guilherme Fernandes Alves, Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import matplotlib.pyplot as plt
import numpy as np

from .compare import Compare


class CompareFlights(Compare):
    """A class to compare the results of multiple flights.

    Parameters
    ----------
    flights : list
        A list of Flight objects to be compared.

    Attributes
    ----------
    flights : list
        A list of Flight objects to be compared.

    """

    def __init__(self, flights):
        """Initializes the CompareFlights class.

        Parameters
        ----------
        flights : list
            A list of Flight objects to be compared.

        Returns
        -------
        None
        """
        super().__init__(flights)

        # Get the maximum time of all the flights
        # Get the maximum apogee time
        max_time = 0
        apogee_time = 0
        for flight in flights:
            # Update the maximum time
            max_time = max(max_time, flight.tFinal)
            apogee_time = max(apogee_time, flight.apogeeTime)

        self.max_time = max_time
        self.apogee_time = apogee_time
        self.flights = self.object_list

        return None

    def positions(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots a comparison of the position of the rocket in the three
        dimensions separately.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """
        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time"],
            y_attributes=["x", "y", "z"],
            n_rows=3,
            n_cols=1,
            figsize=figsize,  # (width, height)
            legend=legend,
            title="Comparison of the position of the rocket",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=["x (m)", "y (m)", "z (m)"],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def velocities(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the velocity of the rocket in the three
        dimensions separately.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time", "time"],
            y_attributes=["speed", "vx", "vy", "vz"],
            n_rows=4,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the velocity of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=["speed (m/s)", "vx (m/s)", "vy (m/s)", "vz (m/s)"],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def stream_velocities(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a stream plot of the free stream velocity of the rocket in the
        three dimensions separately. The free stream velocity is the velocity of
        the rocket relative to the air.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10 * 4 / 3),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time", "time"],
            y_attributes=[
                "freestreamSpeed",
                "streamVelocityX",
                "streamVelocityY",
                "streamVelocityZ",
            ],
            n_rows=4,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the free stream velocity of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Freestream speed (m/s)",
                "Freestream vx (m/s)",
                "Freestream vy (m/s)",
                "Freestream vz (m/s)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def accelerations(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the acceleration of the rocket in the three
        dimensions separately.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time", "time"],
            y_attributes=["acceleration", "ax", "ay", "az"],
            n_rows=4,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the acceleration of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Acceleration (m/s^2)",
                "ax (m/s^2)",
                "ay (m/s^2)",
                "az (m/s^2)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def euler_angles(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots a comparison of the euler angles of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time"],
            y_attributes=["phi", "theta", "psi"],
            n_rows=3,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the euler angles of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                self.flights[0].phi.getOutputs()[0],
                self.flights[0].theta.getOutputs()[0],
                self.flights[0].psi.getOutputs()[0],
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def quaternions(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the quaternions of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time", "time"],
            y_attributes=["e0", "e1", "e2", "e3"],
            n_rows=4,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the quaternions of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "e0 (deg)",
                "e1 (deg)",
                "e2 (deg)",
                "e3 (deg)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def attitude_angles(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots a comparison of the attitude angles of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time"],
            y_attributes=["pathAngle", "attitudeAngle", "lateralAttitudeAngle"],
            n_rows=3,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the attitude angles of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "pathAngle (deg)",
                "attitudeAngle (deg)",
                "lateralAttitudeAngle (deg)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def angular_velocities(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots a comparison of the angular velocities of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time"],
            y_attributes=["w1", "w2", "w3"],
            n_rows=3,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the angular velocities of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "w1 (deg/s)",
                "w2 (deg/s)",
                "w3 (deg/s)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)

        return None

    def angular_accelerations(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots a comparison of the angular accelerations of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time"],
            y_attributes=["alpha1", "alpha2", "alpha3"],
            n_rows=3,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the angular accelerations of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                self.flights[0].alpha1.getOutputs()[0],
                self.flights[0].alpha2.getOutputs()[0],
                self.flights[0].alpha3.getOutputs()[0],
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def aerodynamic_forces(
        self,
        figsize=(7, 10 * 2 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the aerodynamic forces of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time"],
            y_attributes=["aerodynamicDrag", "aerodynamicLift"],
            n_rows=2,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the aerodynamic forces of the flights",
            x_labels=["Time (s)", "Time (s)"],
            y_labels=[
                "Drag Force (N)",
                "Lift Force (N)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def aerodynamic_moments(
        self,
        figsize=(7, 10 * 2 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the aerodynamic moments of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time"],
            y_attributes=["aerodynamicBendingMoment", "aerodynamicSpinMoment"],
            n_rows=2,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the aerodynamic moments of the flights",
            x_labels=["Time (s)", "Time (s)"],
            y_labels=[
                "Bending Moment (N*m)",
                "Spin Moment (N*m)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def energies(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots a comparison of the energies of the rocket for the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time"],
            y_attributes=["kineticEnergy", "potentialEnergy", "totalEnergy"],
            n_rows=3,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the energies of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Kinetic Energy (J)",
                "Potential Energy (J)",
                "Total Energy (J)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def powers(
        self,
        figsize=(7, 10 * 2 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the powers of the rocket for the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time"],
            y_attributes=["thrustPower", "dragPower"],
            n_rows=2,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the powers of the flights",
            x_labels=["Time (s)", "Time (s)"],
            y_labels=["Thrust Power (W)", "Drag Power (W)"],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def rail_buttons_forces(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the forces acting on the rail buttons of the rocket for
        the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        # Needs to be manually done beacuse of specific time array length
        y_attributes = [
            "railButton1NormalForce",
            "railButton1ShearForce",
            "railButton2NormalForce",
            "railButton2ShearForce",
        ]
        n_rows = 4
        n_cols = 1
        title = "Comparison of the forces acting on the rail buttons of the flights"
        x_labels = ["Time (s)", "Time (s)", "Time (s)", "Time (s)"]
        y_labels = [
            "Rail Button 1 Normal Force (N)",
            "Rail Button 1 Shear Force (N)",
            "Rail Button 2 Normal Force (N)",
            "Rail Button 2 Shear Force (N)",
        ]

        n_plots = len(y_attributes)

        # Create the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, y=1.02, x=0.5)

        # Create the subplots
        ax = []
        for i in range(n_plots):
            ax.append(plt.subplot(n_rows, n_cols, i + 1))

        # Adding the plots to each subplot
        for flight in self.flights:
            for i in range(n_plots):
                time = np.linspace(
                    0, flight.outOfRailTime, flight.outOfRailTimeIndex + 1
                )
                ax[i].plot(
                    time,
                    flight.__getattribute__(y_attributes[i])(time),
                    label=flight.name,
                )

        for i, subplot in enumerate(ax):

            # Set the labels for the x and y axis
            subplot.set_xlabel(x_labels[i])
            subplot.set_ylabel(y_labels[i])

            # Set the limits for the x axis
            if x_lim:
                subplot.set_xlim(*x_lim)
            if y_lim:
                subplot.set_ylim(*y_lim)
            subplot.grid(True)  # Add a grid to the plot

        # Find the two closest integers to the square root of the number of object_list
        # to be used as the number of columns and rows of the legend
        n_cols_legend = int(round(len(self.object_list) ** 0.5))
        n_rows_legend = int(round(len(self.object_list) / n_cols_legend))

        # Set the legend
        if legend:  # Add a global legend to the figure
            fig.legend(
                *ax[0].get_legend_handles_labels(),
                loc="lower center",
                ncol=n_cols_legend,
                numpoints=1,
                frameon=True,
                bbox_to_anchor=(0.5, 1.05),
            )

        fig.tight_layout()

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def angles_of_attack(
        self,
        figsize=(7, 10 * 1 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the angles of attack of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time"],
            y_attributes=["angleOfAttack"],
            n_rows=1,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the angles of attack of the flights",
            x_labels=["Time (s)"],
            y_labels=["Angle of Attack (deg)"],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def fluid_mechanics(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots a comparison of the fluid mechanics of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Check if key word is used for x_limit
        if x_lim:
            x_lim[0] = self.apogee_time if x_lim[0] == "apogee" else x_lim[0]
            x_lim[1] = self.apogee_time if x_lim[1] == "apogee" else x_lim[1]

        # Create the figure
        fig, _ = super().create_comparison_figure(
            x_attributes=["time", "time", "time", "time"],
            y_attributes=[
                "MachNumber",
                "ReynoldsNumber",
                "dynamicPressure",
                "totalPressure",
            ],
            n_rows=4,
            n_cols=1,
            figsize=figsize,
            legend=legend,
            title="Comparison of the fluid mechanics of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Mach Number",
                "Reynolds Number",
                "Dynamic Pressure (Pa)",
                "Total Pressure (Pa)",
            ],
            x_lim=x_lim,
            y_lim=y_lim,
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

        return None

    def stability_margin(
        self, figsize=(7, 10), x_lim=None, y_lim=None, legend=True, filename=None
    ):
        """Plots the stability margin of the rocket for the different flights.
        The stability margin here is different than the static margin, it is the
        difference between the center of pressure and the center of gravity of the
        rocket varying with time.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        print("This method is not implemented yet")

        return None

    def attitude_frequency(
        self,
        figsize=(7, 10 * 4 / 3),
        x_lim=None,
        y_lim=None,
        legend=True,
        filename=None,
    ):
        """Plots the frequency of the attitude of the rocket for the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        x_lim : tuple
            A list of two items, where the first item represents the x axis lower limit
            and second item, the x axis upper limit. If set to None, will be calculated
            automatically by matplotlib. If the string "apogee" is used as a item for the
            tuple, the maximum apogee time of all flights will be used instead.
        y_lim : tuple
            A list of two item, where the first item represents the y axis lower limit
            and second item, the y axis upper limit. If set to None, will be calculated
            automatically by matplotlib.
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        print("This method is not implemented yet")

        return None

    @staticmethod
    def compare_trajectories_3d(
        flights, names_list=None, figsize=(7, 7), legend=None, filename=None
    ):
        """Creates a trajectory plot combining the trajectories listed.
        This function was created based two source-codes:
        - Mateus Stano: https://github.com/RocketPy-Team/Hackathon_2020/pull/123
        - Dyllon Preston: https://github.com/Dyllon-P/MBS-Template/blob/main/MBS.py
        Also, some of the credits go to Georgia Tech Experimental Rocketry Club (GTXR)
        as well.
        The final function was created by the RocketPy Team.
        Parameters
        ----------
        flights : list, array
            List of trajectories. Must be in the form of [trajectory_1, trajectory_2, ..., trajectory_n]
            where each element is a list with the arrays regarding positions in x, y and z [x, y, z].
            The trajectories must be in the same reference frame. The z coordinate must be referenced
            to the ground or to the sea level, but it is important that all trajectories are passed
            in the same reference.
        names_list : list
            List of strings with the name of each trajectory inputted. The names must be in
            the same order as the trajectories in flights. If no names are passed, the
            trajectories will be named as "Trajectory 1", "Trajectory 2", ..., "Trajectory n".
        figsize : tuple, optional
            Tuple with the size of the figure. The default is (7,7).
        legend : boolean, optional
            Whether legend will or will not be plotted. Default is True
        filename : string, optional
            If a filename is passed, the figure will be saved in the current directory.
            The default is None.

        Returns
        -------
        None
        """

        # Initialize variables
        maxX, maxY, maxZ, minX, minY, minZ, maxXY, minXY = 0, 0, 0, 0, 0, 0, 0, 0
        names_list = (
            [f" Trajectory {i}" for i in range(len(flights))]
            if not names_list
            else names_list
        )

        # Create the figure
        fig1 = plt.figure(figsize=figsize)
        fig1.suptitle("Flight Trajectories Comparison", fontsize=16, y=0.95, x=0.5)
        ax1 = plt.subplot(
            111,
            projection="3d",
        )

        # Iterate through trajectories
        for index, flight in enumerate(flights):

            x, y, z = flight

            # Update mx and min values to set the limits of the plot
            maxX = max(maxX, max(x))
            maxY = max(maxY, max(y))
            maxZ = max(maxZ, max(z))
            minX = min(minX, min(x))
            minY = min(minY, min(y))
            minZ = min(minZ, min(z))
            maxXY = max(maxXY, max(max(x), max(y)))
            minXY = min(minXY, min(min(x), min(y)))

            # Add Trajectory as a plot in main figure
            ax1.plot(x, y, z, linewidth="2", label=names_list[index])

        # Plot settings
        ax1.scatter(0, 0, 0, color="black", s=10, marker="o")
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Z - Altitude (m)")
        ax1.set_zlim3d([minZ, maxZ])
        ax1.set_ylim3d([minXY, maxXY])
        ax1.set_xlim3d([minXY, maxXY])
        ax1.view_init(15, 45)

        # Add legend
        if legend:
            fig1.legend()

        fig1.tight_layout()

        # Save figure
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        return None

    def trajectories_3d(self, figsize=(7, 7), legend=None, filename=None):
        """Creates a trajectory plot that is the combination of the trajectories of
        the Flight objects passed via a Python list.

        Parameters
        ----------
        figsize: tuple, optional
            Tuple with the size of the figure. The default is (7, 7). The tuple
            must be in the form (width, height).
        legend : boolean, optional
            Whether legend will or will not be included. Default is True
        savefig : string, optional
            If a string is passed, the figure will be saved in the path passed.
        Returns
        -------
        None
        """

        # Iterate through Flight objects and create a list of trajectories
        flights, names_list = self.__retrieve_trajectories()

        # Call compare_trajectories_3d function to do the hard work
        self.compare_trajectories_3d(
            flights=flights,
            names_list=names_list,
            legend=legend,
            filename=filename,
            figsize=figsize,
        )

        return None

    def __retrieve_trajectories(self):
        """Retrieve trajectories from Flight objects.

        Returns
        -------
        flights : list
            List of trajectories. Must be in the form of [trajectory_1, trajectory_2, ..., trajectory_n]
            where each element is a list with the arrays regarding positions in x, y and z [x, y, z].
            The trajectories must be in the same reference frame. The z coordinate must be referenced
            to the ground or to the sea level, but it is important that all trajectories are passed
            in the same reference.
        names_list : list
            List of strings with the name of each trajectory inputted. The names must be in
            the same order as the trajectories in flights. If no names are passed, the
            trajectories will be named as "Trajectory 1", "Trajectory 2", ..., "Trajectory n".
        """

        flights = []
        names_list = []
        for _, flight in enumerate(self.flights):
            # Get trajectories
            try:
                x = flight.x[:, 1]
                y = flight.y[:, 1]
                z = flight.z[:, 1] - flight.env.elevation
            except AttributeError:
                raise AttributeError(
                    "Flight object {} does not have a trajectory.".format(flight.name)
                )
            flights.append([x, y, z])
            names_list.append(flight.name)
        return flights, names_list

    def trajectories_2d(self, plane="xy", figsize=(7, 7), legend=None, filename=None):
        """Creates a 2D trajectory plot that is the combination of the trajectories of
        the Flight objects passed via a Python list.

        Parameters
        ----------
        legend : boolean, optional
            Whether legend will or will not be included. Default is True
        plane : string, optional
            The plane in which the trajectories will be plotted. The default is "xy".
            The options are:
                - "xy": X-Y plane
                - "xz": X-Z plane
                - "yz": Y-Z plane
        filename : string, optional
            If a string is passed, the figure will be saved in the path passed.
            The image format options are: .png, .jpg, .jpeg, .tiff, .bmp, .pdf, .svg, .pgf, .eps
        figsize : tuple, optional
            Tuple with the size of the figure. The default is (7, 7).

        Returns
        -------
        None
        """

        # Iterate through Flight objects and create a list of trajectories
        flights, names_list = self.__retrieve_trajectories()

        f_planes = {
            "xy": self.__plot_xy,
            "xz": self.__plot_xz,
            "yz": self.__plot_yz,
        }

        # Check if the plane is valid and call the corresponding function
        # If the plane is not valid, the default plane is "xy"
        func = f_planes.get(plane, f_planes.get("xy"))

        func(flights, names_list, figsize, legend, filename)

        return None

    def __plot_xy(
        self, flights, names_list, figsize=(7, 7), legend=None, filename=None
    ):
        """Creates a 2D trajectory plot in the X-Y plane that is the combination of the trajectories of
        the Flight objects passed via a Python list.

        Parameters
        ----------
        legend : boolean, optional
            Whether legend will or will not be included. Default is True
        filename : string, optional
            If a string is passed, the figure will be saved in the path passed.
            The image format options are: .png, .jpg, .jpeg, .tiff, .bmp, .pdf, .svg, .pgf, .eps
        figsize : tuple, optional
            Tuple with the size of the figure. The default is (7, 7).

        Returns
        -------
        None
        """

        # Create the figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Flight Trajectories Comparison", fontsize=16, y=0.95, x=0.5)
        ax = plt.subplot(111)

        # Initialize variables to set the limits of the plot
        maxX = maxY = maxXY = -np.inf
        minX = minY = minXY = np.inf

        # Iterate through trajectories
        for index, flight in enumerate(flights):

            x, y, _ = flight

            # Update mx and min values to set the limits of the plot
            maxX = max(maxX, max(x))
            maxY = max(maxY, max(y))
            maxXY = max(maxXY, max(max(x), max(y)))
            minX = min(minX, min(x))
            minY = min(minY, min(y))
            minXY = min(minXY, min(min(x), min(y)))

            # Add Trajectory as a plot in main figure
            ax.plot(x, y, linewidth="2", label=names_list[index])

        # Plot settings
        ax.scatter(0, 0, color="black", s=10, marker="o")
        ax.set_xlabel("X - East (m)")
        ax.set_ylabel("Y - North (m)")
        ax.set_ylim([minXY, maxXY])
        ax.set_xlim([minXY, maxXY])

        # Add legend
        if legend:
            fig.legend()

        fig.tight_layout()

        # Save figure
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        return None

    def __plot_xz(
        self, flights, names_list, figsize=(7, 7), legend=None, filename=None
    ):
        """Creates a 2D trajectory plot in the X-Z plane that is the combination
        of the trajectories of the Flight objects passed via a Python list.

        Parameters
        ----------
        legend : boolean, optional
            Whether legend will or will not be included. Default is True
        filename : string, optional
            If a string is passed, the figure will be saved in the path passed.
            The image format options are: .png, .jpg, .jpeg, .tiff, .bmp, .pdf,
            .svg, .pgf, .eps
        figsize : tuple, optional
            Tuple with the size of the figure. The default is (7, 7).

        Returns
        -------
        None
        """

        # Create the figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Flight Trajectories Comparison", fontsize=16, y=0.95, x=0.5)
        ax = plt.subplot(111)

        # Initialize variables to set the limits of the plot
        maxX = maxZ = maxXY = -np.inf
        minX = minZ = minXY = np.inf

        # Iterate through trajectories
        for index, flight in enumerate(flights):

            x, _, z = flight

            # Update mx and min values to set the limits of the plot
            maxX = max(maxX, max(x))
            maxZ = max(maxZ, max(z))
            maxXY = max(maxXY, max(max(x), max(z)))
            minX = min(minX, min(x))
            minZ = min(minZ, min(z))
            minXY = min(minXY, min(min(x), min(z)))

            # Add Trajectory as a plot in main figure
            ax.plot(x, z, linewidth="2", label=names_list[index])

        # Plot settings
        ax.scatter(0, 0, color="black", s=10, marker="o")
        ax.set_xlabel("X - East (m)")
        ax.set_ylabel("Z - Up (m)")
        ax.set_ylim([minXY, maxXY])
        ax.set_xlim([minXY, maxXY])

        # Add legend
        if legend:
            fig.legend()

        fig.tight_layout()

        # Save figure
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        return None

    def __plot_yz(
        self, flights, names_list, figsize=(7, 7), legend=None, filename=None
    ):
        """Creates a 2D trajectory plot in the Y-Z plane that is the combination
        of the trajectories of the Flight objects passed via a Python list.

        Parameters
        ----------
        legend : boolean, optional
            Whether legend will or will not be included. Default is True
        filename : string, optional
            If a string is passed, the figure will be saved in the path passed.
            The image format options are: .png, .jpg, .jpeg, .tiff, .bmp, .pdf,
            .svg, .pgf, .eps
        figsize : tuple, optional
            Tuple with the size of the figure. The default is (7, 7).

        Returns
        -------
        None
        """

        # Create the figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Flight Trajectories Comparison", fontsize=16, y=0.95, x=0.5)
        ax = plt.subplot(111)

        # Initialize variables to set the limits of the plot
        maxY = maxZ = maxXY = -np.inf
        minY = minZ = minXY = np.inf

        # Iterate through trajectories
        for index, flight in enumerate(flights):

            _, y, z = flight

            # Update mx and min values to set the limits of the plot
            maxY = max(maxY, max(y))
            maxZ = max(maxZ, max(z))
            maxXY = max(maxXY, max(max(y), max(z)))
            minY = min(minY, min(y))
            minZ = min(minZ, min(z))
            minXY = min(minXY, min(min(y), min(z)))

            # Add Trajectory as a plot in main figure
            ax.plot(y, z, linewidth="2", label=names_list[index])

        # Plot settings
        ax.scatter(0, 0, color="black", s=10, marker="o")
        ax.set_xlabel("Y - North (m)")
        ax.set_ylabel("Z - Up (m)")
        ax.set_ylim([minXY, maxXY])
        ax.set_xlim([minXY, maxXY])

        # Add legend
        if legend:
            fig.legend()

        fig.tight_layout()

        # Save figure
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        return None

    def all(self):
        """Prints out all data and graphs available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.trajectories_3d()

        self.positions()

        self.velocities()

        self.stream_velocities()

        self.accelerations()

        self.angular_velocities()

        self.angular_accelerations()

        self.euler_angles()

        self.quaternions()

        self.attitude_angles()

        self.angles_of_attack()

        self.stability_margin()

        self.aerodynamic_forces()

        self.aerodynamic_moments()

        self.rail_buttons_forces()

        self.energies()

        self.powers()

        self.fluid_mechanics()

        self.attitude_frequency()

        return None
