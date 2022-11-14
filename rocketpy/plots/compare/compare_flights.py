__author__ = "Guilherme Fernandes Alves, Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import matplotlib.pyplot as plt


class CompareFlights:
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

    def __init__(self, flights: list) -> None:
        """Initializes the CompareFlights class.

        Parameters
        ----------
        flights : list
            A list of Flight objects to be compared.

        Returns
        -------
        None
        """

        self.flights = flights

        return None

    def __create_comparison_figure(
        self,
        figsize=(7, 10),  # (width, height)
        legend=True,
        n_rows=3,
        n_cols=1,
        n_plots=3,
        title="Comparison",
        x_labels=["Time (s)", "Time (s)", "Time (s)"],
        y_labels=["x (m)", "y (m)", "z (m)"],
        flight_attributes=["x", "y", "z"],
    ):
        """Creates a figure to compare the results of multiple flights.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure, by default (7, 10)
        legend : bool, optional
            Whether to show the legend or not, by default True
        n_rows : int, optional
            The number of rows of the figure, by default 3
        n_cols : int, optional
            The number of columns of the figure, by default 1
        n_plots : int, optional
            The number of plots in the figure, by default 3
        title : str, optional
            The title of the figure, by default "Comparison"
        x_labels : list, optional
            The x labels of each subplot, by default ["Time (s)", "Time (s)", "Time (s)"]
        y_labels : list, optional
            The y labels of each subplot, by default ["x (m)", "y (m)", "z (m)"]
        flight_attributes : list, optional
            The attributes of the Flight class to be plotted, by default ["x", "y", "z"].
            The attributes must be a list of strings. Each string must be a valid
            attribute of the Flight class, i.e., should point to a attribute of
            the Flight class that is a Function object or a numpy array.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object.
        """

        # Create the matplotlib figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, y=1.02, x=0.5)

        # Create the subplots
        ax = []
        for i in range(n_plots):
            ax.append(plt.subplot(n_rows, n_cols, i + 1))

        # Get the maximum time of all the flights
        max_time = 0

        # Adding the plots to each subplot
        for flight in self.flights:
            for i in range(n_plots):
                try:
                    ax[i].plot(
                        flight.time,
                        flight.__getattribute__(flight_attributes[i])[:, 1],
                        label=flight.name,
                    )
                    # Update the maximum time
                    max_time = flight.tFinal if flight.time[-1] > max_time else max_time
                except AttributeError:
                    raise AttributeError(
                        f"Invalid attribute {flight_attributes[i]} for the Flight class."
                    )

        # Set the labels for the x and y axis
        for i, subplot in enumerate(ax):
            subplot.set_xlabel(x_labels[i])
            subplot.set_ylabel(y_labels[i])

        # Set the limits for the x axis
        for subplot in ax:
            subplot.set_xlim(0, max_time)

        # Set the legend
        if legend:
            fig.legend(
                loc="upper center",
                fancybox=True,
                shadow=True,
                fontsize=10,
                bbox_to_anchor=(0.5, 0.995),
            )

        fig.tight_layout()

        return fig, ax

    def positions(self, figsize=(7, 10), legend=True, filename=None):
        """Plots a comparison of the position of the rocket in the three
        dimensions separately.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=3,
            n_cols=1,
            n_plots=3,
            title="Comparison of the position of the rocket",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=["x (m)", "y (m)", "z (m)"],
            flight_attributes=["x", "y", "z"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def velocities(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots a comparison of the velocity of the rocket in the three
        dimensions separately.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the velocity of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=["speed (m/s)", "vx (m/s)", "vy (m/s)", "vz (m/s)"],
            flight_attributes=["speed", "vx", "vy", "vz"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def stream_velocities(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots a stream plot of the free stream velocity of the rocket in the
        three dimensions separately. The free stream velocity is the velocity of
        the rocket relative to the air.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10 * 4 / 3),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the free stream velocity of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Freestream speed (m/s)",
                "Freestream vx (m/s)",
                "Freestream vy (m/s)",
                "Freestream vz (m/s)",
            ],
            flight_attributes=[
                "freestreamSpeed",
                "streamVelocityX",
                "streamVelocityY",
                "streamVelocityZ",
            ],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def accelerations(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots a comparison of the acceleration of the rocket in the three
        dimensions separately.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the acceleration of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Acceleration (m/s^2)",
                "ax (m/s^2)",
                "ay (m/s^2)",
                "az (m/s^2)",
            ],
            flight_attributes=["acceleration", "ax", "ay", "az"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def euler_angles(self, figsize=(7, 10), legend=True, filename=None):
        """Plots a comparison of the euler angles of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=3,
            n_cols=1,
            n_plots=3,
            title="Comparison of the euler angles of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                self.flights[0].phi.getOutputs()[0],
                self.flights[0].theta.getOutputs()[0],
                self.flights[0].psi.getOutputs()[0],
            ],
            flight_attributes=["phi", "theta", "psi"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def quaternions(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots a comparison of the quaternions of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the quaternions of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "e0 (deg)",
                "e1 (deg)",
                "e2 (deg)",
                "e3 (deg)",
            ],
            flight_attributes=["e0", "e1", "e2", "e3"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def attitude_angles(self, figsize=(7, 10), legend=True, filename=None):
        """Plots a comparison of the attitude angles of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=3,
            n_cols=1,
            n_plots=3,
            title="Comparison of the attitude angles of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "pathAngle (deg)",
                "attitudeAngle (deg)",
                "lateralAttitudeAngle (deg)",
            ],
            flight_attributes=["pathAngle", "attitudeAngle", "lateralAttitudeAngle"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def angular_velocities(self, figsize=(7, 10), legend=True, filename=None):
        """Plots a comparison of the angular velocities of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=3,
            n_cols=1,
            n_plots=3,
            title="Comparison of the angular velocities of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "w1 (deg/s)",
                "w2 (deg/s)",
                "w3 (deg/s)",
            ],
            flight_attributes=["w1", "w2", "w3"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()

        return None

    def angular_accelerations(self, figsize=(7, 10), legend=True, filename=None):
        """Plots a comparison of the angular accelerations of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=3,
            n_cols=1,
            n_plots=3,
            title="Comparison of the angular accelerations of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                self.flights[0].alpha1.getOutputs()[0].getUnits(),
                self.flights[0].alpha2.getOutputs()[0].getUnits(),
                self.flights[0].alpha3.getOutputs()[0].getUnits(),
            ],
            flight_attributes=["alpha1", "alpha2", "alpha3"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def aerodynamic_forces(self, figsize=(7, 10 * 2 / 3), legend=True, filename=None):
        """Plots a comparison of the aerodynamic forces of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.
            Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=2,
            n_cols=1,
            n_plots=2,
            title="Comparison of the aerodynamic forces of the flights",
            x_labels=["Time (s)", "Time (s)"],
            y_labels=[
                "Drag Force (N)",
                "Lift Force (N)",
            ],
            flight_attributes=["aerodynamicDrag", "aerodynamicLift"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def aerodynamic_moments(self, figsize=(7, 10 * 2 / 3), legend=True, filename=None):
        """Plots a comparison of the aerodynamic moments of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=2,
            n_cols=1,
            n_plots=2,
            title="Comparison of the aerodynamic moments of the flights",
            x_labels=["Time (s)", "Time (s)"],
            y_labels=[
                "Bending Moment (N*m)",
                "Spin Moment (N*m)",
            ],
            flight_attributes=["aerodynamicBendingMoment", "aerodynamicSpinMoment"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def energies(self, figsize=(7, 10), legend=True, filename=None):
        """Plots a comparison of the energies of the rocket for the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=3,
            n_cols=1,
            n_plots=3,
            title="Comparison of the energies of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Kinetic Energy (J)",
                "Potential Energy (J)",
                "Total Energy (J)",
            ],
            flight_attributes=["kineticEnergy", "potentialEnergy", "totalEnergy"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def powers(self, figsize=(7, 10 * 2 / 3), legend=True, filename=None):
        """Plots a comparison of the powers of the rocket for the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=2,
            n_cols=1,
            n_plots=2,
            title="Comparison of the powers of the flights",
            x_labels=["Time (s)", "Time (s)"],
            y_labels=["Thrust Power (W)", "Drag Power (W)"],
            flight_attributes=["thrustPower", "dragPower"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def rail_buttons_forces(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots a comparison of the forces acting on the rail buttons of the rocket for
        the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the forces acting on the rail buttons of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Rail Button 1 Normal Force (N)",
                "Rail Button 1 Shear Force (N)",
                "Rail Button 2 Normal Force (N)",
                "Rail Button 2 Shear Force (N)",
            ],
            flight_attributes=[
                "railButton1NormalForce",
                "railButton1ShearForce",
                "railButton2NormalForce",
                "railButton2ShearForce",
            ],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def angles_of_attack(self, figsize=(7, 10 * 1 / 3), legend=True, filename=None):
        """Plots a comparison of the angles of attack of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=1,
            n_cols=1,
            n_plots=1,
            title="Comparison of the angles of attack of the flights",
            x_labels=["Time (s)"],
            y_labels=["Angle of Attack (deg)"],
            flight_attributes=["angleOfAttack"],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def fluid_mechanics(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots a comparison of the fluid mechanics of the rocket for the different
        flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the fluid mechanics of the flights",
            x_labels=["Time (s)", "Time (s)", "Time (s)", "Time (s)"],
            y_labels=[
                "Mach Number",
                "Reynolds Number",
                "Dynamic Pressure (Pa)",
                "Total Pressure (Pa)",
            ],
            flight_attributes=[
                "machNumber",
                "reynoldsNumber",
                "dynamicPressure",
                "totalPressure",
            ],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None

    def attitude_frequency(self, figsize=(7, 10 * 4 / 3), legend=True, filename=None):
        """Plots the frequency of the attitude of the rocket for the different flights.

        Parameters
        ----------
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default (7, 10),
            where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by default None.

        Returns
        -------
        None
        """

        # Create the figure
        fig, _ = self.__create_comparison_figure(
            figsize=figsize,
            legend=legend,
            n_rows=4,
            n_cols=1,
            n_plots=4,
            title="Comparison of the attitude frequency of the flights",
            x_labels=[
                "Frequency (Hz)",
                "Frequency (Hz)",
                "Frequency (Hz)",
                "Frequency (Hz)",
            ],
            y_labels=[
                "Attitude Angle Fourier Amplitude",
                "Omega 1 Angle Fourier Amplitude",
                "Omega 2 Angle Fourier Amplitude",
                "Omega 3 Angle Fourier Amplitude",
            ],
            flight_attributes=[
                "attitudeFrequencyResponse",
                "omega1FrequencyResponse",
                "omega2FrequencyResponse",
                "omega3FrequencyResponse",
            ],
        )

        # Saving the plot to a file if a filename is provided, showing the plot otherwise
        if filename:
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()

        return None
