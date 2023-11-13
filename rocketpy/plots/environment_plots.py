import matplotlib.pyplot as plt
import numpy as np


class _EnvironmentPlots:
    """Class that holds plot methods for Environment class.

    Attributes
    ----------
    _EnvironmentPlots.environment : Environment
        Environment object that will be used for the plots.

    _EnvironmentPlots.grid : list
        Height grid for Environment plots.

    """

    def __init__(self, environment):
        """Initializes _EnvironmentPlots class.

        Parameters
        ----------
        environment : Environment
            Instance of the Environment class

        Returns
        -------
        None
        """
        # Create height grid
        self.grid = np.linspace(environment.elevation, environment.max_expected_height)
        self.environment = environment
        return None

    def __wind(self, ax):
        """Adds wind speed and wind direction graphs to the same axis.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis to add the graphs.

        Returns
        -------
        ax : matplotlib.pyplot.axis
            Axis with the graphs.
        """
        ax.plot(
            [self.environment.wind_speed(i) for i in self.grid],
            self.grid,
            "#ff7f0e",
            label="Wind Speed",
        )
        ax.set_xlabel("Wind Speed (m/s)", color="#ff7f0e")
        ax.tick_params("x", colors="#ff7f0e")
        axup = ax.twiny()
        axup.plot(
            [self.environment.wind_direction(i) for i in self.grid],
            self.grid,
            color="#1f77b4",
            label="Wind Direction",
        )
        axup.set_xlabel("Wind Direction (°)", color="#1f77b4")
        axup.tick_params("x", colors="#1f77b4")
        axup.set_xlim(0, 360)
        ax.set_ylim(self.grid[0], self.grid[-1])
        ax.set_ylabel("Height Above Sea Level (m)")
        ax.grid(True)

        return ax

    def __density_speed_of_sound(self, ax):
        """Adds density and speed of sound graphs to the same axis.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis to add the graphs.

        Returns
        -------
        ax : matplotlib.pyplot.axis
            Axis with the graphs.
        """
        ax.plot(
            [self.environment.speed_of_sound(i) for i in self.grid],
            self.grid,
            "#ff7f0e",
            label="Speed of Sound",
        )
        ax.set_xlabel("Speed of Sound (m/s)", color="#ff7f0e")
        ax.tick_params("x", colors="#ff7f0e")
        axup = ax.twiny()
        axup.plot(
            [self.environment.density(i) for i in self.grid],
            self.grid,
            color="#1f77b4",
            label="Density",
        )
        axup.set_xlabel("Density (kg/m³)", color="#1f77b4")
        axup.tick_params("x", colors="#1f77b4")
        ax.set_ylim(self.grid[0], self.grid[-1])
        ax.set_ylabel("Height Above Sea Level (m)")
        ax.grid(True)

        return ax

    def __wind_components(self, ax):
        """Adds wind u and wind v graphs to the same axis.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis to add the graphs.

        Returns
        -------
        ax : matplotlib.pyplot.axis
            Axis with the graphs.
        """
        ax.plot(
            [self.environment.wind_velocity_x(i) for i in self.grid],
            self.grid,
            label="wind u",
        )
        ax.plot(
            [self.environment.wind_velocity_y(i) for i in self.grid],
            self.grid,
            label="wind v",
        )
        ax.legend(loc="best")
        ax.set_ylabel("Height Above Sea Level (m)")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.grid(True)
        ax.set_ylim(self.grid[0], self.grid[-1])

        return ax

    def __pressure_temperature(self, ax):
        """Adds pressure and temperature graphs to the same axis.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis to add the graphs.

        Returns
        -------
        ax : matplotlib.pyplot.axis
            Axis with the graphs.
        """
        ax.plot(
            [self.environment.pressure(i) / 100 for i in self.grid],
            self.grid,
            "#ff7f0e",
            label="Pressure",
        )
        ax.set_xlabel("Pressure (hPa)", color="#ff7f0e")
        ax.tick_params("x", colors="#ff7f0e")
        axup = ax.twiny()
        axup.plot(
            [self.environment.temperature(i) for i in self.grid],
            self.grid,
            color="#1f77b4",
            label="Temperature",
        )
        axup.set_xlabel("Temperature (K)", color="#1f77b4")
        axup.tick_params("x", colors="#1f77b4")
        ax.set_ylabel("Height Above Sea Level (m)")
        ax.grid(True)
        ax.set_ylim(self.grid[0], self.grid[-1])

        return ax

    def gravity_model(self):
        """Plots the gravity model graph that represents the gravitational
        acceleration as a function of height.

        Returns
        -------
        None
        """
        # Create figure
        plt.figure(figsize=(4.5, 4.5))

        # Create gravity model subplot
        ax = plt.subplot(111)
        gravity = [self.environment.gravity(i) for i in self.grid]
        ax.plot(gravity, self.grid)
        ax.set_ylabel("Height Above Sea Level (m)")
        ax.set_xlabel("Gravity Acceleration (m/s²)")
        ax.grid(True)
        ax.set_ylim(self.grid[0], self.grid[-1])
        plt.xticks(rotation=45)

        plt.show()

        return None

    def atmospheric_model(self):
        """Plots all atmospheric model graphs available. This includes wind
        speed and wind direction, density and speed of sound, wind u and wind v,
        and pressure and temperature.

        Returns
        -------
        None
        """

        # Create figure
        plt.figure(figsize=(9, 9))

        # Create wind speed and wind direction subplot
        ax1 = plt.subplot(221)
        ax1 = self.__wind(ax1)

        # Create density and speed of sound subplot
        ax2 = plt.subplot(222)
        ax2 = self.__density_speed_of_sound(ax2)

        # Create wind u and wind v subplot
        ax3 = plt.subplot(223)
        ax3 = self.__wind_components(ax3)

        # Create pressure and temperature subplot
        ax4 = plt.subplot(224)
        ax4 = self.__pressure_temperature(ax4)

        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()

        return None

    def ensemble_member_comparison(self):
        """Plots ensemble member comparisons. It requires that the environment
        model has been set as Ensemble.

        Returns
        -------
        None
        """

        current_member = self.environment.ensemble_member

        # Create figure
        plt.figure(figsize=(9, 13.5))

        # Create wind u subplot
        ax5 = plt.subplot(321)
        for i in range(self.environment.num_ensemble_members):
            self.environment.select_ensemble_member(i)
            ax5.plot(
                [self.environment.wind_velocity_x(i) for i in self.grid],
                self.grid,
                label=i,
            )
        ax5.set_ylabel("Height Above Sea Level (m)")
        ax5.set_xlabel("Wind Speed (m/s)")
        ax5.set_title("Wind U - Ensemble Members")
        ax5.grid(True)

        # Create wind v subplot
        ax6 = plt.subplot(322)
        for i in range(self.environment.num_ensemble_members):
            self.environment.select_ensemble_member(i)
            ax6.plot(
                [self.environment.wind_velocity_y(i) for i in self.grid],
                self.grid,
                label=i,
            )
        ax6.set_ylabel("Height Above Sea Level (m)")
        ax6.set_xlabel("Wind Speed (m/s)")
        ax6.set_title("Wind V - Ensemble Members")
        ax6.grid(True)

        # Create wind speed subplot
        ax7 = plt.subplot(323)
        for i in range(self.environment.num_ensemble_members):
            self.environment.select_ensemble_member(i)
            ax7.plot(
                [self.environment.wind_speed(i) for i in self.grid], self.grid, label=i
            )
        ax7.set_ylabel("Height Above Sea Level (m)")
        ax7.set_xlabel("Wind Speed (m/s)")
        ax7.set_title("Wind Speed Magnitude - Ensemble Members")
        ax7.grid(True)

        # Create wind direction subplot
        ax8 = plt.subplot(324)
        for i in range(self.environment.num_ensemble_members):
            self.environment.select_ensemble_member(i)
            ax8.plot(
                [self.environment.wind_direction(i) for i in self.grid],
                self.grid,
                label=i,
            )
        ax8.set_ylabel("Height Above Sea Level (m)")
        ax8.set_xlabel("Degrees True (°)")
        ax8.set_title("Wind Direction - Ensemble Members")
        ax8.grid(True)

        # Create pressure subplot
        ax9 = plt.subplot(325)
        for i in range(self.environment.num_ensemble_members):
            self.environment.select_ensemble_member(i)
            ax9.plot(
                [self.environment.pressure(i) for i in self.grid], self.grid, label=i
            )
        ax9.set_ylabel("Height Above Sea Level (m)")
        ax9.set_xlabel("Pressure (P)")
        ax9.set_title("Pressure - Ensemble Members")
        ax9.grid(True)

        # Create temperature subplot
        ax10 = plt.subplot(326)
        for i in range(self.environment.num_ensemble_members):
            self.environment.select_ensemble_member(i)
            ax10.plot(
                [self.environment.temperature(i) for i in self.grid], self.grid, label=i
            )
        ax10.set_ylabel("Height Above Sea Level (m)")
        ax10.set_xlabel("Temperature (K)")
        ax10.set_title("Temperature - Ensemble Members")
        ax10.grid(True)

        # Display plot
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()

        # Clean up
        self.environment.select_ensemble_member(current_member)

        return None

    def info(self):
        """Plots a summary of the atmospheric model, including wind speed and
        wind direction, density and speed of sound. This is important for the
        Environment.info() method.

        Returns
        -------
        None
        """
        print("\nAtmospheric Model Plots\n")
        plt.figure(figsize=(9, 4.5))
        # Create wind speed and wind direction subplot
        ax1 = plt.subplot(121)
        ax1 = self.__wind(ax1)

        # Create density and speed of sound subplot
        ax2 = plt.subplot(122)
        ax2 = self.__density_speed_of_sound(ax2)

        plt.subplots_adjust(wspace=0.5)
        plt.show()
        return None

    def all(self):
        """Prints out all graphs available about the Environment. This includes
        a complete description of the atmospheric model and the ensemble members
        comparison if the atmospheric model is an ensemble.

        Returns
        -------
        None
        """

        # Plot graphs
        print("\n\nGravity Model Plots")
        self.gravity_model()

        print("\n\nAtmospheric Model Plots")
        self.atmospheric_model()

        # Plot ensemble member comparison
        if self.environment.atmospheric_model_type == "Ensemble":
            print("\n\nEnsemble Members Comparison")
            self.ensemble_member_comparison()

        return None
