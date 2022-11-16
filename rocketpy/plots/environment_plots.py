__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


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
        self.grid = np.linspace(environment.elevation, environment.maxExpectedHeight)

        self.environment = environment

        return None

    def atmospheric_model(self):
        """Plots all atmospheric model graphs available

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Create figure
        plt.figure(figsize=(9, 9))

        # Create wind speed and wind direction subplot
        ax1 = plt.subplot(221)
        ax1.plot(
            [self.environment.windSpeed(i) for i in self.grid],
            self.grid,
            "#ff7f0e",
            label="Speed of Sound",
        )
        ax1.set_xlabel("Wind Speed (m/s)", color="#ff7f0e")
        ax1.tick_params("x", colors="#ff7f0e")
        ax1up = ax1.twiny()
        ax1up.plot(
            [self.environment.windDirection(i) for i in self.grid],
            self.grid,
            color="#1f77b4",
            label="Density",
        )
        ax1up.set_xlabel("Wind Direction (°)", color="#1f77b4")
        ax1up.tick_params("x", colors="#1f77b4")
        ax1up.set_xlim(0, 360)
        ax1.set_ylabel("Height Above Sea Level (m)")
        ax1.grid(True)

        # Create density and speed of sound subplot
        ax2 = plt.subplot(222)
        ax2.plot(
            [self.environment.speedOfSound(i) for i in self.grid],
            self.grid,
            "#ff7f0e",
            label="Speed of Sound",
        )
        ax2.set_xlabel("Speed of Sound (m/s)", color="#ff7f0e")
        ax2.tick_params("x", colors="#ff7f0e")
        ax2up = ax2.twiny()
        ax2up.plot(
            [self.environment.density(i) for i in self.grid],
            self.grid,
            color="#1f77b4",
            label="Density",
        )
        ax2up.set_xlabel("Density (kg/m³)", color="#1f77b4")
        ax2up.tick_params("x", colors="#1f77b4")
        ax2.set_ylabel("Height Above Sea Level (m)")
        ax2.grid(True)

        # Create wind u and wind v subplot
        ax3 = plt.subplot(223)
        ax3.plot(
            [self.environment.windVelocityX(i) for i in self.grid],
            self.grid,
            label="Wind U",
        )
        ax3.plot(
            [self.environment.windVelocityY(i) for i in self.grid],
            self.grid,
            label="Wind V",
        )
        ax3.legend(loc="best").set_draggable(True)
        ax3.set_ylabel("Height Above Sea Level (m)")
        ax3.set_xlabel("Wind Speed (m/s)")
        ax3.grid(True)

        # Create pressure and temperature subplot
        ax4 = plt.subplot(224)
        ax4.plot(
            [self.environment.pressure(i) / 100 for i in self.grid],
            self.grid,
            "#ff7f0e",
            label="Pressure",
        )
        ax4.set_xlabel("Pressure (hPa)", color="#ff7f0e")
        ax4.tick_params("x", colors="#ff7f0e")
        ax4up = ax4.twiny()
        ax4up.plot(
            [self.environment.temperature(i) for i in self.grid],
            self.grid,
            color="#1f77b4",
            label="Temperature",
        )
        ax4up.set_xlabel("Temperature (K)", color="#1f77b4")
        ax4up.tick_params("x", colors="#1f77b4")
        ax4.set_ylabel("Height Above Sea Level (m)")
        ax4.grid(True)

        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()

        return None

    def ensemble_member_comparison(self):
        """Plots ensemble member comparisons.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        currentMember = self.environment.ensembleMember

        # Create figure
        plt.figure(figsize=(9, 13.5))

        # Create wind u subplot
        ax5 = plt.subplot(321)
        for i in range(self.environment.numEnsembleMembers):
            self.environment.selectEnsembleMember(i)
            ax5.plot(
                [self.environment.windVelocityX(i) for i in self.grid],
                self.grid,
                label=i,
            )
        # ax5.legend(loc='best').set_draggable(True)
        ax5.set_ylabel("Height Above Sea Level (m)")
        ax5.set_xlabel("Wind Speed (m/s)")
        ax5.set_title("Wind U - Ensemble Members")
        ax5.grid(True)

        # Create wind v subplot
        ax6 = plt.subplot(322)
        for i in range(self.environment.numEnsembleMembers):
            self.environment.selectEnsembleMember(i)
            ax6.plot(
                [self.environment.windVelocityY(i) for i in self.grid],
                self.grid,
                label=i,
            )
        # ax6.legend(loc='best').set_draggable(True)
        ax6.set_ylabel("Height Above Sea Level (m)")
        ax6.set_xlabel("Wind Speed (m/s)")
        ax6.set_title("Wind V - Ensemble Members")
        ax6.grid(True)

        # Create wind speed subplot
        ax7 = plt.subplot(323)
        for i in range(self.environment.numEnsembleMembers):
            self.environment.selectEnsembleMember(i)
            ax7.plot(
                [self.environment.windSpeed(i) for i in self.grid], self.grid, label=i
            )
        # ax7.legend(loc='best').set_draggable(True)
        ax7.set_ylabel("Height Above Sea Level (m)")
        ax7.set_xlabel("Wind Speed (m/s)")
        ax7.set_title("Wind Speed Magnitude - Ensemble Members")
        ax7.grid(True)

        # Create wind direction subplot
        ax8 = plt.subplot(324)
        for i in range(self.environment.numEnsembleMembers):
            self.environment.selectEnsembleMember(i)
            ax8.plot(
                [self.environment.windDirection(i) for i in self.grid],
                self.grid,
                label=i,
            )
        # ax8.legend(loc='best').set_draggable(True)
        ax8.set_ylabel("Height Above Sea Level (m)")
        ax8.set_xlabel("Degrees True (°)")
        ax8.set_title("Wind Direction - Ensemble Members")
        ax8.grid(True)

        # Create pressure subplot
        ax9 = plt.subplot(325)
        for i in range(self.environment.numEnsembleMembers):
            self.environment.selectEnsembleMember(i)
            ax9.plot(
                [self.environment.pressure(i) for i in self.grid], self.grid, label=i
            )
        # ax9.legend(loc='best').set_draggable(True)
        ax9.set_ylabel("Height Above Sea Level (m)")
        ax9.set_xlabel("Pressure (P)")
        ax9.set_title("Pressure - Ensemble Members")
        ax9.grid(True)

        # Create temperature subplot
        ax10 = plt.subplot(326)
        for i in range(self.environment.numEnsembleMembers):
            self.environment.selectEnsembleMember(i)
            ax10.plot(
                [self.environment.temperature(i) for i in self.grid], self.grid, label=i
            )
        # ax10.legend(loc='best').set_draggable(True)
        ax10.set_ylabel("Height Above Sea Level (m)")
        ax10.set_xlabel("Temperature (K)")
        ax10.set_title("Temperature - Ensemble Members")
        ax10.grid(True)

        # Display plot
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()

        # Clean up
        self.environment.selectEnsembleMember(currentMember)

        return None

    def all(self):
        """Prints out all graphs available about the Environment.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Plot graphs
        print("\n\nAtmospheric Model Plots")
        self.atmospheric_model()

        # Plot ensemble member comparison
        if self.environment.atmosphericModelType == "Ensemble":
            print("\n\nEnsemble Members Comparison")
            self.ensemble_member_comparison()

        return None
