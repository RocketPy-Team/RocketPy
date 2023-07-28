__author__ = "Oscar Mauricio Prada Ramirez"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import matplotlib.pyplot as plt
import numpy as np


class _RocketPlots:
    """Class that holds plot methods for Rocket class.

    Attributes
    ----------
    _RocketPlots.rocket : Rocket
        Rocket object that will be used for the plots.

    """

    def __init__(self, rocket) -> None:
        """Initializes _RocketPlots class.

        Parameters
        ----------
        rocket : Rocket
            Instance of the Rocket class

        Returns
        -------
        None
        """

        self.rocket = rocket

        return None

    def total_mass(self):
        """Plots total mass of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.total_mass()

        return None

    def reduced_mass(self):
        """Plots reduced mass of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.reduced_mass()

        return None

    def static_margin(self):
        """Plots static margin of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.static_margin()

        return None

    def power_on_drag(self):
        """Plots power on drag of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.power_on_drag()

        return None

    def power_off_drag(self):
        """Plots power off drag of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.power_off_drag()

        return None

    def thrust_to_weight(self):
        """Plots the motor thrust force divided by rocket
            weight as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.thrust_to_weight.plot(
            lower=0, upper=self.rocket.motor.burn_out_time
        )

        return None

    def all(self):
        """Prints out all graphs available about the Rocket. It simply calls
        all the other plotter methods in this class.

        Parameters
        ----------
        None
        Return
        ------
        None
        """

        # Show plots
        print("\nMass Plots")
        self.total_mass()
        self.reduced_mass()
        print("\nAerodynamics Plots")
        self.static_margin()
        self.power_on_drag()
        self.power_off_drag()
        self.thrust_to_weight()

        return None
