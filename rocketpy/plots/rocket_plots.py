__author__ = "Oscar Mauricio Prada Ramirez"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import matplotlib.pyplot as plt
import numpy as np


class _RocketPlots:
    """Class that holds plot methods for Environment class.

    Attributes
    ----------
    _RocketPlots.rocket : Rocket
        Rocket object that will be used for the plots.

    """

    def __init__(self, rocket) -> None:
        """Initializes _EnvironmentPlots class.

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

    def totalMass(self):
        """Plots total mass of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.totalMass()

        return None

    def reducedMass(self):
        """Plots reduced mass of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.reducedMass()

        return None

    def staticMargin(self):
        """Plots static margin of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.staticMargin()

        return None

    def powerOnDrag(self):
        """Plots power on drag of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.powerOnDrag()

        return None

    def powerOffDrag(self):
        """Plots power off drag of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.powerOffDrag()

        return None

    def thrustToWeight(self):
        """Plots the motor thrust force divided by rocket
            weight as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        self.rocket.thrustToWeight.plot(lower=0, upper=self.rocket.motor.burnOutTime)

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
        self.totalMass()
        self.reducedMass()
        print("\nAerodynamics Plots")
        self.staticMargin()
        self.powerOnDrag()
        self.powerOffDrag()
        self.thrustToWeight()

        return None
