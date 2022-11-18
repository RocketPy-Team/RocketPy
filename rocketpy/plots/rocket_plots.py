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

    def powerOnDrag(self):

        self.rocket.powerOnDrag()

        return None

    def all(self):
        """Prints out all graphs available about the Rocket.
        Parameters
        ----------
        None
        Return
        ------
        None
        """

        # Show plots
        print("\nMass Plots")
        self.rocket.totalMass()
        self.rocket.reducedMass()
        print("\nAerodynamics Plots")
        self.rocket.staticMargin()
        self.rocket.powerOnDrag()
        self.rocket.powerOffDrag()
        self.rocket.thrustToWeight.plot(lower=0, upper=self.rocket.motor.burnOutTime)

        return None
