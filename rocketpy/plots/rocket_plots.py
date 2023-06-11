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

    def stabilityMargin(self):
        """Plots static margin of the rocket as a function of time.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # TODO: it would be interesting to make a 3D plot of stability margin
        # (https://matplotlib.org/stable/gallery/mplot3d/surface3d.html)

        x = np.linspace(0, self.rocket.motor.burnOutTime, 20)
        y = np.array([self.rocket.stabilityMargin(t, 0) for t in x])

        plt.plot(x, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Stability Margin (calibers)")
        plt.title("Stability Margin (mach = 0)")
        plt.grid()
        plt.show()

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
        self.stabilityMargin()
        self.powerOnDrag()
        self.powerOffDrag()
        self.thrustToWeight()

        return None
