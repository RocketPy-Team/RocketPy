import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


class _TankPlots:
    """Class that holds plot methods for Tank class.

    Attributes
    ----------
    _TankPlots.tank : Tank
        Tank object that will be used for the plots.

    """

    def __init__(self, tank):
        """Initializes _MotorClass class.

        Parameters
        ----------
        tank : Tank
            Instance of the Tank class

        Returns
        -------
        None
        """

        self.tank = tank
        self.name = tank.name
        self.geometry = tank.geometry

        return None

    def _generate_tank(self, translate=(0, 0), csys=1):
        """Generates a matplotlib patch object that represents the tank.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the tank on. If None, a new figure and axes
            will be created.
        translate : tuple, optional
            Tuple of floats that represents the translation of the tank
            geometry.
        csys : float, optional
            Coordinate system of the tank, this will define the orientation of
            the tank. Default is 1, which means that the tank will be drawn
            with the nose cone pointing left.

        Returns
        -------
        tank : matplotlib.patches.Polygon
            Polygon object that represents the tank.
        """
        # get positions of all points
        x = csys * self.geometry.radius.x_array + translate[0]
        y = csys * self.geometry.radius.y_array + translate[1]
        x = np.concatenate([x, x[::-1]])
        y = np.concatenate([y, -y[::-1]])
        xy = np.column_stack([x, y])

        tank = Polygon(
            xy,
            label=self.name,
            facecolor="dimgray",
            edgecolor="black",
        )
        # Don't set any plot config here. Use the draw methods for that
        return tank

    def draw(self):
        """Draws the tank geometry.

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(facecolor="#EEEEEE")

        ax.add_patch(self._generate_tank())

        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", linewidth=0.5)

        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Radius (m)")
        ax.set_title("Tank Geometry")

        x_max = self.geometry.radius.x_array.max()
        y_max = self.geometry.radius.y_array.max()
        ax.set_xlim(-1.2 * x_max, 1.2 * x_max)
        ax.set_ylim(-1.5 * y_max, 1.5 * y_max)

    def all(self):
        """Prints out all graphs available about the Tank. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """

        return None
