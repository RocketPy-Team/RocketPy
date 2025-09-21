import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from rocketpy.mathutils.function import Function

from .plot_helpers import show_or_save_plot


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
        self.flux_time = tank.flux_time
        self.geometry = tank.geometry

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

    def draw(self, *, filename=None):
        """Draws the tank geometry.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        _, ax = plt.subplots(facecolor="#EEEEEE")

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
        show_or_save_plot(filename)

    def fluid_volume(self, filename=None):
        """Plots both the liquid and gas fluid volumes.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        """
        _, ax = Function.compare_plots(
            [self.tank.liquid_volume, self.tank.gas_volume],
            *self.flux_time,
            title="Fluid Volume (m^3) x Time (s)",
            xlabel="Time (s)",
            ylabel="Volume (m^3)",
            show=False,
            return_object=True,
        )
        ax.legend(["Liquid", "Gas"])
        show_or_save_plot(filename)

    def fluid_height(self, filename=None):
        """Plots both the liquid and gas fluid height.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        """
        _, ax = Function.compare_plots(
            [self.tank.liquid_height, self.tank.gas_height],
            *self.flux_time,
            title="Fluid Height (m) x Time (s)",
            xlabel="Time (s)",
            ylabel="Height (m)",
            show=False,
            return_object=True,
        )
        ax.legend(["Liquid", "Gas"])
        show_or_save_plot(filename)

    def fluid_center_of_mass(self, filename=None):
        """Plots the gas, liquid and combined center of mass.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        """
        _, ax = Function.compare_plots(
            [
                self.tank.liquid_center_of_mass,
                self.tank.gas_center_of_mass,
                self.tank.center_of_mass,
            ],
            *self.flux_time,
            title="Fluid Center of Mass (m) x Time (s)",
            xlabel="Time (s)",
            ylabel="Center of Mass (m)",
            show=False,
            return_object=True,
        )
        # Change style of lines
        ax.lines[0].set_linestyle("--")
        ax.lines[1].set_linestyle("-.")
        ax.legend(["Liquid", "Gas", "Total"])
        show_or_save_plot(filename)

    def all(self):
        """Prints out all graphs available about the Tank. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.draw()
        self.tank.fluid_mass.plot(*self.flux_time)
        self.tank.net_mass_flow_rate.plot(*self.flux_time)
        self.fluid_height()
        self.fluid_volume()
        self.fluid_center_of_mass()
        self.tank.inertia.plot(*self.flux_time)
