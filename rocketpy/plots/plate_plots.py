import matplotlib.pyplot as plt
import numpy as np

from rocketpy.exceptions import InvalidParameterError
from rocketpy.plots.plot_helpers import show_or_save_plot


class _PlatePlots:
    """Class that holds plot methods for the Plate class.

    Attributes
    ----------
    _PlatePlots.plate : Plate
        Plate object that will be used for the plots.
    _PlatePlots.rocket : Rocket, optional
        Rocket object to which the plate belongs. Default is None.
    """

    def __init__(self, plate) -> None:
        """Initializes _PlatePlots class.

        Parameters
        ----------
        plate : Plate
            Plate instance.
        """
        self.plate = plate
        self.rocket = None

    def draw_3d(
        self,
        color: str = "teal",
        marker: str = "h",
        elev: float | int | None = None,
        azim: float | int | None = None,
        filename: str | None = None,
    ) -> None:
        """Plots the 3D scatter plot of the discretized points forming the plate
        surface used to model soft-iron magnetic distortion.

        Parameters
        ----------
        color : str, optional
            Color of the scatter points. A full list of color names can be found
            at: https://matplotlib.org/stable/gallery/color/named_colors
            Default is "teal".
        marker : str, optional
            Shape of the markers representing the discretization points. A full
            list of markers can be found at:
            https://matplotlib.org/stable/api/markers_api.html
            Default is "h".
        elev : float, int, optional
            The elevation angle in degrees rotates the camera above the plane
            pierced by the vertical axis, with a positive angle corresponding
            to a location above that plane. If None, the default view is used.
            Default is None.
        azim : float, int, optional
            The azimuthal angle in degrees rotates the camera about the vertical
            axis. If None, the default view is used. Default is None.
        filename : str, optional
            The path the plot should be saved to. If None, the plot will be shown instead
            of saved. Supported file formats include: eps, jpg, jpeg, pdf, pgf, png, ps,
            raw, rgba, svg, svgz, tif, tiff, and webp. Default is None.
        """
        if self.rocket is None:
            raise InvalidParameterError(
                "Plate points list is empty. Add the plate to a rocket before plotting."
            )
        # from bacs to ucs
        x, y, z = zip(*self.plate.points)
        if self.rocket._csys == -1:
            x = -np.array(x)
        else:
            x = np.array(x)
        z = self.rocket.center_of_dry_mass_position + (self.rocket._csys * np.array(z))
        y = np.array(y)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        # plot individual points
        ax.scatter(x, y, z, color=color, marker=marker, label=self.plate.name)
        ax.view_init(elev=elev, azim=azim)
        if self.rocket._csys == 1:
            ax.invert_xaxis()

        # Labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

        print(f"\n{self.plate.name} representation: ")
        show_or_save_plot(filename)

    def draw(
        self,
        vis_args: dict | None = None,
        plane: str = "xz",
        color: str = "darkgreen",
        filename: str | None = None,
    ) -> None:
        """Plots the plate along with the 2D cross-sectional outline of the
        rocket body.

        Parameters
        ----------
        vis_args : dict, optional
            Determines the visual appearance when drawing the rocket body. If
            None, default parameters are used:

            .. code-block:: python

                {
                    "background": "#EEEEEE",
                    "tail": "black",
                    "nose": "black",
                    "body": "black",
                    "fins": "black",
                    "motor": "black",
                    "buttons": "black",
                    "line_width": 1.0,
                }

            A full list of color names can be found at:
            https://matplotlib.org/stable/gallery/color/named_colors
        plane : str, optional
            Cross-sectional projection plane to represent. Accepted options are
            "xz" and "yz". Default is "xz".
        color : str, optional
            Fill color of the plate. A full list of color names can be found at:
            https://matplotlib.org/stable/gallery/color/named_colors
            Default is "darkgreen".
        filename : str, optional
            The path the plot should be saved to. If None, the plot is shown
            interactively. Supported file formats include: eps, jpg, jpeg, pdf,
            pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, and webp. Default is None.
        """
        if self.rocket is None:
            raise InvalidParameterError(
                "Plate points list is empty. Add the plate to a rocket before plotting."
            )
        if vis_args is None:
            vis_args = {
                "background": "#EEEEEE",
                "tail": "black",
                "nose": "black",
                "body": "black",
                "fins": "black",
                "motor": "black",
                "buttons": "black",
                "line_width": 1.0,
            }

        ax, _, _ = self.rocket.plots._rocket_shape_plot(vis_args, plane)
        self._plot_plate_rocket(ax, plane, color)

        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            handlelength=0.8,
            handleheight=0.6,
        )
        plt.tight_layout()
        show_or_save_plot(filename)

    def _plot_plate_rocket(
        self, ax, plane: str = "xz", color: str = "darkgreen"
    ) -> None:
        """Draws the filled 2D projection of the plate onto the rocket axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib Axes instance where the plate will be drawn.
        plane : str, optional
            Cross-sectional projection plane ("xz" or "yz"). Default is "xz".
        color : str, optional
            Fill color of the plate projection. Default is "darkgreen".
        """
        x, y, z = zip(*self.plate.points)

        x = self.rocket._csys * np.array(x)
        z = self.rocket.center_of_dry_mass_position + (self.rocket._csys * np.array(z))
        y = np.array(y)

        if plane == "xz":
            r = x
        elif plane == "yz":
            r = y
        else:
            raise InvalidParameterError("Plane value can only be 'xz' or 'yz'.")

        unique_z = np.unique(z)
        r_min = np.array([r[z == uz].min() for uz in unique_z])
        r_max = np.array([r[z == uz].max() for uz in unique_z])

        ax.fill_between(
            unique_z, r_min, r_max, color=color, alpha=0.9, label=self.plate.name
        )

    def all(self) -> None:
        """Plots all available graphs for the _PlatePlots instance using default
        parameters.
        """
        self.draw_3d()
        self.draw()
