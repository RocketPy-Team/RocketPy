import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from rocketpy.exceptions import InvalidParameterError
from rocketpy.plots.plot_helpers import show_or_save_plot


class _WirePlots:
    """Class that holds plot methods for the Wire class.

    Attributes
    ----------
    _WirePlots.wire : Wire
        Wire object that will be used for the plots.
    _WirePlots.rocket : Rocket, optional
        Rocket object the wire belongs to. Default is None.
    """

    def __init__(self, wire) -> None:
        """Initializes _WirePlots class.

        Parameters
        ----------
        wire : Wire
            Wire instance.
        """
        self.wire = wire
        self.rocket = None

    def draw(
        self,
        vis_args: dict | None = None,
        plane: str = "xz",
        color: str = "salmon",
        marker: str = "o",
        linestyle: str = "-",
        endpoints_names: bool = True,
        filename: str | None = None,
    ) -> None:
        """Plots the wire and the rocket together.

        Parameters
        ----------
        vis_args : dict, optional
            Determines the visual aspects when drawing the rocket. If ``None``,
            default values are used. Default values are:

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
            Plane to represent. Accepted options are "xz" and "yz".
            Default value is "xz".
        color : str, optional
            Color of the wire line and markers. A full list of color names can
            be found at: https://matplotlib.org/stable/gallery/color/named_colors
            Default is "salmon".
        marker : str, optional
            Shape of the marker representing wire endpoints. A full list of markers
            can be found at: https://matplotlib.org/stable/api/markers_api.html
            Default is "o".
        linestyle : str, optional
            Line style used to represent the wire. A full list of linestyles
            can be found at:
            https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
            Default is "-".
        endpoints_names : bool, optional
            Boolean defining whether the endpoint labels are displayed. If False,
            they will not be displayed. If True, the labels "Endpoint A" and
            "Endpoint B" will be shown. Default is True.
        filename : str, optional
            The path the plot should be saved to. If None, the plot will be shown instead
            of saved. Supported file formats include: eps, jpg, jpeg, pdf, pgf, png, ps,
            raw, rgba, svg, svgz, tif, tiff, and webp. Default is None.∫
        """
        if self.rocket is None:
            raise InvalidParameterError("Add the wire to a rocket before plotting.")
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
        self._draw_wires(ax, plane, color, marker, linestyle, endpoints_names)

        plt.xlim()
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        show_or_save_plot(filename)

    def _draw_wires(
        self,
        ax: Axes,
        plane: str = "xz",
        color: str = "salmon",
        marker: str = "o",
        linestyle: str = "-",
        endpoints_names: bool = True,
    ) -> None:
        """Plots the endpoints and the wire on the specified Matplotlib axes.

        Parameters
        ----------
        ax : Axes
            Matplotlib Axes instance on which the wire will be plotted.
        plane : str, optional
            Plane to represent. Accepted options are "xz" and "yz".
            Default is "xz".
        color : str, optional
            Color of the wire and endpoint markers. A full list of color names can
            be found at: https://matplotlib.org/stable/gallery/color/named_colors
            Default is "salmon".
        marker : str, optional
            Shape of the marker representing wire endpoints. A full list of markers
            can be found at: https://matplotlib.org/stable/api/markers_api.html
            Default is "o".
        linestyle : str, optional
            Line style used to represent the wire. A full list of linestyles
            can be found at:
            https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
            Default is "-".
        endpoints_names : bool, optional
            Boolean defining whether the endpoint labels are displayed. If False,
            they will not be displayed. If True, the labels "Endpoint A" and
            "Endpoint B" will be shown. Default is True.
        """
        # change nose to tail with nose origin
        endpoint_a_x = self.wire._wire_endpoints_bacs[0][0] * self.rocket._csys
        endpoint_b_x = self.wire._wire_endpoints_bacs[1][0] * self.rocket._csys

        endpoint_a_y = self.wire._wire_endpoints_bacs[0][1]
        endpoint_b_y = self.wire._wire_endpoints_bacs[1][1]

        endpoint_a_z = self.rocket.center_of_dry_mass_position + (
            self.wire._wire_endpoints_bacs[0][2] * self.rocket._csys
        )
        endpoint_b_z = self.rocket.center_of_dry_mass_position + (
            self.wire._wire_endpoints_bacs[1][2] * self.rocket._csys
        )
        if plane == "xz":
            r_a = endpoint_a_x
            r_b = endpoint_b_x
        elif plane == "yz":
            r_a = endpoint_a_y
            r_b = endpoint_b_y
        else:
            raise InvalidParameterError("The plane must be 'xz' or 'yz'.")

        z = [endpoint_a_z, endpoint_b_z]
        r = [r_a, r_b]

        # plot lines connecting endpoints
        ax.plot(z, r, color=color, linestyle=linestyle, label=self.wire.name)

        if endpoints_names:
            ax.scatter(
                z, r, marker=marker, color=color, zorder=5, label="Wire endpoints"
            )

            ax.annotate(
                "Endpoint A",
                xy=(endpoint_a_z, r_a),
                xytext=(6, 10),
                textcoords="offset points",
                fontsize=9,
            )

            ax.annotate(
                "Endpoint B",
                xy=(endpoint_b_z, r_b),
                xytext=(6, -14),
                textcoords="offset points",
                fontsize=9,
            )
        else:
            ax.scatter(z, r, marker=marker, color=color, zorder=5)

    def all(self) -> None:
        """Plots all available graphs for the _WirePlots instance using default
        parameters.
        """
        print(f"\n{self.wire.name} representation: ")
        self.draw()
