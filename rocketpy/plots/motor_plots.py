import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


class _MotorPlots:
    """Class that holds plot methods for Motor class.

    Attributes
    ----------
    _MotorPlots.motor : Motor
        Motor object that will be used for the plots.

    """

    def __init__(self, motor):
        """Initializes _MotorClass class.

        Parameters
        ----------
        motor : Motor
            Instance of the Motor class

        Returns
        -------
        None
        """
        self.motor = motor

    def thrust(self, lower_limit=None, upper_limit=None):
        """Plots thrust of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.thrust.plot(lower=lower_limit, upper=upper_limit)

    def total_mass(self, lower_limit=None, upper_limit=None):
        """Plots total_mass of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.total_mass.plot(lower=lower_limit, upper=upper_limit)

    def propellant_mass(self, lower_limit=None, upper_limit=None):
        """Plots propellant_mass of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is None, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is None, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.propellant_mass.plot(lower=lower_limit, upper=upper_limit)

    def center_of_mass(self, lower_limit=None, upper_limit=None):
        """Plots center_of_mass of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.center_of_mass.plot(lower=lower_limit, upper=upper_limit)

    def mass_flow_rate(self, lower_limit=None, upper_limit=None):
        """Plots mass_flow_rate of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.mass_flow_rate.plot(lower=lower_limit, upper=upper_limit)

    def exhaust_velocity(self, lower_limit=None, upper_limit=None):
        """Plots exhaust_velocity of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.exhaust_velocity.plot(lower=lower_limit, upper=upper_limit)

    def I_11(self, lower_limit=None, upper_limit=None):
        """Plots I_11 of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.I_11.plot(lower=lower_limit, upper=upper_limit)

    def I_22(self, lower_limit=None, upper_limit=None):
        """Plots I_22 of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.I_22.plot(lower=lower_limit, upper=upper_limit)

    def I_33(self, lower_limit=None, upper_limit=None):
        """Plots I_33 of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.I_33.plot(lower=lower_limit, upper=upper_limit)

    def I_12(self, lower_limit=None, upper_limit=None):
        """Plots I_12 of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.I_12.plot(lower=lower_limit, upper=upper_limit)

    def I_13(self, lower_limit=None, upper_limit=None):
        """Plots I_13 of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is none, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.I_13.plot(lower=lower_limit, upper=upper_limit)

    def I_23(self, lower_limit=None, upper_limit=None):
        """Plots I_23 of the motor as a function of time.

        Parameters
        ----------
        lower_limit : float
            Lower limit of the plot. Default is None, which means that the plot
            limits will be automatically calculated.
        upper_limit : float
            Upper limit of the plot. Default is None, which means that the plot
            limits will be automatically calculated.

        Return
        ------
        None
        """
        self.motor.I_23.plot(lower=lower_limit, upper=upper_limit)

    def _generate_nozzle(self, translate=(0, 0), csys=1):
        """Generates a patch that represents the nozzle of the motor. It is
        simply a polygon with 5 vertices mirrored in the x axis. The nozzle is
        drawn in the origin and then translated and rotated to the correct
        position.

        Parameters
        ----------
        translate : tuple
            Tuple with the x and y coordinates of the translation that will be
            applied to the nozzle.
        csys : float
            Coordinate system of the motor or rocket. This will define the
            orientation of the nozzle draw. Default is 1, which means that the
            nozzle will be drawn with its outlet pointing to the right.

        Returns
        -------
        patch : matplotlib.patches.Polygon
            Patch that represents the nozzle of the liquid_motor.
        """
        nozzle_radius = self.motor.nozzle_radius
        try:
            throat_radius = self.motor.throat_radius
        except AttributeError:
            # Liquid motors don't have throat radius, set a default value
            throat_radius = nozzle_radius / 3
        # calculate length between throat and nozzle outlet assuming 15º angle
        major_axis = (nozzle_radius - throat_radius) / np.tan(np.deg2rad(15))
        # calculate minor axis assuming a 45º angle
        minor_axis = (nozzle_radius - throat_radius) / np.tan(np.deg2rad(45))

        # calculate x and y coordinates of the nozzle
        x = csys * np.array(
            [0, 0, major_axis, major_axis + minor_axis, major_axis + minor_axis]
        )
        y = csys * np.array([0, nozzle_radius, throat_radius, nozzle_radius, 0])
        # we need to draw the other half of the nozzle
        x = np.concatenate([x, x[::-1]])
        y = np.concatenate([y, -y[::-1]])
        # now we need to sum the position and the translate
        x = x + translate[0]
        y = y + translate[1]

        patch = Polygon(
            np.column_stack([x, y]),
            label="Nozzle",
            facecolor="black",
            edgecolor="black",
        )
        return patch

    def _generate_combustion_chamber(
        self, translate=(0, 0), label="Combustion Chamber"
    ):
        """Generates a patch that represents the combustion chamber of the
        motor. It is simply a polygon with 4 vertices mirrored in the x axis.
        The combustion chamber is drawn in the origin and must be translated.

        Parameters
        ----------
        translate : tuple
            Tuple with the x and y coordinates of the translation that will be
            applied to the combustion chamber.
        label : str
            Label that will be used in the legend of the plot. Default is
            "Combustion Chamber".

        Returns
        -------
        patch : matplotlib.patches.Polygon
            Patch that represents the combustion chamber of the motor.
        """
        chamber_length = (
            self.motor.grain_initial_height + self.motor.grain_separation
        ) * self.motor.grain_number
        x = np.array(
            [
                0,
                chamber_length,
                chamber_length,
            ]
        )
        y = np.array(
            [
                self.motor.grain_outer_radius * 1.3,
                self.motor.grain_outer_radius * 1.3,
                0,
            ]
        )
        # we need to draw the other half of the chamber
        x = np.concatenate([x, x[::-1]])
        y = np.concatenate([y, -y[::-1]])
        # the point of reference for the chamber is its center
        # so we need to subtract half of its length and add the translation
        x = x + translate[0] - chamber_length / 2
        y = y + translate[1]

        patch = Polygon(
            np.column_stack([x, y]),
            label=label,
            facecolor="lightslategray",
            edgecolor="black",
        )
        return patch

    def _generate_grains(self, translate=(0, 0)):
        """Generates a list of patches that represent the grains of the motor.
        Each grain is a polygon with 4 vertices mirrored in the x axis. The top
        and bottom vertices are the same for all grains, but the left and right
        vertices are different for each grain. The grains are drawn in the
        origin and must be translated.

        Parameters
        ----------
        translate : tuple
            Tuple with the x and y coordinates of the translation that will be
            applied to the grains.

        Returns
        -------
        patches : list
            List of patches that represent the grains of the motor.
        """
        patches = []
        numgrains = self.motor.grain_number
        separation = self.motor.grain_separation
        height = self.motor.grain_initial_height
        outer_radius = self.motor.grain_outer_radius
        inner_radius = self.motor.grain_initial_inner_radius
        total_length = (
            self.motor.grain_number
            * (self.motor.grain_initial_height + (self.motor.grain_separation))
            - self.motor.grain_separation
        )

        inner_y = np.array([0, inner_radius, inner_radius, 0])
        outer_y = np.array([inner_radius, outer_radius, outer_radius, inner_radius])
        inner_y = np.concatenate([inner_y, -inner_y[::-1]])
        outer_y = np.concatenate([outer_y, -outer_y[::-1]])
        inner_y = inner_y + translate[1]
        outer_y = outer_y + translate[1]

        initial_grain_position = 0
        for n in range(numgrains):
            grain_start = initial_grain_position
            grain_end = grain_start + height
            initial_grain_position = grain_end + separation
            x = np.array([grain_start, grain_start, grain_end, grain_end])
            # draw the other half of the nozzle
            x = np.concatenate([x, x[::-1]])
            # sum the translate
            x = x + translate[0] - total_length / 2
            patch = Polygon(
                np.column_stack([x, outer_y]),
                facecolor="olive",
                edgecolor="khaki",
            )
            patches.append(patch)

            patch = Polygon(
                np.column_stack([x, inner_y]),
                facecolor="khaki",
                edgecolor="olive",
            )
            if n == 0:
                patch.set_label("Grains")
            patches.append(patch)
        return patches

    def _generate_positioned_tanks(self, translate=(0, 0), csys=1):
        """Generates a list of patches that represent the tanks of the
        liquid_motor.

        Parameters
        ----------
        translate : tuple
            Tuple with the x and y coordinates of the translation that will be
            applied to the tanks.
        csys : float
            Coordinate system of the motor or rocket. This will define the
            orientation of the tanks draw. Default is 1, which means that the
            tanks will be drawn with the nose cone pointing left.

        Returns
        -------
        patches_and_centers : list
            List of tuples where the first item is the patch of the tank, and the
            second item is the geometrical center.
        """
        colors = {
            0: ("black", "dimgray"),
            1: ("darkblue", "cornflowerblue"),
            2: ("darkgreen", "limegreen"),
            3: ("darkorange", "gold"),
            4: ("darkred", "tomato"),
            5: ("darkviolet", "violet"),
        }
        patches_and_centers = []
        for idx, pos_tank in enumerate(self.motor.positioned_tanks):
            tank = pos_tank["tank"]
            position = pos_tank["position"] * csys
            geometrical_center = (position + translate[0], translate[1])
            color_idx = idx % len(colors)  # Use modulo operator to loop through colors
            patch = tank.plots._generate_tank(geometrical_center, csys)
            patch.set_facecolor(colors[color_idx][1])
            patch.set_edgecolor(colors[color_idx][0])
            patch.set_alpha(0.8)
            patches_and_centers.append((patch, geometrical_center))
        return patches_and_centers

    def _draw_center_of_mass(self, ax):
        """Draws a red circle in the center of mass of the motor. This can be
        used for grains center of mass and the center of dry mass.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot the center of mass on.
        """
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)  # symmetry line
        try:
            ax.plot(
                [self.motor.grains_center_of_mass_position],
                [0],
                "ro",
                label="Grains Center of Mass",
            )
        except AttributeError:
            pass
        ax.plot(
            [self.motor.center_of_dry_mass_position],
            [0],
            "bo",
            label="Center of Dry Mass",
        )

    def _generate_motor_region(self, list_of_patches):
        """Generates a patch that represents the motor outline. It is
        simply a polygon with 4 vertices mirrored in the x axis. The outline is
        drawn considering all the patches that represent the motor.

        Parameters
        ----------
        list_of_patches : list
            List of patches that represent the motor outline.

        Returns
        -------
        patch : matplotlib.patches.Polygon
            Patch that represents the motor outline.
        """
        # get max and min x and y values from all motor patches
        x_min = min(patch.xy[:, 0].min() for patch in list_of_patches)
        x_max = max(patch.xy[:, 0].max() for patch in list_of_patches)
        y_min = min(patch.xy[:, 1].min() for patch in list_of_patches)
        y_max = max(patch.xy[:, 1].max() for patch in list_of_patches)

        # calculate x and y coordinates of the motor outline
        x = np.array([x_min, x_max, x_max, x_min])
        y = np.array([y_min, y_min, y_max, y_max])

        # draw the other half of the outline
        x = np.concatenate([x, x[::-1]])
        y = np.concatenate([y, -y[::-1]])

        # create the patch polygon with no fill but outlined with dashed line
        patch = Polygon(
            np.column_stack([x, y]),
            facecolor="#bdbdbd",
            edgecolor="#bdbdbd",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

        return patch

    def _set_plot_properties(self, ax):
        ax.set_aspect("equal")
        ax.set_ymargin(0.8)
        plt.grid(True, linestyle="--", linewidth=0.2)
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    def all(self):
        """Prints out all graphs available about the Motor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.thrust(*self.motor.burn_time)
        # self.mass_flow_rate(*self.motor.burn_time)
        self.exhaust_velocity(*self.motor.burn_time)
        self.total_mass(*self.motor.burn_time)
        self.propellant_mass(*self.motor.burn_time)
        self.center_of_mass(*self.motor.burn_time)
        self.I_11(*self.motor.burn_time)
        self.I_22(*self.motor.burn_time)
        self.I_33(*self.motor.burn_time)
        self.I_12(*self.motor.burn_time)
        self.I_13(*self.motor.burn_time)
        self.I_23(*self.motor.burn_time)
