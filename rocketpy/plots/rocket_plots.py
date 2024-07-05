import warnings

import matplotlib.pyplot as plt
import numpy as np

from rocketpy.motors import EmptyMotor, HybridMotor, LiquidMotor, SolidMotor
from rocketpy.rocket.aero_surface import Fins, NoseCone, Tail

from .plot_helpers import show_or_save_plot


class _RocketPlots:
    """Class that holds plot methods for Rocket class.

    Attributes
    ----------
    _RocketPlots.rocket : Rocket
        Rocket object that will be used for the plots.

    """

    def __init__(self, rocket):
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

        Returns
        -------
        None
        """

        self.rocket.total_mass()

        return None

    def reduced_mass(self):
        """Plots reduced mass of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.reduced_mass()

        return None

    def static_margin(self, filename=None):
        """Plots static margin of the rocket as a function of time.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case the plot will be shown instead of saved. Supported file endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp.

        Returns
        -------
        None
        """

        self.rocket.static_margin(filename=filename)

        return None

    def stability_margin(self):
        """Plots static margin of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.stability_margin.plot_2d(
            lower=0,
            upper=[2, self.rocket.motor.burn_out_time],  # Mach 2 and burnout
            samples=[20, 20],
            disp_type="surface",
            alpha=1,
        )

        return None

    def power_on_drag(self):
        """Plots power on drag of the rocket as a function of time.

        Returns
        -------
        None
        """

        warnings.warn(
            "The method 'power_on_drag' is deprecated as of version "
            + "1.2 and will be removed in version 1.4 "
            + "Use 'plots.drag_curves' instead.",
            DeprecationWarning,
        )

        self.rocket.power_on_drag()

        return None

    def power_off_drag(self):
        """Plots power off drag of the rocket as a function of time.

        Returns
        -------
        None
        """

        warnings.warn(
            "The method 'power_off_drag' is deprecated as of version "
            + "1.2 and will be removed in version 1.4 "
            + "Use 'plots.drag_curves' instead.",
            DeprecationWarning,
        )

        self.rocket.power_off_drag()

        return None

    def drag_curves(self, filename=None):
        """Plots power off and on drag curves of the rocket as a function of time.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case the plot will be shown instead of saved. Supported file endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp.

        Returns
        -------
        None
        """

        try:
            x_power_drag_on = self.rocket.power_on_drag.x_array
            y_power_drag_on = self.rocket.power_on_drag.y_array
        except AttributeError:
            x_power_drag_on = np.linspace(0, 2, 50)
            y_power_drag_on = np.array(
                [self.rocket.power_on_drag.source(x) for x in x_power_drag_on]
            )
        try:
            x_power_drag_off = self.rocket.power_off_drag.x_array
            y_power_drag_off = self.rocket.power_off_drag.y_array
        except AttributeError:
            x_power_drag_off = np.linspace(0, 2, 50)
            y_power_drag_off = np.array(
                [self.rocket.power_off_drag.source(x) for x in x_power_drag_off]
            )

        fig, ax = plt.subplots()
        ax.plot(x_power_drag_on, y_power_drag_on, label="Power on Drag")
        ax.plot(
            x_power_drag_off, y_power_drag_off, label="Power off Drag", linestyle="--"
        )

        ax.set_title("Drag Curves")
        ax.set_ylabel("Drag Coefficient")
        ax.set_xlabel("Mach")
        ax.axvspan(0.8, 1.2, alpha=0.3, color="gray", label="Transonic Region")
        ax.legend(loc="best", shadow=True)
        plt.grid(True)
        show_or_save_plot(filename)

    def thrust_to_weight(self):
        """Plots the motor thrust force divided by rocket
            weight as a function of time.

        Returns
        -------
        None
        """

        self.rocket.thrust_to_weight.plot(
            lower=0, upper=self.rocket.motor.burn_out_time
        )

        return None

    def draw(self, vis_args=None, filename=None):
        """Draws the rocket in a matplotlib figure.

        Parameters
        ----------
        vis_args : dict, optional
            Determines the visual aspects when drawing the rocket. If None,
            default values are used. Default values are:
            {
                "background": "#EEEEEE",
                "tail": "black",
                "nose": "black",
                "body": "black",
                "fins": "black",
                "motor": "black",
                "buttons": "black",
                "line_width": 2.0,
            }
            A full list of color names can be found at:
            https://matplotlib.org/stable/gallery/color/named_colors
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case the plot will be shown instead of saved. Supported file endings are: eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff and webp.
        """
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

        # Create the figure and axis
        _, ax = plt.subplots(figsize=(8, 6), facecolor="#EEEEEE")
        ax.set_aspect("equal")
        ax.set_facecolor(vis_args["background"])
        ax.grid(True, linestyle="--", linewidth=0.5)

        csys = self.rocket._csys
        reverse = csys == 1
        self.rocket.aerodynamic_surfaces.sort_by_position(reverse=reverse)

        drawn_surfaces = self._draw_aerodynamic_surfaces(ax, vis_args)
        last_radius, last_x = self._draw_tubes(ax, drawn_surfaces, vis_args)
        self._draw_motor(last_radius, last_x, ax, vis_args)
        self._draw_rail_buttons(ax, vis_args)
        self._draw_center_of_mass_and_pressure(ax)

        plt.title("Rocket Representation")
        plt.xlim()
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        show_or_save_plot(filename)

    def _draw_aerodynamic_surfaces(self, ax, vis_args):
        """Draws the aerodynamic surfaces and saves the position of the points
        of interest for the tubes."""
        # List of drawn surfaces with the position of points of interest
        # and the radius of the rocket at that point
        drawn_surfaces = []
        # Idea is to get the shape of each aerodynamic surface in their own
        # coordinate system and then plot them in the rocket coordinate system
        # using the position of each surface
        # For the tubes, the surfaces need to be checked in order to check for
        # diameter changes. The final point of the last surface is the final
        # point of the last tube

        for surface, position in self.rocket.aerodynamic_surfaces:
            if isinstance(surface, NoseCone):
                self._draw_nose_cone(ax, surface, position, drawn_surfaces, vis_args)
            elif isinstance(surface, Tail):
                self._draw_tail(ax, surface, position, drawn_surfaces, vis_args)
            elif isinstance(surface, Fins):
                self._draw_fins(ax, surface, position, drawn_surfaces, vis_args)
        return drawn_surfaces

    def _draw_nose_cone(self, ax, surface, position, drawn_surfaces, vis_args):
        """Draws the nosecone and saves the position of the points of interest
        for the tubes."""
        x_nosecone = -self.rocket._csys * surface.shape_vec[0] + position
        y_nosecone = surface.shape_vec[1]
        ax.plot(
            x_nosecone,
            y_nosecone,
            color=vis_args["nose"],
            linewidth=vis_args["line_width"],
        )
        ax.plot(
            x_nosecone,
            -y_nosecone,
            color=vis_args["nose"],
            linewidth=vis_args["line_width"],
        )
        # close the nosecone
        ax.plot(
            [x_nosecone[-1], x_nosecone[-1]],
            [y_nosecone[-1], -y_nosecone[-1]],
            color=vis_args["nose"],
            linewidth=vis_args["line_width"],
        )
        # Add the nosecone to the list of drawn surfaces
        drawn_surfaces.append(
            (surface, x_nosecone[-1], surface.rocket_radius, x_nosecone[-1])
        )

    def _draw_tail(self, ax, surface, position, drawn_surfaces, vis_args):
        """Draws the tail and saves the position of the points of interest
        for the tubes."""
        x_tail = -self.rocket._csys * surface.shape_vec[0] + position
        y_tail = surface.shape_vec[1]
        ax.plot(
            x_tail, y_tail, color=vis_args["tail"], linewidth=vis_args["line_width"]
        )
        ax.plot(
            x_tail, -y_tail, color=vis_args["tail"], linewidth=vis_args["line_width"]
        )
        # close above and below the tail
        ax.plot(
            [x_tail[-1], x_tail[-1]],
            [y_tail[-1], -y_tail[-1]],
            color=vis_args["tail"],
            linewidth=vis_args["line_width"],
        )
        ax.plot(
            [x_tail[0], x_tail[0]],
            [y_tail[0], -y_tail[0]],
            color=vis_args["tail"],
            linewidth=vis_args["line_width"],
        )
        # Add the tail to the list of drawn surfaces
        drawn_surfaces.append((surface, position, surface.bottom_radius, x_tail[-1]))

    def _draw_fins(self, ax, surface, position, drawn_surfaces, vis_args):
        """Draws the fins and saves the position of the points of interest
        for the tubes."""
        num_fins = surface.n
        x_fin = -self.rocket._csys * surface.shape_vec[0] + position
        y_fin = surface.shape_vec[1] + surface.rocket_radius
        rotation_angles = [2 * np.pi * i / num_fins for i in range(num_fins)]

        for angle in rotation_angles:
            # Create a rotation matrix for the current angle around the x-axis
            rotation_matrix = np.array([[1, 0], [0, np.cos(angle)]])

            # Apply the rotation to the original fin points
            rotated_points_2d = np.dot(rotation_matrix, np.vstack((x_fin, y_fin)))

            # Extract x and y coordinates of the rotated points
            x_rotated, y_rotated = rotated_points_2d

            # Project points above the XY plane back into the XY plane (set z-coordinate to 0)
            x_rotated = np.where(
                rotated_points_2d[1] > 0, rotated_points_2d[0], x_rotated
            )
            y_rotated = np.where(
                rotated_points_2d[1] > 0, rotated_points_2d[1], y_rotated
            )
            ax.plot(
                x_rotated,
                y_rotated,
                color=vis_args["fins"],
                linewidth=vis_args["line_width"],
            )

        drawn_surfaces.append((surface, position, surface.rocket_radius, x_rotated[-1]))

    def _draw_tubes(self, ax, drawn_surfaces, vis_args):
        """Draws the tubes between the aerodynamic surfaces."""
        for i, d_surface in enumerate(drawn_surfaces):
            # Draw the tubes, from the end of the first surface to the beginning
            # of the next surface, with the radius of the rocket at that point
            surface, position, radius, last_x = d_surface

            if i == len(drawn_surfaces) - 1:
                # If the last surface is a tail, do nothing
                if isinstance(surface, Tail):
                    continue
                # Else goes to the end of the surface
                else:
                    x_tube = [position, last_x]
                    y_tube = [radius, radius]
                    y_tube_negated = [-radius, -radius]
            else:
                # If it is not the last surface, the tube goes to the beginning
                # of the next surface
                next_surface, next_position, next_radius, next_last_x = drawn_surfaces[
                    i + 1
                ]
                x_tube = [last_x, next_position]
                y_tube = [radius, radius]
                y_tube_negated = [-radius, -radius]

            ax.plot(
                x_tube,
                y_tube,
                color=vis_args["body"],
                linewidth=vis_args["line_width"],
            )
            ax.plot(
                x_tube,
                y_tube_negated,
                color=vis_args["body"],
                linewidth=vis_args["line_width"],
            )
        return radius, last_x

    def _draw_motor(self, last_radius, last_x, ax, vis_args):
        """Draws the motor from motor patches"""
        total_csys = self.rocket._csys * self.rocket.motor._csys
        nozzle_position = (
            self.rocket.motor_position + self.rocket.motor.nozzle_position * total_csys
        )

        # Get motor patches translated to the correct position
        motor_patches = self._generate_motor_patches(total_csys, ax, vis_args)

        # Draw patches
        if not isinstance(self.rocket.motor, EmptyMotor):
            # Add nozzle last so it is in front of the other patches
            nozzle = self.rocket.motor.plots._generate_nozzle(
                translate=(nozzle_position, 0), csys=self.rocket._csys
            )
            motor_patches += [nozzle]

            outline = self.rocket.motor.plots._generate_motor_region(
                list_of_patches=motor_patches
            )
            # add outline first so it is behind the other patches
            ax.add_patch(outline)
            for patch in motor_patches:
                ax.add_patch(patch)

        self._draw_nozzle_tube(last_radius, last_x, nozzle_position, ax, vis_args)

    def _generate_motor_patches(self, total_csys, ax, vis_args):
        """Generates motor patches for drawing"""
        motor_patches = []

        if isinstance(self.rocket.motor, SolidMotor):
            grains_cm_position = (
                self.rocket.motor_position
                + self.rocket.motor.grains_center_of_mass_position * total_csys
            )
            ax.scatter(
                grains_cm_position,
                0,
                color="brown",
                label="Grains Center of Mass",
                s=8,
                zorder=10,
            )

            chamber = self.rocket.motor.plots._generate_combustion_chamber(
                translate=(grains_cm_position, 0), label=None
            )
            grains = self.rocket.motor.plots._generate_grains(
                translate=(grains_cm_position, 0)
            )

            motor_patches += [chamber, *grains]

        elif isinstance(self.rocket.motor, HybridMotor):
            grains_cm_position = (
                self.rocket.motor_position
                + self.rocket.motor.grains_center_of_mass_position * total_csys
            )
            ax.scatter(
                grains_cm_position,
                0,
                color="brown",
                label="Grains Center of Mass",
                s=8,
                zorder=10,
            )

            tanks_and_centers = self.rocket.motor.plots._generate_positioned_tanks(
                translate=(self.rocket.motor_position, 0), csys=total_csys
            )
            chamber = self.rocket.motor.plots._generate_combustion_chamber(
                translate=(grains_cm_position, 0), label=None
            )
            grains = self.rocket.motor.plots._generate_grains(
                translate=(grains_cm_position, 0)
            )
            motor_patches += [chamber, *grains]
            for tank, center in tanks_and_centers:
                ax.scatter(
                    center[0],
                    center[1],
                    color="black",
                    alpha=0.2,
                    s=5,
                    zorder=10,
                )
                motor_patches += [tank]

        elif isinstance(self.rocket.motor, LiquidMotor):
            tanks_and_centers = self.rocket.motor.plots._generate_positioned_tanks(
                translate=(self.rocket.motor_position, 0), csys=total_csys
            )
            for tank, center in tanks_and_centers:
                ax.scatter(
                    center[0],
                    center[1],
                    color="black",
                    alpha=0.2,
                    s=4,
                    zorder=10,
                )
                motor_patches += [tank]

        return motor_patches

    def _draw_nozzle_tube(self, last_radius, last_x, nozzle_position, ax, vis_args):
        """Draws the tube from the last surface to the nozzle position."""
        # Check if nozzle is beyond the last surface, if so draw a tube
        # to it, with the radius of the last surface
        if self.rocket._csys == 1:
            if nozzle_position < last_x:
                x_tube = [last_x, nozzle_position]
                y_tube = [last_radius, last_radius]
                y_tube_negated = [-last_radius, -last_radius]

                ax.plot(
                    x_tube,
                    y_tube,
                    color=vis_args["body"],
                    linewidth=vis_args["line_width"],
                )
                ax.plot(
                    x_tube,
                    y_tube_negated,
                    color=vis_args["body"],
                    linewidth=vis_args["line_width"],
                )
        else:  # if self.rocket._csys == -1:
            if nozzle_position > last_x:
                x_tube = [last_x, nozzle_position]
                y_tube = [last_radius, last_radius]
                y_tube_negated = [-last_radius, -last_radius]

                ax.plot(
                    x_tube,
                    y_tube,
                    color=vis_args["body"],
                    linewidth=vis_args["line_width"],
                )
                ax.plot(
                    x_tube,
                    y_tube_negated,
                    color=vis_args["body"],
                    linewidth=vis_args["line_width"],
                )

    def _draw_rail_buttons(self, ax, vis_args):
        """Draws the rail buttons of the rocket."""
        try:
            buttons, pos = self.rocket.rail_buttons[0]
            lower = pos
            upper = pos + buttons.buttons_distance * self.rocket._csys
            ax.scatter(
                lower, -self.rocket.radius, marker="s", color=vis_args["buttons"], s=15
            )
            ax.scatter(
                upper, -self.rocket.radius, marker="s", color=vis_args["buttons"], s=15
            )
        except IndexError:
            pass

    def _draw_center_of_mass_and_pressure(self, ax):
        """Draws the center of mass and center of pressure of the rocket."""
        # Draw center of mass and center of pressure
        cm = self.rocket.center_of_mass(0)
        ax.scatter(cm, 0, color="#1565c0", label="Center of Mass", s=10)

        cp = self.rocket.cp_position(0)
        ax.scatter(
            cp, 0, label="Static Center of Pressure", color="red", s=10, zorder=10
        )

    def all(self):
        """Prints out all graphs available about the Rocket. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """

        # Rocket draw
        print("\nRocket Draw")
        print("-" * 40)
        self.draw()

        # Mass Plots
        print("\nMass Plots")
        print("-" * 40)
        self.total_mass()
        self.reduced_mass()

        # Aerodynamics Plots
        print("\nAerodynamics Plots")
        print("-" * 40)

        # Drag Plots
        print("Drag Plots")
        print("-" * 20)  # Separator for Drag Plots
        self.drag_curves()

        # Stability Plots
        print("\nStability Plots")
        print("-" * 20)  # Separator for Stability Plots
        self.static_margin()
        self.stability_margin()

        # Thrust-to-Weight Plot
        print("\nThrust-to-Weight Plot")
        print("-" * 40)
        self.thrust_to_weight()

        return None
