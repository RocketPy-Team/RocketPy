import matplotlib.pyplot as plt
import numpy as np

from rocketpy.motors import EmptyMotor, HybridMotor, LiquidMotor, SolidMotor
from rocketpy.rocket.aero_surface import Fins, NoseCone, Tail
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface

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

    def total_mass(self):
        """Plots total mass of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.total_mass()

    def reduced_mass(self):
        """Plots reduced mass of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.reduced_mass()

    def static_margin(self, *, filename=None):
        """Plots static margin of the rocket as a function of time.

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

        self.rocket.static_margin(filename=filename)

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

    # pylint: disable=too-many-statements
    def drag_curves(self, *, filename=None):
        """Plots power off and on drag curves of the rocket as a function of time.

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

        _, ax = plt.subplots()
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
        """
        Plots the motor thrust force divided by rocket weight as a function of time.
        """

        self.rocket.thrust_to_weight.plot(
            lower=0, upper=self.rocket.motor.burn_out_time
        )

    def draw(self, vis_args=None, plane="xz", *, filename=None):
        """Draws the rocket in a matplotlib figure.

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
                    "line_width": 2.0,
                }

            A full list of color names can be found at: \
            https://matplotlib.org/stable/gallery/color/named_colors
        plane : str, optional
            Plane in which the rocket will be drawn. Default is 'xz'. Other
            options is 'yz'. Used only for sensors representation.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        """

        self.__validate_aerodynamic_surfaces()

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

        _, ax = plt.subplots(figsize=(8, 6), facecolor=vis_args["background"])
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", linewidth=0.5)

        csys = self.rocket._csys
        reverse = csys == 1
        self.rocket.aerodynamic_surfaces.sort_by_position(reverse=reverse)

        drawn_surfaces = self._draw_aerodynamic_surfaces(ax, vis_args, plane)
        last_radius, last_x = self._draw_tubes(ax, drawn_surfaces, vis_args)
        self._draw_motor(last_radius, last_x, ax, vis_args)
        self._draw_rail_buttons(ax, vis_args)
        self._draw_center_of_mass_and_pressure(ax)
        self._draw_sensors(ax, self.rocket.sensors, plane)

        plt.title("Rocket Representation")
        plt.xlim()
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        show_or_save_plot(filename)

    def __validate_aerodynamic_surfaces(self):
        if not self.rocket.aerodynamic_surfaces:
            raise ValueError(
                "The rocket must have at least one aerodynamic surface to be drawn."
            )

    def _draw_aerodynamic_surfaces(self, ax, vis_args, plane):
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
                self._draw_nose_cone(ax, surface, position.z, drawn_surfaces, vis_args)
            elif isinstance(surface, Tail):
                self._draw_tail(ax, surface, position.z, drawn_surfaces, vis_args)
            elif isinstance(surface, Fins):
                self._draw_fins(ax, surface, position.z, drawn_surfaces, vis_args)
            elif isinstance(surface, GenericSurface):
                self._draw_generic_surface(
                    ax, surface, position, drawn_surfaces, vis_args, plane
                )
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

    def _draw_generic_surface(
        self,
        ax,
        surface,
        position,
        drawn_surfaces,
        vis_args,  # pylint: disable=unused-argument
        plane,
    ):
        """Draws the generic surface and saves the position of the points of interest
        for the tubes."""
        match plane:
            case "xz":
                # z position of the sensor is the x position in the plot
                x_pos = position[2]
                # x position of the surface is the y position in the plot
                y_pos = position[0]
            case "yz":
                # z position of the surface is the x position in the plot
                x_pos = position[2]
                # y position of the surface is the y position in the plot
                y_pos = position[1]
            case _:  # pragma: no cover
                raise ValueError("Plane must be 'xz' or 'yz'.")

        ax.scatter(
            x_pos,
            y_pos,
            linewidth=2,
            zorder=10,
            label=surface.name,
        )
        drawn_surfaces.append((surface, position.z, self.rocket.radius, x_pos))

    def _draw_tubes(self, ax, drawn_surfaces, vis_args):
        """Draws the tubes between the aerodynamic surfaces."""
        radius = 0
        last_x = 0
        for i, d_surface in enumerate(drawn_surfaces):
            # Draw the tubes, from the end of the first surface to the beginning
            # of the next surface, with the radius of the rocket at that point
            surface, position, radius, last_x = d_surface

            if i == len(drawn_surfaces) - 1:
                # If the last surface is a tail, do nothing
                if isinstance(surface, Tail):
                    continue
                # Else goes to the end of the surface
                x_tube = [position, last_x]
                y_tube = [radius, radius]
                y_tube_negated = [-radius, -radius]
            else:
                # If it is not the last surface, the tube goes to the beginning
                # of the next surface
                # [next_surface, next_position, next_radius, next_last_x]
                next_position = drawn_surfaces[i + 1][1]
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
        motor_patches = self._generate_motor_patches(total_csys, ax)

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

    def _generate_motor_patches(self, total_csys, ax):  # pylint: disable=unused-argument
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
            lower = pos.z
            upper = lower + buttons.buttons_distance * self.rocket._csys
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

    def _draw_sensors(self, ax, sensors, plane):
        """Draw the sensor as a small thick line at the position of the sensor,
        with a vector pointing in the direction normal of the sensor. Get the
        normal vector from the sensor orientation matrix."""
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, sensor_pos in enumerate(sensors):
            sensor = sensor_pos[0]
            pos = sensor_pos[1]
            match plane:
                case "xz":
                    # z position of the sensor is the x position in the plot
                    x_pos = pos[2]
                    normal_x = sensor.normal_vector.z
                    # x position of the sensor is the y position in the plot
                    y_pos = pos[0]
                    normal_y = sensor.normal_vector.x
                case "yz":
                    # z position of the sensor is the x position in the plot
                    x_pos = pos[2]
                    normal_x = sensor.normal_vector.z
                    # y position of the sensor is the y position in the plot
                    y_pos = pos[1]
                    normal_y = sensor.normal_vector.y
                case _:  # pragma: no cover
                    raise ValueError("Plane must be 'xz' or 'yz'.")

            # line length is 2/5 of the rocket radius
            line_length = self.rocket.radius / 2.5

            ax.plot(
                [x_pos, x_pos],
                [y_pos + line_length, y_pos - line_length],
                linewidth=2,
                color=colors[(i + 1) % len(colors)],
                zorder=10,
                label=sensor.name,
            )
            if abs(sensor.normal_vector) != 0:
                ax.quiver(
                    x_pos,
                    y_pos,
                    normal_x,
                    normal_y,
                    color=colors[(i + 1) % len(colors)],
                    scale_units="xy",
                    angles="xy",
                    minshaft=2,
                    headwidth=2,
                    headlength=4,
                    zorder=10,
                )

    def all(self):
        """Prints out all graphs available about the Rocket. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """

        # Rocket draw
        if len(self.rocket.aerodynamic_surfaces) > 0:
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
