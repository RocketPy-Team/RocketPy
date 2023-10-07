import numpy as np
import matplotlib.pyplot as plt

from rocketpy.rocket.aero_surface import Fins, NoseCone, Tail


class _RocketPlots:
    """Class that holds plot methods for Rocket class.

    Attributes
    ----------
    _RocketPlots.rocket : Rocket
        Rocket object that will be used for the plots.

    """

    def __init__(self, rocket) -> None:
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

    def static_margin(self):
        """Plots static margin of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.static_margin()

        return None

    def power_on_drag(self):
        """Plots power on drag of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.power_on_drag()

        return None

    def power_off_drag(self):
        """Plots power off drag of the rocket as a function of time.

        Returns
        -------
        None
        """

        self.rocket.power_off_drag()

        return None

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

    def draw(self, vis_args=None):
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
        """
        # TODO: we need to modularize this function, it is too big
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
        _, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect("equal")
        ax.set_facecolor(vis_args["background"])
        ax.grid(True, linestyle="--", linewidth=0.5)

        csys = self.rocket._csys
        reverse = csys == 1
        self.rocket.aerodynamic_surfaces.sort_by_position(reverse=reverse)

        # List of drawn surfaces with the position of points of interest
        # and the radius of the rocket at that point
        drawn_surfaces = []

        # Ideia is to get the shape of each aerodynamic surface in their own
        # coordinate system and then plot them in the rocket coordinate system
        # using the position of each surface
        # For the tubes, the surfaces need to be checked in order to check for
        # diameter changes. The final point of the last surface is the final
        # point of the last tube

        for surface, position in self.rocket.aerodynamic_surfaces:
            if isinstance(surface, NoseCone):
                x_nosecone = -csys * surface.shape_vec[0] + position
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

            elif isinstance(surface, Tail):
                x_tail = -csys * surface.shape_vec[0] + position
                y_tail = surface.shape_vec[1]

                ax.plot(
                    x_tail,
                    y_tail,
                    color=vis_args["tail"],
                    linewidth=vis_args["line_width"],
                )
                ax.plot(
                    x_tail,
                    -y_tail,
                    color=vis_args["tail"],
                    linewidth=vis_args["line_width"],
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
                drawn_surfaces.append(
                    (surface, position, surface.bottom_radius, x_tail[-1])
                )

            # Draw fins
            elif isinstance(surface, Fins):
                num_fins = surface.n
                x_fin = -csys * surface.shape_vec[0] + position
                y_fin = surface.shape_vec[1] + surface.rocket_radius

                # Calculate the rotation angles for the other two fins (symmetrically)
                rotation_angles = [2 * np.pi * i / num_fins for i in range(num_fins)]

                # Apply rotation transformations to get points for the other fins in 2D space
                for angle in rotation_angles:
                    # Create a rotation matrix for the current angle around the x-axis
                    rotation_matrix = np.array([[1, 0], [0, np.cos(angle)]])

                    # Apply the rotation to the original fin points
                    rotated_points_2d = np.dot(
                        rotation_matrix, np.vstack((x_fin, y_fin))
                    )

                    # Extract x and y coordinates of the rotated points
                    x_rotated, y_rotated = rotated_points_2d

                    # Project points above the XY plane back into the XY plane (set z-coordinate to 0)
                    x_rotated = np.where(
                        rotated_points_2d[1] > 0, rotated_points_2d[0], x_rotated
                    )
                    y_rotated = np.where(
                        rotated_points_2d[1] > 0, rotated_points_2d[1], y_rotated
                    )

                    # Plot the fins
                    ax.plot(
                        x_rotated,
                        y_rotated,
                        color=vis_args["fins"],
                        linewidth=vis_args["line_width"],
                    )

                # Add the fin to the list of drawn surfaces
                drawn_surfaces.append(
                    (surface, position, surface.rocket_radius, x_rotated[-1])
                )

        # Draw tubes
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

        # TODO - Draw motor
        nozzle_position = (
            self.rocket.motor_position
            + self.rocket.motor.nozzle_position
            * self.rocket._csys
            * self.rocket.motor._csys
        )
        ax.scatter(
            nozzle_position, 0, label="Nozzle Outlet", color="brown", s=10, zorder=10
        )
        # Check if nozzle is beyond the last surface, if so draw a tube
        # to it, with the radius of the last surface
        if self.rocket._csys == 1:
            if nozzle_position < last_x:
                x_tube = [last_x, nozzle_position]
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
        else:  # if self.rocket._csys == -1:
            if nozzle_position > last_x:
                x_tube = [last_x, nozzle_position]
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

        # Draw rail buttons
        try:
            buttons, pos = self.rocket.rail_buttons[0]
            lower = pos
            upper = pos + buttons.buttons_distance * csys
            ax.scatter(
                lower, -self.rocket.radius, marker="s", color=vis_args["buttons"], s=15
            )
            ax.scatter(
                upper, -self.rocket.radius, marker="s", color=vis_args["buttons"], s=15
            )
        except IndexError:
            pass

        # Draw center of mass and center of pressure
        cm = self.rocket.center_of_mass(0)
        ax.scatter(cm, 0, color="black", label="Center of Mass", s=30)
        ax.scatter(cm, 0, facecolors="none", edgecolors="black", s=100)

        cp = self.rocket.cp_position
        ax.scatter(cp, 0, label="Center Of Pressure", color="red", s=30, zorder=10)
        ax.scatter(cp, 0, facecolors="none", edgecolors="red", s=100, zorder=10)

        # Set plot attributes
        plt.title(f"Rocket Geometry")
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

        return None

    def all(self):
        """Prints out all graphs available about the Rocket. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """

        # Show plots
        print("\nMass Plots")
        self.total_mass()
        self.reduced_mass()
        print("\nAerodynamics Plots")
        self.static_margin()
        self.power_on_drag()
        self.power_off_drag()
        self.thrust_to_weight()

        return None
