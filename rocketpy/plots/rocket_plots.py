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
                "body": "dimgrey",
                "fins": "black",
                "motor": "black",
                "buttons": "black",
                "line_width": 2.0,
            }
            A full list of color names can be found at:
            https://matplotlib.org/stable/gallery/color/named_colors
        """
        if vis_args is None:
            vis_args = {
                "background": "#EEEEEE",
                "tail": "black",
                "nose": "black",
                "body": "dimgrey",
                "fins": "black",
                "motor": "black",
                "buttons": "black",
                "line_width": 2.0,
            }

        # Create the figure and axis
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect("equal")
        ax.set_facecolor(vis_args["background"])
        ax.grid(True, linestyle="--", linewidth=0.5)

        csys = self.rocket._csys

        # Draw rocket body
        reverse = csys == 1
        self.rocket.aerodynamic_surfaces.sort_by_position(reverse=reverse)
        y_tube, x_tube = [], []
        for surface, position in self.rocket.aerodynamic_surfaces:
            if isinstance(surface, (NoseCone, Tail)):
                # Append the x and y coordinates of the surface shape_vec to the respective lists
                x_tube.extend((-csys) * surface.shape_vec[0] + position)
                y_tube.extend(surface.shape_vec[1])
            if isinstance(surface, Fins):
                pass

        # Negate each element in the y_tube list using a list comprehension
        y_tube_negated = [-y for y in y_tube]
        plt.plot(
            x_tube, y_tube, color=vis_args["body"], linewidth=vis_args["line_width"]
        )
        plt.plot(
            x_tube,
            y_tube_negated,
            color=vis_args["body"],
            linewidth=vis_args["line_width"],
        )

        # Get nozzle position (kinda not working for EmptyMotor class)
        x_nozzle = self.rocket.motor_position + self.rocket.motor.nozzle_position

        # Find the last point of the rocket
        idx = -1 if csys == 1 else 0
        surface, position = self.rocket.aerodynamic_surfaces[idx]
        length = surface.shape_vec[0][-1] - surface.shape_vec[0][0]
        x_last = position + (-1 * csys) * length
        y_last = surface.shape_vec[1][-1]

        plt.plot(
            [x_nozzle, x_last],
            [0, y_last],
            color=vis_args["body"],
            linewidth=vis_args["line_width"],
        )
        plt.plot(
            [x_nozzle, x_last],
            [0, -y_last],
            color=vis_args["body"],
            linewidth=vis_args["line_width"],
        )

        # Draw nosecone
        nosecones = self.rocket.aerodynamic_surfaces.get_tuple_by_type(NoseCone)
        for nose, position in nosecones:
            x_nosecone = -csys * nose.shape_vec[0] + position
            y_nosecone = nose.shape_vec[1]

            plt.plot(
                x_nosecone,
                y_nosecone,
                color=vis_args["nose"],
                linewidth=vis_args["line_width"] - 0.05,
            )
            plt.plot(
                x_nosecone,
                -y_nosecone,
                color=vis_args["nose"],
                linewidth=vis_args["line_width"] - 0.05,
            )

        # Draw transitions
        tails = self.rocket.aerodynamic_surfaces.get_tuple_by_type(Tail)
        for tail, position in tails:
            x_tail = -csys * tail.shape_vec[0] + position
            y_tail = tail.shape_vec[1]

            plt.plot(
                x_tail,
                y_tail,
                color=vis_args["tail"],
                linewidth=vis_args["line_width"],
            )
            plt.plot(
                x_tail,
                -y_tail,
                color=vis_args["tail"],
                linewidth=vis_args["line_width"],
            )

        # Draw fins
        fins = self.rocket.aerodynamic_surfaces.get_tuple_by_type(Fins)
        for fin, position in fins:
            x_fin = -csys * fin.shape_vec[0] + position
            y_fin = fin.shape_vec[1] + self.rocket.radius

            plt.plot(
                x_fin,
                y_fin,
                color=vis_args["fins"],
                linewidth=vis_args["line_width"],
            )
            plt.plot(
                x_fin,
                -y_fin,
                color=vis_args["fins"],
                linewidth=vis_args["line_width"],
            )

        # Draw rail buttons
        buttons, pos = self.rocket.rail_buttons[0]
        lower = pos
        upper = pos + buttons.buttons_distance * csys
        plt.scatter(
            lower, -self.rocket.radius, marker="s", color=vis_args["buttons"], s=10
        )
        plt.scatter(
            upper, -self.rocket.radius, marker="s", color=vis_args["buttons"], s=10
        )

        # Draw center of mass and center of pressure
        cm = self.rocket.center_of_mass(0)
        plt.scatter(cm, 0, color="black", label="Center of Mass", s=30)
        plt.scatter(cm, 0, facecolors="none", edgecolors="black", s=100)

        cp = self.rocket.cp_position
        plt.scatter(cp, 0, label="Center Of Pressure", color="red", s=30, zorder=10)
        plt.scatter(cp, 0, facecolors="none", edgecolors="red", s=100, zorder=10)

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
