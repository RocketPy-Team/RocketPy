import numpy as np


class _EllipticalMixin:
    """Mixin class for elliptical fins. This class holds methods and properties
    specific to elliptical fin shapes. It is designed to be used in conjunction
    with other classes that define the overall behavior of the fins."""

    def evaluate_geometrical_parameters(self):  # pylint: disable=too-many-statements
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement."""

        # Compute auxiliary geometrical parameters
        # pylint: disable=invalid-name
        Af = (np.pi * self.root_chord / 2 * self.span) / 2  # Fin area
        gamma_c = 0  # Zero for elliptical fins
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        Yma = (
            self.span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
        )  # Span wise coord of mean aero chord
        roll_geometrical_constant = (
            self.root_chord
            * self.span
            * (
                3 * np.pi * self.span**2
                + 32 * self.rocket_radius * self.span
                + 12 * np.pi * self.rocket_radius**2
            )
            / 48
        )

        # Fin–body interference correction parameters
        tau = (self.span + self.rocket_radius) / self.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        if self.span > self.rocket_radius:
            roll_damping_interference_factor = 1 + (
                (self.rocket_radius**2)
                * (
                    2
                    * (self.rocket_radius**2)
                    * np.sqrt(self.span**2 - self.rocket_radius**2)
                    * np.log(
                        (
                            2
                            * self.span
                            * np.sqrt(self.span**2 - self.rocket_radius**2)
                            + 2 * self.span**2
                        )
                        / self.rocket_radius
                    )
                    - 2
                    * (self.rocket_radius**2)
                    * np.sqrt(self.span**2 - self.rocket_radius**2)
                    * np.log(2 * self.span)
                    + 2 * self.span**3
                    - np.pi * self.rocket_radius * self.span**2
                    - 2 * (self.rocket_radius**2) * self.span
                    + np.pi * self.rocket_radius**3
                )
            ) / (
                2
                * (self.span**2)
                * (self.span / 3 + np.pi * self.rocket_radius / 4)
                * (self.span**2 - self.rocket_radius**2)
            )
        elif self.span < self.rocket_radius:
            roll_damping_interference_factor = 1 - (
                self.rocket_radius**2
                * (
                    2 * self.span**3
                    - np.pi * self.span**2 * self.rocket_radius
                    - 2 * self.span * self.rocket_radius**2
                    + np.pi * self.rocket_radius**3
                    + 2
                    * self.rocket_radius**2
                    * np.sqrt(-(self.span**2) + self.rocket_radius**2)
                    * np.arctan(
                        (self.span) / (np.sqrt(-(self.span**2) + self.rocket_radius**2))
                    )
                    - np.pi
                    * self.rocket_radius**2
                    * np.sqrt(-(self.span**2) + self.rocket_radius**2)
                )
            ) / (
                2
                * self.span
                * (-(self.span**2) + self.rocket_radius**2)
                * (self.span**2 / 3 + np.pi * self.span * self.rocket_radius / 4)
            )
        else:
            roll_damping_interference_factor = (28 - 3 * np.pi) / (4 + 3 * np.pi)

        roll_forcing_interference_factor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + ((np.pi * (tau**2 + 1) ** 2) / (tau**2 * (tau - 1) ** 2))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + ((tau**2 + 1) ** 2)
            / (tau**2 * (tau - 1) ** 2)
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )

        # Store values
        # pylint: disable=invalid-name
        self.Af = Af  # Fin area
        self.AR = AR  # Fin aspect ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.roll_geometrical_constant = roll_geometrical_constant
        self.tau = tau
        self.lift_interference_factor = lift_interference_factor
        self.roll_damping_interference_factor = roll_damping_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor

        self.evaluate_shape()

    def evaluate_shape(self):
        angles = np.arange(0, 180, 5)
        x_array = self.root_chord / 2 + self.root_chord / 2 * np.cos(np.radians(angles))
        y_array = self.span * np.sin(np.radians(angles))
        self.shape_vec = [x_array, y_array]

    def info(self):
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        self.prints.all()
        self.plots.all()
