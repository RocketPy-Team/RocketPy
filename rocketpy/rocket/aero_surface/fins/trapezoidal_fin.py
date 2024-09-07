import math

import numpy as np

from .fin import Fin


class TrapezoidalFin(Fin):
    def __init__(
        self,
        angular_position,
        root_chord,
        tip_chord,
        span,
        rocket_radius,
        cant_angle=0,
        sweep_length=None,
        sweep_angle=None,
        airfoil=None,
        name="Fins",
    ):
        super().__init__(
            angular_position,
            root_chord,
            span,
            rocket_radius,
            cant_angle,
            airfoil,
            name,
        )

        # Check if sweep angle or sweep length is given
        if sweep_length is not None and sweep_angle is not None:
            raise ValueError("Cannot use sweep_length and sweep_angle together")
        elif sweep_angle is not None:
            sweep_length = math.tan(sweep_angle * math.pi / 180) * span
        elif sweep_length is None:
            sweep_length = root_chord - tip_chord
        else:  # Sweep length is given
            pass

        self._tip_chord = tip_chord
        self._sweep_length = sweep_length
        self._sweep_angle = sweep_angle

        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()
        self.evaluate_rotation_matrix()

    @property
    def tip_chord(self):
        return self._tip_chord

    @tip_chord.setter
    def tip_chord(self, value):
        self._tip_chord = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def sweep_angle(self):
        return self._sweep_angle

    @sweep_angle.setter
    def sweep_angle(self, value):
        self._sweep_angle = value
        self._sweep_length = math.tan(value * math.pi / 180) * self.span
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def sweep_length(self):
        return self._sweep_length

    @sweep_length.setter
    def sweep_length(self, value):
        self._sweep_length = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    def evaluate_center_of_pressure(self):
        """Calculates and returns the center of pressure of the fin set in local
        coordinates. The center of pressure position is saved and stored as a
        tuple.

        Returns
        -------
        None
        """
        # Center of pressure position in local coordinates
        cpz = (self.sweep_length / 3) * (
            (self.root_chord + 2 * self.tip_chord) / (self.root_chord + self.tip_chord)
        ) + (1 / 6) * (
            self.root_chord
            + self.tip_chord
            - self.root_chord * self.tip_chord / (self.root_chord + self.tip_chord)
        )
        self.cpx = 0
        self.cpy = self.Yma
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)

    def evaluate_geometrical_parameters(self):
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Returns
        -------
        None
        """

        Yr = self.root_chord + self.tip_chord
        Af = Yr * self.span / 2  # Fin area
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        gamma_c = math.atan(
            (self.sweep_length + 0.5 * self.tip_chord - 0.5 * self.root_chord)
            / (self.span)
        )
        Yma = (
            (self.span / 3) * (self.root_chord + 2 * self.tip_chord) / Yr
        )  # Span wise coord of mean aero chord

        # Fin–body interference correction parameters
        tau = (self.span + self.rocket_radius) / self.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        λ = self.tip_chord / self.root_chord

        # Parameters for Roll Moment.
        # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/master/docs/technical/aerodynamics/Roll_Equations.pdf
        roll_geometrical_constant = (
            (self.root_chord + 3 * self.tip_chord) * self.span**3
            + 4
            * (self.root_chord + 2 * self.tip_chord)
            * self.rocket_radius
            * self.span**2
            + 6 * (self.root_chord + self.tip_chord) * self.span * self.rocket_radius**2
        ) / 12
        roll_damping_interference_factor = 1 + (
            ((tau - λ) / (tau)) - ((1 - λ) / (tau - 1)) * math.log(tau)
        ) / (
            ((tau + 1) * (tau - λ)) / (2) - ((1 - λ) * (tau**3 - 1)) / (3 * (tau - 1))
        )
        roll_forcing_interference_factor = (1 / math.pi**2) * (
            (math.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + ((math.pi * (tau**2 + 1) ** 2) / (tau**2 * (tau - 1) ** 2))
            * math.asin((tau**2 - 1) / (tau**2 + 1))
            - (2 * math.pi * (tau + 1)) / (tau * (tau - 1))
            + ((tau**2 + 1) ** 2)
            / (tau**2 * (tau - 1) ** 2)
            * (math.asin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * math.asin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * math.log((tau**2 + 1) / (2 * tau))
        )

        # Store values
        self.Yr = Yr
        self.Af = Af  # Fin area
        self.AR = AR  # Aspect Ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.roll_geometrical_constant = roll_geometrical_constant
        self.tau = tau
        self.lift_interference_factor = lift_interference_factor
        self.λ = λ
        self.roll_damping_interference_factor = roll_damping_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor

        self.evaluate_shape()

    def evaluate_shape(self):
        if self.sweep_length:
            points = [
                (0, 0),
                (self.sweep_length, self.span),
                (self.sweep_length + self.tip_chord, self.span),
                (self.root_chord, 0),
            ]
        else:
            points = [
                (0, 0),
                (self.root_chord - self.tip_chord, self.span),
                (self.root_chord, self.span),
                (self.root_chord, 0),
            ]

        x_array, y_array = zip(*points)
        self.shape_vec = [np.array(x_array), np.array(y_array)]

    def info(self):
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        self.prints.all()
        self.plots.all()
