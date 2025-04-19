import numpy as np


class _TrapezoidalMixin:
    """Mixin class for trapezoidal fins.
    This class holds methods and properties specific to trapezoidal fin shapes.
    It is designed to be used in conjunction with other classes that define the
    overall behavior of the fins.
    """

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
        self._sweep_length = np.tan(value * np.pi / 180) * self.span
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

    def _initialize(self, sweep_length, sweep_angle, root_chord, tip_chord, span):
        # Check if sweep angle or sweep length is given
        if sweep_length is not None and sweep_angle is not None:
            raise ValueError("Cannot use sweep_length and sweep_angle together")
        elif sweep_angle is not None:
            sweep_length = np.tan(sweep_angle * np.pi / 180) * span
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

    def evaluate_geometrical_parameters(self):
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Returns
        -------
        None
        """
        # pylint: disable=invalid-name
        Yr = self.root_chord + self.tip_chord
        Af = Yr * self.span / 2  # Fin area
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        gamma_c = np.arctan(
            (self.sweep_length + 0.5 * self.tip_chord - 0.5 * self.root_chord)
            / (self.span)
        )
        Yma = (
            (self.span / 3) * (self.root_chord + 2 * self.tip_chord) / Yr
        )  # Span wise coord of mean aero chord

        # Fin–body interference correction parameters
        tau = (self.span + self.rocket_radius) / self.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        lambda_ = self.tip_chord / self.root_chord

        # Parameters for Roll Moment.
        # Documented at: https://docs.rocketpy.org/en/latest/technical/
        roll_geometrical_constant = (
            (self.root_chord + 3 * self.tip_chord) * self.span**3
            + 4
            * (self.root_chord + 2 * self.tip_chord)
            * self.rocket_radius
            * self.span**2
            + 6 * (self.root_chord + self.tip_chord) * self.span * self.rocket_radius**2
        ) / 12
        roll_damping_interference_factor = 1 + (
            ((tau - lambda_) / (tau)) - ((1 - lambda_) / (tau - 1)) * np.log(tau)
        ) / (
            ((tau + 1) * (tau - lambda_)) / (2)
            - ((1 - lambda_) * (tau**3 - 1)) / (3 * (tau - 1))
        )
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
        self.Yr = Yr
        self.Af = Af  # Fin area
        self.AR = AR  # Aspect Ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.roll_geometrical_constant = roll_geometrical_constant
        self.tau = tau
        self.lift_interference_factor = lift_interference_factor
        self.λ = lambda_  # pylint: disable=non-ascii-name
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

    def to_dict(self, include_outputs=False):
        data = super().to_dict(include_outputs)
        data["tip_chord"] = self.tip_chord

        if include_outputs:
            data.update(
                {
                    "sweep_length": self.sweep_length,
                    "sweep_angle": self.sweep_angle,
                    "shape_vec": self.shape_vec,
                    "Af": self.Af,
                    "AR": self.AR,
                    "gamma_c": self.gamma_c,
                    "Yma": self.Yma,
                    "roll_geometrical_constant": self.roll_geometrical_constant,
                    "tau": self.tau,
                    "lift_interference_factor": self.lift_interference_factor,
                    "roll_damping_interference_factor": self.roll_damping_interference_factor,
                    "roll_forcing_interference_factor": self.roll_forcing_interference_factor,
                }
            )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            n=data["n"],
            root_chord=data["root_chord"],
            tip_chord=data["tip_chord"],
            span=data["span"],
            rocket_radius=data["rocket_radius"],
            cant_angle=data["cant_angle"],
            airfoil=data["airfoil"],
            name=data["name"],
        )
