"""Geometry strategy classes for fin aerodynamic surfaces."""

import warnings
from abc import ABC, abstractmethod

import numpy as np


class _FinGeometry(ABC):
    """Base geometry strategy for fin shapes."""

    def __init__(self, owner):
        self.owner = owner

    @abstractmethod
    def evaluate_geometrical_parameters(self):
        """Evaluate and store geometry-dependent aerodynamic parameters."""

    @abstractmethod
    def evaluate_shape(self):
        """Evaluate the shape vector used by plotting and outputs."""

    def get_data(self, include_outputs=False):
        """Return geometry-specific serialization data."""
        _ = include_outputs
        return {}


class _TrapezoidalGeometry(_FinGeometry):
    """Geometry strategy for trapezoidal fins."""

    def __init__(
        self,
        owner,
        tip_chord,
        sweep_length=None,
        sweep_angle=None,
    ):
        super().__init__(owner)
        if sweep_length is not None and sweep_angle is not None:
            raise ValueError("Cannot use sweep_length and sweep_angle together")

        if sweep_angle is not None:
            sweep_length = np.tan(np.radians(sweep_angle)) * owner.span
        elif sweep_length is None:
            sweep_length = owner.root_chord - tip_chord

        self._tip_chord = tip_chord
        self._sweep_length = sweep_length
        self._sweep_angle = sweep_angle
        self.owner._tip_chord = self._tip_chord
        self.owner._sweep_length = self._sweep_length
        self.owner._sweep_angle = self._sweep_angle

    @property
    def tip_chord(self):
        return self._tip_chord

    @tip_chord.setter
    def tip_chord(self, value):
        self._tip_chord = value
        self.owner._tip_chord = value

    @property
    def sweep_length(self):
        return self._sweep_length

    @sweep_length.setter
    def sweep_length(self, value):
        self._sweep_length = value
        self.owner._sweep_length = value

    @property
    def sweep_angle(self):
        return self._sweep_angle

    @sweep_angle.setter
    def sweep_angle(self, value):
        self._sweep_angle = value
        self._sweep_length = np.tan(np.radians(value)) * self.owner.span
        self.owner._sweep_angle = self._sweep_angle
        self.owner._sweep_length = self._sweep_length

    def evaluate_geometrical_parameters(self):
        """Calculates and saves trapezoidal fin geometric parameters."""
        # pylint: disable=invalid-name
        owner = self.owner
        Yr = owner.root_chord + self.tip_chord
        Af = Yr * owner.span / 2
        AR = 2 * owner.span**2 / Af
        gamma_c = np.arctan(
            (
                self.sweep_length
                + 0.5 * self.tip_chord
                - 0.5 * owner.root_chord
            )
            / owner.span
        )
        Yma = (owner.span / 3) * (owner.root_chord + 2 * self.tip_chord) / Yr

        tau = (owner.span + owner.rocket_radius) / owner.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        lambda_ = self.tip_chord / owner.root_chord

        roll_geometrical_constant = (
            (owner.root_chord + 3 * self.tip_chord) * owner.span**3
            + 4
            * (owner.root_chord + 2 * self.tip_chord)
            * owner.rocket_radius
            * owner.span**2
            + 6
            * (owner.root_chord + self.tip_chord)
            * owner.span
            * owner.rocket_radius**2
        ) / 12
        roll_damping_interference_factor = 1 + (
            ((tau - lambda_) / tau)
            - ((1 - lambda_) / (tau - 1)) * np.log(tau)
        ) / (
            ((tau + 1) * (tau - lambda_)) / 2
            - ((1 - lambda_) * (tau**3 - 1)) / (3 * (tau - 1))
        )
        roll_forcing_interference_factor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + (
                np.pi
                * (tau**2 + 1) ** 2
                / (tau**2 * (tau - 1) ** 2)
            )
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + (
                (tau**2 + 1) ** 2
                / (tau**2 * (tau - 1) ** 2)
            )
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )

        owner.Yr = Yr
        owner.Af = Af
        owner.AR = AR
        owner.gamma_c = gamma_c
        owner.Yma = Yma
        owner.roll_geometrical_constant = roll_geometrical_constant
        owner.tau = tau
        owner.lift_interference_factor = lift_interference_factor
        owner.λ = lambda_  # pylint: disable=non-ascii-name
        owner.roll_damping_interference_factor = (
            roll_damping_interference_factor
        )
        owner.roll_forcing_interference_factor = (
            roll_forcing_interference_factor
        )

        self.evaluate_shape()

    def evaluate_shape(self):
        owner = self.owner
        if self.sweep_length:
            points = [
                (0, 0),
                (self.sweep_length, owner.span),
                (self.sweep_length + self.tip_chord, owner.span),
                (owner.root_chord, 0),
            ]
        else:
            points = [
                (0, 0),
                (owner.root_chord - self.tip_chord, owner.span),
                (owner.root_chord, owner.span),
                (owner.root_chord, 0),
            ]

        x_array, y_array = zip(*points)
        owner.shape_vec = [np.array(x_array), np.array(y_array)]

    def get_data(self, include_outputs=False):
        data = {
            "tip_chord": self.tip_chord,
            "sweep_length": self.sweep_length,
            "sweep_angle": self.sweep_angle,
        }
        if include_outputs:
            data.update(
                {
                    "shape_vec": self.owner.shape_vec,
                    "Af": self.owner.Af,
                    "AR": self.owner.AR,
                    "gamma_c": self.owner.gamma_c,
                    "Yma": self.owner.Yma,
                    "roll_geometrical_constant": (
                        self.owner.roll_geometrical_constant
                    ),
                    "tau": self.owner.tau,
                    "lift_interference_factor": (
                        self.owner.lift_interference_factor
                    ),
                    "roll_damping_interference_factor": (
                        self.owner.roll_damping_interference_factor
                    ),
                    "roll_forcing_interference_factor": (
                        self.owner.roll_forcing_interference_factor
                    ),
                }
            )
        return data


class _EllipticalGeometry(_FinGeometry):
    """Geometry strategy for elliptical fins."""

    def evaluate_geometrical_parameters(self):  # pylint: disable=too-many-statements
        """Calculates and saves elliptical fin geometric parameters."""
        owner = self.owner

        # pylint: disable=invalid-name
        Af = (np.pi * owner.root_chord / 2 * owner.span) / 2
        gamma_c = 0
        AR = 2 * owner.span**2 / Af
        Yma = owner.span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
        roll_geometrical_constant = (
            owner.root_chord
            * owner.span
            * (
                3 * np.pi * owner.span**2
                + 32 * owner.rocket_radius * owner.span
                + 12 * np.pi * owner.rocket_radius**2
            )
            / 48
        )

        tau = (owner.span + owner.rocket_radius) / owner.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        if owner.span > owner.rocket_radius:
            roll_damping_interference_factor = 1 + (
                owner.rocket_radius**2
                * (
                    2
                    * owner.rocket_radius**2
                    * np.sqrt(owner.span**2 - owner.rocket_radius**2)
                    * np.log(
                        (
                            2
                            * owner.span
                            * np.sqrt(owner.span**2 - owner.rocket_radius**2)
                            + 2 * owner.span**2
                        )
                        / owner.rocket_radius
                    )
                    - 2
                    * owner.rocket_radius**2
                    * np.sqrt(owner.span**2 - owner.rocket_radius**2)
                    * np.log(2 * owner.span)
                    + 2 * owner.span**3
                    - np.pi * owner.rocket_radius * owner.span**2
                    - 2 * owner.rocket_radius**2 * owner.span
                    + np.pi * owner.rocket_radius**3
                )
            ) / (
                2
                * owner.span**2
                * (owner.span / 3 + np.pi * owner.rocket_radius / 4)
                * (owner.span**2 - owner.rocket_radius**2)
            )
        elif owner.span < owner.rocket_radius:
            roll_damping_interference_factor = 1 - (
                owner.rocket_radius**2
                * (
                    2 * owner.span**3
                    - np.pi * owner.span**2 * owner.rocket_radius
                    - 2 * owner.span * owner.rocket_radius**2
                    + np.pi * owner.rocket_radius**3
                    + 2
                    * owner.rocket_radius**2
                    * np.sqrt(-(owner.span**2) + owner.rocket_radius**2)
                    * np.arctan(
                        owner.span
                        / np.sqrt(-(owner.span**2) + owner.rocket_radius**2)
                    )
                    - np.pi
                    * owner.rocket_radius**2
                    * np.sqrt(-(owner.span**2) + owner.rocket_radius**2)
                )
            ) / (
                2
                * owner.span
                * (-(owner.span**2) + owner.rocket_radius**2)
                * (
                    owner.span**2 / 3
                    + np.pi * owner.span * owner.rocket_radius / 4
                )
            )
        else:
            roll_damping_interference_factor = (28 - 3 * np.pi) / (4 + 3 * np.pi)

        roll_forcing_interference_factor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + (
                np.pi
                * (tau**2 + 1) ** 2
                / (tau**2 * (tau - 1) ** 2)
            )
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + (
                (tau**2 + 1) ** 2
                / (tau**2 * (tau - 1) ** 2)
            )
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )

        owner.Af = Af
        owner.AR = AR
        owner.gamma_c = gamma_c
        owner.Yma = Yma
        owner.roll_geometrical_constant = roll_geometrical_constant
        owner.tau = tau
        owner.lift_interference_factor = lift_interference_factor
        owner.roll_damping_interference_factor = (
            roll_damping_interference_factor
        )
        owner.roll_forcing_interference_factor = (
            roll_forcing_interference_factor
        )

        self.evaluate_shape()

    def evaluate_shape(self):
        owner = self.owner
        angles = np.arange(0, 180, 5)
        x_array = owner.root_chord / 2 + owner.root_chord / 2 * np.cos(
            np.radians(angles)
        )
        y_array = owner.span * np.sin(np.radians(angles))
        owner.shape_vec = [x_array, y_array]

    def get_data(self, include_outputs=False):
        if not include_outputs:
            return {}
        return {
            "Af": self.owner.Af,
            "AR": self.owner.AR,
            "gamma_c": self.owner.gamma_c,
            "Yma": self.owner.Yma,
            "roll_geometrical_constant": self.owner.roll_geometrical_constant,
            "tau": self.owner.tau,
            "lift_interference_factor": self.owner.lift_interference_factor,
            "roll_damping_interference_factor": (
                self.owner.roll_damping_interference_factor
            ),
            "roll_forcing_interference_factor": (
                self.owner.roll_forcing_interference_factor
            ),
        }


class _FreeFormGeometry(_FinGeometry):
    """Geometry strategy for free-form fins."""

    def __init__(self, owner, shape_points):
        super().__init__(owner)
        self.shape_points = shape_points

    @staticmethod
    def infer_dimensions(shape_points):
        """Infer root chord and span from free-form points."""
        down = False
        for i in range(1, len(shape_points)):
            if shape_points[i][1] > shape_points[i - 1][1] and down:
                warnings.warn(
                    "Jagged fin shape detected. This may cause small "
                    "inaccuracies center of pressure and pitch moment "
                    "calculations."
                )
                break
            if shape_points[i][1] < shape_points[i - 1][1]:
                down = True

        root_chord = abs(shape_points[0][0] - shape_points[-1][0])
        ys = [point[1] for point in shape_points]
        span = max(ys) - min(ys)
        return root_chord, span

    def evaluate_geometrical_parameters(self):  # pylint: disable=too-many-statements
        """Calculates and saves free-form fin geometric parameters."""
        owner = self.owner

        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        Af = 0
        for i in range(len(self.shape_points) - 1):
            x1, y1 = self.shape_points[i]
            x2, y2 = self.shape_points[i + 1]
            Af += (y1 + y2) * (x1 - x2)
        Af = abs(Af) / 2
        if Af < 1e-6:
            raise ValueError("Fin area is too small. Check the shape_points.")

        AR = 2 * owner.span**2 / Af
        tau = (owner.span + owner.rocket_radius) / owner.rocket_radius
        lift_interference_factor = 1 + 1 / tau

        roll_forcing_interference_factor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + (
                np.pi
                * (tau**2 + 1) ** 2
                / (tau**2 * (tau - 1) ** 2)
            )
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + (
                (tau**2 + 1) ** 2
                / (tau**2 * (tau - 1) ** 2)
            )
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )

        points_per_line = 40
        chord_lead = np.ones(points_per_line) * np.inf
        chord_trail = np.ones(points_per_line) * -np.inf
        chord_length = np.zeros(points_per_line)

        for p in range(1, len(self.shape_points)):
            x1, y1 = self.shape_points[p - 1]
            x2, y2 = self.shape_points[p]

            prev_idx = int(y1 / owner.span * (points_per_line - 1))
            curr_idx = int(y2 / owner.span * (points_per_line - 1))
            prev_idx = np.clip(prev_idx, 0, points_per_line - 1)
            curr_idx = np.clip(curr_idx, 0, points_per_line - 1)

            if prev_idx > curr_idx:
                prev_idx, curr_idx = curr_idx, prev_idx

            for i in range(prev_idx, curr_idx + 1):
                y = i * owner.span / (points_per_line - 1)
                if y1 != y2:
                    x = np.clip(
                        (y - y2) / (y1 - y2) * x1 + (y1 - y) / (y1 - y2) * x2,
                        min(x1, x2),
                        max(x1, x2),
                    )
                else:
                    x = x1

                chord_lead[i] = min(chord_lead[i], x)
                chord_trail[i] = max(chord_trail[i], x)

                if y1 < y2:
                    chord_length[i] -= x
                else:
                    chord_length[i] += x

        invalid_lead = np.isnan(chord_lead) | np.isinf(chord_lead)
        invalid_trail = np.isnan(chord_trail) | np.isinf(chord_trail)
        chord_lead[invalid_lead | invalid_trail] = 0
        chord_trail[invalid_lead | invalid_trail] = 0

        chord_length[chord_length < 0] = 0
        chord_length[np.isnan(chord_length)] = 0
        max_chord = chord_trail - chord_lead
        chord_length = np.minimum(chord_length, max_chord)

        radius = owner.rocket_radius
        total_area = 0
        mac_length = 0
        mac_lead = 0
        mac_span = 0
        cos_gamma_sum = 0
        roll_geometrical_constant = 0
        roll_damping_numerator = 0
        roll_damping_denominator = 0

        dy = owner.span / (points_per_line - 1)
        for i in range(points_per_line):
            chord = chord_trail[i] - chord_lead[i]
            y = i * dy

            mac_length += chord * chord
            mac_span += y * chord
            mac_lead += chord_lead[i] * chord
            total_area += chord
            roll_geometrical_constant += chord_length[i] * (radius + y) ** 2
            roll_damping_numerator += radius**3 * chord / (radius + y) ** 2
            roll_damping_denominator += (radius + y) * chord

            if i > 0:
                dx = (chord_trail[i] + chord_lead[i]) / 2 - (
                    chord_trail[i - 1] + chord_lead[i - 1]
                ) / 2
                cos_gamma_sum += dy / np.hypot(dx, dy)

        mac_length *= dy
        mac_span *= dy
        mac_lead *= dy
        total_area *= dy
        roll_geometrical_constant *= dy
        roll_damping_numerator *= dy
        roll_damping_denominator *= dy

        mac_length /= total_area
        mac_span /= total_area
        mac_lead /= total_area
        cos_gamma = cos_gamma_sum / (points_per_line - 1)

        owner.Af = Af
        owner.AR = AR
        owner.gamma_c = np.arccos(cos_gamma)
        owner.Yma = mac_span
        owner.mac_length = mac_length
        owner.mac_lead = mac_lead
        owner.tau = tau
        owner.roll_geometrical_constant = roll_geometrical_constant
        owner.lift_interference_factor = lift_interference_factor
        owner.roll_forcing_interference_factor = roll_forcing_interference_factor
        owner.roll_damping_interference_factor = 1 + (
            roll_damping_numerator / roll_damping_denominator
        )

        self.evaluate_shape()

    def evaluate_shape(self):
        x_array, y_array = zip(*self.shape_points)
        self.owner.shape_vec = [np.array(x_array), np.array(y_array)]

    def get_data(self, include_outputs=False):
        data = {"shape_points": self.shape_points}
        if include_outputs:
            data.update(
                {
                    "Af": self.owner.Af,
                    "AR": self.owner.AR,
                    "gamma_c": self.owner.gamma_c,
                    "Yma": self.owner.Yma,
                    "mac_length": self.owner.mac_length,
                    "mac_lead": self.owner.mac_lead,
                    "roll_geometrical_constant": (
                        self.owner.roll_geometrical_constant
                    ),
                    "tau": self.owner.tau,
                    "lift_interference_factor": (
                        self.owner.lift_interference_factor
                    ),
                    "roll_forcing_interference_factor": (
                        self.owner.roll_forcing_interference_factor
                    ),
                    "roll_damping_interference_factor": (
                        self.owner.roll_damping_interference_factor
                    ),
                }
            )
        return data
