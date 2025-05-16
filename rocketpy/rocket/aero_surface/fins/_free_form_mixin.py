import warnings

import numpy as np


class _FreeFormMixin:
    """Mixin class for free form fins. This class holds methods and properties
    specific to free form fin shapes. It is designed to be used in conjunction
    with other classes that define the overall behavior of the fins."""

    def _initialize(self, shape_points):
        self.shape_points = shape_points

        down = False
        for i in range(1, len(shape_points)):
            if shape_points[i][1] > shape_points[i - 1][1] and down:
                warnings.warn(
                    "Jagged fin shape detected. This may cause small inaccuracies "
                    "center of pressure and pitch moment calculations."
                )
                break
            if shape_points[i][1] < shape_points[i - 1][1]:
                down = True
            i += 1

        root_chord = abs(shape_points[0][0] - shape_points[-1][0])
        ys = [point[1] for point in shape_points]
        span = max(ys) - min(ys)
        return (
            root_chord,
            span,
        )

    def evaluate_geometrical_parameters(self):  # pylint: disable=too-many-statements
        """Calculates and saves the fin set's geometrical parameters such as the
        fin area, aspect ratio, and parameters related to roll movement. This
        method uses the same calculations to those in OpenRocket for free-form
        fin shapes."""
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        # Calculate the fin area (Af) using the Shoelace theorem (polygon area formula)
        Af = 0
        for i in range(len(self.shape_points) - 1):
            x1, y1 = self.shape_points[i]
            x2, y2 = self.shape_points[i + 1]
            Af += (y1 + y2) * (x1 - x2)
        Af = abs(Af) / 2
        if Af < 1e-6:
            raise ValueError("Fin area is too small. Check the shape_points.")

        # Calculate aspect ratio (AR) and lift interference factors
        AR = 2 * self.span**2 / Af  # Aspect ratio
        tau = (self.span + self.rocket_radius) / self.rocket_radius
        lift_interference_factor = 1 + 1 / tau

        # Calculate roll forcing interference factor using OpenRocket's approach
        roll_forcing_interference_factor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + ((np.pi * (tau**2 + 1) ** 2) / (tau**2 * (tau - 1) ** 2))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + ((tau**2 + 1) ** 2 / (tau**2 * (tau - 1) ** 2))
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )

        # Define number of interpolation points along the span of the fin
        points_per_line = 40  # Same as OpenRocket

        # Initialize arrays for leading/trailing edge and chord lengths
        chord_lead = np.ones(points_per_line) * np.inf  # Leading edge x coordinates
        chord_trail = np.ones(points_per_line) * -np.inf  # Trailing edge x coordinates
        chord_length = np.zeros(
            points_per_line
        )  # Chord length for each spanwise section

        # Discretize fin shape and calculate chord length, leading, and trailing edges
        for p in range(1, len(self.shape_points)):
            x1, y1 = self.shape_points[p - 1]
            x2, y2 = self.shape_points[p]

            # Compute corresponding points along the fin span (clamp to valid range)
            prev_idx = int(y1 / self.span * (points_per_line - 1))
            curr_idx = int(y2 / self.span * (points_per_line - 1))
            prev_idx = np.clip(prev_idx, 0, points_per_line - 1)
            curr_idx = np.clip(curr_idx, 0, points_per_line - 1)

            if prev_idx > curr_idx:
                prev_idx, curr_idx = curr_idx, prev_idx

            # Compute intersection of fin edge with each spanwise section
            for i in range(prev_idx, curr_idx + 1):
                y = i * self.span / (points_per_line - 1)
                if y1 != y2:
                    x = np.clip(
                        (y - y2) / (y1 - y2) * x1 + (y1 - y) / (y1 - y2) * x2,
                        min(x1, x2),
                        max(x1, x2),
                    )
                else:
                    x = x1  # Handle horizontal segments

                # Update leading and trailing edge positions
                chord_lead[i] = min(chord_lead[i], x)
                chord_trail[i] = max(chord_trail[i], x)

                # Update chord length
                if y1 < y2:
                    chord_length[i] -= x
                else:
                    chord_length[i] += x

        # Replace infinities and handle invalid values in chord_lead and chord_trail
        for i in range(points_per_line):
            if (
                np.isinf(chord_lead[i])
                or np.isinf(chord_trail[i])
                or np.isnan(chord_lead[i])
                or np.isnan(chord_trail[i])
            ):
                chord_lead[i] = 0
                chord_trail[i] = 0
            if chord_length[i] < 0 or np.isnan(chord_length[i]):
                chord_length[i] = 0
            if chord_length[i] > chord_trail[i] - chord_lead[i]:
                chord_length[i] = chord_trail[i] - chord_lead[i]

        # Initialize integration variables for various aerodynamic and roll properties
        radius = self.rocket_radius
        total_area = 0
        mac_length = 0  # Mean aerodynamic chord length
        mac_lead = 0  # Mean aerodynamic chord leading edge
        mac_span = 0  # Mean aerodynamic chord spanwise position (Yma)
        cos_gamma_sum = 0  # Sum of cosine of the sweep angle
        roll_geometrical_constant = 0
        roll_damping_numerator = 0
        roll_damping_denominator = 0

        # Perform integration over spanwise sections
        dy = self.span / (points_per_line - 1)
        for i in range(points_per_line):
            chord = chord_trail[i] - chord_lead[i]
            y = i * dy

            # Update integration variables
            mac_length += chord * chord
            mac_span += y * chord
            mac_lead += chord_lead[i] * chord
            total_area += chord
            roll_geometrical_constant += chord_length[i] * (radius + y) ** 2
            roll_damping_numerator += radius**3 * chord / (radius + y) ** 2
            roll_damping_denominator += (radius + y) * chord

            # Update cosine of sweep angle (cos_gamma)
            if i > 0:
                dx = (chord_trail[i] + chord_lead[i]) / 2 - (
                    chord_trail[i - 1] + chord_lead[i - 1]
                ) / 2
                cos_gamma_sum += dy / np.hypot(dx, dy)

        # Finalize mean aerodynamic chord properties
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

        # Store computed values
        self.Af = Af  # Fin area
        self.AR = AR  # Aspect ratio
        self.gamma_c = np.arccos(cos_gamma)  # Sweep angle
        self.Yma = mac_span  # Mean aerodynamic chord spanwise position
        self.mac_length = mac_length
        self.mac_lead = mac_lead
        self.tau = tau
        self.roll_geometrical_constant = roll_geometrical_constant
        self.lift_interference_factor = lift_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor
        self.roll_damping_interference_factor = 1 + (
            roll_damping_numerator / roll_damping_denominator
        )

        # Evaluate the shape and finalize geometry
        self.evaluate_shape()

    def evaluate_shape(self):
        x_array, y_array = zip(*self.shape_points)
        self.shape_vec = [np.array(x_array), np.array(y_array)]

    def to_dict(self, include_outputs=False):
        data = super().to_dict(include_outputs)
        data["shape_points"] = self.shape_points

        if include_outputs:
            data.update(
                {
                    "Af": self.Af,
                    "AR": self.AR,
                    "gamma_c": self.gamma_c,
                    "Yma": self.Yma,
                    "mac_length": self.mac_length,
                    "mac_lead": self.mac_lead,
                    "roll_geometrical_constant": self.roll_geometrical_constant,
                    "tau": self.tau,
                    "lift_interference_factor": self.lift_interference_factor,
                    "roll_forcing_interference_factor": self.roll_forcing_interference_factor,
                    "roll_damping_interference_factor": self.roll_damping_interference_factor,
                }
            )
        return data

    def info(self):
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        self.prints.all()
        self.plots.all()
