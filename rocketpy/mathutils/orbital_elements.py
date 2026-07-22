"""Classical orbital elements derived from Cartesian states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrbitalElements:
    """Osculating classical orbital elements in SI units and radians."""

    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    argument_of_periapsis: float
    true_anomaly: float

    @property
    def semi_latus_rectum(self) -> float:
        return self.semi_major_axis * (1.0 - self.eccentricity**2)

    @classmethod
    def from_state(
        cls,
        position,
        velocity,
        gravitational_parameter: float = 3.986004418e14,
        tolerance: float = 1e-10,
    ) -> "OrbitalElements":
        """Calculate robust classical elements from a GCRF Cartesian state."""
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        radius = np.linalg.norm(position)
        speed_squared = float(np.dot(velocity, velocity))
        if radius <= tolerance:
            raise ValueError("Position norm must be positive.")

        momentum = np.cross(position, velocity)
        momentum_norm = np.linalg.norm(momentum)
        if momentum_norm <= tolerance:
            raise ValueError("Orbital elements are undefined for radial motion.")

        node = np.cross([0.0, 0.0, 1.0], momentum)
        node_norm = np.linalg.norm(node)
        eccentricity_vector = (
            ((speed_squared - gravitational_parameter / radius) * position)
            - np.dot(position, velocity) * velocity
        ) / gravitational_parameter
        eccentricity = float(np.linalg.norm(eccentricity_vector))
        specific_energy = speed_squared / 2.0 - gravitational_parameter / radius
        semi_major_axis = (
            np.inf
            if abs(specific_energy) <= tolerance
            else -gravitational_parameter / (2.0 * specific_energy)
        )
        inclination = float(np.arccos(np.clip(momentum[2] / momentum_norm, -1, 1)))

        if node_norm > tolerance:
            raan = float(np.arctan2(node[1], node[0]) % (2.0 * np.pi))
        else:
            raan = 0.0

        if eccentricity > tolerance and node_norm > tolerance:
            argument_of_periapsis = _oriented_angle(node, eccentricity_vector, momentum)
        elif eccentricity > tolerance:
            argument_of_periapsis = float(
                np.arctan2(eccentricity_vector[1], eccentricity_vector[0])
                % (2.0 * np.pi)
            )
        else:
            argument_of_periapsis = 0.0

        if eccentricity > tolerance:
            true_anomaly = _oriented_angle(eccentricity_vector, position, momentum)
        elif node_norm > tolerance:
            true_anomaly = _oriented_angle(node, position, momentum)
        else:
            true_anomaly = float(np.arctan2(position[1], position[0]) % (2.0 * np.pi))

        return cls(
            semi_major_axis=float(semi_major_axis),
            eccentricity=eccentricity,
            inclination=inclination,
            raan=raan,
            argument_of_periapsis=argument_of_periapsis,
            true_anomaly=true_anomaly,
        )

    def to_state(
        self, gravitational_parameter: float = 3.986004418e14
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the elements to a GCRF Cartesian position and velocity."""
        p = self.semi_latus_rectum
        if p <= 0:
            raise ValueError("Semi-latus rectum must be positive.")
        cosine = np.cos(self.true_anomaly)
        sine = np.sin(self.true_anomaly)
        position_perifocal = (
            p / (1.0 + self.eccentricity * cosine) * np.array([cosine, sine, 0.0])
        )
        velocity_perifocal = np.sqrt(gravitational_parameter / p) * np.array(
            [-sine, self.eccentricity + cosine, 0.0]
        )

        cos_raan, sin_raan = np.cos(self.raan), np.sin(self.raan)
        cos_arg, sin_arg = (
            np.cos(self.argument_of_periapsis),
            np.sin(self.argument_of_periapsis),
        )
        cos_inc, sin_inc = np.cos(self.inclination), np.sin(self.inclination)
        rotation = np.array(
            [
                [
                    cos_raan * cos_arg - sin_raan * sin_arg * cos_inc,
                    -cos_raan * sin_arg - sin_raan * cos_arg * cos_inc,
                    sin_raan * sin_inc,
                ],
                [
                    sin_raan * cos_arg + cos_raan * sin_arg * cos_inc,
                    -sin_raan * sin_arg + cos_raan * cos_arg * cos_inc,
                    -cos_raan * sin_inc,
                ],
                [sin_arg * sin_inc, cos_arg * sin_inc, cos_inc],
            ]
        )
        return rotation @ position_perifocal, rotation @ velocity_perifocal


def _oriented_angle(first, second, normal) -> float:
    first_unit = first / np.linalg.norm(first)
    second_unit = second / np.linalg.norm(second)
    cosine = np.clip(np.dot(first_unit, second_unit), -1.0, 1.0)
    sine = np.dot(np.cross(first_unit, second_unit), normal / np.linalg.norm(normal))
    return float(np.arctan2(sine, cosine) % (2.0 * np.pi))
