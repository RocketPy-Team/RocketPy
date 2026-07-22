"""Orbital output views attached to Earth-centered Flight simulations."""

from __future__ import annotations

from functools import cached_property

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.orbital_elements import OrbitalElements


class FlightOrbit:
    """Osculating orbital-element history for a :class:`rocketpy.Flight`."""

    def __init__(self, flight):
        self.flight = flight

    @cached_property
    def _elements_array(self):
        rows = []
        mu = self.flight.env.earth_datum.gravitational_parameter
        for solution in self.flight.solution_array:
            elements = OrbitalElements.from_state(
                solution[1:4], solution[4:7], gravitational_parameter=mu
            )
            rows.append(
                [
                    solution[0],
                    elements.semi_major_axis,
                    elements.eccentricity,
                    elements.inclination,
                    elements.raan,
                    elements.argument_of_periapsis,
                    elements.true_anomaly,
                ]
            )
        return np.asarray(rows)

    def at_time(self, time: float) -> OrbitalElements:
        """Return interpolated osculating elements at ``time``."""
        return OrbitalElements.from_state(
            [self.flight.x(time), self.flight.y(time), self.flight.z(time)],
            [self.flight.vx(time), self.flight.vy(time), self.flight.vz(time)],
            self.flight.env.earth_datum.gravitational_parameter,
        )

    @property
    def initial(self) -> OrbitalElements:
        """Initial osculating orbit."""
        return self.at_time(self.flight.t_initial)

    @property
    def final(self) -> OrbitalElements:
        """Final osculating orbit."""
        return self.at_time(self.flight.t_final)

    def _function(self, column, output):
        return Function(
            self._elements_array[:, [0, column]],
            inputs="Time (s)",
            outputs=output,
            interpolation="linear",
            extrapolation="constant",
        )

    @cached_property
    def semi_major_axis(self):
        """Semi-major axis in meters as a function of time."""
        return self._function(1, "Semi-Major Axis (m)")

    @cached_property
    def eccentricity(self):
        """Eccentricity as a function of time."""
        return self._function(2, "Eccentricity")

    @cached_property
    def inclination(self):
        """Inclination in radians as a function of time."""
        return self._function(3, "Inclination (rad)")

    @cached_property
    def raan(self):
        """Right ascension of ascending node in radians versus time."""
        return self._function(4, "RAAN (rad)")

    @cached_property
    def argument_of_periapsis(self):
        """Argument of periapsis in radians versus time."""
        return self._function(5, "Argument of Periapsis (rad)")

    @cached_property
    def true_anomaly(self):
        """True anomaly in radians as a function of time."""
        return self._function(6, "True Anomaly (rad)")

    @cached_property
    def periapsis_radius(self):
        """Osculating periapsis radius in meters as a function of time."""
        values = self._elements_array[:, 1] * (1.0 - self._elements_array[:, 2])
        return Function(
            np.column_stack((self._elements_array[:, 0], values)),
            inputs="Time (s)",
            outputs="Periapsis Radius (m)",
            interpolation="linear",
            extrapolation="constant",
        )

    @cached_property
    def apoapsis_radius(self):
        """Osculating apoapsis radius in meters for bound trajectories."""
        eccentricity = self._elements_array[:, 2]
        values = np.where(
            eccentricity < 1.0,
            self._elements_array[:, 1] * (1.0 + eccentricity),
            np.inf,
        )
        return Function(
            np.column_stack((self._elements_array[:, 0], values)),
            inputs="Time (s)",
            outputs="Apoapsis Radius (m)",
            interpolation="linear",
            extrapolation="constant",
        )

    @cached_property
    def period(self):
        """Osculating period in seconds for bound trajectories."""
        semi_major_axis = self._elements_array[:, 1]
        eccentricity = self._elements_array[:, 2]
        mu = self.flight.env.earth_datum.gravitational_parameter
        values = np.where(
            (semi_major_axis > 0.0) & (eccentricity < 1.0),
            2.0 * np.pi * np.sqrt(np.maximum(semi_major_axis, 0.0) ** 3 / mu),
            np.inf,
        )
        return Function(
            np.column_stack((self._elements_array[:, 0], values)),
            inputs="Time (s)",
            outputs="Orbital Period (s)",
            interpolation="linear",
            extrapolation="constant",
        )
