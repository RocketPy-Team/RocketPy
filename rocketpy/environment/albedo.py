"""Earth radiation-pressure models."""

from __future__ import annotations

import math

import numpy as np

from rocketpy.environment.third_body import ASTRONOMICAL_UNIT, ThirdBody
from rocketpy.mathutils.compilation import numbify
from rocketpy.mathutils.frame_vector import FrameVector
from rocketpy.mathutils.reference_frame import ReferenceFrame


class EarthRadiationPressure:
    """Knocke finite-element Earth albedo and infrared pressure model.

    The visible terrestrial cap is integrated over a fixed 10-by-20 surface
    grid. Reflected shortwave radiation and emitted longwave radiation are
    both included.
    """

    name = "earth_radiation_pressure"

    def __init__(self, sun):
        if not isinstance(sun, ThirdBody):
            raise TypeError("sun must be an explicit ThirdBody.")
        self.sun = sun

    def acceleration(self, epoch, state, vehicle, earth):
        """Return Earth radiation-pressure acceleration in GCRF."""
        sun_position = self.sun.position(epoch)
        incident_direction = FrameVector(-state.position, ReferenceFrame.GCRF)
        coefficient = vehicle.evaluate_radiation_coefficient(
            epoch, state, incident_direction
        )
        area = vehicle.evaluate_radiation_area(epoch, state, incident_direction)
        mass = vehicle.mass_at(state.elapsed_time)
        return self._evaluate_knocke_erp(
            epoch.jd_tdb,
            np.array(state.position, dtype=float, copy=True),
            np.array(sun_position, dtype=float, copy=True),
            float(area),
            float(coefficient),
            mass,
        )

    @staticmethod
    @numbify(
        nopython=True,
        signature=(
            "float64[:](float64, float64[:], float64[:], float64, float64, float64)"
        ),
    )
    def _evaluate_knocke_erp(jd, r_sat, r_sun, area_sat, coefficient, mass):
        """Evaluate the finite-element Earth-radiation integration."""
        earth_radius = 6378136.0
        satellite_norm = math.sqrt(r_sat[0] ** 2 + r_sat[1] ** 2 + r_sat[2] ** 2)
        sun_norm = math.sqrt(r_sun[0] ** 2 + r_sun[1] ** 2 + r_sun[2] ** 2)
        if satellite_norm <= earth_radius:
            return np.zeros(3, dtype=np.float64)

        sun_unit = r_sun / sun_norm
        satellite_unit = r_sat / satellite_norm
        solar_pressure = 4.56e-6 / (sun_norm / ASTRONOMICAL_UNIT) ** 2
        cap_angle = math.asin(earth_radius / satellite_norm)
        local_z = satellite_unit
        auxiliary = np.zeros(3, dtype=np.float64)
        if abs(local_z[2]) < 0.99:
            auxiliary[2] = 1.0
        else:
            auxiliary[0] = 1.0
        local_x = np.array(
            [
                local_z[1] * auxiliary[2] - local_z[2] * auxiliary[1],
                local_z[2] * auxiliary[0] - local_z[0] * auxiliary[2],
                local_z[0] * auxiliary[1] - local_z[1] * auxiliary[0],
            ],
            dtype=np.float64,
        )
        local_x /= math.sqrt(local_x[0] ** 2 + local_x[1] ** 2 + local_x[2] ** 2)
        local_y = np.array(
            [
                local_z[1] * local_x[2] - local_z[2] * local_x[1],
                local_z[2] * local_x[0] - local_z[0] * local_x[2],
                local_z[0] * local_x[1] - local_z[1] * local_x[0],
            ],
            dtype=np.float64,
        )

        cosine_phase = math.cos(math.tau / 365.25 * (jd - 2451535.0))
        rings = 10
        sectors = 20
        delta_alpha = cap_angle / rings
        delta_beta = math.tau / sectors
        pressure_vector = np.zeros(3, dtype=np.float64)
        for ring in range(rings):
            alpha = (ring + 0.5) * delta_alpha
            sin_alpha = math.sin(alpha)
            cos_alpha = math.cos(alpha)
            area_element = earth_radius**2 * sin_alpha * delta_alpha * delta_beta
            for sector in range(sectors):
                beta = (sector + 0.5) * delta_beta
                normal = (
                    sin_alpha * math.cos(beta) * local_x
                    + sin_alpha * math.sin(beta) * local_y
                    + cos_alpha * local_z
                )
                element_to_satellite = r_sat - earth_radius * normal
                distance = math.sqrt(
                    element_to_satellite[0] ** 2
                    + element_to_satellite[1] ** 2
                    + element_to_satellite[2] ** 2
                )
                direction = element_to_satellite / distance
                satellite_cosine = (
                    normal[0] * direction[0]
                    + normal[1] * direction[1]
                    + normal[2] * direction[2]
                )
                if satellite_cosine <= 0.0:
                    continue
                latitude_sine = normal[2]
                legendre_p2 = 0.5 * (3.0 * latitude_sine**2 - 1.0)
                emissivity = max(
                    0.0,
                    min(
                        1.0,
                        0.68 - 0.07 * cosine_phase * latitude_sine - 0.18 * legendre_p2,
                    ),
                )
                surface_pressure = emissivity * solar_pressure / 4.0
                sun_cosine = (
                    normal[0] * sun_unit[0]
                    + normal[1] * sun_unit[1]
                    + normal[2] * sun_unit[2]
                )
                if sun_cosine > 0.0:
                    albedo = max(
                        0.0,
                        min(
                            1.0,
                            0.34
                            + 0.10 * cosine_phase * latitude_sine
                            + 0.29 * legendre_p2,
                        ),
                    )
                    surface_pressure += albedo * solar_pressure * sun_cosine
                pressure_vector += (
                    surface_pressure
                    / math.pi
                    * area_element
                    * satellite_cosine
                    / distance**2
                    * direction
                )
        return pressure_vector * (coefficient * area_sat / mass)
