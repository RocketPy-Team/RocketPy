"""Perturbing acceleration models for Earth-centered Flight dynamics."""

from __future__ import annotations

import math

import numpy as np
from scipy.constants import speed_of_light

from rocketpy.environment.celestial_body import ASTRONOMICAL_UNIT, CelestialBody


class ThirdBodyGravity:
    """Differential third-body gravitational acceleration in GCRF."""

    name = "third_body_gravity"

    def __init__(self, bodies=None):
        self.bodies = bodies

    def acceleration(self, epoch, state, rocket, environment):
        del rocket
        bodies = environment.celestial_bodies if self.bodies is None else self.bodies
        total = np.zeros(3)
        for body in bodies:
            body = CelestialBody.builtin(body) if isinstance(body, str) else body
            body_position = body.position(epoch)
            relative = body_position - state.position
            relative_norm = np.linalg.norm(relative)
            body_norm = np.linalg.norm(body_position)
            if relative_norm == 0 or body_norm == 0:
                continue
            total += body.gravitational_parameter * (
                relative / relative_norm**3 - body_position / body_norm**3
            )
        return total


class SolarRadiationPressure:
    """Cannonball solar-radiation pressure with conical-disk occultation."""

    name = "solar_radiation_pressure"

    def __init__(self, sun=None):
        self.sun = sun

    def acceleration(self, epoch, state, rocket, environment):
        sun = self.sun or _find_sun(environment.celestial_bodies)
        sun_position = sun.position(epoch)
        satellite_to_sun = sun_position - state.position
        distance = np.linalg.norm(satellite_to_sun)
        if distance == 0:
            return np.zeros(3)
        illuminated = occultation_fraction(
            observer=state.position,
            source=sun_position,
            source_radius=sun.radius,
            blocker=np.zeros(3),
            blocker_radius=environment.earth_datum.semi_major_axis + 21000.0,
        )
        for blocker in environment.celestial_bodies:
            blocker = (
                CelestialBody.builtin(blocker) if isinstance(blocker, str) else blocker
            )
            if blocker.name.lower() == sun.name.lower() or not blocker.radius:
                continue
            illuminated = min(
                illuminated,
                occultation_fraction(
                    observer=state.position,
                    source=sun_position,
                    source_radius=sun.radius,
                    blocker=blocker.position(epoch),
                    blocker_radius=blocker.radius,
                ),
            )
        if illuminated == 0:
            return np.zeros(3)
        coefficient = rocket.evaluate_radiation_coefficient(
            epoch, state, satellite_to_sun
        )
        area = rocket.evaluate_radiation_area(epoch, state, satellite_to_sun)
        mass = float(rocket.total_mass.get_value_opt(state.elapsed_time))
        pressure = (sun.solar_pressure_at_reference or 4.56e-6) * (
            ASTRONOMICAL_UNIT / distance
        ) ** 2
        # Radiation pushes away from the Sun.
        return (
            -illuminated
            * pressure
            * coefficient
            * area
            / mass
            * satellite_to_sun
            / distance
        )


class PlanetaryRadiationPressure:
    """Lambertian radiation pressure reflected by configured third bodies."""

    name = "planetary_radiation_pressure"

    def __init__(self, bodies=None, sources=None):
        self.bodies = bodies
        self.sources = sources

    def acceleration(self, epoch, state, rocket, environment):
        configured = environment.celestial_bodies
        bodies = configured if self.bodies is None else self.bodies
        sources = configured if self.sources is None else self.sources
        bodies = [_coerce_body(body) for body in bodies]
        sources = [_coerce_body(body) for body in sources]
        reflectors = [body for body in bodies if body.radius and body.albedo > 0]
        light_sources = [
            body for body in sources if body.solar_pressure_at_reference is not None
        ]
        total = np.zeros(3)
        for source in light_sources:
            source_position = source.position(epoch)
            for reflector in reflectors:
                if reflector is source:
                    continue
                reflector_position = reflector.position(epoch)
                body_to_vehicle = state.position - reflector_position
                body_to_source = source_position - reflector_position
                vehicle_distance = np.linalg.norm(body_to_vehicle)
                source_distance = np.linalg.norm(body_to_source)
                if vehicle_distance == 0 or source_distance == 0:
                    continue
                cosine_phase = np.clip(
                    np.dot(body_to_vehicle, body_to_source)
                    / (vehicle_distance * source_distance),
                    -1.0,
                    1.0,
                )
                phase = math.acos(cosine_phase)
                lambertian = (
                    2.0
                    / (3.0 * math.pi)
                    * (math.sin(phase) + (math.pi - phase) * cosine_phase)
                )
                source_pressure = (
                    source.solar_pressure_at_reference
                    * (ASTRONOMICAL_UNIT / source_distance) ** 2
                )
                reflected_pressure = (
                    source_pressure
                    * reflector.albedo
                    * (reflector.radius / vehicle_distance) ** 2
                    * lambertian
                )
                direction = body_to_vehicle / vehicle_distance
                coefficient = rocket.evaluate_radiation_coefficient(
                    epoch, state, -body_to_vehicle
                )
                area = rocket.evaluate_radiation_area(epoch, state, -body_to_vehicle)
                mass = float(rocket.total_mass.get_value_opt(state.elapsed_time))
                total += reflected_pressure * coefficient * area / mass * direction
        return total


class RelativisticCorrection:
    """First post-Newtonian correction for a spherical central body."""

    name = "relativistic_correction"

    def acceleration(self, epoch, state, rocket, environment):
        del epoch, rocket
        position = state.position
        velocity = state.velocity
        radius = np.linalg.norm(position)
        mu = environment.earth_datum.gravitational_parameter
        velocity_squared = np.dot(velocity, velocity)
        radial_velocity = np.dot(position, velocity)
        return (
            mu
            / (speed_of_light**2 * radius**3)
            * (
                (4.0 * mu / radius - velocity_squared) * position
                + 4.0 * radial_velocity * velocity
            )
        )


class EarthRadiationPressure:
    """Knocke finite-element Earth albedo and infrared pressure model."""

    name = "earth_radiation_pressure"

    def __init__(self, sun=None, rings=10, sectors=20):
        self.sun = sun
        self.rings = int(rings)
        self.sectors = int(sectors)
        if self.rings <= 0 or self.sectors <= 0:
            raise ValueError("Earth-radiation grid dimensions must be positive.")

    def acceleration(self, epoch, state, rocket, environment):
        sun = self.sun or _find_sun(environment.celestial_bodies)
        sun_position = sun.position(epoch)
        satellite_position = state.position
        satellite_radius = np.linalg.norm(satellite_position)
        earth_radius = environment.earth_datum.semi_major_axis
        if satellite_radius <= earth_radius:
            return np.zeros(3)

        sun_distance = np.linalg.norm(sun_position)
        sun_unit = sun_position / sun_distance
        satellite_unit = satellite_position / satellite_radius
        solar_pressure = 4.56e-6 * (ASTRONOMICAL_UNIT / sun_distance) ** 2
        cap_angle = math.asin(earth_radius / satellite_radius)

        local_z = satellite_unit
        auxiliary = (
            np.array([0.0, 0.0, 1.0])
            if abs(local_z[2]) < 0.99
            else np.array([1.0, 0.0, 0.0])
        )
        local_x = np.cross(local_z, auxiliary)
        local_x /= np.linalg.norm(local_x)
        local_y = np.cross(local_z, local_x)

        seasonal_phase = math.tau / 365.25 * (epoch.jd_tdb - 2451535.0)
        cosine_phase = math.cos(seasonal_phase)
        delta_alpha = cap_angle / self.rings
        delta_beta = math.tau / self.sectors
        pressure_vector = np.zeros(3)

        for ring in range(self.rings):
            alpha = (ring + 0.5) * delta_alpha
            sin_alpha, cos_alpha = math.sin(alpha), math.cos(alpha)
            area_element = earth_radius**2 * sin_alpha * delta_alpha * delta_beta
            for sector in range(self.sectors):
                beta = (sector + 0.5) * delta_beta
                normal = (
                    sin_alpha * math.cos(beta) * local_x
                    + sin_alpha * math.sin(beta) * local_y
                    + cos_alpha * local_z
                )
                element_position = earth_radius * normal
                element_to_satellite = satellite_position - element_position
                distance = np.linalg.norm(element_to_satellite)
                direction = element_to_satellite / distance
                satellite_cosine = np.dot(normal, direction)
                if satellite_cosine <= 0:
                    continue
                latitude_sine = normal[2]
                legendre_p2 = 0.5 * (3.0 * latitude_sine**2 - 1.0)
                emissivity = np.clip(
                    0.68 - 0.07 * cosine_phase * latitude_sine - 0.18 * legendre_p2,
                    0.0,
                    1.0,
                )
                surface_pressure = emissivity * solar_pressure / 4.0
                sun_cosine = np.dot(normal, sun_unit)
                if sun_cosine > 0:
                    albedo = np.clip(
                        0.34 + 0.10 * cosine_phase * latitude_sine + 0.29 * legendre_p2,
                        0.0,
                        1.0,
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

        incident_direction = -satellite_position
        coefficient = rocket.evaluate_radiation_coefficient(
            epoch, state, incident_direction
        )
        area = rocket.evaluate_radiation_area(epoch, state, incident_direction)
        mass = float(rocket.total_mass.get_value_opt(state.elapsed_time))
        return coefficient * area / mass * pressure_vector


def occultation_fraction(
    observer,
    source,
    source_radius,
    blocker,
    blocker_radius,
):
    """Return the visible fraction of a circular source in ``[0, 1]``."""
    observer = np.asarray(observer, dtype=float)
    source_vector = np.asarray(source, dtype=float) - observer
    blocker_vector = np.asarray(blocker, dtype=float) - observer
    source_distance = np.linalg.norm(source_vector)
    blocker_distance = np.linalg.norm(blocker_vector)
    if blocker_distance <= blocker_radius:
        return 0.0
    if source_distance <= source_radius or blocker_distance > source_distance:
        return 1.0
    source_angle = math.asin(min(1.0, source_radius / source_distance))
    blocker_angle = math.asin(min(1.0, blocker_radius / blocker_distance))
    separation = math.acos(
        np.clip(
            np.dot(source_vector, blocker_vector)
            / (source_distance * blocker_distance),
            -1.0,
            1.0,
        )
    )
    if separation >= source_angle + blocker_angle:
        return 1.0
    if separation <= abs(blocker_angle - source_angle):
        if blocker_angle >= source_angle:
            return 0.0
        return 1.0 - (blocker_angle / source_angle) ** 2
    x = (separation**2 + source_angle**2 - blocker_angle**2) / (2.0 * separation)
    height = math.sqrt(max(0.0, source_angle**2 - x**2))
    overlap = (
        source_angle**2 * math.acos(np.clip(x / source_angle, -1.0, 1.0))
        + blocker_angle**2
        * math.acos(np.clip((separation - x) / blocker_angle, -1.0, 1.0))
        - separation * height
    )
    return float(np.clip(1.0 - overlap / (math.pi * source_angle**2), 0.0, 1.0))


def _find_sun(bodies):
    for body in bodies:
        candidate = CelestialBody.builtin(body) if isinstance(body, str) else body
        if candidate.name.lower() == "sun":
            return candidate
    return CelestialBody.builtin("sun")


def _coerce_body(body):
    return CelestialBody.builtin(body) if isinstance(body, str) else body
