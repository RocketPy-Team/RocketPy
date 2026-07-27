"""Composed Earth and Space domains used by Earth-centered Flight."""

from __future__ import annotations

import math

import numpy as np
from scipy.constants import speed_of_light

from rocketpy.environment.atmosphere import (
    Atmosphere,
    AtmosphereLayer,
    AtmosphericState,
    ExponentialAtmosphereLayer,
    ZeroAtmosphereLayer,
)
from rocketpy.environment.gravity import (
    Gravity,
    SphericalGravity,
    SphericalHarmonicGravity,
    ZeroGravity,
    ZonalGravity,
)
from rocketpy.environment.third_body import (
    SpiceEphemeris,
    ThirdBody,
    occultation_fraction,
)
from rocketpy.mathutils.epoch import Epoch
from rocketpy.mathutils.flight_state import FlightState
from rocketpy.mathutils.frame_vector import FrameVector
from rocketpy.mathutils.reference_frame import (
    GRS80,
    WGS72,
    WGS84,
    Datum,
    EarthDatum,
    FlatEarthDatum,
    ReferenceFrame,
    SimpleDatum,
    transform_kinematics,
)


class Earth:
    """Coordinator for Earth geometry, gravity, atmosphere and albedo."""

    def __init__(
        self,
        geopotential="ellipsoidal",
        atmosphere="exponential",
        datum=WGS84,
        albedo=None,
        relativistic_correction=False,
    ):
        self.datum = datum
        self.geopotential = geopotential
        self.atmosphere = atmosphere
        self.albedo = albedo
        self.relativistic_correction = bool(relativistic_correction)

    @property
    def datum(self):
        return self._datum

    @datum.setter
    def datum(self, value):
        if isinstance(value, str):
            factories = {
                "flat": FlatEarthDatum,
                "flat_earth": FlatEarthDatum,
                "simple": SimpleDatum,
                "simple_wgs84": SimpleDatum,
                "wgs84": Datum,
                "grs80": lambda: Datum(
                    name=GRS80.name,
                    semi_major_axis=GRS80.semi_major_axis,
                    flattening=GRS80.flattening,
                    angular_velocity=GRS80.angular_velocity,
                    gravitational_parameter=GRS80.gravitational_parameter,
                ),
                "wgs72": lambda: Datum(
                    name=WGS72.name,
                    semi_major_axis=WGS72.semi_major_axis,
                    flattening=WGS72.flattening,
                    angular_velocity=WGS72.angular_velocity,
                    gravitational_parameter=WGS72.gravitational_parameter,
                ),
                "full": Datum,
            }
            try:
                value = factories[value.lower()]()
            except KeyError as exc:
                raise ValueError(f"Unknown Earth datum: {value!r}.") from exc
        if not isinstance(value, EarthDatum):
            raise TypeError("datum must be an EarthDatum.")
        if isinstance(value, FlatEarthDatum):
            raise ValueError(
                "Earth requires a geocentric datum. FlatEarthDatum is reserved "
                "for standalone Environment low_altitude simulations."
            )
        self._datum = value

    @property
    def geopotential(self):
        return self._geopotential

    @geopotential.setter
    def geopotential(self, value):
        if value is None:
            value = "zero"
        if isinstance(value, str):
            factories = {
                "zero": ZeroGravity,
                "point_mass": SphericalGravity,
                "spherical": SphericalGravity,
                "ellipsoidal": ZonalGravity,
                "egm2008": SphericalHarmonicGravity,
            }
            try:
                value = factories[value.lower()]()
            except KeyError as exc:
                raise ValueError(f"Unknown geopotential model: {value!r}.") from exc
        if not isinstance(value, Gravity):
            raise TypeError("geopotential must be a Gravity model or string.")
        self._geopotential = value

    @property
    def gravity(self):
        """Compatibility alias for :attr:`geopotential`."""
        return self.geopotential

    @gravity.setter
    def gravity(self, value):
        self.geopotential = value

    @property
    def atmosphere(self):
        return self._atmosphere

    @atmosphere.setter
    def atmosphere(self, value):
        if value is None:
            value = ZeroAtmosphereLayer()
        elif isinstance(value, str):
            factories = {
                "zero": ZeroAtmosphereLayer,
                "vacuum": ZeroAtmosphereLayer,
                "exponential": ExponentialAtmosphereLayer,
            }
            try:
                value = factories[value.lower()]()
            except KeyError as exc:
                raise ValueError(f"Unknown atmosphere layer: {value!r}.") from exc
        if isinstance(value, AtmosphereLayer):
            value = Atmosphere({0.0: value})
        elif isinstance(value, dict):
            value = Atmosphere(value)
        if not isinstance(value, Atmosphere):
            raise TypeError(
                "atmosphere must be Atmosphere, AtmosphereLayer, a layer map, "
                "a supported string, or None."
            )
        self._atmosphere = value

    @property
    def albedo(self):
        return self._albedo

    @albedo.setter
    def albedo(self, value):
        if value in (None, False, "zero"):
            value = None
        elif isinstance(value, str):
            raise ValueError(
                "The Knocke model needs an explicit Sun body; pass "
                "EarthRadiationPressure(sun=...)."
            )
        self._albedo = value

    @property
    def local_environment(self):
        from rocketpy.environment.environment import Environment

        for layer in self.atmosphere.atmosphere_layers_map.values():
            if isinstance(layer, Environment):
                return layer
        return None

    @property
    def earth_datum(self):
        return self.datum

    @property
    def earth_radius(self):
        return self.datum.semi_major_axis

    @property
    def elevation(self):
        return (
            float(self.local_environment.elevation)
            if self.local_environment is not None
            else 0.0
        )

    @property
    def latitude(self):
        return (
            float(self.local_environment.latitude)
            if self.local_environment is not None
            else 0.0
        )

    @property
    def longitude(self):
        return (
            float(self.local_environment.longitude)
            if self.local_environment is not None
            else 0.0
        )

    def transform_kinematics(
        self,
        epoch,
        position,
        velocity=None,
        acceleration=None,
        *,
        source,
        target,
        datum=None,
    ):
        return transform_kinematics(
            epoch,
            position,
            velocity,
            acceleration,
            source=source,
            target=target,
            datum=self.datum if datum is None else datum,
        )

    def prepare(self, start_epoch, end_epoch=None):
        """Prepare frame and atmospheric data before integration."""
        self.datum.prepare_epoch(start_epoch)
        self.atmosphere.prepare(start_epoch, end_epoch)
        return self

    def gravity_acceleration(self, epoch, state):
        if self.geopotential.requires_absolute_epoch:
            epoch.require_absolute(type(self.geopotential).__name__)
        return self.geopotential.acceleration(
            epoch, state.position, frame=state.frame, datum=self.datum
        )

    def evaluate_atmosphere(self, epoch, state):
        """Evaluate the composed atmosphere at an Earth-centered state."""
        if not isinstance(epoch, Epoch):
            raise TypeError("epoch must be an Epoch instance.")
        if not isinstance(state, FlightState):
            raise TypeError("state must be a FlightState instance.")
        if self.atmosphere.requires_absolute_epoch:
            epoch.require_absolute(type(self.atmosphere).__name__)
        if state.frame == ReferenceFrame.FLAT_EARTH:
            raise ValueError("Earth atmospheric evaluation requires GCRF or ITRF.")

        position_itrf, velocity_itrf, _ = self.transform_kinematics(
            epoch,
            state.position,
            state.velocity,
            source=state.frame,
            target=ReferenceFrame.ITRF,
        )
        latitude, longitude, altitude = self.datum.itrs_to_geodetic(position_itrf)
        if state.frame == ReferenceFrame.GCRF:
            position_gcrf, velocity_gcrf = state.position, state.velocity
        else:
            position_gcrf, velocity_gcrf, _ = self.transform_kinematics(
                epoch,
                state.position,
                state.velocity,
                source=state.frame,
                target=ReferenceFrame.GCRF,
            )
        atmosphere_state = self.atmosphere.evaluate(
            epoch,
            altitude,
            latitude_rad=latitude,
            longitude_rad=longitude,
            position_gcrf=position_gcrf,
            velocity_gcrf=velocity_gcrf,
        )
        if state.frame == ReferenceFrame.ITRF:
            return atmosphere_state
        wind_gcrf, _, _ = self.transform_kinematics(
            epoch,
            atmosphere_state.wind_velocity,
            source=ReferenceFrame.ITRF,
            target=ReferenceFrame.GCRF,
        )
        return AtmosphericState(
            pressure=atmosphere_state.pressure,
            temperature=atmosphere_state.temperature,
            density=atmosphere_state.density,
            speed_of_sound=atmosphere_state.speed_of_sound,
            dynamic_viscosity=atmosphere_state.dynamic_viscosity,
            wind_velocity=wind_gcrf,
        )

    def albedo_acceleration(self, epoch, state, vehicle):
        """Evaluate the configured terrestrial-radiation model directly."""
        if self.albedo is None:
            return np.zeros(3)
        evaluator = getattr(self.albedo, "acceleration", self.albedo)
        return np.asarray(evaluator(epoch, state, vehicle, self), dtype=float)

    def relativistic_acceleration(self, state):
        """Evaluate the first post-Newtonian correction when enabled."""
        if not self.relativistic_correction:
            return np.zeros(3)
        position = state.position
        velocity = state.velocity
        radius = np.linalg.norm(position)
        mu = self.datum.gravitational_parameter
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


class Space:
    """Aggregate third-body gravity, occultation and radiation pressure."""

    EFFECTIVE_EARTH_RADIUS = 6378136.0 + 21000.0

    def __init__(self, third_bodies=None, *, ephemeris=None):
        requested_bodies = () if third_bodies is None else tuple(third_bodies)
        if isinstance(ephemeris, str):
            ephemeris = SpiceEphemeris(ephemeris)
        if ephemeris is None and any(
            isinstance(body, str) for body in requested_bodies
        ):
            ephemeris = SpiceEphemeris()
        bodies = []
        for body in requested_bodies:
            if isinstance(body, str):
                body = ThirdBody.builtin(body, ephemeris=ephemeris)
            if not isinstance(body, ThirdBody):
                raise TypeError("third_bodies must contain names or ThirdBody objects.")
            bodies.append(body)
        self.third_bodies = bodies
        self._light_emitting_bodies = [
            body for body in bodies if body.pressure_function is not None
        ]
        self._blocker_bodies = [
            body for body in bodies if body.radius is not None and body.radius > 0.0
        ]
        self._reflecting_bodies = [
            body for body in self._blocker_bodies if body.albedo > 0.0
        ]

    @property
    def bodies(self):
        """Compatibility alias for :attr:`third_bodies`."""
        return self.third_bodies

    def gravity_acceleration(self, epoch, state):
        return sum(
            (
                body.gravity_acceleration(epoch, state.position)
                for body in self.third_bodies
            ),
            np.zeros(3),
        )

    def radiation_pressure_acceleration(self, epoch, state, vehicle, earth):
        """Evaluate direct and reflected pressure from configured bodies."""
        total = np.zeros(3)
        for source in self._light_emitting_bodies:
            source_position = source.position(epoch)
            fraction = occultation_fraction(
                state.position,
                source_position,
                source.radius,
                np.zeros(3),
                earth.datum.semi_major_axis + 21000.0,
            )
            for blocker in self._blocker_bodies:
                if blocker is source:
                    continue
                fraction = min(
                    fraction,
                    occultation_fraction(
                        state.position,
                        source_position,
                        source.radius,
                        blocker.position(epoch),
                        blocker.radius,
                    ),
                )

            vehicle_to_source = source_position - state.position
            source_distance = np.linalg.norm(vehicle_to_source)
            if source_distance > 0.0 and fraction > 0.0:
                incident = FrameVector(vehicle_to_source, ReferenceFrame.GCRF)
                coefficient = vehicle.evaluate_radiation_coefficient(
                    epoch, state, incident
                )
                area = vehicle.evaluate_radiation_area(epoch, state, incident)
                mass = vehicle.mass_at(state.elapsed_time)
                total -= (
                    fraction
                    * source.pressure_at(epoch.jd_tdb, source_distance)
                    * coefficient
                    * area
                    / mass
                    * vehicle_to_source
                    / source_distance
                )

            for reflector in self._reflecting_bodies:
                if reflector is source:
                    continue
                reflector_position = reflector.position(epoch)
                body_to_vehicle = state.position - reflector_position
                body_to_source = source_position - reflector_position
                vehicle_distance = np.linalg.norm(body_to_vehicle)
                source_distance = np.linalg.norm(body_to_source)
                if vehicle_distance == 0.0 or source_distance == 0.0:
                    continue
                cosine_phase = np.clip(
                    np.dot(body_to_vehicle, body_to_source)
                    / (vehicle_distance * source_distance),
                    -1.0,
                    1.0,
                )
                phase = math.acos(cosine_phase)
                phase_function = (
                    2.0
                    / (3.0 * math.pi)
                    * (math.sin(phase) + (math.pi - phase) * cosine_phase)
                )
                pressure = (
                    source.pressure_at(epoch.jd_tdb, source_distance)
                    * reflector.albedo
                    * (reflector.radius / vehicle_distance) ** 2
                    * phase_function
                )
                incident = FrameVector(-body_to_vehicle, ReferenceFrame.GCRF)
                coefficient = vehicle.evaluate_radiation_coefficient(
                    epoch, state, incident
                )
                area = vehicle.evaluate_radiation_area(epoch, state, incident)
                mass = vehicle.mass_at(state.elapsed_time)
                total += (
                    pressure
                    * coefficient
                    * area
                    / mass
                    * body_to_vehicle
                    / vehicle_distance
                )
        return total
