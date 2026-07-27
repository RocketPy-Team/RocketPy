"""Reference frames and kinematic transformations used by RocketPy."""

from __future__ import annotations

import math
from enum import Enum
from functools import lru_cache
from importlib import resources

import numpy as np

from .compilation import numbify
from .epoch import Epoch

_FRAME_KERNEL_SIGNATURE = (
    "Tuple((float64[::1], float64[::1], float64[::1]))("
    "float64[::1], float64[::1], float64[::1], float64[:, ::1], "
    "float64[::1], boolean, boolean)"
)
_FREE_ROTATION_SIGNATURE = (
    "Tuple((float64[::1], float64[::1], float64[::1]))("
    "float64[::1], float64[::1], float64[::1], float64[:, ::1])"
)


class ReferenceFrame(str, Enum):
    """Built-in reference-frame identifiers."""

    FLAT_EARTH = "flat_earth"
    GCRF = "gcrf"
    ITRF = "itrf"
    TEME = "teme"

    @classmethod
    def coerce(cls, value: "ReferenceFrame | str | None") -> "ReferenceFrame":
        """Normalize public frame aliases."""
        if value is None:
            return cls.FLAT_EARTH
        if isinstance(value, cls):
            return value
        aliases = {"eci": cls.GCRF, "ecef": cls.ITRF, "legacy": cls.FLAT_EARTH}
        normalized = str(value).lower()
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(frame.value for frame in cls)
            raise ValueError(
                f"Unknown reference frame {value!r}. Choose {choices}."
            ) from exc


class EarthDatum:
    """Earth shape and constants shared by all datum frame policies."""

    integration_frame = ReferenceFrame.GCRF
    fixed_frame = ReferenceFrame.ITRF
    requires_absolute_epoch = False
    supports_apparent_gravity = False
    is_geocentric = True

    def __init__(
        self,
        name="WGS84",
        semi_major_axis=6378137.0,
        flattening=1.0 / 298.257223563,
        angular_velocity=7.2921150e-5,
        gravitational_parameter=3.986004418e14,
        *,
        a=None,
        f=None,
        w=None,
        GM=None,
    ):
        semi_major_axis = semi_major_axis if a is None else a
        flattening = flattening if f is None else f
        angular_velocity = angular_velocity if w is None else w
        gravitational_parameter = gravitational_parameter if GM is None else GM
        self.name = str(name)
        self.semi_major_axis = float(semi_major_axis)
        self.flattening = float(flattening)
        self.angular_velocity = float(angular_velocity)
        self.gravitational_parameter = float(gravitational_parameter)

    @property
    def semi_minor_axis(self) -> float:
        return self.semi_major_axis * (1.0 - self.flattening)

    @property
    def eccentricity_squared(self) -> float:
        return self.flattening * (2.0 - self.flattening)

    @property
    def a(self):
        """Semi-major axis in metres."""
        return self.semi_major_axis

    @property
    def f(self):
        """Ellipsoid flattening."""
        return self.flattening

    @property
    def w(self):
        """Nominal angular velocity in rad/s."""
        return self.angular_velocity

    @property
    def GM(self):
        """Gravitational parameter in m³/s²."""
        return self.gravitational_parameter

    def geodetic_to_itrs(self, latitude_rad, longitude_rad, altitude) -> np.ndarray:
        """Convert broadcastable geodetic coordinates to ITRF positions.

        Scalar inputs return shape ``(3,)``. Array inputs are broadcast and
        return shape ``(..., 3)``.
        """
        latitude_rad, longitude_rad, altitude = np.broadcast_arrays(
            np.asarray(latitude_rad, dtype=float),
            np.asarray(longitude_rad, dtype=float),
            np.asarray(altitude, dtype=float),
        )
        sin_lat = np.sin(latitude_rad)
        cos_lat = np.cos(latitude_rad)
        radius = self.semi_major_axis / np.sqrt(
            1.0 - self.eccentricity_squared * sin_lat**2
        )
        return np.stack(
            (
                (radius + altitude) * cos_lat * np.cos(longitude_rad),
                (radius + altitude) * cos_lat * np.sin(longitude_rad),
                (radius * (1.0 - self.eccentricity_squared) + altitude) * sin_lat,
            ),
            axis=-1,
        )

    def itrs_to_geodetic(self, position) -> tuple:
        """Convert one or many ITRF positions to geodetic coordinates.

        ``position`` must have shape ``(..., 3)``. The returned latitude,
        longitude and altitude have the corresponding leading shape.
        """
        position = np.asarray(position, dtype=float)
        if position.shape == () or position.shape[-1] != 3:
            raise ValueError("ITRF positions must have shape (..., 3).")
        x, y, z = position[..., 0], position[..., 1], position[..., 2]
        longitude = np.arctan2(y, x)
        horizontal = np.hypot(x, y)
        latitude = np.arctan2(z, horizontal * (1.0 - self.eccentricity_squared))
        altitude = np.zeros_like(latitude)
        for _ in range(12):
            sin_lat = np.sin(latitude)
            radius = self.semi_major_axis / np.sqrt(
                1.0 - self.eccentricity_squared * sin_lat**2
            )
            cos_lat = np.cos(latitude)
            polar = np.abs(cos_lat) < 1e-14
            altitude = np.where(
                polar,
                np.abs(z) - self.semi_minor_axis,
                np.divide(
                    horizontal,
                    cos_lat,
                    out=np.zeros_like(horizontal),
                    where=~polar,
                )
                - radius,
            )
            next_latitude = np.arctan2(
                z,
                horizontal
                * (1.0 - self.eccentricity_squared * radius / (radius + altitude)),
            )
            next_latitude = np.where(polar, np.sign(z) * np.pi / 2.0, next_latitude)
            if np.all(np.abs(next_latitude - latitude) < 1e-13):
                latitude = next_latitude
                break
            latitude = next_latitude
        if position.ndim == 1:
            return float(latitude), float(longitude), float(altitude)
        return latitude, longitude, altitude

    def prepare_epoch(self, epoch):
        """Prepare any external orientation data before integration."""
        if self.requires_absolute_epoch:
            epoch.prepare_earth_orientation()
        return epoch

    def orientation_matrix(self, epoch):
        """Return the inertial-to-fixed position rotation."""
        raise NotImplementedError

    def transform_kinematics(
        self,
        epoch,
        position,
        velocity=None,
        acceleration=None,
        *,
        source,
        target,
    ):
        if ReferenceFrame.TEME in (
            ReferenceFrame.coerce(source),
            ReferenceFrame.coerce(target),
        ):
            return transform_kinematics(
                epoch,
                position,
                velocity,
                acceleration,
                source=source,
                target=target,
                datum=self,
            )
        return _transform_with_datum(
            self,
            epoch,
            position,
            velocity,
            acceleration,
            source=source,
            target=target,
        )

    def transform_vector(self, epoch, vector, *, source, target):
        """Rotate a free vector between datum-owned coordinate frames.

        Unlike a position/velocity transformation, a free vector has no
        origin and receives no transport-rate terms. This is appropriate for
        forces, relative velocities at one point, and direction vectors.
        """
        transformed, _, _ = transform_kinematics(
            epoch,
            vector,
            source=source,
            target=target,
            datum=self,
        )
        return transformed

    @staticmethod
    def _check_vector_shape(value, name):
        value = np.asarray(value, dtype=float)
        if value.shape == () or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (..., 3).")
        return value

    def topocentric_matrix(self, latitude_rad, longitude_rad, convention="enu"):
        """Return the ITRF-to-topocentric free-vector rotation matrix."""
        latitude_rad = float(latitude_rad)
        longitude_rad = float(longitude_rad)
        sin_latitude, cos_latitude = np.sin(latitude_rad), np.cos(latitude_rad)
        sin_longitude, cos_longitude = np.sin(longitude_rad), np.cos(longitude_rad)
        convention = str(convention).lower()
        if convention == "enu":
            return np.array(
                [
                    [-sin_longitude, cos_longitude, 0.0],
                    [
                        -sin_latitude * cos_longitude,
                        -sin_latitude * sin_longitude,
                        cos_latitude,
                    ],
                    [
                        cos_latitude * cos_longitude,
                        cos_latitude * sin_longitude,
                        sin_latitude,
                    ],
                ]
            )
        if convention == "sez":
            return np.array(
                [
                    [
                        sin_latitude * cos_longitude,
                        sin_latitude * sin_longitude,
                        -cos_latitude,
                    ],
                    [sin_longitude, -cos_longitude, 0.0],
                    [
                        cos_latitude * cos_longitude,
                        cos_latitude * sin_longitude,
                        sin_latitude,
                    ],
                ]
            )
        raise ValueError("Topocentric convention must be 'enu' or 'sez'.")

    def to_topocentric(
        self,
        position_itrs,
        velocity_itrs=None,
        acceleration_itrs=None,
        *,
        latitude_rad,
        longitude_rad,
        altitude=0.0,
        convention="enu",
    ):
        """Convert ITRF kinematics to a local topocentric frame."""
        position_itrs = self._check_vector_shape(position_itrs, "position_itrs")
        rotation = self.topocentric_matrix(latitude_rad, longitude_rad, convention)
        origin = self.geodetic_to_itrs(latitude_rad, longitude_rad, altitude)
        position = (position_itrs - origin) @ rotation.T
        velocity = (
            None
            if velocity_itrs is None
            else self._check_vector_shape(velocity_itrs, "velocity_itrs") @ rotation.T
        )
        acceleration = (
            None
            if acceleration_itrs is None
            else self._check_vector_shape(acceleration_itrs, "acceleration_itrs")
            @ rotation.T
        )
        return position, velocity, acceleration

    def from_topocentric(
        self,
        position,
        velocity=None,
        acceleration=None,
        *,
        latitude_rad,
        longitude_rad,
        altitude=0.0,
        convention="enu",
    ):
        """Convert local topocentric kinematics to ITRF."""
        position = self._check_vector_shape(position, "position")
        rotation = self.topocentric_matrix(latitude_rad, longitude_rad, convention)
        origin = self.geodetic_to_itrs(latitude_rad, longitude_rad, altitude)
        position_itrs = position @ rotation + origin
        velocity_itrs = (
            None
            if velocity is None
            else self._check_vector_shape(velocity, "velocity") @ rotation
        )
        acceleration_itrs = (
            None
            if acceleration is None
            else self._check_vector_shape(acceleration, "acceleration") @ rotation
        )
        return position_itrs, velocity_itrs, acceleration_itrs

    @staticmethod
    def launch_matrix(azimuth_rad):
        """Return the ENU-to-downrange/crossrange/up rotation matrix."""
        sine, cosine = np.sin(float(azimuth_rad)), np.cos(float(azimuth_rad))
        return np.array(
            [
                [sine, cosine, 0.0],
                [cosine, -sine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def to_launch_frame(
        self,
        position_itrs,
        velocity_itrs=None,
        acceleration_itrs=None,
        *,
        latitude_rad,
        longitude_rad,
        altitude=0.0,
        azimuth_rad=0.0,
    ):
        """Convert ITRF kinematics to downrange/crossrange/up coordinates."""
        values = self.to_topocentric(
            position_itrs,
            velocity_itrs,
            acceleration_itrs,
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
            altitude=altitude,
            convention="enu",
        )
        rotation = self.launch_matrix(azimuth_rad)
        return tuple(None if value is None else value @ rotation.T for value in values)

    def from_launch_frame(
        self,
        position,
        velocity=None,
        acceleration=None,
        *,
        latitude_rad,
        longitude_rad,
        altitude=0.0,
        azimuth_rad=0.0,
    ):
        """Convert downrange/crossrange/up kinematics to ITRF."""
        rotation = self.launch_matrix(azimuth_rad)
        values = tuple(
            None
            if value is None
            else self._check_vector_shape(value, "launch vector") @ rotation
            for value in (position, velocity, acceleration)
        )
        return self.from_topocentric(
            *values,
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
            altitude=altitude,
            convention="enu",
        )

    @staticmethod
    def rtn_matrix(reference_position, reference_velocity):
        """Return the GCRF-to-RTN free-vector rotation matrix."""
        reference_position = np.asarray(reference_position, dtype=float)
        reference_velocity = np.asarray(reference_velocity, dtype=float)
        radius = np.linalg.norm(reference_position)
        momentum = np.cross(reference_position, reference_velocity)
        momentum_norm = np.linalg.norm(momentum)
        if radius == 0.0 or momentum_norm == 0.0:
            raise ValueError("RTN axes require nonzero radius and angular momentum.")
        radial = reference_position / radius
        normal = momentum / momentum_norm
        transverse = np.cross(normal, radial)
        return np.vstack((radial, transverse, normal))

    def to_rtn(
        self,
        reference_position,
        reference_velocity,
        reference_acceleration,
        position,
        velocity=None,
        acceleration=None,
    ):
        """Convert target GCRF kinematics to relative RTN coordinates."""
        reference_position = np.asarray(reference_position, dtype=float)
        reference_velocity = np.asarray(reference_velocity, dtype=float)
        reference_acceleration = np.asarray(reference_acceleration, dtype=float)
        position = np.asarray(position, dtype=float)
        rotation = self.rtn_matrix(reference_position, reference_velocity)
        radius = np.linalg.norm(reference_position)
        momentum = np.cross(reference_position, reference_velocity)
        angular_velocity = momentum / radius**2
        relative_position = position - reference_position
        position_rtn = rotation @ relative_position
        velocity_rtn = None
        rotating_velocity = None
        if velocity is not None:
            relative_velocity = np.asarray(velocity, dtype=float) - reference_velocity
            rotating_velocity = relative_velocity - np.cross(
                angular_velocity, relative_position
            )
            velocity_rtn = rotation @ rotating_velocity
        acceleration_rtn = None
        if acceleration is not None:
            if rotating_velocity is None:
                raise ValueError("RTN acceleration conversion also requires velocity.")
            relative_acceleration = (
                np.asarray(acceleration, dtype=float) - reference_acceleration
            )
            angular_acceleration = (
                np.cross(reference_position, reference_acceleration) / radius**2
                - 2.0
                * np.dot(reference_position, reference_velocity)
                / radius**2
                * angular_velocity
            )
            rotating_acceleration = (
                relative_acceleration
                - np.cross(angular_acceleration, relative_position)
                - np.cross(
                    angular_velocity,
                    np.cross(angular_velocity, relative_position),
                )
                - 2.0 * np.cross(angular_velocity, rotating_velocity)
            )
            acceleration_rtn = rotation @ rotating_acceleration
        return position_rtn, velocity_rtn, acceleration_rtn

    def from_rtn(
        self,
        reference_position,
        reference_velocity,
        reference_acceleration,
        position,
        velocity=None,
        acceleration=None,
    ):
        """Convert relative RTN kinematics to target GCRF coordinates."""
        reference_position = np.asarray(reference_position, dtype=float)
        reference_velocity = np.asarray(reference_velocity, dtype=float)
        reference_acceleration = np.asarray(reference_acceleration, dtype=float)
        rotation = self.rtn_matrix(reference_position, reference_velocity)
        radius = np.linalg.norm(reference_position)
        angular_velocity = np.cross(reference_position, reference_velocity) / radius**2
        relative_position = rotation.T @ np.asarray(position, dtype=float)
        target_position = reference_position + relative_position
        target_velocity = None
        rotating_velocity = None
        if velocity is not None:
            rotating_velocity = rotation.T @ np.asarray(velocity, dtype=float)
            target_velocity = (
                reference_velocity
                + rotating_velocity
                + np.cross(angular_velocity, relative_position)
            )
        target_acceleration = None
        if acceleration is not None:
            if rotating_velocity is None:
                raise ValueError("RTN acceleration conversion also requires velocity.")
            angular_acceleration = (
                np.cross(reference_position, reference_acceleration) / radius**2
                - 2.0
                * np.dot(reference_position, reference_velocity)
                / radius**2
                * angular_velocity
            )
            rotating_acceleration = rotation.T @ np.asarray(acceleration, dtype=float)
            target_acceleration = (
                reference_acceleration
                + rotating_acceleration
                + np.cross(angular_acceleration, relative_position)
                + np.cross(
                    angular_velocity,
                    np.cross(angular_velocity, relative_position),
                )
                + 2.0 * np.cross(angular_velocity, rotating_velocity)
            )
        return target_position, target_velocity, target_acceleration

    def orientation_kinematics(self, epoch, step=0.25):
        """Return inertial-to-fixed rotation and its first two derivatives."""
        matrix = self.orientation_matrix(epoch)
        if self.integration_frame == self.fixed_frame:
            return matrix, np.zeros((3, 3)), np.zeros((3, 3))
        before = self.orientation_matrix(epoch - step)
        after = self.orientation_matrix(epoch + step)
        return (
            matrix,
            (after - before) / (2.0 * step),
            (after - 2.0 * matrix + before) / (step**2),
        )

    def inertial_angular_velocity(self, epoch):
        """Return fixed-frame angular velocity relative to inertial, in inertial axes."""
        matrix, matrix_dot, _ = self.orientation_kinematics(epoch)
        cross_matrix = -matrix.T @ matrix_dot
        return np.array(
            [
                cross_matrix[2, 1],
                cross_matrix[0, 2],
                cross_matrix[1, 0],
            ]
        )

    def _to_ecef_coordinates(self, epoch, position, velocity=None, acceleration=None):
        return self.transform_kinematics(
            epoch,
            position,
            velocity,
            acceleration,
            source=self.integration_frame,
            target=self.fixed_frame,
        )

    def _to_eci_coordinates(self, epoch, position, velocity=None, acceleration=None):
        return self.transform_kinematics(
            epoch,
            position,
            velocity,
            acceleration,
            source=self.fixed_frame,
            target=self.integration_frame,
        )


class FlatEarthDatum(EarthDatum):
    """Non-rotating local datum used by traditional low_altitude flights.

    Fixed and inertial coordinates are the same launch-local ENU frame.
    Somigliana apparent gravity is allowed only with this datum.
    """

    integration_frame = ReferenceFrame.FLAT_EARTH
    fixed_frame = ReferenceFrame.FLAT_EARTH
    supports_apparent_gravity = True
    is_geocentric = False

    def orientation_matrix(self, epoch):
        del epoch
        return np.eye(3)


class SimpleDatum(EarthDatum):
    """Earth datum using only Earth Rotation Angle for ECI/ECEF conversion."""

    requires_absolute_epoch = True

    def orientation_matrix(self, epoch):
        angle = epoch.earth_rotation_angle
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])

    def orientation_kinematics(self, epoch, step=None):
        """Return the ERA rotation and its analytic time derivatives."""
        del step
        matrix = self.orientation_matrix(epoch)
        angle = epoch.earth_rotation_angle
        cosine = np.cos(angle)
        sine = np.sin(angle)
        omega = self.angular_velocity * (1.0 - epoch.length_of_day / 86400.0)
        matrix_dot = omega * np.array(
            [
                [-sine, cosine, 0.0],
                [-cosine, -sine, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        matrix_ddot = omega**2 * np.array(
            [
                [-cosine, -sine, 0.0],
                [sine, -cosine, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        return matrix, matrix_dot, matrix_ddot


class Datum(SimpleDatum):
    """IAU 2000/2006 GCRF/ITRF datum with implicit IERS orientation data.

    The implementation uses the IERS 2010 X/Y/s series, Earth Rotation Angle
    and polar motion. Matrix time derivatives are evaluated numerically so
    position, velocity and acceleration transformations share one coherent
    kinematic definition.
    """

    def __init__(self, *args, cache=2048, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_precession_nutation_coefficients()
        if cache:
            self._orientation_kinematics_cached = lru_cache(maxsize=int(cache))(
                self._orientation_kinematics_uncached
            )
        else:
            self._orientation_kinematics_cached = self._orientation_kinematics_uncached

    def _load_precession_nutation_coefficients(self):
        coefficient_directory = resources.files("rocketpy.environment").joinpath("data")

        def load(prefix, filename):
            with np.load(coefficient_directory.joinpath(filename)) as data:
                return tuple(
                    np.ascontiguousarray(data[f"{prefix}_coeff_{index}"])
                    for index in range(5)
                ) + tuple(
                    np.ascontiguousarray(data[f"{prefix}_astro_{index}"])
                    for index in range(5)
                )

        self._x_coefficients = load("x", "presnut_tab52a.npz")
        self._y_coefficients = load("y", "presnut_tab52b.npz")
        self._s_coefficients = load("s", "presnut_tab52d.npz")

    def orientation_matrix(self, epoch):
        epoch.require_absolute("Full GCRF/ITRF transformation")
        return self._orientation_kinematics_cached(float(epoch.seconds))[0]

    def orientation_kinematics(self, epoch, step=None):
        """Return the IAU rotation and analytic Earth-rotation derivatives.

        Precession, nutation and polar-motion rates are negligible over a
        numerical integration step and are treated as locally constant.
        Earth Rotation Angle derivatives include the observed excess length
        of day.
        """
        del step
        epoch.require_absolute("Full GCRF/ITRF transformation")
        return self._orientation_kinematics_cached(float(epoch.seconds))

    def _orientation_kinematics_uncached(self, epoch_seconds):
        epoch = Epoch(float(epoch_seconds), is_absolute=True)
        xp, yp, dx, dy = epoch.earth_orientation
        tio_locator = -4.7e-05 * np.pi / (180.0 * 3600.0) * epoch.tt_century
        cx, sx = np.cos(xp), np.sin(xp)
        cy, sy = np.cos(yp), np.sin(yp)
        cs, ss = np.cos(tio_locator), np.sin(tio_locator)
        polar_motion = np.array(
            [
                [cx * cs, -cy * ss + sy * sx * cs, -sy * ss - cy * sx * cs],
                [cx * ss, cy * cs + sy * sx * ss, sy * cs - cy * sx * ss],
                [sx, -sy * cx, cy * cx],
            ]
        )

        x, y, cio = _compute_precession_nutation(
            epoch.tt_century,
            *self._x_coefficients,
            *self._y_coefficients,
            *self._s_coefficients,
        )
        x += dx
        y += dy
        cio -= 0.5 * x * y
        denominator = 1.0 - x * x - y * y
        if denominator <= 0:
            raise ValueError("Invalid IAU precession-nutation coordinates.")
        coefficient = 1.0 / (
            1.0 + np.cos(np.arctan(np.sqrt((x * x + y * y) / denominator)))
        )
        celestial_pole = np.array(
            [
                [1 - coefficient * x * x, -coefficient * x * y, x],
                [-coefficient * x * y, 1 - coefficient * y * y, y],
                [-x, -y, 1 - coefficient * (x * x + y * y)],
            ]
        )
        cosine, sine = np.cos(cio), np.sin(cio)
        cio_rotation = np.array(
            [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        precession_nutation = celestial_pole @ cio_rotation

        era = epoch.earth_rotation_angle
        cosine, sine = np.cos(era), np.sin(era)
        fixed_to_inertial_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        omega = self.angular_velocity * (1.0 - epoch.length_of_day / 86400.0)
        rotation_dot = omega * np.array(
            [[-sine, -cosine, 0.0], [cosine, -sine, 0.0], [0.0, 0.0, 0.0]]
        )
        rotation_ddot = omega**2 * np.array(
            [[-cosine, sine, 0.0], [-sine, -cosine, 0.0], [0.0, 0.0, 0.0]]
        )
        left = polar_motion.T
        right = precession_nutation.T
        return (
            left @ fixed_to_inertial_rotation.T @ right,
            left @ rotation_dot.T @ right,
            left @ rotation_ddot.T @ right,
        )


WGS84 = Datum()
GRS80 = Datum(
    name="GRS80",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257222101,
    angular_velocity=7.292115e-5,
    gravitational_parameter=3.986005e14,
)
WGS72 = Datum(
    name="WGS72",
    semi_major_axis=6378135.0,
    flattening=1.0 / 298.26,
    angular_velocity=7.2921151467e-5,
    gravitational_parameter=3.986005e14,
)
SIMPLE_WGS84 = SimpleDatum()
FLAT_WGS84 = FlatEarthDatum()


def transform_kinematics(
    epoch: Epoch,
    position: np.ndarray,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    *,
    source: ReferenceFrame | str,
    target: ReferenceFrame | str,
    datum: EarthDatum = WGS84,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Transform position, velocity and acceleration through ``datum``."""
    source = ReferenceFrame.coerce(source)
    target = ReferenceFrame.coerce(target)

    if ReferenceFrame.TEME in (source, target):
        if source == ReferenceFrame.TEME and target != ReferenceFrame.GCRF:
            gcrf = transform_kinematics(
                epoch,
                position,
                velocity,
                acceleration,
                source=ReferenceFrame.TEME,
                target=ReferenceFrame.GCRF,
                datum=datum,
            )
            return transform_kinematics(
                epoch,
                *gcrf,
                source=ReferenceFrame.GCRF,
                target=target,
                datum=datum,
            )
        if target == ReferenceFrame.TEME and source != ReferenceFrame.GCRF:
            gcrf = transform_kinematics(
                epoch,
                position,
                velocity,
                acceleration,
                source=source,
                target=ReferenceFrame.GCRF,
                datum=datum,
            )
            return transform_kinematics(
                epoch,
                *gcrf,
                source=ReferenceFrame.GCRF,
                target=ReferenceFrame.TEME,
                datum=datum,
            )
        angle = epoch.greenwich_mean_sidereal_time - epoch.earth_rotation_angle
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        if source == ReferenceFrame.TEME:
            rotation = rotation.T
        output = _rotate_free_kinematics(
            position,
            np.zeros(3, dtype=np.float64) if velocity is None else velocity,
            np.zeros(3, dtype=np.float64) if acceleration is None else acceleration,
            np.require(rotation, dtype=np.float64, requirements=["C", "W"]),
        )
        return (
            output[0],
            None if velocity is None else output[1],
            (None if acceleration is None else output[2]),
        )

    return datum.transform_kinematics(
        epoch,
        position,
        velocity,
        acceleration,
        source=source,
        target=target,
    )


def _transform_with_datum(
    datum,
    epoch,
    position,
    velocity=None,
    acceleration=None,
    *,
    source,
    target,
):
    source = ReferenceFrame.coerce(source)
    target = ReferenceFrame.coerce(target)
    position = np.asarray(position, dtype=float)
    velocity_array = None if velocity is None else np.asarray(velocity, dtype=float)
    acceleration_array = (
        None if acceleration is None else np.asarray(acceleration, dtype=float)
    )
    if source == target:
        return (
            position.copy(),
            _copy_optional(velocity_array),
            _copy_optional(acceleration_array),
        )
    if ReferenceFrame.FLAT_EARTH in (source, target):
        raise ValueError(
            "Flat-Earth coordinates need a launch-site origin and orientation "
            "and cannot be transformed as Earth-centered coordinates."
        )
    supported = {datum.integration_frame, datum.fixed_frame}
    if {source, target} != supported:
        raise ValueError(
            f"{type(datum).__name__} cannot transform {source.value} to "
            f"{target.value}; it owns {datum.integration_frame.value} and "
            f"{datum.fixed_frame.value}."
        )
    if datum.integration_frame == datum.fixed_frame:
        return (
            position.copy(),
            _copy_optional(velocity_array),
            _copy_optional(acceleration_array),
        )

    matrix, matrix_dot, matrix_ddot = datum.orientation_kinematics(epoch)

    if source == datum.integration_frame:
        position_out = matrix @ position
        velocity_out = None
        if velocity_array is not None:
            velocity_out = matrix @ velocity_array + matrix_dot @ position
        acceleration_out = None
        if acceleration_array is not None:
            inertial_velocity = (
                np.zeros(3) if velocity_array is None else velocity_array
            )
            acceleration_out = (
                matrix @ acceleration_array
                + 2.0 * matrix_dot @ inertial_velocity
                + matrix_ddot @ position
            )
        return position_out, velocity_out, acceleration_out

    transpose = matrix.T
    position_out = transpose @ position
    velocity_out = None
    inertial_velocity = np.zeros(3)
    if velocity_array is not None or acceleration_array is not None:
        fixed_velocity = np.zeros(3) if velocity_array is None else velocity_array
        inertial_velocity = transpose @ (fixed_velocity - matrix_dot @ position_out)
        if velocity_array is not None:
            velocity_out = inertial_velocity
    acceleration_out = None
    if acceleration_array is not None:
        acceleration_out = transpose @ (
            acceleration_array
            - 2.0 * matrix_dot @ inertial_velocity
            - matrix_ddot @ position_out
        )
    return position_out, velocity_out, acceleration_out


@numbify(
    signature=_FRAME_KERNEL_SIGNATURE,
    nopython=True,
    cache=True,
)
def _gcrf_to_itrf_kernel(
    position,
    velocity,
    acceleration,
    rotation,
    omega,
    has_velocity,
    has_acceleration,
):
    """Numerical GCRF-to-ITRF kernel with rotating-frame derivatives."""
    position_out = rotation @ position
    velocity_out = rotation @ velocity - np.cross(omega, position_out)
    acceleration_out = np.zeros(3)
    if has_acceleration:
        rotating_velocity = (
            velocity_out if has_velocity else -np.cross(omega, position_out)
        )
        acceleration_out = (
            rotation @ acceleration
            - 2.0 * np.cross(omega, rotating_velocity)
            - np.cross(omega, np.cross(omega, position_out))
        )
    return position_out, velocity_out, acceleration_out


@numbify(
    signature=_FRAME_KERNEL_SIGNATURE,
    nopython=True,
    cache=True,
)
def _itrf_to_gcrf_kernel(
    position,
    velocity,
    acceleration,
    rotation,
    omega,
    has_velocity,
    has_acceleration,
):
    """Numerical ITRF-to-GCRF kernel with rotating-frame derivatives."""
    rotation_transpose = np.ascontiguousarray(rotation.T)
    velocity_for_acceleration = velocity if has_velocity else np.zeros(3)
    position_out = rotation_transpose @ position
    velocity_out = rotation_transpose @ (velocity + np.cross(omega, position))
    acceleration_out = np.zeros(3)
    if has_acceleration:
        acceleration_out = rotation_transpose @ (
            acceleration
            + 2.0 * np.cross(omega, velocity_for_acceleration)
            + np.cross(omega, np.cross(omega, position))
        )
    return position_out, velocity_out, acceleration_out


@numbify(
    signature=_FREE_ROTATION_SIGNATURE,
    nopython=True,
    cache=True,
)
def _rotate_free_kinematics(position, velocity, acceleration, rotation):
    """Rotate three free vectors with a common orthogonal matrix."""
    return rotation @ position, rotation @ velocity, rotation @ acceleration


def _copy_optional(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else value.copy()


def quaternion_from_matrix(matrix) -> np.ndarray:
    """Return a scalar-first unit quaternion for a proper rotation matrix."""
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


_PRECESSION_NUTATION_SIGNATURE = (
    "float64[::1](float64, " + ", ".join(["float64[:, ::1]"] * 30) + ")"
)


@numbify(
    signature=_PRECESSION_NUTATION_SIGNATURE,
    nopython=True,
    cache=True,
)
def _compute_precession_nutation(
    centuries,
    xc0,
    xc1,
    xc2,
    xc3,
    xc4,
    xa0,
    xa1,
    xa2,
    xa3,
    xa4,
    yc0,
    yc1,
    yc2,
    yc3,
    yc4,
    ya0,
    ya1,
    ya2,
    ya3,
    ya4,
    sc0,
    sc1,
    sc2,
    sc3,
    sc4,
    sa0,
    sa1,
    sa2,
    sa3,
    sa4,
):
    """Evaluate the IERS 2010 X/Y/s precession-nutation series.

    Based on IERS Conventions 2010, tables 5.2a, 5.2b and 5.2d.
    """
    t = centuries
    arguments = np.array(
        [
            485868.249036
            + 1717915923.2178 * t
            + 31.8792 * t**2
            + 0.051635 * t**3
            - 0.00024470 * t**4,
            1287104.79305
            + 129596581.0481 * t
            - 0.5532 * t**2
            + 0.000136 * t**3
            - 0.00001149 * t**4,
            335779.526232
            + 1739527262.8478 * t
            - 12.7512 * t**2
            - 0.001037 * t**3
            + 0.00000417 * t**4,
            1072260.70369
            + 1602961601.2090 * t
            - 6.3706 * t**2
            + 0.006593 * t**3
            - 0.00003169 * t**4,
            450160.398036
            - 6962890.5431 * t
            + 7.4722 * t**2
            + 0.007702 * t**3
            - 0.00005939 * t**4,
            (252.250905494 + 149472.6746358 * t) * 3600.0,
            (181.979800853 + 58517.8156748 * t) * 3600.0,
            (100.466448494 + 35999.3728521 * t) * 3600.0,
            (355.433274605 + 19140.299314 * t) * 3600.0,
            (34.351483900 + 3034.90567464 * t) * 3600.0,
            (50.0774713998 + 1222.11379404 * t) * 3600.0,
            (314.055005137 + 428.466998313 * t) * 3600.0,
            (304.348665499 + 218.486200208 * t) * 3600.0,
            (1.39697137214 * t + 0.0003086 * t**2) * 3600.0,
        ]
    )
    arguments = (arguments % (360.0 * 3600.0)) * (np.pi / (180.0 * 3600.0))

    x_value = (
        -16617.0
        + 2004191898.0 * t
        - 429782.9 * t**2
        - 198618.34 * t**3
        + 7.578 * t**4
        + 5.9285 * t**5
    )
    y_value = (
        -6951.0
        - 25896.0 * t
        - 22407274.7 * t**2
        + 1900.59 * t**3
        + 1112.526 * t**4
        + 0.1358 * t**5
    )
    s_value = (
        94.0
        + 3808.65 * t
        - 122.68 * t**2
        - 72574.11 * t**3
        + 27.98 * t**4
        + 15.62 * t**5
    )
    x_coefficients = (xc0, xc1, xc2, xc3, xc4)
    x_arguments = (xa0, xa1, xa2, xa3, xa4)
    y_coefficients = (yc0, yc1, yc2, yc3, yc4)
    y_arguments = (ya0, ya1, ya2, ya3, ya4)
    s_coefficients = (sc0, sc1, sc2, sc3, sc4)
    s_arguments = (sa0, sa1, sa2, sa3, sa4)
    for power in range(5):
        factor = t**power
        for index in range(len(x_coefficients[power])):
            phase = 0.0
            for argument in range(14):
                phase += x_arguments[power][index, argument] * arguments[argument]
            x_value += (
                x_coefficients[power][index, 0] * math.sin(phase)
                + x_coefficients[power][index, 1] * math.cos(phase)
            ) * factor
        for index in range(len(y_coefficients[power])):
            phase = 0.0
            for argument in range(14):
                phase += y_arguments[power][index, argument] * arguments[argument]
            y_value += (
                y_coefficients[power][index, 0] * math.sin(phase)
                + y_coefficients[power][index, 1] * math.cos(phase)
            ) * factor
        for index in range(len(s_coefficients[power])):
            phase = 0.0
            for argument in range(14):
                phase += s_arguments[power][index, argument] * arguments[argument]
            s_value += (
                s_coefficients[power][index, 0] * math.sin(phase)
                + s_coefficients[power][index, 1] * math.cos(phase)
            ) * factor
    return np.array([x_value, y_value, s_value]) * (
        np.pi / (180.0 * 3600.0) / 1_000_000.0
    )
