"""Reference frames and kinematic transformations used by RocketPy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


@dataclass(frozen=True)
class EarthDatum:
    """Earth ellipsoid and rotation constants, in SI units."""

    name: str = "WGS84"
    semi_major_axis: float = 6378137.0
    flattening: float = 1.0 / 298.257223563
    angular_velocity: float = 7.2921150e-5
    gravitational_parameter: float = 3.986004418e14

    @property
    def semi_minor_axis(self) -> float:
        return self.semi_major_axis * (1.0 - self.flattening)

    @property
    def eccentricity_squared(self) -> float:
        return self.flattening * (2.0 - self.flattening)

    def geodetic_to_itrs(
        self, latitude_rad: float, longitude_rad: float, altitude: float
    ) -> np.ndarray:
        """Convert geodetic coordinates to an ITRF position in meters."""
        sin_lat = np.sin(latitude_rad)
        cos_lat = np.cos(latitude_rad)
        radius = self.semi_major_axis / np.sqrt(
            1.0 - self.eccentricity_squared * sin_lat**2
        )
        return np.array(
            [
                (radius + altitude) * cos_lat * np.cos(longitude_rad),
                (radius + altitude) * cos_lat * np.sin(longitude_rad),
                (radius * (1.0 - self.eccentricity_squared) + altitude) * sin_lat,
            ],
            dtype=float,
        )

    def itrs_to_geodetic(self, position: np.ndarray) -> tuple[float, float, float]:
        """Convert an ITRF position to latitude, longitude and altitude."""
        x, y, z = np.asarray(position, dtype=float)
        longitude = float(np.arctan2(y, x))
        horizontal = float(np.hypot(x, y))
        latitude = float(np.arctan2(z, horizontal * (1.0 - self.eccentricity_squared)))
        altitude = 0.0
        for _ in range(12):
            sin_lat = np.sin(latitude)
            radius = self.semi_major_axis / np.sqrt(
                1.0 - self.eccentricity_squared * sin_lat**2
            )
            cos_lat = np.cos(latitude)
            if abs(cos_lat) < 1e-14:
                altitude = abs(z) - self.semi_minor_axis
                break
            altitude = horizontal / cos_lat - radius
            next_latitude = np.arctan2(
                z,
                horizontal
                * (1.0 - self.eccentricity_squared * radius / (radius + altitude)),
            )
            if abs(next_latitude - latitude) < 1e-13:
                latitude = float(next_latitude)
                break
            latitude = float(next_latitude)
        return latitude, longitude, float(altitude)


WGS84 = EarthDatum()


def _rotation_gcrf_to_itrf(epoch: Epoch) -> np.ndarray:
    angle = epoch.earth_rotation_angle
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])


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
    """Transform position, velocity and acceleration between built-in frames.

    GCRF/ITRF transformations currently include Earth rotation and its first
    two kinematic derivatives. Precession, nutation and polar motion are
    delegated to future Earth-orientation providers.
    """
    source = ReferenceFrame.coerce(source)
    target = ReferenceFrame.coerce(target)
    position = np.require(position, dtype=np.float64, requirements=["C", "W"])
    velocity = (
        None
        if velocity is None
        else np.require(velocity, dtype=np.float64, requirements=["C", "W"])
    )
    acceleration = (
        None
        if acceleration is None
        else np.require(acceleration, dtype=np.float64, requirements=["C", "W"])
    )

    if source == target:
        return position.copy(), _copy_optional(velocity), _copy_optional(acceleration)

    if ReferenceFrame.FLAT_EARTH in (source, target):
        raise ValueError(
            "Flat-Earth coordinates need a launch-site origin and orientation "
            "and cannot be transformed as Earth-centered coordinates."
        )

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

    if {source, target} != {ReferenceFrame.GCRF, ReferenceFrame.ITRF}:
        raise ValueError(f"Unsupported frame transformation: {source} to {target}.")

    rotation = _rotation_gcrf_to_itrf(epoch)
    omega = np.array([0.0, 0.0, datum.angular_velocity])

    velocity_array = np.zeros(3, dtype=np.float64) if velocity is None else velocity
    acceleration_array = (
        np.zeros(3, dtype=np.float64) if acceleration is None else acceleration
    )
    kernel = (
        _gcrf_to_itrf_kernel if source == ReferenceFrame.GCRF else _itrf_to_gcrf_kernel
    )
    output = kernel(
        position,
        velocity_array,
        acceleration_array,
        np.require(rotation, dtype=np.float64, requirements=["C", "W"]),
        omega,
        velocity is not None,
        acceleration is not None,
    )
    return (
        output[0],
        None if velocity is None else output[1],
        (None if acceleration is None else output[2]),
    )


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


def enu_to_itrf_matrix(latitude_rad: float, longitude_rad: float) -> np.ndarray:
    """Return the rotation matrix from local ENU axes to ITRF axes."""
    sin_latitude, cos_latitude = np.sin(latitude_rad), np.cos(latitude_rad)
    sin_longitude, cos_longitude = np.sin(longitude_rad), np.cos(longitude_rad)
    return np.array(
        [
            [
                -sin_longitude,
                -sin_latitude * cos_longitude,
                cos_latitude * cos_longitude,
            ],
            [
                cos_longitude,
                -sin_latitude * sin_longitude,
                cos_latitude * sin_longitude,
            ],
            [0.0, cos_latitude, sin_latitude],
        ]
    )


def gcrf_to_rtn_matrix(position, velocity) -> np.ndarray:
    """Return a free-vector rotation from GCRF to radial/transverse/normal."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    radius = np.linalg.norm(position)
    momentum = np.cross(position, velocity)
    momentum_norm = np.linalg.norm(momentum)
    if radius == 0.0 or momentum_norm == 0.0:
        raise ValueError("RTN axes require nonzero radius and angular momentum.")
    radial = position / radius
    normal = momentum / momentum_norm
    transverse = np.cross(normal, radial)
    return np.vstack((radial, transverse, normal))


def itrf_to_topocentric(
    position,
    *,
    latitude_rad,
    longitude_rad,
    altitude=0.0,
    datum=WGS84,
):
    """Convert an ITRF position to launch-site East/North/Up coordinates."""
    site = datum.geodetic_to_itrs(latitude_rad, longitude_rad, altitude)
    return enu_to_itrf_matrix(latitude_rad, longitude_rad).T @ (
        np.asarray(position, dtype=float) - site
    )
