"""Typed state values shared by local and Earth-centered Flight dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .epoch import Epoch
from .reference_frame import (
    WGS84,
    ReferenceFrame,
    transform_kinematics,
)


@dataclass(frozen=True)
class FlightState:
    """A RocketPy 13-component rigid-body state.

    Position is in meters, velocity in m/s, quaternion is scalar-first, and
    angular velocity is expressed in the rocket body frame in rad/s.
    """

    epoch: Epoch
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    frame: ReferenceFrame = ReferenceFrame.FLAT_EARTH
    elapsed_time: float = 0.0

    def __post_init__(self):
        for name, value, size in (
            ("position", self.position, 3),
            ("velocity", self.velocity, 3),
            ("quaternion", self.quaternion, 4),
            ("angular_velocity", self.angular_velocity, 3),
        ):
            array = np.array(value, dtype=float, copy=True)
            if array.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},).")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "frame", ReferenceFrame.coerce(self.frame))
        object.__setattr__(self, "elapsed_time", float(self.elapsed_time))

    @classmethod
    def cartesian(
        cls,
        *,
        epoch: Epoch,
        position,
        velocity,
        frame: ReferenceFrame | str,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        elapsed_time=0.0,
    ) -> "FlightState":
        """Construct a state from Cartesian translational coordinates."""
        return cls(
            epoch=epoch,
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            frame=ReferenceFrame.coerce(frame),
            elapsed_time=elapsed_time,
        )

    @classmethod
    def geodetic(
        cls,
        *,
        epoch: Epoch,
        latitude,
        longitude,
        altitude,
        velocity_enu=(0.0, 0.0, 0.0),
        frame: ReferenceFrame | str = ReferenceFrame.GCRF,
        datum=WGS84,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        elapsed_time=0.0,
    ) -> "FlightState":
        """Construct an Earth-centered state from geodetic coordinates.

        Latitude and longitude are in degrees. ``velocity_enu`` is relative to
        the rotating Earth in local east/north/up axes.
        """
        target = ReferenceFrame.coerce(frame)
        if target not in (ReferenceFrame.ITRF, ReferenceFrame.GCRF):
            raise ValueError("Geodetic states can be created in ITRF or GCRF.")
        latitude_rad = np.radians(float(latitude))
        longitude_rad = np.radians(float(longitude))
        position_itrf = datum.geodetic_to_itrs(
            latitude_rad, longitude_rad, float(altitude)
        )
        _, velocity_itrf, _ = datum.from_topocentric(
            np.zeros(3),
            np.asarray(velocity_enu, dtype=float),
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
            altitude=float(altitude),
        )
        position, velocity, _ = transform_kinematics(
            epoch,
            position_itrf,
            velocity_itrf,
            source=ReferenceFrame.ITRF,
            target=target,
            datum=datum,
        )
        return cls.cartesian(
            epoch=epoch,
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            frame=target,
            elapsed_time=elapsed_time,
        )

    def to_array(self) -> np.ndarray:
        """Return RocketPy's established 13-component state array."""
        return np.concatenate(
            (self.position, self.velocity, self.quaternion, self.angular_velocity)
        )

    @classmethod
    def from_array(
        cls,
        values,
        *,
        epoch: Epoch,
        frame: ReferenceFrame | str,
        elapsed_time=0.0,
    ) -> "FlightState":
        """Construct a typed state from a 13-component RocketPy state."""
        values = np.asarray(values, dtype=float)
        if values.shape != (13,):
            raise ValueError("RocketPy flight states must contain 13 values.")
        return cls(
            epoch=epoch,
            position=values[:3],
            velocity=values[3:6],
            quaternion=values[6:10],
            angular_velocity=values[10:13],
            frame=ReferenceFrame.coerce(frame),
            elapsed_time=elapsed_time,
        )
