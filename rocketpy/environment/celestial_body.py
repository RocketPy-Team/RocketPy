"""Celestial-body definitions and deterministic ephemeris providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rocketpy.mathutils.epoch import Epoch


ASTRONOMICAL_UNIT = 149597870700.0


class AnalyticalEphemeris:
    """Low-fidelity, dependency-free geocentric Sun and Moon ephemerides."""

    def position(self, name: str, epoch: Epoch) -> np.ndarray:
        epoch.require_absolute("Analytical ephemeris evaluation")
        days = epoch.jd_tdb - 2451545.0
        name = name.lower()
        if name == "sun":
            mean_longitude = np.radians((280.460 + 0.9856474 * days) % 360.0)
            mean_anomaly = np.radians((357.528 + 0.9856003 * days) % 360.0)
            ecliptic_longitude = mean_longitude + np.radians(
                1.915 * np.sin(mean_anomaly) + 0.020 * np.sin(2 * mean_anomaly)
            )
            obliquity = np.radians(23.439 - 0.0000004 * days)
            distance = ASTRONOMICAL_UNIT * (
                1.00014
                - 0.01671 * np.cos(mean_anomaly)
                - 0.00014 * np.cos(2 * mean_anomaly)
            )
            return distance * np.array(
                [
                    np.cos(ecliptic_longitude),
                    np.cos(obliquity) * np.sin(ecliptic_longitude),
                    np.sin(obliquity) * np.sin(ecliptic_longitude),
                ]
            )
        if name == "moon":
            # Circular inclined approximation for deterministic low-fidelity
            # perturbation studies. High-fidelity work should use SPICE.
            longitude = np.radians((218.316 + 13.176396 * days) % 360.0)
            node = np.radians((125.045 - 0.0529538 * days) % 360.0)
            inclination = np.radians(5.145)
            argument = longitude - node
            distance = 384400000.0
            return distance * np.array(
                [
                    np.cos(node) * np.cos(argument)
                    - np.sin(node) * np.sin(argument) * np.cos(inclination),
                    np.sin(node) * np.cos(argument)
                    + np.cos(node) * np.sin(argument) * np.cos(inclination),
                    np.sin(argument) * np.sin(inclination),
                ]
            )
        raise ValueError(
            f"The analytical ephemeris does not support {name!r}; use SpiceEphemeris."
        )


class SpiceEphemeris:
    """SPICE ephemeris using explicitly supplied kernel files."""

    def __init__(self, *kernels):
        if not kernels:
            raise ValueError("At least one SPICE kernel path is required.")
        self.kernels = tuple(Path(kernel) for kernel in kernels)
        missing = [str(kernel) for kernel in self.kernels if not kernel.exists()]
        if missing:
            raise FileNotFoundError(f"Missing SPICE kernels: {', '.join(missing)}")
        self._loaded = False

    def position(self, name: str, epoch: Epoch) -> np.ndarray:
        epoch.require_absolute("SPICE ephemeris evaluation")
        try:
            import spiceypy as spice
        except ImportError as exc:
            raise ImportError(
                "SpiceEphemeris requires the 'spiceypy' package."
            ) from exc
        if not self._loaded:
            for kernel in self.kernels:
                spice.furnsh(str(kernel))
            self._loaded = True
        ephemeris_time = (epoch.jd_tdb - 2451545.0) * 86400.0
        state, _ = spice.spkezr(name, ephemeris_time, "J2000", "NONE", "EARTH")
        return np.asarray(state[:3], dtype=float) * 1000.0


@dataclass(frozen=True)
class CelestialBody:
    """A celestial body contributing gravity or radiation pressure."""

    name: str
    gravitational_parameter: float
    radius: float | None = None
    albedo: float = 0.0
    solar_pressure_at_reference: float | None = None
    ephemeris: object = field(default_factory=AnalyticalEphemeris)

    def position(self, epoch: Epoch) -> np.ndarray:
        """Return geocentric GCRF position in meters."""
        return np.asarray(self.ephemeris.position(self.name, epoch), dtype=float)

    @classmethod
    def builtin(cls, name: str, ephemeris=None) -> "CelestialBody":
        """Create a built-in solar-system body."""
        normalized = name.lower()
        data = {
            "sun": (1.32712440042e20, 696340e3, 0.0, 4.56e-6),
            "moon": (4.902800118e12, 1737.4e3, 0.11, None),
            "mercury": (2.2031868551e13, 2439.7e3, 0.088, None),
            "venus": (3.24858592e14, 6051.8e3, 0.76, None),
            "mars barycenter": (4.2828375816e13, 3389.5e3, 0.25, None),
            "jupiter barycenter": (1.267127641e17, 69911e3, 0.503, None),
            "saturn barycenter": (3.799405848418e16, 58232e3, 0.342, None),
            "uranus barycenter": (5.7945564e15, 25362e3, 0.300, None),
            "neptune barycenter": (6.83652710058e15, 24622e3, 0.290, None),
            "pluto barycenter": (9.755e11, 1188.3e3, 0.72, None),
        }
        try:
            mu, radius, albedo, pressure = data[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown built-in celestial body: {name!r}.") from exc
        return cls(
            name=normalized,
            gravitational_parameter=mu,
            radius=radius,
            albedo=albedo,
            solar_pressure_at_reference=pressure,
            ephemeris=ephemeris or AnalyticalEphemeris(),
        )
