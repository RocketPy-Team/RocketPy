"""Third-body models and explicit ephemeris providers."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from urllib import request

import numpy as np

from rocketpy._logging import logger
from rocketpy.mathutils.compilation import numbify

ASTRONOMICAL_UNIT = 149597870700.0
DEFAULT_SPICE_KERNEL = "de440s.bsp"
SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets"
SPICE_KERNELS = {
    "de432s": "de432s.bsp",
    "de440": "de440.bsp",
    "de440s": "de440s.bsp",
    "de442": "de442.bsp",
    "de442s": "de442s.bsp",
}
DEFAULT_SPICE_KERNEL_URL = f"{SPICE_KERNEL_BASE_URL}/{DEFAULT_SPICE_KERNEL}"

_BODY_DATA = {
    "sun": {
        "gravitational_parameter": 1.32712440042e20,
        "radius": 696340e3,
        "pressure_function": lambda _, distance: (
            4.56e-6 * (ASTRONOMICAL_UNIT / distance) ** 2
        ),
    },
    "mercury": {"gravitational_parameter": 2.2031868551e13},
    "venus": {"gravitational_parameter": 3.24858592e14},
    "moon": {
        "gravitational_parameter": 4.902800118e12,
        "radius": 1737.4e3,
        "albedo": 0.11,
    },
    "mars barycenter": {"gravitational_parameter": 4.2828375816e13},
    "jupiter barycenter": {"gravitational_parameter": 1.267127641e17},
    "saturn barycenter": {"gravitational_parameter": 3.799405848418e16},
    "uranus barycenter": {"gravitational_parameter": 5.7945564e15},
    "neptune barycenter": {"gravitational_parameter": 6.83652710058e15},
    "pluto barycenter": {"gravitational_parameter": 9.755e11},
}


class SpiceEphemeris:
    """SPICE ephemeris backed by prepared local kernel files.

    A single kernel argument may be either a local path or one of the bundled
    download names: ``"de432s"``, ``"de440"``, ``"de440s"``, ``"de442"`` or
    ``"de442s"``. If no argument is supplied, the compact DE440s planetary
    kernel is selected. Named kernels are downloaded once into RocketPy's cache
    and loaded immediately. The download is logged and occurs while the
    ephemeris (normally :class:`Space`) is being constructed, never during
    :class:`Flight` integration.

    Positions are geometric J2000 geocentric vectors in metres, matching the
    GCRF approximation used by the current orbital propagator.
    """

    def __init__(
        self,
        *kernels,
        cache_directory=None,
        kernel_url=None,
    ):
        if len(kernels) == 1 and self._is_kernel_name(kernels[0]):
            filename = SPICE_KERNELS[str(kernels[0]).lower()]
            kernels = (
                self._prepare_named_kernel(filename, cache_directory, kernel_url),
            )
        if not kernels:
            kernels = (
                self._prepare_named_kernel(
                    DEFAULT_SPICE_KERNEL, cache_directory, kernel_url
                ),
            )
        self.kernels = tuple(Path(kernel) for kernel in kernels)
        missing = [str(kernel) for kernel in self.kernels if not kernel.exists()]
        if missing:
            raise FileNotFoundError(f"Missing SPICE kernels: {', '.join(missing)}")
        self._loaded = False
        self._load()

    @staticmethod
    def _cache_directory(cache_directory=None):
        if cache_directory is not None:
            return Path(cache_directory).expanduser()
        configured = os.environ.get("ROCKETPY_CACHE_DIR")
        if configured:
            return Path(configured).expanduser()
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        return base / "rocketpy"

    @staticmethod
    def _is_kernel_name(value):
        return isinstance(value, str) and value.lower() in SPICE_KERNELS

    @classmethod
    def _prepare_named_kernel(cls, filename, cache_directory, kernel_url=None):
        cache = cls._cache_directory(cache_directory)
        cache.mkdir(parents=True, exist_ok=True)
        kernel = cache / filename
        if kernel.is_file() and kernel.stat().st_size > 0:
            logger.info("Using cached SPICE kernel %s", kernel)
            return kernel

        kernel_url = kernel_url or f"{SPICE_KERNEL_BASE_URL}/{filename}"
        temporary = kernel.with_suffix(f"{kernel.suffix}.part")
        logger.info("Downloading SPICE kernel from %s to %s", kernel_url, kernel)
        try:
            request.urlretrieve(kernel_url, temporary)
            if temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded SPICE kernel is empty.")
            temporary.replace(kernel)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to prepare SPICE kernel from {kernel_url}: {exc}"
            ) from exc
        logger.info(
            "SPICE kernel ready at %s (%.1f MiB)",
            kernel,
            kernel.stat().st_size / (1024.0**2),
        )
        return kernel

    def _load(self):
        try:
            import spiceypy as spice
        except ImportError as exc:
            raise ImportError(
                "SpiceEphemeris requires the optional 'spiceypy' package."
            ) from exc
        if not self._loaded:
            for kernel in self.kernels:
                spice.furnsh(str(kernel))
            self._loaded = True
        return spice

    @lru_cache(maxsize=8192)
    def _position_tuple_at_jd(self, name, jd_tdb):
        spice = self._load()
        ephemeris_time = (float(jd_tdb) - 2451545.0) * 86400.0
        state, _ = spice.spkezr(name, ephemeris_time, "J2000", "NONE", "EARTH")
        return tuple(np.asarray(state[:3], dtype=float) * 1000.0)

    def position_at_jd(self, name, jd_tdb):
        """Return a cached geocentric J2000 position in metres."""
        return np.asarray(
            self._position_tuple_at_jd(str(name).lower(), float(jd_tdb)),
            dtype=float,
        )

    def position(self, name, epoch):
        """Return a body's geocentric J2000 position for a RocketPy Epoch."""
        epoch.require_absolute("SPICE ephemeris evaluation")
        return self.position_at_jd(name, epoch.jd_tdb)


class ThirdBody:
    """External celestial body used by :class:`rocketpy.Space`."""

    def __init__(
        self,
        name,
        gravitational_parameter,
        position_function,
        radius=None,
        pressure_function=None,
        albedo=0.0,
    ):
        if not callable(position_function):
            raise TypeError("position_function must be callable.")
        self.name = str(name)
        self.gravitational_parameter = float(gravitational_parameter)
        self.position_function = position_function
        self.radius = None if radius is None else float(radius)
        self.pressure_function = pressure_function
        self.albedo = float(albedo)

    @property
    def GM(self):
        return self.gravitational_parameter

    def _evaluate_position(self, jd_tdb):
        return np.asarray(self.position_function(float(jd_tdb)), dtype=float)

    def position(self, epoch):
        """Return geocentric GCRF position in metres."""
        epoch.require_absolute(f"{self.name} ephemeris evaluation")
        return self._evaluate_position(epoch.jd_tdb)

    def gravity_acceleration(self, epoch, observer):
        """Return differential third-body acceleration in GCRF."""
        observer = np.asarray(observer, dtype=float)
        body_position = self.position(epoch)
        relative = body_position - observer
        relative_norm = np.linalg.norm(relative)
        body_norm = np.linalg.norm(body_position)
        if relative_norm == 0.0 or body_norm == 0.0:
            return np.zeros(3)
        return self.gravitational_parameter * (
            relative / relative_norm**3 - body_position / body_norm**3
        )

    def pressure_at(self, jd_tdb, distance):
        """Return radiation pressure at a distance from this body."""
        if self.pressure_function is None:
            return 0.0
        return float(self.pressure_function(float(jd_tdb), float(distance)))

    @classmethod
    def builtin(
        cls,
        name,
        *,
        position_function=None,
        ephemeris=None,
    ):
        """Create a body with standard constants and an explicit ephemeris."""
        normalized = str(name).lower()
        try:
            data = _BODY_DATA[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown built-in celestial body: {name!r}.") from exc
        if position_function is None:
            if ephemeris is None:
                raise ValueError(
                    f"Built-in body {name!r} needs position_function or ephemeris."
                )

            def position_function(jd, body=normalized):
                return ephemeris.position_at_jd(body, jd)

        return cls(normalized, position_function=position_function, **data)

    @staticmethod
    @numbify(
        nopython=True,
        signature="float64(float64[:], float64[:], float64, float64[:], float64)",
    )
    def calculate_occultation_fraction(
        observer,
        source,
        source_radius,
        blocker,
        blocker_radius,
    ):
        """Return the visible fraction of one disk occulted by another."""
        sx = source[0] - observer[0]
        sy = source[1] - observer[1]
        sz = source[2] - observer[2]
        bx = blocker[0] - observer[0]
        by = blocker[1] - observer[1]
        bz = blocker[2] - observer[2]
        source_distance = math.sqrt(sx * sx + sy * sy + sz * sz)
        blocker_distance = math.sqrt(bx * bx + by * by + bz * bz)
        if blocker_distance <= blocker_radius:
            return 0.0
        if source_distance <= source_radius or blocker_distance > source_distance:
            return 1.0
        source_angle = math.asin(min(1.0, source_radius / source_distance))
        blocker_angle = math.asin(min(1.0, blocker_radius / blocker_distance))
        cosine = (sx * bx + sy * by + sz * bz) / (source_distance * blocker_distance)
        separation = math.acos(max(-1.0, min(1.0, cosine)))
        if separation >= source_angle + blocker_angle:
            return 1.0
        if separation <= abs(blocker_angle - source_angle):
            if blocker_angle >= source_angle:
                return 0.0
            return 1.0 - (blocker_angle / source_angle) ** 2
        x = (separation**2 + source_angle**2 - blocker_angle**2) / (2.0 * separation)
        height = math.sqrt(max(0.0, source_angle**2 - x**2))
        overlap = (
            source_angle**2 * math.acos(max(-1.0, min(1.0, x / source_angle)))
            + blocker_angle**2
            * math.acos(max(-1.0, min(1.0, (separation - x) / blocker_angle)))
            - separation * height
        )
        return max(0.0, min(1.0, 1.0 - overlap / (math.pi * source_angle**2)))


def occultation_fraction(observer, source, source_radius, blocker, blocker_radius):
    """Compatibility function delegating to the compiled body kernel."""
    return ThirdBody.calculate_occultation_fraction(
        np.array(observer, dtype=float, copy=True),
        np.array(source, dtype=float, copy=True),
        float(source_radius),
        np.array(blocker, dtype=float, copy=True),
        float(blocker_radius),
    )
