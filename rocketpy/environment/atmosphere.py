"""Composable atmospheric models for local and Earth-centered simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rocketpy.mathutils.epoch import Epoch


VACUUM_DENSITY = 1e-30


@dataclass(frozen=True)
class AtmosphericState:
    """Atmospheric properties at a point, expressed in SI units."""

    pressure: float
    temperature: float
    density: float
    speed_of_sound: float
    dynamic_viscosity: float
    wind_velocity: np.ndarray

    def __post_init__(self):
        wind = np.array(self.wind_velocity, dtype=float, copy=True)
        if wind.shape != (3,):
            raise ValueError("wind_velocity must contain three components.")
        wind.setflags(write=False)
        object.__setattr__(self, "wind_velocity", wind)


class Atmosphere(ABC):
    """Interface implemented by RocketPy atmospheric models."""

    requires_absolute_epoch = False

    @abstractmethod
    def evaluate(
        self,
        epoch: Epoch,
        altitude: float,
        *,
        latitude_rad: float | None = None,
        longitude_rad: float | None = None,
        position_gcrf: np.ndarray | None = None,
        velocity_gcrf: np.ndarray | None = None,
    ) -> AtmosphericState:
        """Evaluate atmospheric properties at ``altitude`` in meters."""


class FunctionAtmosphere(Atmosphere):
    """Adapter over RocketPy's established height-dependent Functions."""

    def __init__(
        self,
        *,
        pressure,
        temperature,
        density,
        speed_of_sound,
        dynamic_viscosity,
        wind_velocity_x,
        wind_velocity_y,
    ):
        self.pressure = pressure
        self.temperature = temperature
        self.density = density
        self.speed_of_sound = speed_of_sound
        self.dynamic_viscosity = dynamic_viscosity
        self.wind_velocity_x = wind_velocity_x
        self.wind_velocity_y = wind_velocity_y

    def evaluate(self, epoch, altitude, **kwargs) -> AtmosphericState:
        del epoch, kwargs
        return AtmosphericState(
            pressure=float(self.pressure.get_value_opt(altitude)),
            temperature=float(self.temperature.get_value_opt(altitude)),
            density=max(float(self.density.get_value_opt(altitude)), VACUUM_DENSITY),
            speed_of_sound=float(self.speed_of_sound.get_value_opt(altitude)),
            dynamic_viscosity=float(self.dynamic_viscosity.get_value_opt(altitude)),
            wind_velocity=np.array(
                [
                    self.wind_velocity_x.get_value_opt(altitude),
                    self.wind_velocity_y.get_value_opt(altitude),
                    0.0,
                ]
            ),
        )


class VacuumAtmosphere(Atmosphere):
    """Atmosphere model for vacuum propagation."""

    def evaluate(self, epoch, altitude, **kwargs) -> AtmosphericState:
        del epoch, altitude, kwargs
        return AtmosphericState(
            pressure=0.0,
            temperature=0.0,
            density=VACUUM_DENSITY,
            speed_of_sound=np.inf,
            dynamic_viscosity=0.0,
            wind_velocity=np.zeros(3),
        )


class ExponentialAtmosphere(Atmosphere):
    """US Standard Atmosphere 1976 exponential density approximation."""

    _LAYERS = np.array(
        [
            [0, 1.225, 7.249],
            [25, 3.899e-2, 6.349],
            [30, 1.774e-2, 6.682],
            [40, 3.972e-3, 7.554],
            [50, 1.057e-3, 8.382],
            [60, 3.206e-4, 7.714],
            [70, 8.770e-5, 6.549],
            [80, 1.905e-5, 5.799],
            [90, 3.396e-6, 5.382],
            [100, 5.297e-7, 5.877],
            [110, 9.661e-8, 7.263],
            [120, 2.438e-8, 9.473],
            [130, 8.484e-9, 12.636],
            [140, 3.845e-9, 16.149],
            [150, 2.070e-9, 22.523],
            [180, 5.464e-10, 29.740],
            [200, 2.789e-10, 37.105],
            [250, 7.248e-11, 45.546],
            [300, 2.418e-11, 53.628],
            [350, 9.518e-12, 53.298],
            [400, 3.725e-12, 58.515],
            [450, 1.585e-12, 60.828],
            [500, 6.967e-13, 63.822],
            [600, 1.454e-13, 71.835],
            [700, 3.614e-14, 88.667],
            [800, 1.170e-14, 124.64],
            [900, 5.245e-15, 181.05],
            [1000, 3.019e-15, 268.0],
        ],
        dtype=float,
    )

    def evaluate(self, epoch, altitude, **kwargs) -> AtmosphericState:
        del epoch, kwargs
        altitude_km = max(float(altitude) / 1000.0, 0.0)
        index = np.searchsorted(self._LAYERS[:, 0], altitude_km, side="right") - 1
        index = max(0, min(index, len(self._LAYERS) - 1))
        reference_altitude, reference_density, scale_height = self._LAYERS[index]
        density = reference_density * np.exp(
            -(altitude_km - reference_altitude) / scale_height
        )
        return AtmosphericState(
            pressure=0.0,
            temperature=0.0,
            density=max(float(density), VACUUM_DENSITY),
            speed_of_sound=np.inf,
            dynamic_viscosity=0.0,
            wind_velocity=np.zeros(3),
        )


class LayeredAtmosphere(Atmosphere):
    """Select atmospheric models by geodetic altitude."""

    def __init__(self, layers: dict[float, Atmosphere]):
        if not layers:
            raise ValueError("LayeredAtmosphere requires at least one layer.")
        self.layers = dict(sorted((float(h), model) for h, model in layers.items()))
        self._boundaries = np.array(tuple(self.layers), dtype=float)
        self._models = tuple(self.layers.values())
        self.requires_absolute_epoch = any(
            model.requires_absolute_epoch for model in self._models
        )

    def evaluate(self, epoch, altitude, **kwargs) -> AtmosphericState:
        index = np.searchsorted(self._boundaries, altitude, side="right") - 1
        index = max(0, min(int(index), len(self._models) - 1))
        return self._models[index].evaluate(epoch, altitude, **kwargs)


class HarrisPriesterAtmosphere(Atmosphere):
    """Harris-Priester thermosphere with the diurnal density bulge."""

    requires_absolute_epoch = True
    _LAYERS = np.array(
        [
            [100, 497400, 497400],
            [120, 24900, 24900],
            [130, 8377, 8710],
            [140, 3899, 4059],
            [150, 2122, 2215],
            [160, 1263, 1344],
            [170, 800.8, 875.8],
            [180, 528.3, 601.0],
            [190, 361.7, 429.7],
            [200, 255.7, 316.2],
            [220, 134.1, 185.3],
            [240, 74.88, 115.7],
            [260, 44.03, 75.55],
            [280, 26.97, 50.95],
            [300, 17.08, 35.26],
            [320, 10.99, 25.11],
            [340, 7.214, 18.19],
            [360, 4.824, 13.37],
            [380, 3.274, 9.955],
            [400, 2.249, 7.492],
            [440, 1.091, 4.355],
            [480, 0.5474, 2.612],
            [520, 0.2819, 1.605],
            [560, 0.1488, 1.005],
            [600, 0.0807, 0.639],
            [640, 0.04519, 0.4121],
            [680, 0.02632, 0.2691],
            [720, 0.01607, 0.1779],
            [760, 0.01036, 0.119],
            [800, 0.007069, 0.08059],
            [840, 0.00468, 0.05741],
            [880, 0.0032, 0.0421],
            [920, 0.00221, 0.0313],
            [960, 0.00156, 0.0236],
            [1000, 0.00115, 0.0181],
        ],
        dtype=float,
    )

    def __init__(self, sun_ephemeris=None):
        from .celestial_body import AnalyticalEphemeris

        self.sun_ephemeris = sun_ephemeris or AnalyticalEphemeris()

    def evaluate(
        self,
        epoch,
        altitude,
        *,
        position_gcrf=None,
        velocity_gcrf=None,
        **kwargs,
    ) -> AtmosphericState:
        del kwargs
        epoch.require_absolute("Harris-Priester atmosphere evaluation")
        if position_gcrf is None or velocity_gcrf is None:
            raise ValueError("Harris-Priester requires GCRF position and velocity.")
        altitude_km = altitude / 1000.0
        if altitude_km >= self._LAYERS[-1, 0]:
            density = VACUUM_DENSITY
        else:
            index = np.searchsorted(self._LAYERS[:, 0], altitude_km) - 1
            index = max(0, min(index, len(self._LAYERS) - 2))
            h0, minimum0, maximum0 = self._LAYERS[index]
            h1, minimum1, maximum1 = self._LAYERS[index + 1]
            minimum_scale = -(h1 - h0) / np.log(minimum1 / minimum0)
            maximum_scale = -(h1 - h0) / np.log(maximum1 / maximum0)
            minimum = minimum0 * np.exp((altitude_km - h0) / minimum_scale)
            maximum = maximum0 * np.exp((altitude_km - h0) / maximum_scale)

            sun = self.sun_ephemeris.position("sun", epoch)
            sun_right_ascension = np.arctan2(sun[1], sun[0])
            sun_declination = np.arcsin(sun[2] / np.linalg.norm(sun))
            apex_ra = sun_right_ascension + np.radians(30.0)
            apex = np.array(
                [
                    np.cos(sun_declination) * np.cos(apex_ra),
                    np.cos(sun_declination) * np.sin(apex_ra),
                    np.sin(sun_declination),
                ]
            )
            position_unit = position_gcrf / np.linalg.norm(position_gcrf)
            momentum = np.cross(position_gcrf, velocity_gcrf)
            cosine_inclination = momentum[2] / np.linalg.norm(momentum)
            exponent = 2.0 + 4.0 * (1.0 - cosine_inclination**2)
            cosine_half_angle_squared = 0.5 * (
                1.0 + np.clip(np.dot(position_unit, apex), -1.0, 1.0)
            )
            density = (
                minimum
                + (maximum - minimum) * cosine_half_angle_squared ** (exponent / 2.0)
            ) * 1e-12
        return AtmosphericState(
            pressure=0.0,
            temperature=0.0,
            density=max(float(density), VACUUM_DENSITY),
            speed_of_sound=np.inf,
            dynamic_viscosity=0.0,
            wind_velocity=np.zeros(3),
        )


class NRLMSISE00(Atmosphere):
    """Optional NRLMSISE-00/2.0 atmosphere powered by ``pymsis``."""

    requires_absolute_epoch = True

    def __init__(self, version: int = 0, *, f107=None, f107a=None, ap=None):
        self.version = version
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap

    def evaluate(
        self,
        epoch,
        altitude,
        *,
        latitude_rad=None,
        longitude_rad=None,
        **kwargs,
    ) -> AtmosphericState:
        del kwargs
        epoch.require_absolute("NRLMSISE atmosphere evaluation")
        if latitude_rad is None or longitude_rad is None:
            raise ValueError("NRLMSISE requires geodetic latitude and longitude.")
        try:
            import pymsis
        except ImportError as exc:
            raise ImportError(
                "NRLMSISE requires the optional 'pymsis' package."
            ) from exc

        arguments = {}
        if self.f107 is not None:
            arguments["f107s"] = [self.f107]
        if self.f107a is not None:
            arguments["f107as"] = [self.f107a]
        if self.ap is not None:
            arguments["aps"] = [self.ap]
        result = pymsis.calculate(
            dates=[np.datetime64(epoch.to_datetime().replace(tzinfo=None))],
            lons=[np.degrees(longitude_rad)],
            lats=[np.degrees(latitude_rad)],
            alts=[altitude / 1000.0],
            version=self.version,
            **arguments,
        )
        density = float(result[..., pymsis.Variable.MASS_DENSITY][0])
        temperature = float(result[..., pymsis.Variable.TEMPERATURE][0])
        return AtmosphericState(
            pressure=0.0,
            temperature=temperature,
            density=max(density, VACUUM_DENSITY),
            speed_of_sound=np.inf,
            dynamic_viscosity=0.0,
            wind_velocity=np.zeros(3),
        )
