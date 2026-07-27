"""Composable atmospheric models for local and Earth-centered simulations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rocketpy._logging import logger
from rocketpy.mathutils.compilation import numbify
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


class AtmosphereLayer(ABC):
    """Interface implemented by one atmospheric model.

    A layer evaluates local properties at a requested altitude. Selection and
    blending between layers belong to :class:`Atmosphere`.
    """

    requires_absolute_epoch = False

    def prepare(self, start_epoch: Epoch, end_epoch: Epoch | None = None):
        """Prepare external data required over an evaluation interval.

        Most layers are self-contained and therefore need no preparation.
        Data-backed layers override this hook so :class:`Flight` can complete
        network and cache work before numerical integration starts.
        """
        del start_epoch, end_epoch
        return self

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


class FunctionAtmosphereLayer(AtmosphereLayer):
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


class ZeroAtmosphereLayer(AtmosphereLayer):
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


class ExponentialAtmosphereLayer(AtmosphereLayer):
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
        density = self._evaluate_density(float(altitude), self._LAYERS)
        return AtmosphericState(
            pressure=0.0,
            temperature=0.0,
            density=max(float(density), VACUUM_DENSITY),
            speed_of_sound=np.inf,
            dynamic_viscosity=0.0,
            wind_velocity=np.zeros(3),
        )

    @staticmethod
    @numbify(
        nopython=True,
        signature="float64(float64, float64[:, ::1])",
    )
    def _evaluate_density(altitude, layer_data):
        """Numba-compatible core of the tabulated exponential model."""
        altitude_km = max(altitude / 1000.0, 0.0)
        index = np.searchsorted(layer_data[:, 0], altitude_km, side="right") - 1
        index = max(0, min(index, len(layer_data) - 1))
        reference_altitude, reference_density, scale_height = layer_data[index]
        density = reference_density * math.exp(
            -(altitude_km - reference_altitude) / scale_height
        )
        return max(density, VACUUM_DENSITY)


class Atmosphere:
    """Compose atmospheric layers by geodetic altitude.

    Density is blended logarithmically across transition regions, while wind
    and the remaining thermodynamic properties are blended linearly.  This
    works in both directions: ascent into a global model and descent into
    newly fetched local weather use the very same layer map.

    Parameters
    ----------
    layers : dict
        Mapping from the lower altitude of each layer to an
        :class:`AtmosphereLayer`. A local :class:`rocketpy.Environment`
        implements that interface and is normally used as the lowest layer.
    transition_width : float, optional
        Absolute transition half-width in meters.
    automatic_fall_environment : bool, optional
        If ``True`` and the lowest layer is an
        :class:`rocketpy.Environment`, Earth-centered ``Flight`` creates a
        one-shot descent event at 20 km. The event constructs the same
        Environment type and atmospheric-model type at the current coordinates,
        then replaces the lowest layer. If the lowest layer is not an
        Environment, no event is added.
    """

    AUTOMATIC_FALL_ENVIRONMENT_ALTITUDE = 20_000.0

    def __init__(
        self,
        atmosphere_layers_map: dict[float, AtmosphereLayer],
        *,
        transition_width: float | None = None,
        automatic_fall_environment: bool = False,
    ):
        if not atmosphere_layers_map:
            raise ValueError("Atmosphere requires at least one layer.")
        if transition_width is not None and transition_width < 0:
            raise ValueError("transition_width cannot be negative.")
        if not all(
            isinstance(model, AtmosphereLayer)
            for model in atmosphere_layers_map.values()
        ):
            raise TypeError("Every layer must implement AtmosphereLayer.")
        self.transition_width = (
            None if transition_width is None else float(transition_width)
        )
        if not isinstance(automatic_fall_environment, bool):
            raise TypeError("automatic_fall_environment must be a bool.")
        self.automatic_fall_environment = automatic_fall_environment
        self._set_layers(atmosphere_layers_map)

    def _set_layers(self, layers):
        self.atmosphere_layers_map = dict(
            sorted((float(h), model) for h, model in layers.items())
        )
        self._boundaries = np.array(tuple(self.atmosphere_layers_map), dtype=float)
        self._models = tuple(self.atmosphere_layers_map.values())
        self.requires_absolute_epoch = any(
            model.requires_absolute_epoch for model in self._models
        )

    @property
    def layers(self):
        """Compatibility view of :attr:`atmosphere_layers_map`."""
        return self.atmosphere_layers_map

    def model_at(self, altitude: float) -> AtmosphereLayer:
        """Return the discrete model owning ``altitude`` (without blending)."""
        index = np.searchsorted(self._boundaries, altitude, side="right") - 1
        index = max(0, min(int(index), len(self._models) - 1))
        return self._models[index]

    @property
    def lowest_layer(self) -> AtmosphereLayer:
        """Return the first layer in the altitude map."""
        return self._models[0]

    @property
    def creates_automatic_fall_environment(self) -> bool:
        """Whether Flight should add the automatic 20 km descent event."""
        from rocketpy.environment.environment import Environment

        return self.automatic_fall_environment and isinstance(
            self.lowest_layer, Environment
        )

    def create_fall_environment(self, *, epoch, latitude, longitude):
        """Replace the lowest Environment at descent coordinates.

        Custom Environment subclasses used with the automatic mode must accept
        the standard Environment constructor keywords. Forecast, reanalysis,
        ensemble and Windy models are requested again using the original model
        source at the new coordinates. More specialized construction or
        data-selection workflows should use an explicit user Event.
        """
        from rocketpy.environment.environment import Environment

        source = self.lowest_layer
        if not isinstance(source, Environment):
            return None
        regional = type(source)(
            date=epoch.to_datetime(),
            latitude=float(latitude),
            longitude=float(longitude),
            elevation=0.0,
            datum=source.datum,
            timezone="UTC",
            max_expected_height=source.max_expected_height,
        )
        model_type = source.atmospheric_model_type
        if model_type.lower() == "custom_atmosphere":
            regional.set_atmospheric_model(
                type=model_type,
                pressure=source.pressure,
                temperature=source.temperature,
                wind_u=source.wind_velocity_x,
                wind_v=source.wind_velocity_y,
            )
        elif model_type.lower() != "standard_atmosphere":
            regional.set_atmospheric_model(
                type=model_type,
                file=source.atmospheric_model_file,
                dictionary=source.atmospheric_model_dict,
            )
        lowest_boundary = next(iter(self.atmosphere_layers_map))
        layers = dict(self.atmosphere_layers_map)
        layers[lowest_boundary] = regional
        self._set_layers(layers)
        return regional

    def install_regional_layer(
        self,
        atmosphere: AtmosphereLayer,
        *,
        boundary: float,
        transition_width: float | None = None,
    ):
        """Replace the low-altitude region while retaining global layers.

        This mutates the composition deliberately so a running ``Flight`` and
        its ``Earth`` keep the same atmosphere object when descent weather is
        installed.
        """
        if not isinstance(atmosphere, AtmosphereLayer):
            raise TypeError("atmosphere must implement AtmosphereLayer.")
        boundary = float(boundary)
        upper_model = self.model_at(boundary)
        retained = {
            height: model
            for height, model in self.atmosphere_layers_map.items()
            if height > boundary
        }
        self._set_layers({-np.inf: atmosphere, boundary: upper_model, **retained})
        if transition_width is not None and transition_width < 0:
            raise ValueError("transition_width cannot be negative.")
        self.transition_width = (
            None if transition_width is None else float(transition_width)
        )
        return self

    def _transition_half_width(self, boundary):
        del boundary
        if self.transition_width is not None:
            return self.transition_width
        return 0.0

    def prepare(self, start_epoch: Epoch, end_epoch: Epoch | None = None):
        """Prepare every distinct layer for the requested epoch interval."""
        prepared = set()
        for model in self._models:
            identity = id(model)
            if identity in prepared:
                continue
            model.prepare(start_epoch, end_epoch)
            prepared.add(identity)
        return self

    @staticmethod
    def _blend(lower, upper, weight):
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight == 0.0:
            return lower
        if weight == 1.0:
            return upper

        def linear(name):
            lower_value = getattr(lower, name)
            upper_value = getattr(upper, name)
            if np.isinf(lower_value) or np.isinf(upper_value):
                return np.inf
            return (1.0 - weight) * lower_value + weight * upper_value

        density = math.exp(
            (1.0 - weight) * math.log(max(lower.density, VACUUM_DENSITY))
            + weight * math.log(max(upper.density, VACUUM_DENSITY))
        )
        return AtmosphericState(
            pressure=linear("pressure"),
            temperature=linear("temperature"),
            density=density,
            speed_of_sound=linear("speed_of_sound"),
            dynamic_viscosity=linear("dynamic_viscosity"),
            wind_velocity=(
                (1.0 - weight) * lower.wind_velocity + weight * upper.wind_velocity
            ),
        )

    def evaluate(self, epoch, altitude, **kwargs) -> AtmosphericState:
        index = np.searchsorted(self._boundaries, altitude, side="right") - 1
        index = max(0, min(int(index), len(self._models) - 1))
        selected = self._models[index]

        # Check the upper and lower boundary around the selected layer. There
        # can be only one active transition for sensible, non-overlapping maps;
        # when transitions overlap, the closest boundary wins deterministically.
        candidates = []
        if index > 0:
            candidates.append((index, index - 1, index))
        if index < len(self._models) - 1:
            candidates.append((index + 1, index, index + 1))
        active = []
        for boundary_index, lower_index, upper_index in candidates:
            boundary = self._boundaries[boundary_index]
            half_width = self._transition_half_width(boundary)
            if half_width > 0 and abs(float(altitude) - boundary) <= half_width:
                active.append(
                    (
                        abs(float(altitude) - boundary),
                        boundary,
                        half_width,
                        lower_index,
                        upper_index,
                    )
                )
        if not active:
            return selected.evaluate(epoch, altitude, **kwargs)

        _, boundary, half_width, lower_index, upper_index = min(active)
        lower = self._models[lower_index].evaluate(epoch, altitude, **kwargs)
        upper = self._models[upper_index].evaluate(epoch, altitude, **kwargs)
        weight = (float(altitude) - (boundary - half_width)) / (2 * half_width)
        return self._blend(lower, upper, weight)


class HarrisPriesterAtmosphereLayer(AtmosphereLayer):
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
            [210, 183.9, 239.6],
            [220, 134.1, 185.3],
            [230, 99.49, 145.5],
            [240, 74.88, 115.7],
            [250, 57.09, 93.08],
            [260, 44.03, 75.55],
            [270, 34.3, 61.82],
            [280, 26.97, 50.95],
            [290, 21.39, 42.26],
            [300, 17.08, 35.26],
            [320, 10.99, 25.11],
            [340, 7.214, 18.19],
            [360, 4.824, 13.37],
            [380, 3.274, 9.955],
            [400, 2.249, 7.492],
            [420, 1.558, 5.684],
            [440, 1.091, 4.355],
            [460, 0.7701, 3.362],
            [480, 0.5474, 2.612],
            [500, 0.3916, 2.042],
            [520, 0.2819, 1.605],
            [540, 0.2042, 1.267],
            [560, 0.1488, 1.005],
            [580, 0.1092, 0.7997],
            [600, 0.0807, 0.639],
            [620, 0.06012, 0.5123],
            [640, 0.04519, 0.4121],
            [660, 0.0343, 0.3325],
            [680, 0.02632, 0.2691],
            [700, 0.02043, 0.2185],
            [720, 0.01607, 0.1779],
            [740, 0.01281, 0.1452],
            [760, 0.01036, 0.119],
            [780, 0.008496, 0.09776],
            [800, 0.007069, 0.08059],
            [840, 0.00468, 0.05741],
            [880, 0.0032, 0.0421],
            [920, 0.00221, 0.0313],
            [960, 0.00156, 0.0236],
            [1000, 0.00115, 0.0181],
        ],
        dtype=float,
    )

    def __init__(self, sun_position):
        """Create the layer with a TDB-Julian-date Sun position function."""
        if not callable(sun_position):
            raise TypeError("sun_position must be callable.")
        self.sun_position = sun_position

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
            sun = np.asarray(self.sun_position(epoch.jd_tdb), dtype=float)
            sun_right_ascension = np.arctan2(sun[1], sun[0])
            sun_declination = np.arcsin(sun[2] / np.linalg.norm(sun))
            density = self._evaluate_density(
                float(altitude_km),
                np.array(position_gcrf, dtype=float, copy=True),
                np.array(velocity_gcrf, dtype=float, copy=True),
                float(sun_right_ascension),
                float(sun_declination),
                self._LAYERS,
            )
        return AtmosphericState(
            pressure=0.0,
            temperature=0.0,
            density=max(float(density), VACUUM_DENSITY),
            speed_of_sound=np.inf,
            dynamic_viscosity=0.0,
            wind_velocity=np.zeros(3),
        )

    @staticmethod
    @numbify(
        nopython=True,
        signature=(
            "float64(float64, float64[:], float64[:], float64, "
            "float64, float64[:, ::1])"
        ),
    )
    def _evaluate_density(altitude_km, position, velocity, sun_ra, sun_dec, layers):
        """Numba-compatible Harris-Priester density kernel."""
        index = np.searchsorted(layers[:, 0], altitude_km) - 1
        index = max(0, min(index, len(layers) - 2))
        h0, minimum0, maximum0 = layers[index]
        h1, minimum1, maximum1 = layers[index + 1]
        minimum_scale = -(h1 - h0) / math.log(minimum1 / minimum0)
        maximum_scale = -(h1 - h0) / math.log(maximum1 / maximum0)
        minimum = minimum0 * math.exp(-(altitude_km - h0) / minimum_scale)
        maximum = maximum0 * math.exp(-(altitude_km - h0) / maximum_scale)

        hx = position[1] * velocity[2] - position[2] * velocity[1]
        hy = position[2] * velocity[0] - position[0] * velocity[2]
        hz = position[0] * velocity[1] - position[1] * velocity[0]
        momentum_norm = math.sqrt(hx * hx + hy * hy + hz * hz)
        cosine_inclination = max(-1.0, min(1.0, hz / momentum_norm))
        exponent = 2.0 + 4.0 * (1.0 - cosine_inclination**2)

        radius = math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
        apex_ra = sun_ra + math.radians(30.0)
        cosine_half_angle_squared = 0.5 * (
            1.0
            + position[0] / radius * math.cos(sun_dec) * math.cos(apex_ra)
            + position[1] / radius * math.cos(sun_dec) * math.sin(apex_ra)
            + position[2] / radius * math.sin(sun_dec)
        )
        density = minimum + (maximum - minimum) * (
            cosine_half_angle_squared ** (exponent / 2.0)
        )
        return max(density * 1e-12, VACUUM_DENSITY)


class NRLMSISE00AtmosphereLayer(AtmosphereLayer):
    """Optional NRLMSISE-00/2.0 atmosphere powered by ``pymsis``."""

    requires_absolute_epoch = True

    def __init__(self, version: int = 0, *, f107=None, f107a=None, ap=None):
        self.version = version
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap
        self._space_weather_cache = {}

    @property
    def requires_space_weather(self):
        """Whether any empirical solar or geomagnetic driver is implicit."""
        return self.f107 is None or self.f107a is None or self.ap is None

    @staticmethod
    def _day_key(epoch):
        return np.datetime64(epoch.to_datetime().replace(tzinfo=None), "D")

    def prepare(self, start_epoch: Epoch, end_epoch: Epoch | None = None):
        """Cache daily solar and geomagnetic drivers before propagation.

        Parameters
        ----------
        start_epoch, end_epoch : Epoch
            Inclusive UTC interval that may be evaluated. ``end_epoch`` may be
            omitted for a single-day preparation.
        """
        if not self.requires_space_weather:
            return self
        start_epoch.require_absolute("NRLMSISE preparation")
        end_epoch = start_epoch if end_epoch is None else end_epoch
        end_epoch.require_absolute("NRLMSISE preparation")
        first = self._day_key(start_epoch)
        last = self._day_key(end_epoch)
        if last < first:
            first, last = last, first
        days = np.arange(
            first,
            last + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        missing = [day for day in days if day not in self._space_weather_cache]
        if not missing:
            return self
        try:
            from pymsis.utils import get_f107_ap
        except ImportError as exc:
            raise ImportError(
                "NRLMSISE requires the optional 'pymsis' package."
            ) from exc
        logger.info(
            "Preparing NRLMSISE space-weather drivers for %s through %s.",
            str(missing[0]),
            str(missing[-1]),
        )
        f107_values, f107a_values, ap_values = get_f107_ap(
            np.asarray(missing, dtype="datetime64[D]")
        )
        for index, day in enumerate(missing):
            self._space_weather_cache[day] = (
                f107_values[index],
                f107a_values[index],
                ap_values[index],
            )
        return self

    def _space_weather_at(self, epoch):
        day = self._day_key(epoch)
        if day not in self._space_weather_cache:
            self.prepare(epoch)
        return self._space_weather_cache[day]

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

        prepared = (
            self._space_weather_at(epoch) if self.requires_space_weather else None
        )
        arguments = {
            "f107s": [self.f107 if self.f107 is not None else prepared[0]],
            "f107as": [self.f107a if self.f107a is not None else prepared[1]],
            "aps": np.atleast_2d(self.ap if self.ap is not None else prepared[2]),
        }
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
