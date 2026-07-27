from datetime import UTC, datetime

import numpy as np
import pytest

from rocketpy import (
    Atmosphere,
    AtmosphereLayer,
    AtmosphericState,
    DefaultGravity,
    Earth,
    EarthRadiationPressure,
    Environment,
    Epoch,
    ExponentialAtmosphereLayer,
    FlightState,
    HarrisPriesterAtmosphereLayer,
    NRLMSISE00AtmosphereLayer,
    Space,
    SphericalHarmonicGravity,
    SpiceEphemeris,
    ThirdBody,
    ZonalGravity,
)
from rocketpy.environment.gravity import _harmonic_acceleration


class _ConstantAtmosphere(AtmosphereLayer):
    def __init__(self, density, wind=(0, 0, 0)):
        self.value = AtmosphericState(
            pressure=density * 1e5,
            temperature=250.0,
            density=density,
            speed_of_sound=300.0,
            dynamic_viscosity=1e-5,
            wind_velocity=np.asarray(wind),
        )

    def evaluate(self, epoch, altitude, **kwargs):
        del epoch, altitude, kwargs
        return self.value


def test_environment_is_a_local_layer_inside_earth_atmosphere():
    """Local weather composes with a global atmosphere without owning Earth."""
    # Arrange
    environment = Environment(date=datetime(2026, 1, 2, tzinfo=UTC))
    earth = Earth(
        atmosphere=Atmosphere(
            {-np.inf: environment, 80_000: ExponentialAtmosphereLayer()}
        )
    )
    orbital_state = FlightState.cartesian(
        epoch=environment.epoch,
        position=[earth.datum.semi_major_axis + 500e3, 0.0, 0.0],
        velocity=[0.0, 7600.0, 0.0],
        frame="gcrf",
    )

    # Act
    local_atmosphere = environment.evaluate(environment.epoch, 100.0)
    orbital_gravity = earth.gravity_acceleration(environment.epoch, orbital_state)
    atmosphere = earth.evaluate_atmosphere(environment.epoch, orbital_state)

    # Assert
    assert isinstance(environment, AtmosphereLayer)
    assert local_atmosphere.density > 0.0
    assert orbital_gravity[0] < 0.0
    assert atmosphere.density == pytest.approx(
        ExponentialAtmosphereLayer().evaluate(environment.epoch, 500e3).density,
        rel=2e-2,
    )


def test_zonal_gravity_j3_correction_remains_acceleration_sized():
    """The J3 term has acceleration dimensions and stays a small perturbation."""
    # Arrange
    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    position = np.array([5.0e6, 2.0e6, 4.0e6])
    without_j3 = ZonalGravity(j3=0.0)
    with_j3 = ZonalGravity()

    # Act
    correction = with_j3.acceleration(
        epoch, position, frame="gcrf"
    ) - without_j3.acceleration(epoch, position, frame="gcrf")

    # Assert
    assert 1e-8 < np.linalg.norm(correction) < 1e-3


def test_spherical_harmonic_gravity_and_tides_are_finite():
    """Bundled EGM2008 coefficients and optional tides produce finite vectors."""
    # Arrange
    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    position = np.array([6.9e6, 1.0e5, 2.0e5])
    sun = ThirdBody.builtin(
        "sun", position_function=lambda _: np.array([1.5e11, 0.0, 0.0])
    )
    moon = ThirdBody.builtin(
        "moon", position_function=lambda _: np.array([3.8e8, 0.0, 0.0])
    )
    static_model = SphericalHarmonicGravity(degree=4, order=4)
    tide_model = SphericalHarmonicGravity(
        degree=4, order=4, include_tides=True, sun=sun, moon=moon
    )

    # Act
    static = static_model.acceleration(epoch, position, frame="gcrf")
    dynamic = tide_model.acceleration(epoch, position, frame="gcrf")

    # Assert
    assert np.all(np.isfinite(static))
    assert np.all(np.isfinite(dynamic))
    assert np.linalg.norm(dynamic - static) > 0.0
    assert _harmonic_acceleration.__numba_signature__.startswith("float64[::1]")


def test_aligned_numerical_kernels_are_numbified():
    """Hot atmospheric, occultation and ERP loops expose eager signatures."""
    assert ExponentialAtmosphereLayer._evaluate_density.__numba_signature__.startswith(
        "float64("
    )
    assert (
        HarrisPriesterAtmosphereLayer._evaluate_density.__numba_signature__.startswith(
            "float64("
        )
    )
    assert ThirdBody.calculate_occultation_fraction.__numba_signature__.startswith(
        "float64("
    )
    assert EarthRadiationPressure._evaluate_knocke_erp.__numba_signature__.startswith(
        "float64[:]"
    )


def test_layered_atmosphere_smooths_and_replaces_only_regional_layer():
    lower = _ConstantAtmosphere(1.0, (10, 0, 0))
    upper = _ConstantAtmosphere(1e-4, (0, 10, 0))
    atmosphere = Atmosphere({-np.inf: lower, 100_000: upper}, transition_width=10_000)

    midpoint = atmosphere.evaluate(Epoch.relative_origin(), 100_000)
    assert midpoint.density == pytest.approx(1e-2)
    assert midpoint.wind_velocity == pytest.approx((5, 5, 0))

    replacement = _ConstantAtmosphere(0.5, (20, 0, 0))
    identity = id(atmosphere)
    atmosphere.install_regional_layer(replacement, boundary=80_000)
    assert id(atmosphere) == identity
    assert atmosphere.model_at(10_000) is replacement
    assert atmosphere.model_at(90_000) is lower
    assert atmosphere.model_at(110_000) is upper


def test_layered_atmosphere_default_disables_blending_without_failing():
    """Multiple layers select discretely when no transition width is supplied."""
    lower = _ConstantAtmosphere(1.0)
    upper = _ConstantAtmosphere(1e-4)
    atmosphere = Atmosphere({-np.inf: lower, 100_000: upper})
    epoch = Epoch.relative_origin()

    assert atmosphere.evaluate(epoch, 99_999).density == pytest.approx(1.0)
    assert atmosphere.evaluate(epoch, 100_000).density == pytest.approx(1e-4)


def test_harris_priester_density_decreases_continuously_between_table_rows():
    """Logarithmic density interpolation has the expected altitude direction."""
    position = np.array([7.0e6, 1.0e5, 4.0e5])
    velocity = np.array([-100.0, 7_500.0, 800.0])
    table = HarrisPriesterAtmosphereLayer._LAYERS

    density_210 = HarrisPriesterAtmosphereLayer._evaluate_density(
        210.0, position, velocity, 1.2, 0.3, table
    )
    density_215 = HarrisPriesterAtmosphereLayer._evaluate_density(
        215.0, position, velocity, 1.2, 0.3, table
    )
    density_220 = HarrisPriesterAtmosphereLayer._evaluate_density(
        220.0, position, velocity, 1.2, 0.3, table
    )

    assert 210.0 in table[:, 0]
    assert 215.0 not in table[:, 0]
    assert density_210 > density_215 > density_220


def test_harris_priester_contains_the_complete_reference_altitude_grid():
    expected_altitudes = [
        100,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        210,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        300,
        320,
        340,
        360,
        380,
        400,
        420,
        440,
        460,
        480,
        500,
        520,
        540,
        560,
        580,
        600,
        620,
        640,
        660,
        680,
        700,
        720,
        740,
        760,
        780,
        800,
        840,
        880,
        920,
        960,
        1000,
    ]

    assert HarrisPriesterAtmosphereLayer._LAYERS[:, 0].tolist() == expected_altitudes


def test_nrlmsise_prepares_each_required_day_once(monkeypatch):
    calls = []

    def fake_space_weather(days):
        calls.append(np.asarray(days))
        count = len(days)
        return (
            np.full(count, 150.0),
            np.full(count, 145.0),
            np.full((count, 7), 4.0),
        )

    monkeypatch.setattr("pymsis.utils.get_f107_ap", fake_space_weather)
    layer = NRLMSISE00AtmosphereLayer()
    start = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    end = start + 2 * 86_400

    layer.prepare(start, end)
    layer.prepare(start, end)

    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert len(layer._space_weather_cache) == 3


def test_earth_and_space_remain_authoritative_domain_owners():
    earth = Earth(atmosphere=_ConstantAtmosphere(1e-6))
    positions = {
        "sun": np.array([149597870700.0, 0.0, 0.0]),
        "moon": np.array([384400000.0, 0.0, 0.0]),
    }
    space = Space(
        [
            ThirdBody.builtin(
                name, position_function=lambda _, name=name: positions[name]
            )
            for name in positions
        ]
    )
    assert all(isinstance(body, ThirdBody) for body in space.bodies)
    original = earth.atmosphere
    earth.atmosphere = _ConstantAtmosphere(2e-6)
    assert earth.atmosphere is not original

    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    state = FlightState.cartesian(
        epoch=epoch,
        position=[earth.datum.semi_major_axis + 500e3, 0, 0],
        velocity=[0, 7600, 0],
        frame="gcrf",
    )
    assert np.all(np.isfinite(space.gravity_acceleration(epoch, state)))


def test_named_space_bodies_share_implicit_spice_ephemeris(monkeypatch):
    """Named bodies prepare one default ephemeris during Space construction."""
    calls = []

    class FakeEphemeris:
        def __init__(self):
            calls.append("prepared")

        @staticmethod
        def position_at_jd(name, jd_tdb):
            del jd_tdb
            return {
                "sun": np.array([1.5e11, 0.0, 0.0]),
                "moon": np.array([3.8e8, 0.0, 0.0]),
            }[name]

    monkeypatch.setattr(
        "rocketpy.environment.models.SpiceEphemeris",
        FakeEphemeris,
    )
    space = Space(["sun", "moon"])

    assert calls == ["prepared"]
    assert [body.name for body in space.bodies] == ["sun", "moon"]
    assert space.bodies[0]._evaluate_position(2451545.0)[0] == pytest.approx(1.5e11)


def test_space_accepts_named_spice_kernel(monkeypatch):
    requested = []

    class FakeEphemeris:
        def __init__(self, kernel):
            requested.append(kernel)

        @staticmethod
        def position_at_jd(name, jd_tdb):
            del name, jd_tdb
            return np.array([1.5e11, 0.0, 0.0])

    monkeypatch.setattr(
        "rocketpy.environment.models.SpiceEphemeris",
        FakeEphemeris,
    )
    space = Space(["sun"], ephemeris="de442s")

    assert requested == ["de442s"]
    assert space.bodies[0].position_function(2451545.0)[0] == pytest.approx(1.5e11)


def test_spice_position_queries_are_cached():
    calls = []

    class FakeSpice:
        @staticmethod
        def spkezr(name, ephemeris_time, frame, correction, observer):
            calls.append((name, ephemeris_time, frame, correction, observer))
            return np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]), 0.0

    ephemeris = SpiceEphemeris.__new__(SpiceEphemeris)
    ephemeris._loaded = True
    ephemeris._load = lambda: FakeSpice()
    ephemeris._position_tuple_at_jd.cache_clear()

    first = ephemeris.position_at_jd("sun", 2451545.0)
    second = ephemeris.position_at_jd("SUN", 2451545.0)

    assert first == pytest.approx([1000.0, 2000.0, 3000.0])
    assert second == pytest.approx(first)
    assert len(calls) == 1
