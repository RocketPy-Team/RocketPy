from datetime import UTC, datetime

import numpy as np
import pytest

from rocketpy import (
    DefaultGravity,
    Environment,
    Epoch,
    ExponentialAtmosphere,
    FlightState,
    SphericalHarmonicGravity,
    ZonalGravity,
)
from rocketpy.environment.gravity import _harmonic_acceleration


def test_environment_default_composition_supports_local_and_orbital_states():
    """Default Environment composition covers legacy and Earth-centered gravity."""
    # Arrange
    environment = Environment(date=datetime(2026, 1, 2, tzinfo=UTC))
    local_state = FlightState.cartesian(
        epoch=environment.epoch,
        position=[0.0, 0.0, 100.0],
        velocity=np.zeros(3),
        frame="flat_earth",
    )
    orbital_state = FlightState.cartesian(
        epoch=environment.epoch,
        position=[environment.earth_datum.semi_major_axis + 500e3, 0.0, 0.0],
        velocity=[0.0, 7600.0, 0.0],
        frame="gcrf",
    )

    # Act
    local_gravity = environment.gravity_acceleration(environment.epoch, local_state)
    orbital_gravity = environment.gravity_acceleration(environment.epoch, orbital_state)
    atmosphere = environment.evaluate_atmosphere(environment.epoch, orbital_state)

    # Assert
    assert isinstance(environment.gravity_model, DefaultGravity)
    assert local_gravity[2] < 0.0
    assert orbital_gravity[0] < 0.0
    assert atmosphere.density == pytest.approx(
        ExponentialAtmosphere().evaluate(environment.epoch, 500e3).density,
        rel=2e-2,
    )


def test_zonal_gravity_j3_correction_remains_acceleration_sized():
    """The J3 term has acceleration dimensions and stays a small perturbation."""
    # Arrange
    epoch = Epoch.relative_origin()
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
    static_model = SphericalHarmonicGravity(degree=4, order=4)
    tide_model = SphericalHarmonicGravity(degree=4, order=4, include_tides=True)

    # Act
    static = static_model.acceleration(epoch, position, frame="gcrf")
    dynamic = tide_model.acceleration(epoch, position, frame="gcrf")

    # Assert
    assert np.all(np.isfinite(static))
    assert np.all(np.isfinite(dynamic))
    assert np.linalg.norm(dynamic - static) > 0.0
    assert _harmonic_acceleration.__numba_signature__.startswith("float64[::1]")
