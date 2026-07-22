from datetime import UTC, datetime

import numpy as np
import pytest

from rocketpy.mathutils import (
    Epoch,
    FlightState,
    OrbitalElements,
    ReferenceFrame,
    transform_kinematics,
)
from rocketpy.mathutils.reference_frame import _gcrf_to_itrf_kernel


def test_epoch_preserves_absolute_time_during_arithmetic():
    """Absolute Epoch arithmetic preserves its UTC instant and time scale."""
    # Arrange
    instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    epoch = Epoch.from_datetime(instant)

    # Act
    shifted = epoch + 12.5

    # Assert
    assert shifted - epoch == pytest.approx(12.5)
    assert shifted.to_datetime().timestamp() == pytest.approx(
        instant.timestamp() + 12.5
    )


def test_gcrf_itrf_kinematic_transform_round_trip():
    """Position, velocity, and acceleration survive a GCRF/ITRF round trip."""
    # Arrange
    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    position = np.array([7.0e6, -1.2e6, 8.0e5])
    velocity = np.array([1200.0, 7300.0, -400.0])
    acceleration = np.array([-7.0, 1.2, -0.8])

    # Act
    fixed = transform_kinematics(
        epoch,
        position,
        velocity,
        acceleration,
        source=ReferenceFrame.GCRF,
        target=ReferenceFrame.ITRF,
    )
    recovered = transform_kinematics(
        epoch,
        *fixed,
        source=ReferenceFrame.ITRF,
        target=ReferenceFrame.GCRF,
    )

    # Assert
    assert recovered[0] == pytest.approx(position, rel=0, abs=1e-8)
    assert recovered[1] == pytest.approx(velocity, rel=0, abs=1e-10)
    assert recovered[2] == pytest.approx(acceleration, rel=0, abs=1e-12)
    assert _gcrf_to_itrf_kernel.__numba_signature__.startswith("Tuple")


def test_gcrf_teme_transform_round_trip():
    """The built-in approximate TEME output transformation is reversible."""
    # Arrange
    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    position = np.array([7.0e6, -1.2e6, 8.0e5])
    velocity = np.array([1200.0, 7300.0, -400.0])

    # Act
    teme = transform_kinematics(
        epoch,
        position,
        velocity,
        source="gcrf",
        target="teme",
    )
    recovered = transform_kinematics(
        epoch,
        teme[0],
        teme[1],
        source="teme",
        target="gcrf",
    )

    # Assert
    assert recovered[0] == pytest.approx(position, rel=0, abs=1e-8)
    assert recovered[1] == pytest.approx(velocity, rel=0, abs=1e-10)


def test_flat_earth_cannot_be_transformed_without_launch_site_definition():
    """Cross-frame conversion rejects an under-specified flat-Earth frame."""
    # Arrange
    epoch = Epoch.relative_origin()

    # Act / Assert
    with pytest.raises(ValueError, match="launch-site origin"):
        transform_kinematics(
            epoch,
            np.zeros(3),
            source=ReferenceFrame.FLAT_EARTH,
            target=ReferenceFrame.GCRF,
        )


def test_orbital_elements_cartesian_round_trip():
    """Non-singular classical elements round-trip through Cartesian state."""
    # Arrange
    expected = OrbitalElements(
        semi_major_axis=7.2e6,
        eccentricity=0.05,
        inclination=0.7,
        raan=1.1,
        argument_of_periapsis=0.4,
        true_anomaly=2.0,
    )

    # Act
    position, velocity = expected.to_state()
    actual = OrbitalElements.from_state(position, velocity)

    # Assert
    assert actual.semi_major_axis == pytest.approx(expected.semi_major_axis)
    assert actual.eccentricity == pytest.approx(expected.eccentricity)
    assert actual.inclination == pytest.approx(expected.inclination)
    assert actual.raan == pytest.approx(expected.raan)
    assert actual.argument_of_periapsis == pytest.approx(expected.argument_of_periapsis)
    assert actual.true_anomaly == pytest.approx(expected.true_anomaly)


def test_flight_state_copies_input_arrays_and_records_elapsed_time():
    """FlightState is immutable and carries time for time-varying rocket mass."""
    # Arrange
    position = np.array([7.0e6, 0.0, 0.0])

    # Act
    state = FlightState.cartesian(
        epoch=Epoch.relative_origin(),
        position=position,
        velocity=[0.0, 7500.0, 0.0],
        frame="gcrf",
        elapsed_time=42.0,
    )
    position[0] = 0.0

    # Assert
    assert state.position[0] == pytest.approx(7.0e6)
    assert state.elapsed_time == pytest.approx(42.0)
    with pytest.raises(ValueError):
        state.position[0] = 1.0


def test_transform_kinematics_accepts_readonly_arrays():
    """Kinematic transformation handles read-only NumPy array inputs cleanly."""
    # Arrange
    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    position = np.array([7.0e6, -1.2e6, 8.0e5])
    position.flags.writeable = False
    velocity = np.array([1200.0, 7300.0, -400.0])
    velocity.flags.writeable = False

    # Act
    fixed = transform_kinematics(
        epoch,
        position,
        velocity,
        source=ReferenceFrame.GCRF,
        target=ReferenceFrame.ITRF,
    )

    # Assert
    assert fixed[0] is not None
    assert fixed[1] is not None
