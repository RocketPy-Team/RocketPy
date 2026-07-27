from datetime import UTC, datetime

import numpy as np
import pytest

from rocketpy.environment import Earth
from rocketpy.mathutils import (
    GRS80,
    WGS72,
    Datum,
    Epoch,
    FlatEarthDatum,
    FlightState,
    OrbitalElements,
    ReferenceFrame,
    SimpleDatum,
    transform_kinematics,
)
from rocketpy.mathutils.epoch import ARCSEC_TO_RAD, DEFAULT_EARTH_ORIENTATION
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


def test_epoch_uses_historical_leap_seconds():
    """TAI-UTC changes at the correctly represented 2017 leap boundary."""
    before = Epoch.from_datetime(datetime(2016, 12, 31, 12, tzinfo=UTC))
    after = Epoch.from_datetime(datetime(2017, 1, 1, tzinfo=UTC))

    assert (before.jd_tai - before.jd_utc) * 86400 == pytest.approx(36, abs=2e-5)
    assert (after.jd_tai - after.jd_utc) * 86400 == pytest.approx(37, abs=2e-5)
    assert (after.jd_tt - after.jd_tai) * 86400 == pytest.approx(32.184, abs=2e-5)


def test_datum_policies_have_explicit_inertial_contracts():
    """Flat, ERA-only and full datums expose distinct frame capabilities."""
    flat = FlatEarthDatum()
    simple = SimpleDatum()
    full = Datum()

    assert flat.integration_frame is ReferenceFrame.FLAT_EARTH
    assert flat.fixed_frame is ReferenceFrame.FLAT_EARTH
    assert flat.supports_apparent_gravity
    assert simple.integration_frame is ReferenceFrame.GCRF
    assert simple.fixed_frame is ReferenceFrame.ITRF
    assert not simple.supports_apparent_gravity
    assert full.integration_frame is ReferenceFrame.GCRF


def test_named_datum_presets_expose_expected_reference_parameters():
    assert GRS80.name == "GRS80"
    assert GRS80.semi_major_axis == pytest.approx(6_378_137.0)
    assert GRS80.flattening == pytest.approx(1 / 298.257222101)
    assert WGS72.name == "WGS72"
    assert WGS72.semi_major_axis == pytest.approx(6_378_135.0)
    assert WGS72.flattening == pytest.approx(1 / 298.26)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("wgs84", "WGS84"), ("grs80", "GRS80"), ("wgs72", "WGS72")],
)
def test_earth_accepts_named_ellipsoid_presets(name, expected):
    assert Earth(datum=name).datum.name == expected


def test_geodetic_conversion_is_vectorized_and_round_trips():
    datum = Datum()
    latitude = np.radians([0.0, 45.0, -80.0, 90.0])
    longitude = np.radians([0.0, 20.0, 170.0, -30.0])
    altitude = np.array([0.0, 100.0, 500e3, 2_000.0])

    positions = datum.geodetic_to_itrs(latitude, longitude, altitude)
    result = datum.itrs_to_geodetic(positions)

    assert positions.shape == (4, 3)
    assert result[0] == pytest.approx(latitude, abs=1e-12)
    assert result[1] == pytest.approx(longitude, abs=1e-12)
    assert result[2] == pytest.approx(altitude, abs=1e-6)


def test_topocentric_launch_and_rtn_interfaces_round_trip():
    datum = Datum()
    latitude = np.radians(32.0)
    longitude = np.radians(-106.0)
    site_altitude = 1_400.0
    local = np.array([1_000.0, -200.0, 300.0])
    local_velocity = np.array([10.0, 20.0, -3.0])

    itrs = datum.from_topocentric(
        local,
        local_velocity,
        latitude_rad=latitude,
        longitude_rad=longitude,
        altitude=site_altitude,
    )
    recovered = datum.to_topocentric(
        itrs[0],
        itrs[1],
        latitude_rad=latitude,
        longitude_rad=longitude,
        altitude=site_altitude,
    )
    assert recovered[0] == pytest.approx(local, abs=1e-9)
    assert recovered[1] == pytest.approx(local_velocity, abs=1e-12)

    launch = datum.to_launch_frame(
        itrs[0],
        itrs[1],
        latitude_rad=latitude,
        longitude_rad=longitude,
        altitude=site_altitude,
        azimuth_rad=np.radians(75.0),
    )
    recovered_itrs = datum.from_launch_frame(
        launch[0],
        launch[1],
        latitude_rad=latitude,
        longitude_rad=longitude,
        altitude=site_altitude,
        azimuth_rad=np.radians(75.0),
    )
    assert recovered_itrs[0] == pytest.approx(itrs[0], abs=1e-9)
    assert recovered_itrs[1] == pytest.approx(itrs[1], abs=1e-12)

    reference_position = np.array([7.0e6, 0.0, 0.0])
    reference_velocity = np.array([0.0, 7_500.0, 0.0])
    reference_acceleration = np.array([-8.0, 0.0, 0.0])
    target_position = reference_position + np.array([100.0, 200.0, -50.0])
    target_velocity = reference_velocity + np.array([0.1, -0.2, 0.3])
    target_acceleration = reference_acceleration + np.array([1e-3, 2e-3, -3e-3])
    relative = datum.to_rtn(
        reference_position,
        reference_velocity,
        reference_acceleration,
        target_position,
        target_velocity,
        target_acceleration,
    )
    recovered_target = datum.from_rtn(
        reference_position,
        reference_velocity,
        reference_acceleration,
        *relative,
    )
    assert recovered_target[0] == pytest.approx(target_position, abs=1e-9)
    assert recovered_target[1] == pytest.approx(target_velocity, abs=1e-12)
    assert recovered_target[2] == pytest.approx(target_acceleration, abs=1e-12)


def test_full_datum_matches_astropy_reference(monkeypatch):
    """The IAU datum reproduces April 2004 reference state."""
    epoch = Epoch.from_datetime(datetime(2004, 4, 6, 7, 51, 28, 386009, tzinfo=UTC))
    # Freeze the interpolated IERS so the test remains deterministic
    # and does not require network access.
    row = np.array(
        [
            epoch.mjd_utc,
            -0.14053799452617766,
            0.3344723979628682,
            -0.4404269373256251,
            1.4702788170412184,
            -0.10039846937358379,
            -0.05018529687821866,
        ]
    )
    monkeypatch.setattr(
        DEFAULT_EARTH_ORIENTATION,
        "_eop",
        np.vstack((row, row + np.array([1.0, 0, 0, 0, 0, 0, 0]))),
    )
    monkeypatch.setattr(DEFAULT_EARTH_ORIENTATION, "_prepared", True)
    monkeypatch.setattr(DEFAULT_EARTH_ORIENTATION, "_warned_bounds", set())

    position, velocity, acceleration = Datum()._to_eci_coordinates(
        epoch,
        np.array([-1033479.383, 7901295.2754, 6380356.5958]),
        np.array([-3225.63652, -2872.45145, 5531.924446]),
        np.array([0.99316485, -7.59307723, -6.13146816]),
    )

    assert epoch.earth_orientation == pytest.approx(
        [
            row[1] * ARCSEC_TO_RAD,
            row[2] * ARCSEC_TO_RAD,
            row[5] * ARCSEC_TO_RAD / 1000,
            row[6] * ARCSEC_TO_RAD / 1000,
        ],
        abs=1e-15,
    )
    assert position == pytest.approx(
        [5102509.1950914, 6123011.24923484, 6378136.88387638], abs=1e-4
    )
    assert velocity == pytest.approx(
        [-4743.22015896, 790.53672821, 5533.75574004], abs=3e-4
    )
    assert acceleration == pytest.approx(
        [-4.99161364, -6.54368085, -6.12927736], abs=1e-6
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
