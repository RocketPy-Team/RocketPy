from pathlib import Path

import numpy as np
import pytest

PRESSURANT_PARAMS = (0.135 / 2, 0.981)
PROPELLANT_PARAMS = (0.0744, 0.8068)
SPHERICAL_PARAMS = (0.05, 0.1)

BASE_PATH = Path("tests/fixtures/motor/data/")

parametrize_fixtures = pytest.mark.parametrize(
    "params",
    [
        (
            "pressurant_tank_geometry",
            PRESSURANT_PARAMS,
            BASE_PATH / "cylindrical_pressurant_tank_expected.csv",
        ),
        (
            "propellant_tank_geometry",
            PROPELLANT_PARAMS,
            BASE_PATH / "cylindrical_oxidizer_tank_expected.csv",
        ),
        (
            "spherical_oxidizer_geometry",
            SPHERICAL_PARAMS,
            BASE_PATH / "spherical_oxidizer_tank_expected.csv",
        ),
    ],
)


@parametrize_fixtures
def test_tank_bounds(params, request):
    """Test basic geometric properties of the tanks."""
    geometry, (expected_radius, expected_height), _ = params
    geometry = request.getfixturevalue(geometry)

    expected_total_height = expected_height

    assert np.isclose(geometry.radius(0), expected_radius, atol=1e-6)
    assert np.isclose(geometry.total_height, expected_total_height, atol=1e-6)


@parametrize_fixtures
def test_tank_coordinates(params, request):
    """Test basic coordinate values of the tanks."""
    geometry, (_, height), _ = params
    geometry = request.getfixturevalue(geometry)

    expected_bottom = -height / 2
    expected_top = height / 2

    assert np.isclose(geometry.bottom, expected_bottom, atol=1e-6)
    assert np.isclose(geometry.top, expected_top, atol=1e-6)


@parametrize_fixtures
def test_tank_total_volume(params, request):
    """Test the total volume of the tanks comparing to the analytically
    calculated values.
    """
    geometry, (radius, height), _ = params
    geometry = request.getfixturevalue(geometry)

    expected_total_volume = (
        np.pi * radius**2 * (height - 2 * radius) + 4 / 3 * np.pi * radius**3
    )

    assert np.isclose(geometry.total_volume, expected_total_volume, atol=1e-6)


@parametrize_fixtures
def test_tank_volume(params, request):
    """Test the volume of the tanks at different heights comparing to the
    CAD generated values.
    """
    geometry, *_, file_path = params
    geometry = request.getfixturevalue(geometry)

    expected_data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    heights = expected_data[:, 0]
    expected_volumes = expected_data[:, 1]

    assert np.allclose(expected_volumes, geometry.volume(heights), atol=1e-6)


@parametrize_fixtures
def test_tank_centroid(params, request):
    """Test the centroid of the tanks at different heights comparing to the
    analytically calculated values.
    """
    geometry, *_, file_path = params
    geometry = request.getfixturevalue(geometry)

    expected_data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    heights = expected_data[:, 0]
    expected_volumes = expected_data[:, 1]
    expected_centroids = expected_data[:, 2]

    for i, h in enumerate(heights[1:], 1):  # Avoid empty geometry
        # Loss of accuracy when volume is close to zero
        assert np.isclose(
            expected_centroids[i],
            geometry.volume_moment(geometry.bottom, h)(h) / expected_volumes[i],
            atol=1e-3,
        )


@parametrize_fixtures
def test_tank_inertia(params, request):
    """Test the inertia of the tanks at different heights comparing to the
    analytically calculated values.
    """
    geometry, *_, file_path = params
    geometry = request.getfixturevalue(geometry)

    expected_data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    heights = expected_data[:, 0]
    expected_inertia = expected_data[:, 3]

    for i, h in enumerate(heights):  # Avoid empty geometry
        assert np.isclose(
            expected_inertia[i],
            geometry.Ix_volume(geometry.bottom, h)(h),
            atol=1e-5,
        )
