from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy.motors import TankGeometry

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

    assert np.isclose(geometry.radius(0), expected_radius)
    assert np.isclose(geometry.total_height, expected_total_height)


@parametrize_fixtures
def test_tank_coordinates(params, request):
    """Test basic coordinate values of the tanks."""
    geometry, (_, height), _ = params
    geometry = request.getfixturevalue(geometry)

    expected_bottom = -height / 2
    expected_top = height / 2

    assert np.isclose(geometry.bottom, expected_bottom)
    assert np.isclose(geometry.top, expected_top)


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

    assert np.isclose(geometry.total_volume, expected_total_volume)


@parametrize_fixtures
def test_tank_volume(params, request):
    """Test the volume of the tanks at different heights comparing to the
    CAD generated values.
    """
    geometry, *_, file_path = params
    geometry = request.getfixturevalue(geometry)

    heights, expected_volumes = np.loadtxt(
        file_path, delimiter=",", skiprows=1, usecols=(0, 1), unpack=True
    )

    assert np.allclose(expected_volumes, geometry.volume(heights))


@parametrize_fixtures
def test_tank_centroid(params, request):
    """Test the centroid of the tanks at different heights comparing to the
    analytically calculated values.
    """
    geometry, *_, file_path = params
    geometry = request.getfixturevalue(geometry)

    heights, expected_volumes, expected_centroids = np.loadtxt(
        file_path, delimiter=",", skiprows=1, usecols=(0, 1, 2), unpack=True
    )

    # For higher accuracy: geometry.volume_moment(geometry.bottom, h)(h)
    assert np.allclose(
        expected_centroids * expected_volumes,
        geometry.volume_moment(geometry.bottom, geometry.top)(heights),
    )


@parametrize_fixtures
def test_tank_inertia(params, request):
    """Test the inertia of the tanks at different heights comparing to the
    analytically calculated values.
    """
    geometry, *_, file_path = params
    geometry = request.getfixturevalue(geometry)

    heights, expected_inertia = np.loadtxt(
        file_path, delimiter=",", skiprows=1, usecols=(0, 3), unpack=True
    )

    # For higher accuracy: geometry.Ix_volume(geometry.bottom, h)(h)
    assert np.allclose(
        expected_inertia[1:],
        geometry.Ix_volume(geometry.bottom, geometry.top)(heights[1:]),
        rtol=1e-5,
        atol=1e-9,
    )


@patch("matplotlib.pyplot.show")
def test_tank_geometry_plots_info(mock_show):  # pylint: disable=unused-argument
    assert TankGeometry({(0, 5): 1}).plots.all() is None
