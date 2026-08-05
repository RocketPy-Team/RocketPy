import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.tools import (
    calculate_cubic_hermite_coefficients,
    euler313_to_quaternions,
    find_roots_cubic_function,
    haversine,
    inverted_haversine,
    tuple_handler,
)


@pytest.mark.parametrize(
    "angles, expected_quaternions",
    [((0, 0, 0), (1, 0, 0, 0)), ((90, 90, 90), (0, 0.7071068, 0, 0.7071068))],
)
def test_euler_to_quaternions(angles, expected_quaternions):
    q0, q1, q2, q3 = euler313_to_quaternions(*np.deg2rad(angles))
    assert round(q0, 7) == expected_quaternions[0]
    assert round(q1, 7) == expected_quaternions[1]
    assert round(q2, 7) == expected_quaternions[2]
    assert round(q3, 7) == expected_quaternions[3]


def test_calculate_cubic_hermite_coefficients():
    """Test the calculate_cubic_hermite_coefficients method of the Function class."""
    # Function: f(x) = x**3 + 2x**2 -1 ; derivative: f'(x) = 3x**2 + 4x
    x = np.array([-3, -2, -1, 0, 1])
    y = np.array([-10, -1, 0, -1, 2])

    # Selects two points as x0 and x1
    x0, x1 = 0, 1
    y0, y1 = -1, 2
    yp0, yp1 = 0, 7

    a, b, c, d = calculate_cubic_hermite_coefficients(x0, x1, y0, yp0, y1, yp1)

    assert np.isclose(a, 1)
    assert np.isclose(b, 2)
    assert np.isclose(c, 0)
    assert np.isclose(d, -1)
    assert np.allclose(
        a * x**3 + b * x**2 + c * x + d,
        y,
    )


def test_cardanos_root_finding():
    """Tests the find_roots_cubic_function method of the Function class."""
    # Function: f(x) = x**3 + 2x**2 -1
    # roots: (-1 - 5**0.5) / 2; -1; (-1 + 5**0.5) / 2

    roots = list(find_roots_cubic_function(a=1, b=2, c=0, d=-1))
    roots.sort(key=lambda x: x.real)

    assert np.isclose(roots[0].real, (-1 - 5**0.5) / 2)
    assert np.isclose(roots[1].real, -1)
    assert np.isclose(roots[2].real, (-1 + 5**0.5) / 2)

    assert np.isclose(roots[0].imag, 0)
    assert np.isclose(roots[1].imag, 0)
    assert np.isclose(roots[2].imag, 0)


@pytest.mark.parametrize(
    "lat0, lon0, lat1, lon1, expected_distance",
    [
        (0, 0, 0, 0, 0),
        (45, 45, 45, 45, 0),
        (-23.508958, -46.720080, -23.522939, -46.558253, 16591.438),
    ],
)  # These values were calculated with google earth
def test_haversine(lat0, lon0, lat1, lon1, expected_distance):
    distance = haversine(lat0, lon0, lat1, lon1)
    assert np.isclose(distance, expected_distance, rtol=1e-2)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (5, (0, 5)),
        (3.5, (0, 3.5)),
        ([7], (0, 7)),
        ((8,), (0, 8)),
        ([2, 4], (2, 4)),
        ((1, 3), (1, 3)),
    ],
)
def test_tuple_handler(input_value, expected_output):
    assert tuple_handler(input_value) == expected_output


@pytest.mark.parametrize(
    "input_value, expected_exception",
    [
        ([1, 2, 3], ValueError),
        ((4, 5, 6), ValueError),
    ],
)
def test_tuple_handler_exceptions(input_value, expected_exception):
    with pytest.raises(expected_exception):
        tuple_handler(input_value)


@pytest.mark.parametrize("pressure_conversion_factor", ["hPa", "mbar", "Pa", 100])
def test_valid_pressure_conversion_factor(pressure_conversion_factor):
    env = Environment(
        gravity=9.81,
        latitude=47.213476,
        longitude=9.003336,
        date=(2020, 2, 22, 13),
        elevation=407,
    )
    env.set_atmospheric_model(
        type="Reanalysis",
        file="data/weather/bella_lui_weather_data_ERA5.nc",
        dictionary="ECMWF",
        pressure_conversion_factor=pressure_conversion_factor,
    )


@pytest.mark.parametrize("pressure_conversion_factor", [-1, "mPa"])
def test_invalid_pressure_conversion_factor(pressure_conversion_factor):
    env = Environment(
        gravity=9.81,
        latitude=47.213476,
        longitude=9.003336,
        date=(2020, 2, 22, 13),
        elevation=407,
    )

    with pytest.raises(ValueError):
        env.set_atmospheric_model(
            type="Reanalysis",
            file="data/weather/bella_lui_weather_data_ERA5.nc",
            dictionary="ECMWF",
            pressure_conversion_factor=pressure_conversion_factor,
        )


def test_inverted_haversine_scalar():
    """Test inverted_haversine with scalar arguments matches haversine distance."""
    # Arrange
    lat0, lon0 = -23.508958, -46.720080
    lat1, lon1 = -23.522939, -46.558253
    earth_radius = 6378100.0
    distance = haversine(lat0, lon0, lat1, lon1, earth_radius)
    bearing = 90.0

    # Act
    lat_result, lon_result = inverted_haversine(
        lat0, lon0, distance, bearing, earth_radius
    )

    # Assert
    recalculated_distance = haversine(lat0, lon0, lat_result, lon_result, earth_radius)
    assert recalculated_distance == pytest.approx(distance, abs=1e-2)


def test_inverted_haversine_array():
    """Test inverted_haversine with NumPy arrays returns correct array results."""
    # Arrange
    lat0, lon0 = -23.508958, -46.720080
    distances = np.array([0.0, 5000.0, 16591.438])
    bearings = np.array([0.0, 45.0, 90.0])
    earth_radius = 6378100.0

    # Act
    lat_results, lon_results = inverted_haversine(
        lat0, lon0, distances, bearings, earth_radius
    )

    # Assert
    assert isinstance(lat_results, np.ndarray)
    assert isinstance(lon_results, np.ndarray)
    assert len(lat_results) == 3
    assert len(lon_results) == 3

    # Check scalar consistency for each element
    for i, distance in enumerate(distances):
        lat_scalar, lon_scalar = inverted_haversine(
            lat0, lon0, distance, bearings[i], earth_radius
        )
        assert lat_results[i] == pytest.approx(lat_scalar)
        assert lon_results[i] == pytest.approx(lon_scalar)
