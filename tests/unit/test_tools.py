import math

import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.tools import (
    _seed_sequence_to_int,
    calculate_confidence_ellipse,
    calculate_cubic_hermite_coefficients,
    convert_local_extent_to_wgs84,
    convert_mercator_extent_to_local,
    euler313_to_quaternions,
    find_roots_cubic_function,
    generate_monte_carlo_ellipses,
    haversine,
    inverted_haversine,
    mercator_to_wgs84,
    normalize_quaternions,
    quaternions_to_nutation,
    quaternions_to_precession,
    quaternions_to_spin,
    tuple_handler,
)


WEB_MERCATOR_EARTH_RADIUS = 6378137.0


def _wgs84_to_mercator(latitude, longitude):
    """Convert WGS84 coordinates to the spherical Mercator test fixture."""
    x = WEB_MERCATOR_EARTH_RADIUS * math.radians(longitude)
    y = WEB_MERCATOR_EARTH_RADIUS * math.log(
        math.tan(math.pi / 4 + math.radians(latitude) / 2)
    )
    return x, y


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


def test_quaternions_to_euler_angles_support_flight_arrays():
    quaternions = np.array(
        [
            (0.5, -(0.5**0.5), 0.0, 0.5),
            (0.5, -0.5, -0.5, 0.5),
        ]
    )
    e0, e1, e2, e3 = quaternions.T

    assert quaternions_to_precession(e0, e1, e2, e3) == pytest.approx([45, 90])
    assert quaternions_to_nutation(e1, e2) == pytest.approx([-90, -90])
    assert quaternions_to_spin(e0, e1, e2, e3) == pytest.approx([45, 0])


def test_normalize_quaternions_handles_scaled_and_zero_inputs():
    normalized = normalize_quaternions((1, 2, 3, 4))

    assert normalized == pytest.approx(np.array([1, 2, 3, 4]) / np.sqrt(30))
    assert np.linalg.norm(normalized) == pytest.approx(1)
    assert normalize_quaternions((0, 0, 0, 0)) == (1, 0, 0, 0)


def test_calculate_confidence_ellipse_axes():
    x = np.array([-2.0, -2.0, 2.0, 2.0])
    y = np.array([-1.0, 1.0, -1.0, 1.0])

    theta, width, height = calculate_confidence_ellipse(x, y, n_std=2)

    assert abs(np.cos(np.deg2rad(theta))) == pytest.approx(1)
    assert width == pytest.approx(4 * np.sqrt(16 / 3))
    assert height == pytest.approx(4 * np.sqrt(4 / 3))


def test_generate_monte_carlo_ellipses_builds_scaled_patches():
    apogee_x = np.array([8.0, 8.0, 12.0, 12.0])
    apogee_y = np.array([19.0, 21.0, 19.0, 21.0])
    impact_x = np.array([-8.0, -8.0, -2.0, -2.0])
    impact_y = np.array([2.5, 5.5, 2.5, 5.5])

    impact_ellipses, apogee_ellipses = generate_monte_carlo_ellipses(
        apogee_x,
        apogee_y,
        impact_x,
        impact_y,
        n_apogee=[1, 2],
        n_impact=[1],
        apogee_rgb=(0.2, 0.6, 0.4),
        impact_rgb=(0.8, 0.1, 0.3),
        opacity=0.35,
    )

    assert len(apogee_ellipses) == 2
    assert len(impact_ellipses) == 1
    assert apogee_ellipses[0].center == pytest.approx((10, 20))
    assert impact_ellipses[0].center == pytest.approx((-5, 4))
    assert apogee_ellipses[1].width == pytest.approx(2 * apogee_ellipses[0].width)
    assert apogee_ellipses[1].height == pytest.approx(2 * apogee_ellipses[0].height)
    assert apogee_ellipses[0].get_facecolor() == pytest.approx((0.2, 0.6, 0.4, 0.35))
    assert impact_ellipses[0].get_facecolor() == pytest.approx((0.8, 0.1, 0.3, 0.35))


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


@pytest.mark.parametrize(
    "x, y, expected_latitude, expected_longitude",
    [
        (0.0, 0.0, 0.0, 0.0),
        (
            20037508.342789244,
            20037508.342789244,
            85.0511287798066,
            180.0,
        ),
    ],
)
def test_mercator_to_wgs84_known_coordinates(
    x, y, expected_latitude, expected_longitude
):
    latitude, longitude = mercator_to_wgs84(
        x,
        y,
        earth_radius=WEB_MERCATOR_EARTH_RADIUS,
    )

    assert latitude == pytest.approx(expected_latitude)
    assert longitude == pytest.approx(expected_longitude)


def test_local_extent_round_trip_through_wgs84_and_mercator():
    origin_latitude = -23.5
    origin_longitude = -46.6
    local_extent = [-1000.0, 2000.0, -500.0, 1500.0]

    west, south, east, north = convert_local_extent_to_wgs84(
        local_extent,
        origin_latitude,
        origin_longitude,
        earth_radius=WEB_MERCATOR_EARTH_RADIUS,
    )
    min_x, min_y = _wgs84_to_mercator(south, west)
    max_x, max_y = _wgs84_to_mercator(north, east)
    recovered_extent = convert_mercator_extent_to_local(
        [min_x, max_x, min_y, max_y],
        origin_latitude,
        origin_longitude,
        earth_radius=WEB_MERCATOR_EARTH_RADIUS,
    )

    assert west < origin_longitude < east
    assert south < origin_latitude < north
    assert recovered_extent == pytest.approx(local_extent, abs=0.2)


@pytest.mark.parametrize(
    "geographic_extent, expected_sign",
    [
        ((-47.0, -46.8, -24.0, -23.8), -1),
        ((-46.4, -46.2, -23.3, -23.1), 1),
    ],
)
def test_mercator_extent_to_local_preserves_offset_sign(
    geographic_extent, expected_sign
):
    origin_latitude = -23.5
    origin_longitude = -46.6
    west, east, south, north = geographic_extent
    min_x, min_y = _wgs84_to_mercator(south, west)
    max_x, max_y = _wgs84_to_mercator(north, east)

    local_extent = convert_mercator_extent_to_local(
        [min_x, max_x, min_y, max_y],
        origin_latitude,
        origin_longitude,
        earth_radius=WEB_MERCATOR_EARTH_RADIUS,
    )

    assert local_extent[0] < local_extent[1]
    assert local_extent[2] < local_extent[3]
    assert all(expected_sign * value > 0 for value in local_extent)


def test_seed_sequence_to_int_keeps_the_full_width():
    """All four words have to reach the seed.

    Taking only the first one would still hand every component a different
    number, so every seeding test would pass over a 32-bit collapse that puts
    two streams back together near 2**16 of them.
    """
    root = np.random.SeedSequence(12345)
    a, b = root.spawn(2)
    words = a.generate_state(4, dtype=np.uint32)
    # Every word, in one comparison. Asserting only that the high bits are not
    # zero leaves a 64 or 96 bit truncation passing.
    expected = sum(int(word) << (32 * at) for at, word in enumerate(words))

    seed = _seed_sequence_to_int(a)

    assert seed == expected
    assert seed.bit_length() <= 128
    assert seed != _seed_sequence_to_int(b)
    assert seed == _seed_sequence_to_int(a), "reading it twice moved the seed"
