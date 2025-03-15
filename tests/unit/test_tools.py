import numpy as np
import pytest

from rocketpy.tools import (
    calculate_cubic_hermite_coefficients,
    euler313_to_quaternions,
    find_roots_cubic_function,
    haversine,
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
