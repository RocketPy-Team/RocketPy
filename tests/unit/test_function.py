import numpy as np
import pytest

from rocketpy import Function


@pytest.mark.parametrize("a", [-1, 0, 0.5, 1, 2, 2.5, 3.5, 4, 5])
@pytest.mark.parametrize("b", [-1, 0, 0.5, 1, 2, 2.5, 3.5, 4, 5])
def test_integral_linear_interpolation(linearly_interpolated_func, a, b):
    """Test the integral method of the Function class.

    Parameters
    ----------
    linear_func : rocketpy.Function
        A Function object created from a list of values.
    """
    # Test integral
    assert isinstance(linearly_interpolated_func.integral(a, b, numerical=True), float)
    assert np.isclose(
        linearly_interpolated_func.integral(a, b, numerical=False),
        linearly_interpolated_func.integral(a, b, numerical=True),
        atol=1e-3,
    )


@pytest.mark.parametrize("func", ["linear_func", "spline_interpolated_func"])
@pytest.mark.parametrize("a", [-1, -0.5, 0, 0.5, 1, 2, 2.5, 3.5, 4, 5])
@pytest.mark.parametrize("b", [-1, -0.5, 0, 0.5, 1, 2, 2.5, 3.5, 4, 5])
def test_integral_spline_interpolation(request, func, a, b):
    """Test the integral method of the Function class.

    Parameters
    ----------
    spline_func : rocketpy.Function
        A Function object created from a list of values.
    a : float
        Lower limit of the integral.
    b : float
        Upper limit of the integral.
    """
    # Test integral
    # Get the function from the fixture
    func = request.getfixturevalue(func)
    assert np.isclose(
        func.integral(a, b, numerical=False),
        func.integral(a, b, numerical=True),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "x,y,z_expected",
    [
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0.5, 0.5, 1 / 3),
        (0.25, 0.25, 25 / (25 + 2 * 5**0.5)),
        ([0, 0.5], [0, 0.5], [1, 1 / 3]),
    ],
)
def test_2d_shepard_interpolation(x, y, z_expected):
    """Test the shepard interpolation method of the Function class."""
    # Test plane x + y + z = 1
    source = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])
    z = func(x, y)
    z_opt = func.get_value_opt(x, y)
    assert np.isclose(z, z_opt, atol=1e-8).all()
    assert np.isclose(z, z_expected, atol=1e-8).all()


@pytest.mark.parametrize(
    "x,y,z,w_expected",
    [
        (0, 0, 0, 1),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0.5, 0.5, 0.5, 1 / 4),
        (0.25, 0.25, 0.25, 0.700632626832),
        ([0, 0.5], [0, 0.5], [0, 0.5], [1, 1 / 4]),
    ],
)
def test_3d_shepard_interpolation(x, y, z, w_expected):
    """Test the shepard interpolation method of the Function class."""
    # Test plane x + y + z + w = 1
    source = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    func = Function(source=source, inputs=["x", "y", "z"], outputs=["w"])
    w = func(x, y, z)
    w_opt = func.get_value_opt(x, y, z)
    assert np.isclose(w, w_opt, atol=1e-8).all()
    assert np.isclose(w_expected, w, atol=1e-8).all()


@pytest.mark.parametrize(
    "func_input, derivative_input, expected_derivative",
    [
        (1, 0, 0),  # Test case 1: Function(1)
        (lambda x: x, 0, 1),  # Test case 2: Function(lambda x: x)
        (lambda x: x**2, 1, 2),  # Test case 3: Function(lambda x: x**2)
    ],
)
def test_differentiate(func_input, derivative_input, expected_derivative):
    """Test the differentiate method of the Function class.

    Parameters
    ----------
    func_input : function
        A function object created from a list of values.
    derivative_input : int
        Point at which to differentiate.
    expected_derivative : float
        Expected value of the derivative.
    """
    func = Function(func_input)
    assert isinstance(func.differentiate(derivative_input), float)
    assert np.isclose(func.differentiate(derivative_input), expected_derivative)


def test_get_value():
    """Tests the get_value method of the Function class.
    Both with respect to return instances and expected behaviour.
    """
    func = Function(lambda x: 2 * x)
    assert isinstance(func.get_value(1), int or float)


def test_identity_function():
    """Tests the identity_function method of the Function class.
    Both with respect to return instances and expected behaviour.
    """

    func = Function(lambda x: x**2)
    assert isinstance(func.identity_function(), Function)


def test_derivative_function():
    """Tests the derivative_function method of the Function class.
    Both with respect to return instances and expected behaviour.
    """
    square = Function(lambda x: x**2)
    assert isinstance(square.derivative_function(), Function)


def test_integral():
    """Tests the integral method of the Function class.
    Both with respect to return instances and expected behaviour.
    """

    zero_func = Function(0)
    assert isinstance(zero_func.integral(2, 4, numerical=True), float)
    assert zero_func.integral(2, 4, numerical=True) == 0

    square = Function(lambda x: x**2)
    assert isinstance
    assert square.integral(2, 4, numerical=True) == -square.integral(
        4, 2, numerical=True
    )
    assert square.integral(2, 4, numerical=False) == -square.integral(
        4, 2, numerical=False
    )


def test_integral_function():
    """Tests the integral_function method of the Function class.
    Both with respect to return instances and expected behaviour.
    """
    zero_func = Function(0)
    assert isinstance(zero_func, Function)


@pytest.mark.parametrize(
    "x, y, z",
    [
        (0, 0, 11.22540929),
        (1, 2, 10),
        (2, 3, 15.272727273),
        (3, 4, 20),
        (4, 5, 24.727272727),
        (5, 6, 30),
        (10, 10, 25.7201184),
    ],
)
def test_get_value_opt(x, y, z):
    """Test the get_value_opt method of the Function class. Currently only tests
    a 2D function with shepard interpolation method. The tests are made on
    points that are present or not in the source array.

    Parameters
    ----------
    x : scalar
        The x-coordinate.
    y : scalar
        The y-coordinate.
    z : float
        The expected interpolated value at (x, y).
    """
    x_data = np.array([1.0, 3.0, 5.0])
    y_data = np.array([2.0, 4.0, 6.0])
    z_data = np.array([10.0, 20.0, 30.0])
    source = np.column_stack((x_data, y_data, z_data))
    func = Function(source, interpolation="shepard", extrapolation="natural")
    assert isinstance(func.get_value_opt(x, y), float)
    assert np.isclose(func.get_value_opt(x, y), z, atol=1e-6)
