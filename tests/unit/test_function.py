"""Unit tests for the Function class. Each method in tis module tests an 
individual method of the Function class. The tests are made on both the
expected behaviour and the return instances."""

import os

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


@pytest.mark.parametrize(
    "func_input, derivative_input, expected_derivative",
    [
        (1, 0, 0),  # Test case 1: Function(1)
        (lambda x: x, 0, 1),  # Test case 2: Function(lambda x: x)
        (lambda x: x**2, 1, 2),  # Test case 3: Function(lambda x: x**2)
    ],
)
def test_differentiate_complex_step(func_input, derivative_input, expected_derivative):
    """Test the differentiate_complex_step method of the Function class.

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
    assert isinstance(func.differentiate_complex_step(derivative_input), float)
    assert np.isclose(
        func.differentiate_complex_step(derivative_input), expected_derivative
    )


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


@pytest.mark.parametrize(
    "func",
    [
        "linearly_interpolated_func",
        "spline_interpolated_func",
        "func_2d_from_csv",
        "lambda_quad_func",
    ],
)
def test_savetxt(request, func):
    """Test the savetxt method of various Function objects.

    This test function verifies that the `savetxt` method correctly writes the
    function's data to a CSV file and that a new function object created from
    this file has the same data as the original function object.

    Notes
    -----
    The test performs the following steps:
    1. It invokes the `savetxt` method of the given function object.
    2. It then reads this file to create a new function object.
    3. The test asserts that the data of the new function matches the original.
    4. Finally, the test cleans up by removing the created CSV file.

    Raises
    ------
    AssertionError
        If the `savetxt` method fails to save the file, or if the data of the
        newly read function does not match the data of the original function.
    """
    func = request.getfixturevalue(func)
    assert (
        func.savetxt(
            filename="test_func.csv",
            lower=0,
            upper=9,
            samples=10,
            fmt="%.6f",
            delimiter=",",
            newline="\n",
            encoding=None,
        )
        is None
    ), "Couldn't save the file using the Function.savetxt method."

    read_func = Function(
        "test_func.csv", interpolation="linear", extrapolation="natural"
    )
    if callable(func.source):
        source = np.column_stack(
            (np.linspace(0, 9, 10), func.source(np.linspace(0, 9, 10)))
        )
        assert np.allclose(source, read_func.source)
    else:
        assert np.allclose(func.source, read_func.source)

    # clean up the file
    os.remove("test_func.csv")


@pytest.mark.parametrize("samples", [2, 50, 1000])
def test_set_discrete_mutator(samples):
    """Tests the set_discrete method of the Function class."""
    func = Function(lambda x: x**3)
    discretized_func = func.set_discrete(-10, 10, samples, mutate_self=True)

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert discretized_func.source.shape == (samples, 2)
    assert func.source.shape == (samples, 2)


@pytest.mark.parametrize("samples", [2, 50, 1000])
def test_set_discrete_non_mutator(samples):
    """Tests the set_discrete method of the Function class.
    The mutator argument is set to False.
    """
    func = Function(lambda x: x**3)
    discretized_func = func.set_discrete(-10, 10, samples, mutate_self=False)

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert discretized_func.source.shape == (samples, 2)
    assert callable(func.source)


def test_set_discrete_based_on_model_mutator(linear_func):
    """Tests the set_discrete_based_on_model method of the Function class.
    The mutator argument is set to True.
    """
    func = Function(lambda x: x**3)
    discretized_func = func.set_discrete_based_on_model(linear_func, mutate_self=True)

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert discretized_func.source.shape == (4, 2)
    assert func.source.shape == (4, 2)


def test_set_discrete_based_on_model_non_mutator(linear_func):
    """Tests the set_discrete_based_on_model method of the Function class.
    The mutator argument is set to False.
    """
    func = Function(lambda x: x**3)
    discretized_func = func.set_discrete_based_on_model(linear_func, mutate_self=False)

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert discretized_func.source.shape == (4, 2)
    assert callable(func.source)


@pytest.mark.parametrize(
    "x, y, expected_x, expected_y",
    [
        (
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([10, 20, 30, 40, 50000, 60]),
            np.array([1, 2, 3, 4, 6]),
            np.array([10, 20, 30, 40, 60]),
        ),
    ],
)
def test_remove_outliers_iqr(x, y, expected_x, expected_y):
    """Test the function remove_outliers_iqr which is expected to remove
    outliers from the data based on the Interquartile Range (IQR) method.
    """
    func = Function(source=np.column_stack((x, y)))
    filtered_func = func.remove_outliers_iqr(threshold=1.5)

    # Check if the outliers are removed
    assert np.array_equal(filtered_func.x_array, expected_x)
    assert np.array_equal(filtered_func.y_array, expected_y)

    # Check if the other attributes are preserved
    assert filtered_func.__inputs__ == func.__inputs__
    assert filtered_func.__outputs__ == func.__outputs__
    assert filtered_func.__interpolation__ == func.__interpolation__
    assert filtered_func.__extrapolation__ == func.__extrapolation__
    assert filtered_func.title == func.title


def test_set_get_value_opt():
    """Test the set_value_opt and get_value_opt methods of the Function class."""
    func = Function(lambda x: x**2)
    func.source = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25]])
    func.x_array = np.array([1, 2, 3, 4, 5])
    func.y_array = np.array([1, 4, 9, 16, 25])
    func.x_initial = 1
    func.x_final = 5
    func.set_interpolation("linear")
    func.set_get_value_opt()
    assert func.get_value_opt(2.5) == 6.5


def test_get_image_dim(linear_func):
    """Test the get_img_dim method of the Function class."""
    assert linear_func.get_image_dim() == 1


def test_get_domain_dim(linear_func):
    """Test the get_domain_dim method of the Function class."""
    assert linear_func.get_domain_dim() == 1


def test_bool(linear_func):
    """Test the __bool__ method of the Function class."""
    assert bool(linear_func) == True
