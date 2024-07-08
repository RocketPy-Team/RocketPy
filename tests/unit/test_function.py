"""Unit tests for the Function class. Each method in tis module tests an
individual method of the Function class. The tests are made on both the
expected behaviour and the return instances."""

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Function

plt.rcParams.update({"figure.max_open_warning": 0})


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
    "func_input, derivative_input, expected_first_derivative",
    [
        (1, 0, 0),  # Test case 1: Function(1)
        (lambda x: x, 0, 1),  # Test case 2: Function(lambda x: x)
        (lambda x: x**2, 1, 2),  # Test case 3: Function(lambda x: x**2)
        (lambda x: -(x**3), 2, -12),  # Test case 4: Function(lambda x: -x**3)
    ],
)
def test_differentiate_complex_step(
    func_input, derivative_input, expected_first_derivative
):
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
    assert isinstance(func.differentiate_complex_step(x=derivative_input), float)
    assert np.isclose(
        func.differentiate_complex_step(x=derivative_input, order=1),
        expected_first_derivative,
    )


def test_get_value():
    """Tests the get_value method of the Function class.
    Both with respect to return instances and expected behaviour.
    """
    func = Function(lambda x: 2 * x)
    assert isinstance(func.get_value(1), (int, float))


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
    assert bool(linear_func)


def test_getters(func_from_csv, func_2d_from_csv):
    """Test the different getters of the Function class.

    Parameters
    ----------
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    """
    assert func_from_csv.get_inputs() == ["Scalar"]
    assert func_from_csv.get_outputs() == ["Scalar"]
    assert func_from_csv.get_interpolation_method() == "spline"
    assert func_from_csv.get_extrapolation_method() == "constant"
    assert np.isclose(func_from_csv.get_value(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.get_value_opt(0), 0.0, atol=1e-6)

    assert func_2d_from_csv.get_inputs() == ["Input 1", "Input 2"]
    assert func_2d_from_csv.get_outputs() == ["Scalar"]
    assert func_2d_from_csv.get_interpolation_method() == "shepard"
    assert func_2d_from_csv.get_extrapolation_method() == "natural"
    assert np.isclose(func_2d_from_csv.get_value(0.1, 0.8), 0.058, atol=1e-6)
    assert np.isclose(func_2d_from_csv.get_value_opt(0.1, 0.8), 0.058, atol=1e-6)


def test_setters(func_from_csv, func_2d_from_csv):
    """Test the different setters of the Function class.

    Parameters
    ----------
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    """
    # Test set methods
    func_from_csv.set_inputs(["Scalar2"])
    assert func_from_csv.get_inputs() == ["Scalar2"]
    func_from_csv.set_outputs(["Scalar2"])
    assert func_from_csv.get_outputs() == ["Scalar2"]
    func_from_csv.set_interpolation("linear")
    assert func_from_csv.get_interpolation_method() == "linear"
    func_from_csv.set_extrapolation("natural")
    assert func_from_csv.get_extrapolation_method() == "natural"

    func_2d_from_csv.set_inputs(["Scalar1", "Scalar2"])
    assert func_2d_from_csv.get_inputs() == ["Scalar1", "Scalar2"]
    func_2d_from_csv.set_outputs(["Scalar3"])
    assert func_2d_from_csv.get_outputs() == ["Scalar3"]
    func_2d_from_csv.set_interpolation("shepard")
    assert func_2d_from_csv.get_interpolation_method() == "shepard"
    func_2d_from_csv.set_extrapolation("natural")
    assert func_2d_from_csv.get_extrapolation_method() == "natural"


def test_interpolation_methods(linear_func):
    """Tests some of the interpolation methods of the Function class. Methods
    not tested here are already being called in other tests.

    Parameters
    ----------
    linear_func : rocketpy.Function
        A Function object created from a list of values.
    """
    # Test Akima
    assert isinstance(linear_func.set_interpolation("akima"), Function)
    linear_func.set_interpolation("akima")
    assert isinstance(linear_func.get_interpolation_method(), str)
    assert linear_func.get_interpolation_method() == "akima"
    assert np.isclose(linear_func.get_value(0), 0.0, atol=1e-6)

    # Test polynomial

    assert isinstance(linear_func.set_interpolation("polynomial"), Function)
    linear_func.set_interpolation("polynomial")
    assert isinstance(linear_func.get_interpolation_method(), str)
    assert linear_func.get_interpolation_method() == "polynomial"
    assert np.isclose(linear_func.get_value(0), 0.0, atol=1e-6)


def test_extrapolation_methods(linear_func):
    """Test some of the extrapolation methods of the Function class. Methods
    not tested here are already being called in other tests.

    Parameters
    ----------
    linear_func : rocketpy.Function
        A Function object created from a list of values.
    """
    # Test zero
    linear_func.set_extrapolation("zero")
    assert linear_func.get_extrapolation_method() == "zero"
    assert np.isclose(linear_func.get_value(-1), 0, atol=1e-6)

    # Test constant
    assert isinstance(linear_func.set_extrapolation("constant"), Function)
    linear_func.set_extrapolation("constant")
    assert isinstance(linear_func.get_extrapolation_method(), str)
    assert linear_func.get_extrapolation_method() == "constant"
    assert np.isclose(linear_func.get_value(-1), 0, atol=1e-6)

    # Test natural for linear interpolation
    linear_func.set_interpolation("linear")
    assert isinstance(linear_func.set_extrapolation("natural"), Function)
    linear_func.set_extrapolation("natural")
    assert isinstance(linear_func.get_extrapolation_method(), str)
    assert linear_func.get_extrapolation_method() == "natural"
    assert np.isclose(linear_func.get_value(-1), -1, atol=1e-6)

    # Test natural for spline interpolation
    linear_func.set_interpolation("spline")
    assert isinstance(linear_func.set_extrapolation("natural"), Function)
    linear_func.set_extrapolation("natural")
    assert isinstance(linear_func.get_extrapolation_method(), str)
    assert linear_func.get_extrapolation_method() == "natural"
    assert np.isclose(linear_func.get_value(-1), -1, atol=1e-6)

    # Test natural for akima interpolation
    linear_func.set_interpolation("akima")
    assert isinstance(linear_func.set_extrapolation("natural"), Function)
    linear_func.set_extrapolation("natural")
    assert isinstance(linear_func.get_extrapolation_method(), str)
    assert linear_func.get_extrapolation_method() == "natural"
    assert np.isclose(linear_func.get_value(-1), -1, atol=1e-6)

    # Test natural for polynomial interpolation
    linear_func.set_interpolation("polynomial")
    assert isinstance(linear_func.set_extrapolation("natural"), Function)
    linear_func.set_extrapolation("natural")
    assert isinstance(linear_func.get_extrapolation_method(), str)
    assert linear_func.get_extrapolation_method() == "natural"
    assert np.isclose(linear_func.get_value(-1), -1, atol=1e-6)


@pytest.mark.parametrize("a", [-1, 0, 1])
@pytest.mark.parametrize("b", [-1, 0, 1])
def test_multivariable_dataset(a, b):
    """Test the Function class with a multivariable dataset."""
    # Test plane f(x,y) = x + y
    source = [
        (-1, -1, -2),
        (-1, 0, -1),
        (-1, 1, 0),
        (0, -1, -1),
        (0, 0, 0),
        (0, 1, 1),
        (1, -1, 0),
        (1, 0, 1),
        (1, 1, 2),
    ]
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])

    # Assert interpolation and extrapolation methods
    assert func.get_interpolation_method() == "shepard"
    assert func.get_extrapolation_method() == "natural"

    # Assert values
    assert np.isclose(func(a, b), a + b, atol=1e-6)


@pytest.mark.parametrize(
    "x,y,z_expected",
    [
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
    func = Function(
        source=source, inputs=["x", "y"], outputs=["z"], interpolation="shepard"
    )
    z = func(x, y)
    z_opt = func.get_value_opt(x, y)
    assert np.isclose(z, z_opt, atol=1e-8).all()
    assert np.isclose(z_expected, z, atol=1e-8).all()


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
    func = Function(
        source=source, inputs=["x", "y", "z"], outputs=["w"], interpolation="shepard"
    )
    w = func(x, y, z)
    w_opt = func.get_value_opt(x, y, z)
    assert np.isclose(w, w_opt, atol=1e-8).all()
    assert np.isclose(w_expected, w, atol=1e-8).all()


@pytest.mark.parametrize("a", [-1, -0.5, 0, 0.5, 1])
@pytest.mark.parametrize("b", [-1, -0.5, 0, 0.5, 1])
def test_multivariable_function(a, b):
    """Test the Function class with a multivariable function."""

    def source(x, y):
        return np.sin(x + y)

    func = Function(source=source, inputs=["x", "y"], outputs=["z"])

    # Assert values
    assert np.isclose(func(a, b), np.sin(a + b), atol=1e-6)


def test_set_discrete_2d():
    """Tests the set_discrete method of the Function for
    two dimensional domains.
    """
    func = Function(lambda x, y: x**2 + y**2)
    discretized_func = func.set_discrete([-5, -7], [8, 10], [50, 100])

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert discretized_func.source.shape == (50 * 100, 3)
    assert np.isclose(discretized_func.source[0, 0], -5)
    assert np.isclose(discretized_func.source[0, 1], -7)
    assert np.isclose(discretized_func.source[-1, 0], 8)
    assert np.isclose(discretized_func.source[-1, 1], 10)


def test_set_discrete_2d_simplified():
    """Tests the set_discrete method of the Function for
    two dimensional domains with simplified inputs.
    """
    source = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])
    discretized_func = func.set_discrete(-1, 1, 10)

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert discretized_func.source.shape == (100, 3)
    assert np.isclose(discretized_func.source[0, 0], -1)
    assert np.isclose(discretized_func.source[0, 1], -1)
    assert np.isclose(discretized_func.source[-1, 0], 1)
    assert np.isclose(discretized_func.source[-1, 1], 1)


def test_set_discrete_based_on_2d_model(func_2d_from_csv):
    """Tests the set_discrete_based_on_model method with a 2d model
    Function.
    """
    func = Function(lambda x, y: x**2 + y**2)
    discretized_func = func.set_discrete_based_on_model(func_2d_from_csv)

    assert isinstance(discretized_func, Function)
    assert isinstance(func, Function)
    assert np.array_equal(
        discretized_func.source[:, :2], func_2d_from_csv.source[:, :2]
    )
    assert discretized_func.__interpolation__ == func_2d_from_csv.__interpolation__
    assert discretized_func.__extrapolation__ == func_2d_from_csv.__extrapolation__


@pytest.mark.parametrize(
    "x,y,z_expected",
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0.5, 0.5, 1 / 3),
        (0.25, 0.25, 25 / (25 + 2 * 5**0.5)),
        ([0, 0.5], [0, 0.5], [1, 1 / 3]),
    ],
)
def test_shepard_interpolation(x, y, z_expected):
    """Test the shepard interpolation method of the Function class."""
    # Test plane x + y + z = 1
    source = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])
    z = func(x, y)
    assert np.isclose(z, z_expected, atol=1e-8).all()


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float64(0.1), np.array([1])])
def test_sum_arithmetic_priority(other):
    """Test the arithmetic priority of the add operation of the Function class,
    specially comparing to the numpy array operations.
    """
    func_lambda = Function(lambda x: x**2)
    func_array = Function([(0, 0), (1, 1), (2, 4)])

    assert isinstance(func_lambda + func_array, Function)
    assert isinstance(func_array + func_lambda, Function)
    assert isinstance(func_lambda + other, Function)
    assert isinstance(other + func_lambda, Function)
    assert isinstance(func_array + other, Function)
    assert isinstance(other + func_array, Function)


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float64(0.1), np.array([1])])
def test_sub_arithmetic_priority(other):
    """Test the arithmetic priority of the sub operation of the Function class,
    specially comparing to the numpy array operations.
    """
    func_lambda = Function(lambda x: x**2)
    func_array = Function([(0, 0), (1, 1), (2, 4)])

    assert isinstance(func_lambda - func_array, Function)
    assert isinstance(func_array - func_lambda, Function)
    assert isinstance(func_lambda - other, Function)
    assert isinstance(other - func_lambda, Function)
    assert isinstance(func_array - other, Function)
    assert isinstance(other - func_array, Function)


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float64(0.1), np.array([1])])
def test_mul_arithmetic_priority(other):
    """Test the arithmetic priority of the mul operation of the Function class,
    specially comparing to the numpy array operations.
    """
    func_lambda = Function(lambda x: x**2)
    func_array = Function([(0, 0), (1, 1), (2, 4)])

    assert isinstance(func_lambda * func_array, Function)
    assert isinstance(func_array * func_lambda, Function)
    assert isinstance(func_lambda * other, Function)
    assert isinstance(other * func_lambda, Function)
    assert isinstance(func_array * other, Function)
    assert isinstance(other * func_array, Function)


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float64(0.1), np.array([1])])
def test_truediv_arithmetic_priority(other):
    """Test the arithmetic priority of the truediv operation of the Function class,
    specially comparing to the numpy array operations.
    """
    func_lambda = Function(lambda x: x**2)
    func_array = Function([(1, 1), (2, 4)])

    assert isinstance(func_lambda / func_array, Function)
    assert isinstance(func_array / func_lambda, Function)
    assert isinstance(func_lambda / other, Function)
    assert isinstance(other / func_lambda, Function)
    assert isinstance(func_array / other, Function)
    assert isinstance(other / func_array, Function)


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float64(0.1), np.array([1])])
def test_pow_arithmetic_priority(other):
    """Test the arithmetic priority of the pow operation of the Function class,
    specially comparing to the numpy array operations.
    """
    func_lambda = Function(lambda x: x**2)
    func_array = Function([(0, 0), (1, 1), (2, 4)])

    assert isinstance(func_lambda**func_array, Function)
    assert isinstance(func_array**func_lambda, Function)
    assert isinstance(func_lambda**other, Function)
    assert isinstance(other**func_lambda, Function)
    assert isinstance(func_array**other, Function)
    assert isinstance(other**func_array, Function)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_low_pass_filter(alpha):
    """Test the low_pass_filter method of the Function class.

    Parameters
    ----------
    alpha : float
        Attenuation coefficient, 0 < alpha < 1.
    """
    # Create a test function, sinus here
    source = np.array(
        [(1, np.sin(1)), (2, np.sin(2)), (3, np.sin(3)), (4, np.sin(4)), (5, np.sin(5))]
    )
    func = Function(source)

    # Apply low pass filter
    filtered_func = func.low_pass_filter(alpha)

    # Check that the method works as intended and returns the right object with no issue
    assert isinstance(filtered_func, Function), "The returned type is not a Function"
    assert np.array_equal(
        filtered_func.source[0], source[0]
    ), "The initial value is not the expected value"
    assert len(filtered_func.source) == len(
        source
    ), "The filtered Function and the Function have different lengths"
    assert (
        filtered_func.__interpolation__ == func.__interpolation__
    ), "The interpolation method was unexpectedly changed"
    assert (
        filtered_func.__extrapolation__ == func.__extrapolation__
    ), "The extrapolation method was unexpectedly changed"
    for i in range(1, len(source)):
        expected = alpha * source[i][1] + (1 - alpha) * filtered_func.source[i - 1][1]
        assert np.isclose(filtered_func.source[i][1], expected, atol=1e-6), (
            f"The filtered value at index {i} is not the expected value. "
            f"Expected: {expected}, Actual: {filtered_func.source[i][1]}"
        )
