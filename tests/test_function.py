from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Function

plt.rcParams.update({"figure.max_open_warning": 0})


# Test Function creation from .csv file
def test_function_from_csv(func_from_csv, func_2d_from_csv):
    """Test the Function class creation from a .csv file.

    Parameters
    ----------
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    func_2d_from_csv : rocketpy.Function
        A Function object created from a .csv file with 2 inputs.
    """
    # Assert the function is zero at 0 but with a certain tolerance
    assert np.isclose(func_from_csv(0), 0.0, atol=1e-6)
    assert np.isclose(func_2d_from_csv(0, 0), 0.0, atol=1e-6)
    # Check the __str__ method
    assert func_from_csv.__str__() == "Function from R1 to R1 : (Scalar) → (Scalar)"
    assert (
        func_2d_from_csv.__str__()
        == "Function from R2 to R1 : (Input 1, Input 2) → (Scalar)"
    )
    # Check the __repr__ method
    assert func_from_csv.__repr__() == "'Function from R1 to R1 : (Scalar) → (Scalar)'"
    assert (
        func_2d_from_csv.__repr__()
        == "'Function from R2 to R1 : (Input 1, Input 2) → (Scalar)'"
    )


@pytest.mark.parametrize(
    "csv_file",
    [
        "tests/fixtures/function/1d_quotes.csv",
        "tests/fixtures/function/1d_no_quotes.csv",
    ],
)
def test_func_from_csv_with_header(csv_file):
    """Tests if a Function can be created from a CSV file with a single header
    line. It tests cases where the fields are separated by quotes and without
    quotes."""
    f = Function(csv_file)
    assert f.__repr__() == "'Function from R1 to R1 : (Scalar) → (Scalar)'"
    assert np.isclose(f(0), 100)
    assert np.isclose(f(0) + f(1), 300), "Error summing the values of the function"


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
    assert np.isclose(func_2d_from_csv.get_value(0, 0), 0.0, atol=1e-6)
    assert np.isclose(func_2d_from_csv.get_value_opt(0, 0), 0.0, atol=1e-6)


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
    func_2d_from_csv.set_extrapolation("zero")
    assert func_2d_from_csv.get_extrapolation_method() == "zero"


@patch("matplotlib.pyplot.show")
def test_plots(mock_show, func_from_csv, func_2d_from_csv):
    """Test different plot methods of the Function class.

    Parameters
    ----------
    mock_show : Mock
        Mock of the matplotlib.pyplot.show method.
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    """
    # Test plot methods
    assert func_from_csv.plot() == None
    assert func_2d_from_csv.plot() == None
    # Test plot methods with limits
    assert func_from_csv.plot(-1, 1) == None
    assert func_2d_from_csv.plot(-1, 1) == None
    # Test compare_plots
    func2 = Function(
        source="tests/fixtures/airfoils/e473-10e6-degrees.csv",
        inputs=["Scalar"],
        outputs=["Scalar"],
        interpolation="linear",
        extrapolation="natural",
    )
    assert (
        func_from_csv.compare_plots([func_from_csv, func2], return_object=False) == None
    )


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

    # Test natural
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


@pytest.mark.parametrize("a", [-1, -0.5, 0, 0.5, 1])
@pytest.mark.parametrize("b", [-1, -0.5, 0, 0.5, 1])
def test_multivariable_function(a, b):
    """Test the Function class with a multivariable function."""
    # Test plane f(x,y) = sin(x + y)
    source = lambda x, y: np.sin(x + y)
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])

    # Assert values
    assert np.isclose(func(a, b), np.sin(a + b), atol=1e-6)


@patch("matplotlib.pyplot.show")
def test_multivariable_dataset_plot(mock_show):
    """Test the plot method of the Function class with a multivariable dataset."""
    # Test plane f(x,y) = x - y
    source = [
        (-1, -1, -1),
        (-1, 0, -1),
        (-1, 1, -2),
        (0, 1, 1),
        (0, 0, 0),
        (0, 1, -1),
        (1, -1, 2),
        (1, 0, 1),
        (1, 1, 0),
    ]
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])

    # Assert plot
    assert func.plot() == None


@patch("matplotlib.pyplot.show")
def test_multivariable_function_plot(mock_show):
    """Test the plot method of the Function class with a multivariable function."""
    # Test plane f(x,y) = sin(x + y)
    source = lambda x, y: np.sin(x * y)
    func = Function(source=source, inputs=["x", "y"], outputs=["z"])

    # Assert plot
    assert func.plot() == None


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


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float_(0.1), np.array([1])])
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


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float_(0.1), np.array([1])])
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


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float_(0.1), np.array([1])])
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


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float_(0.1), np.array([1])])
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


@pytest.mark.parametrize("other", [1, 0.1, np.int_(1), np.float_(0.1), np.array([1])])
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
