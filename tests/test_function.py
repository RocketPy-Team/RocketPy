from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Function

plt.rcParams.update({"figure.max_open_warning": 0})


# Test Function creation from .csv file
def test_function_from_csv(func_from_csv):
    """Test the Function class creation from a .csv file.

    Parameters
    ----------
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    """
    # Assert the function is zero at 0 but with a certain tolerance
    assert np.isclose(func_from_csv(0), 0.0, atol=1e-6)
    # Check the __str__ method
    assert func_from_csv.__str__() == "Function from R1 to R1 : (Scalar) → (Scalar)"
    # Check the __repr__ method
    assert func_from_csv.__repr__() == "'Function from R1 to R1 : (Scalar) → (Scalar)'"


def test_getters(func_from_csv):
    """Test the different getters of the Function class.

    Parameters
    ----------
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    """
    assert func_from_csv.get_inputs() == ["Scalar"]
    assert func_from_csv.get_outputs() == ["Scalar"]
    assert func_from_csv.get_interpolation_method() == "linear"
    assert func_from_csv.get_extrapolation_method() == "natural"
    assert np.isclose(func_from_csv.get_value(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.get_value_opt_deprecated(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.get_value_opt(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.get_value_opt2(0), 0.0, atol=1e-6)


def test_setters(func_from_csv):
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


@patch("matplotlib.pyplot.show")
def test_plots(mock_show, func_from_csv):
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
    """Test some of the interpolation methods of the Function class. Methods
    not tested here are already being called in other tests.

    Parameters
    ----------
    linear_func : rocketpy.Function
        A Function object created from a list of values.
    """
    # Test Akima
    linear_func.set_interpolation("akima")
    assert linear_func.get_interpolation_method() == "akima"
    assert np.isclose(linear_func.get_value(0), 0.0, atol=1e-6)

    # Test polynomial
    linear_func.set_interpolation("polynomial")
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
    linear_func.set_extrapolation("constant")
    assert linear_func.get_extrapolation_method() == "constant"
    assert np.isclose(linear_func.get_value(-1), 0, atol=1e-6)

    # Test natural
    linear_func.set_extrapolation("natural")
    assert linear_func.get_extrapolation_method() == "natural"
    assert np.isclose(linear_func.get_value(-1), -1, atol=1e-6)


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
