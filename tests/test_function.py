from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Function

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.fixture
def linear_func():
    """Create a linear function based on a list of points. The function
    represents y = x and may be used on different tests.

    Returns
    -------
    Function
        A linear function representing y = x.
    """
    return Function(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
    )


@pytest.fixture
def func_from_csv():
    func = Function(
        source="tests/fixtures/airfoils/e473-10e6-degrees.csv",
        inputs=["Scalar"],
        outputs=["Scalar"],
        interpolation="linear",
        extrapolation="linear",
    )
    return func


# Test Function creation from .csv file
def test_function_from_csv(func_from_csv):
    func = func_from_csv
    # Assert func is zero at 0 but with tolerance
    assert np.isclose(func_from_csv(0), 0.0, atol=1e-6)
    # Check the __str__ method
    assert func_from_csv.__str__() == "Function from R1 to R1 : (Scalar) → (Scalar)"
    # Check the __repr__ method
    assert func_from_csv.__repr__() == "Function from R1 to R1 : (Scalar) → (Scalar)"


def test_getters(func_from_csv):
    assert func_from_csv.getInputs() == ["Scalar"]
    assert func_from_csv.getOutputs() == ["Scalar"]
    assert func_from_csv.getInterpolationMethod() == "linear"
    assert func_from_csv.getExtrapolationMethod() == "linear"
    assert np.isclose(func_from_csv.getValue(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.getValueOpt_deprecated(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.getValueOpt(0), 0.0, atol=1e-6)
    assert np.isclose(func_from_csv.getValueOpt2(0), 0.0, atol=1e-6)


def test_setters(func_from_csv):
    # Test set methods
    func_from_csv.setInputs(["Scalar2"])
    assert func_from_csv.getInputs() == ["Scalar2"]
    func_from_csv.setOutputs(["Scalar2"])
    assert func_from_csv.getOutputs() == ["Scalar2"]
    func_from_csv.setInterpolation("linear")
    assert func_from_csv.getInterpolationMethod() == "linear"
    func_from_csv.setExtrapolation("linear")
    assert func_from_csv.getExtrapolationMethod() == "linear"


@patch("matplotlib.pyplot.show")
def test_plots(mock_show, func_from_csv):
    # Test plot methods
    assert func_from_csv.plot() == None
    # Test comparePlots
    func2 = Function(
        source="tests/fixtures/airfoils/e473-10e6-degrees.csv",
        inputs=["Scalar"],
        outputs=["Scalar"],
        interpolation="linear",
        extrapolation="linear",
    )
    assert (
        func_from_csv.comparePlots([func_from_csv, func2], returnObject=False) == None
    )


def test_interpolation_methods(linear_func):

    # Test Akima
    linear_func.setInterpolation("akima")
    assert linear_func.getInterpolationMethod() == "akima"
    assert np.isclose(linear_func.getValue(0), 0.0, atol=1e-6)

    # Test polynomial
    linear_func.setInterpolation("polynomial")
    assert linear_func.getInterpolationMethod() == "polynomial"
    assert np.isclose(linear_func.getValue(0), 0.0, atol=1e-6)


def test_extrapolation_methods(linear_func):

    # Test zero
    linear_func.setExtrapolation("zero")
    assert linear_func.getExtrapolationMethod() == "zero"
    assert np.isclose(linear_func.getValue(-1), 0, atol=1e-6)

    # Test constant
    linear_func.setExtrapolation("constant")
    assert linear_func.getExtrapolationMethod() == "constant"
    assert np.isclose(linear_func.getValue(-1), 0, atol=1e-6)

    # Test natural
    linear_func.setExtrapolation("natural")
    assert linear_func.getExtrapolationMethod() == "natural"
    assert np.isclose(linear_func.getValue(-1), -1, atol=1e-6)


def test_integral(linear_func):
    # Test integral
    assert np.isclose(linear_func.integral(0, 1), 0.5, atol=1e-6)
