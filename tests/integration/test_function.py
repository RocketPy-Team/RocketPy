import os
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Function

plt.rcParams.update({"figure.max_open_warning": 0})


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
        "test_func.csv",
        interpolation="linear" if func.get_domain_dim() == 1 else "shepard",
        extrapolation="natural",
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
    assert str(func_from_csv) == "Function from R1 to R1 : (Scalar) → (Scalar)"
    assert (
        str(func_2d_from_csv)
        == "Function from R2 to R1 : (Input 1, Input 2) → (Scalar)"
    )
    # Check the __repr__ method
    assert repr(func_from_csv) == "'Function from R1 to R1 : (Scalar) → (Scalar)'"
    assert (
        repr(func_2d_from_csv)
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
    assert repr(f) == "'Function from R1 to R1 : (time) → (value)'"
    assert np.isclose(f(0), 100)
    assert np.isclose(f(0) + f(1), 300), "Error summing the values of the function"


@patch("matplotlib.pyplot.show")
def test_plots(  # pylint: disable=unused-argument
    mock_show, func_from_csv, func_2d_from_csv
):
    """Test different plot methods of the Function class.

    Parameters
    ----------
    mock_show : Mock
        Mock of the matplotlib.pyplot.show method.
    func_from_csv : rocketpy.Function
        A Function object created from a .csv file.
    """
    # Test plot methods
    assert func_from_csv.plot() is None
    assert func_2d_from_csv.plot() is None
    # Test plot methods with limits
    assert func_from_csv.plot(-1, 1) is None
    assert func_2d_from_csv.plot(-1, 1) is None
    # Test compare_plots
    func2 = Function(
        source="data/airfoils/e473-10e6-degrees.csv",
        inputs=["Scalar"],
        outputs=["Scalar"],
        interpolation="linear",
        extrapolation="natural",
    )
    assert (
        func_from_csv.compare_plots([func_from_csv, func2], return_object=False) is None
    )


@patch("matplotlib.pyplot.show")
def test_multivariable_dataset_plot(mock_show):  # pylint: disable=unused-argument
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
    assert func.plot() is None


@patch("matplotlib.pyplot.show")
def test_multivariable_function_plot(mock_show):  # pylint: disable=unused-argument
    """Test the plot method of the Function class with a multivariable function."""

    def source(x, y):
        # Test plane f(x,y) = sin(x + y)
        return np.sin(x * y)

    func = Function(source=source, inputs=["x", "y"], outputs=["z"])

    # Assert plot
    assert func.plot() is None
