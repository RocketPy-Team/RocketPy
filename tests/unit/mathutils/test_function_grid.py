"""Unit tests for Function.from_grid() method and grid interpolation."""

import warnings

import numpy as np
import pytest

from rocketpy import Function


def test_from_grid_1d():
    """Test from_grid with 1D data (edge case)."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2

    func = Function.from_grid(y_data, [x], inputs=["x"], outputs="y")

    # Test interpolation
    assert abs(func(1.5) - 2.25) < 0.5  # Should be close to 1.5^2


def test_from_grid_2d():
    """Test from_grid with 2D data."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])

    # Create grid: f(x, y) = x + 2*y
    X, Y = np.meshgrid(x, y, indexing="ij")
    z_data = X + 2 * Y

    func = Function.from_grid(z_data, [x, y], inputs=["x", "y"], outputs="z")

    # Test exact points
    assert func(0.0, 0.0) == 0.0
    assert func(1.0, 1.0) == 3.0
    assert func(2.0, 2.0) == 6.0

    # Test interpolation
    result = func(1.0, 0.5)
    expected = 1.0 + 2 * 0.5  # = 2.0
    assert abs(result - expected) < 0.01


def test_from_grid_3d_drag_coefficient():
    """Test from_grid with 3D drag coefficient data (Mach, Reynolds, Alpha)."""
    # Create sample aerodynamic data
    mach = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    reynolds = np.array([1e5, 5e5, 1e6])
    alpha = np.array([0.0, 2.0, 4.0, 6.0])

    # Create a simple drag coefficient model
    # Cd increases with Mach and alpha, slight dependency on Reynolds
    M, Re, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    cd_data = 0.3 + 0.1 * M - 1e-7 * Re + 0.01 * A

    cd_func = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    # Test at grid points
    assert abs(cd_func(0.0, 1e5, 0.0) - 0.29) < 0.01  # 0.3 - 1e-7*1e5
    assert abs(cd_func(1.0, 5e5, 0.0) - 0.35) < 0.01  # 0.3 + 0.1*1 - 1e-7*5e5

    # Test interpolation between points
    result = cd_func(0.5, 3e5, 1.0)
    # Expected roughly: 0.3 + 0.1*0.5 - 1e-7*3e5 + 0.01*1.0 = 0.32
    assert 0.31 < result < 0.34


def test_from_grid_extrapolation_constant():
    """Test that constant extrapolation clamps to edge values."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])  # y = x^2

    func = Function.from_grid(
        y, [x], inputs=["x"], outputs="y", extrapolation="constant"
    )

    # Test below lower bound - should return value at x=0
    assert func(-1.0) == 0.0

    # Test above upper bound - should return value at x=2
    assert func(3.0) == 4.0


def test_from_grid_validation_errors():
    """Test that from_grid raises appropriate errors for invalid inputs."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])

    # Mismatched dimensions
    X, Y = np.meshgrid(x, y, indexing="ij")
    z_data = X + Y

    # Wrong number of axes
    with pytest.raises(ValueError, match="Number of axes"):
        Function.from_grid(z_data, [x], inputs=["x"], outputs="z")

    # Wrong axis length
    with pytest.raises(ValueError, match="Axis 1 has"):
        Function.from_grid(
            z_data, [x, np.array([0.0, 1.0])], inputs=["x", "y"], outputs="z"
        )

    # Wrong number of inputs
    with pytest.raises(ValueError, match="Number of inputs"):
        Function.from_grid(z_data, [x, y], inputs=["x"], outputs="z")


def test_from_grid_default_inputs():
    """Test that from_grid uses default input names when not provided."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])

    X, Y = np.meshgrid(x, y, indexing="ij")
    z_data = X + Y

    func = Function.from_grid(z_data, [x, y])

    # Should use default names
    assert "x0" in func.__inputs__
    assert "x1" in func.__inputs__


def test_from_grid_backward_compatibility():
    """Test that regular Function creation still works after adding from_grid."""
    # Test 1D function from list
    func1 = Function([[0, 0], [1, 1], [2, 4], [3, 9]])
    assert func1(1.5) > 0  # Should interpolate

    # Test 2D function from array
    data = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3]])
    func2 = Function(data)
    assert func2(0.5, 0.5) > 0  # Should interpolate

    # Test callable function
    func3 = Function(lambda x: x**2)
    assert func3(2) == 4


def test_regular_grid_without_grid_interpolator_warns():
    """Test that setting `regular_grid` without a grid interpolator warns.

    This test constructs a Function from scattered points (no structured
    grid). If `regular_grid` interpolation is later selected without a
    grid interpolator being configured, the implementation currently
    falls back to shepard interpolation and should emit a warning. The
    test ensures a warning is raised in this scenario.
    """
    # Create a 2D function with scattered points (not structured grid)
    source = [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]
    func = Function(
        source=source, inputs=["x", "y"], outputs="z", interpolation="shepard"
    )

    # Now manually change interpolation to regular_grid without setting up the grid
    # This simulates the fallback scenario
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        func.set_interpolation("regular_grid")

        # Check that a warning was issued
        assert len(w) == 1
        assert "falling back to shepard interpolation" in str(w[0].message)


def test_shepard_fallback_2d_interpolation():
    """Test that shepard_fallback produces correct interpolation for 2D data.

    This test verifies the fallback interpolation works correctly when
    regular_grid is set without a grid interpolator.
    """
    # Create a 2D function: z = x + y
    source = [
        (0, 0, 0),  # f(0, 0) = 0
        (1, 0, 1),  # f(1, 0) = 1
        (0, 1, 1),  # f(0, 1) = 1
        (1, 1, 2),  # f(1, 1) = 2
    ]

    # First, create with shepard to get baseline results
    func_shepard = Function(
        source=source, inputs=["x", "y"], outputs="z", interpolation="shepard"
    )

    # Create another function and trigger the fallback
    func_fallback = Function(
        source=source, inputs=["x", "y"], outputs="z", interpolation="shepard"
    )

    # Trigger fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings for this test
        func_fallback.set_interpolation("regular_grid")

    # Test that both produce the same results at exact points
    assert func_fallback(0, 0) == func_shepard(0, 0)
    assert func_fallback(1, 1) == func_shepard(1, 1)

    # Test interpolation at an intermediate point
    result_fallback = func_fallback(0.5, 0.5)
    result_shepard = func_shepard(0.5, 0.5)
    assert np.isclose(result_fallback, result_shepard, atol=1e-6)


def test_shepard_fallback_3d_interpolation():
    """Test that shepard_fallback produces correct interpolation for 3D data.

    This test verifies the fallback interpolation works correctly for
    3-dimensional input data.
    """
    # Create a 3D function: w = x + y + z
    source = [
        (0, 0, 0, 0),  # f(0, 0, 0) = 0
        (1, 0, 0, 1),  # f(1, 0, 0) = 1
        (0, 1, 0, 1),  # f(0, 1, 0) = 1
        (0, 0, 1, 1),  # f(0, 0, 1) = 1
        (1, 1, 1, 3),  # f(1, 1, 1) = 3
    ]

    # Create with shepard to get baseline results
    func_shepard = Function(
        source=source,
        inputs=["x", "y", "z"],
        outputs="w",
        interpolation="shepard",
    )

    # Create another function and trigger the fallback
    func_fallback = Function(
        source=source,
        inputs=["x", "y", "z"],
        outputs="w",
        interpolation="shepard",
    )

    # Trigger fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        func_fallback.set_interpolation("regular_grid")

    # Test that both produce the same results at exact points
    assert func_fallback(0, 0, 0) == func_shepard(0, 0, 0)
    assert func_fallback(1, 1, 1) == func_shepard(1, 1, 1)

    # Test interpolation at an intermediate point
    result_fallback = func_fallback(0.5, 0.5, 0.5)
    result_shepard = func_shepard(0.5, 0.5, 0.5)
    assert np.isclose(result_fallback, result_shepard, atol=1e-6)


def test_shepard_fallback_at_exact_data_points():
    """Test that shepard_fallback returns exact values at data points.

    When querying at exact data points, the fallback should return the
    exact value stored at that point.
    """
    # Create a 2D function
    source = [
        (0, 0, 10),
        (1, 0, 20),
        (0, 1, 30),
        (1, 1, 40),
    ]

    func = Function(
        source=source, inputs=["x", "y"], outputs="z", interpolation="shepard"
    )

    # Trigger fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        func.set_interpolation("regular_grid")

    # Test exact data points - should return exact values
    assert func(0, 0) == 10
    assert func(1, 0) == 20
    assert func(0, 1) == 30
    assert func(1, 1) == 40


def test_from_grid_unsorted_axis_warns():
    """Test that from_grid warns when axes are not sorted in ascending order."""
    y_data = np.array([0.0, 1.0, 4.0])

    # Test with unsorted axis (descending order)
    unsorted_axis = np.array([2.0, 1.0, 0.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Function.from_grid(y_data, [unsorted_axis], inputs=["x"], outputs="y")

        # Check that a warning was issued
        assert len(w) == 1
        assert "not strictly sorted in ascending order" in str(w[0].message)


def test_from_grid_repeated_values_warns():
    """Test that from_grid warns when axes have repeated values.

    Note: RegularGridInterpolator requires strictly ascending or descending
    axes. Repeated values will cause scipy to raise a ValueError after our
    warning is issued.
    """
    y_data = np.array([0.0, 1.0, 4.0])

    # Test with repeated values (not strictly ascending)
    repeated_axis = np.array([0.0, 1.0, 1.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Scipy will raise ValueError after our warning, so we expect both
        try:
            Function.from_grid(y_data, [repeated_axis], inputs=["x"], outputs="y")
        except ValueError as e:
            # scipy raises this error for non-strictly-sorted axes
            assert "strictly ascending" in str(e).lower() or "dimension 0" in str(e)

        # Check that a warning was issued before the error
        assert len(w) == 1
        assert "not strictly sorted in ascending order" in str(w[0].message)


def test_from_grid_flatten_for_compatibility_false():
    """Test that flatten_for_compatibility=False skips flattening."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])

    X, Y = np.meshgrid(x, y, indexing="ij")
    z_data = X + Y

    func = Function.from_grid(
        z_data,
        [x, y],
        inputs=["x", "y"],
        outputs="z",
        flatten_for_compatibility=False,
    )

    # Check that flattened attributes are None
    assert func._domain is None
    assert func._image is None
    assert func.source is None
    assert func.y_array is None

    # But the function should still work correctly
    assert func(0.0, 0.0) == 0.0
    assert func(1.0, 1.0) == 2.0
    assert func(2.0, 1.0) == 3.0
