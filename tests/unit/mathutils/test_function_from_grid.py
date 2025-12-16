import numpy as np
import pytest

from rocketpy.mathutils.function import Function


def test_from_grid_unsupported_extrapolation_raises():
    """from_grid should reject unsupported extrapolation names with ValueError."""
    mach = np.array([0.0, 1.0])
    reynolds = np.array([1e5, 2e5])
    grid = np.zeros((mach.size, reynolds.size))

    with pytest.raises(ValueError):
        Function.from_grid(grid, [mach, reynolds], extrapolation="unsupported_mode")


def test_from_grid_is_multidimensional_property():
    """Ensure `is_multidimensional` is True for ND grid Functions and False for 1D."""
    mach = np.array([0.0, 0.5, 1.0])
    reynolds = np.array([1e5, 2e5, 3e5])
    alpha = np.array([0.0, 2.0])

    M, R, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    cd_data = 0.1 + 0.2 * M + 1e-7 * R + 0.01 * A

    func_nd = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    assert hasattr(func_nd, "is_multidimensional")
    assert func_nd.is_multidimensional is True

    # 1D Function constructed from a two-column array should not be multidimensional
    src = np.column_stack((mach, 0.5 + 0.1 * mach))
    func_1d = Function(src, inputs=["Mach"], outputs="Cd")
    assert hasattr(func_1d, "is_multidimensional")
    assert func_1d.is_multidimensional is False
