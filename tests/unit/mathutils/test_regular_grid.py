"""Tests of the regular-grid interpolation: the ``_RegularGrid`` helper on its
own, and the way ``Function`` finds a grid in its data and uses it."""

import warnings

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from rocketpy import Function
from rocketpy.mathutils._regular_grid import _RegularGrid
from rocketpy.units import convert_units


def _points(axes, values):
    """Write values on a grid as a table of points, one row per node."""
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh] + [np.ravel(values)])


@pytest.fixture
def plane_points():
    """``z = x + 10 * y`` on a 4x4 grid, as a table of points."""
    axis = np.linspace(0.0, 3.0, 4)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    return _points([axis, axis], x_grid + 10 * y_grid)


# ----------------------------------------------------------------- _RegularGrid


def test_from_points_reads_rows_in_any_order():
    axes = [np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0]), np.array([5.0, 7.0])]
    values = np.arange(12.0).reshape(3, 2, 2)
    rows = _points(axes, values)
    shuffled = rows[np.random.default_rng(0).permutation(len(rows))]

    grid = _RegularGrid.from_points(shuffled)

    assert grid is not None
    for found, expected in zip(grid.axes, axes):
        assert np.array_equal(found, expected)
    assert np.array_equal(grid.values, values)


@pytest.mark.parametrize(
    "points",
    [
        _points([[0.0, 1.0, 2.0], [0.0, 1.0]], np.arange(6.0))[:-1],  # a node missing
        np.vstack(
            [
                _points([[0.0, 1.0], [0.0, 1.0]], np.arange(4.0))[:-1],
                [[0.0, 0.0, 9.0]],  # a node repeated in place of another
            ]
        ),
        np.array([[0.0, 5.0, 1.0], [1.0, 5.0, 2.0], [2.0, 5.0, 3.0]]),  # y never varies
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]),  # a single variable
        np.array(
            [[0.0, 0.0, 1.0], [1.0, np.nan, 2.0], [0.0, 1.0, 3.0], [1.0, 1.0, 4.0]]
        ),
    ],
)
def test_from_points_rejects_what_is_not_a_grid(points):
    assert _RegularGrid.from_points(points) is None


@pytest.mark.parametrize("method", ["shepard", "rbf"])
def test_from_points_leaves_scattered_methods_alone(plane_points, method):
    assert _RegularGrid.from_points(plane_points, method) is None


@pytest.mark.parametrize(
    "requested, expected",
    [(None, "linear"), ("linear", "linear"), ("spline", "cubic"), ("akima", "pchip")],
)
def test_method_names_are_mapped_onto_the_grid(plane_points, requested, expected):
    assert _RegularGrid.from_points(plane_points, requested).method == expected


@pytest.mark.parametrize("method", ["akima", "spline", "pchip", "cubic"])
def test_a_coarse_grid_falls_back_to_linear(method):
    """The smooth methods need four points per axis. SciPy raises below that."""
    axis = np.linspace(0.0, 2.0, 3)
    points = _points([axis, axis], np.zeros((3, 3)))
    with pytest.warns(UserWarning, match="falling back to 'linear'"):
        grid = _RegularGrid.from_points(points, method, name="cN")
    assert grid.method == "linear"


def test_an_unknown_method_says_so(plane_points):
    with pytest.warns(UserWarning, match="not supported for the regular grid"):
        grid = _RegularGrid.from_points(plane_points, "splien")
    assert grid.method == "linear"


@pytest.mark.parametrize("extrapolation", ["constant", "natural", "zero"])
@pytest.mark.parametrize("sizes", [(5, 2), (4, 3, 6), (3, 2, 4, 3)])
def test_single_point_lookup_matches_scipy(sizes, extrapolation):
    """The single-point evaluation must give SciPy's values inside the grid, on
    its lines and edges, and outside it for every extrapolation."""
    rng = np.random.default_rng(len(sizes))
    axes = [np.sort(rng.uniform(-1, 1, size)) for size in sizes]
    values = rng.normal(size=sizes)
    grid = _RegularGrid(axes, values, "linear", extrapolation)
    reference = RegularGridInterpolator(
        axes, values, bounds_error=False, fill_value=None
    )

    points = rng.uniform(-1.6, 1.6, size=(200, len(sizes)))
    on_the_grid = [
        [axis[0] for axis in axes],
        [axis[-1] for axis in axes],
        [axis[len(axis) // 2] for axis in axes],
    ]
    points = np.vstack([points, on_the_grid])

    for point in points:
        inside = all(axis[0] <= x <= axis[-1] for axis, x in zip(axes, point))
        if extrapolation == "zero" and not inside:
            expected = 0.0
        elif extrapolation == "constant":
            clamped = [np.clip(x, axis[0], axis[-1]) for axis, x in zip(axes, point)]
            expected = reference(clamped)[0]
        else:
            expected = reference(point)[0]
        assert grid.evaluate(*point) == pytest.approx(expected, rel=1e-10, abs=1e-10)

    columns = [points[:, i] for i in range(len(sizes))]
    one_by_one = [grid.evaluate(*point) for point in points]
    assert grid.evaluate(*columns) == pytest.approx(one_by_one)


def test_points_from_axes_accepts_any_axis_order():
    rows = _RegularGrid.points_from_axes(
        [[2.0, 0.0, 1.0], [0.0, 1.0]], [[4.0, 5.0], [0.0, 1.0], [2.0, 3.0]]
    )
    grid = _RegularGrid.from_points(rows)
    assert np.array_equal(grid.values, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])


# --------------------------------------------------------------------- Function


def test_function_finds_the_grid_in_points_and_csv(plane_points, tmp_path):
    from_points = Function(plane_points[::-1], ["x", "y"], "z")

    filename = tmp_path / "plane.csv"
    np.savetxt(filename, plane_points, delimiter=",", header="x,y,z", comments="")
    from_csv = Function(str(filename))

    for function in (from_points, from_csv):
        assert function.is_regular_grid
        assert function.get_interpolation_method() == "linear"
        assert function(1.5, 0.25) == pytest.approx(4.0)
        assert function.get_value_opt(1.5, 0.25) == pytest.approx(4.0)


def test_function_keeps_the_order_of_the_given_points(plane_points):
    rows = plane_points[::-1]
    assert np.array_equal(Function(rows, ["x", "y"], "z").get_source(), rows)


def test_incomplete_grid_stays_scattered(plane_points):
    function = Function(plane_points[:-1], ["x", "y"], "z")
    assert not function.is_regular_grid
    assert function.get_interpolation_method() == "shepard"


def test_one_input_is_never_a_grid():
    function = Function([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    assert not function.is_regular_grid
    assert function.get_interpolation_method() == "spline"


@pytest.mark.parametrize("method", ["shepard", "rbf"])
def test_scattered_interpolation_is_an_escape_hatch(plane_points, method):
    function = Function(plane_points, ["x", "y"], "z", interpolation=method)
    assert not function.is_regular_grid
    assert function.get_interpolation_method() == method


def test_changing_the_method_reads_the_data_again(plane_points):
    function = Function(plane_points, ["x", "y"], "z")

    function.set_interpolation("shepard")
    assert not function.is_regular_grid
    assert function(1.5, 0.25) != pytest.approx(4.0, abs=1e-9)

    function.set_interpolation("akima")
    assert function.is_regular_grid
    assert function.get_interpolation_method() == "pchip"
    assert function(1.5, 0.25) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "extrapolation, expected", [("zero", 0.0), ("constant", 33.0), ("natural", 44.0)]
)
def test_changing_the_extrapolation_reaches_the_grid(
    plane_points, extrapolation, expected
):
    function = Function(plane_points, ["x", "y"], "z")
    function.set_extrapolation(extrapolation)
    assert function.get_value_opt(4.0, 4.0) == pytest.approx(expected)
    assert function(4.0, 4.0) == pytest.approx(expected)


def test_arithmetic_keeps_the_grid(plane_points):
    grid = Function(plane_points, ["x", "y"], "z", interpolation="akima")
    value = grid.get_value_opt(1.5, 0.25)
    results = [
        (grid * 2, 2 * value),
        (2 * grid, 2 * value),
        (grid + 1, value + 1),
        (1 - grid, 1 - value),
        (grid / 2, value / 2),
        (-grid, -value),
        (grid + grid, 2 * value),
    ]
    for result, expected in results:
        assert result.is_regular_grid
        assert result.get_interpolation_method() == "pchip"
        assert result.get_value_opt(1.5, 0.25) == pytest.approx(expected)


def test_copying_a_function_keeps_its_grid_and_method(plane_points):
    grid = Function(plane_points, ["x", "y"], "z", interpolation="akima")
    copy = Function(grid)
    assert copy.is_regular_grid
    assert copy.get_interpolation_method() == "pchip"
    assert copy(1.5, 0.25) == pytest.approx(grid(1.5, 0.25))


def test_a_new_source_is_read_on_its_own_terms(plane_points):
    function = Function(plane_points, ["x", "y"], "z")
    function.set_source(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [2.0, 2.0, 9.0]])
    )
    assert not function.is_regular_grid
    assert function(0.0, 0.0) == pytest.approx(1.0)


def test_discretizing_two_inputs_gives_a_grid_that_survives_saving():
    function = Function(lambda x, y: x + 10 * y, ["x", "y"], "z")
    function.set_discrete(lower=[0, 0], upper=[3, 3], samples=6)
    assert function.is_regular_grid
    assert function(1.5, 0.25) == pytest.approx(4.0)

    restored = Function.from_dict(function.to_dict())
    assert restored.is_regular_grid
    assert restored(1.5, 0.25) == pytest.approx(4.0)


def test_save_and_load_keep_the_grid_and_its_method(plane_points):
    original = Function(plane_points, ["x", "y"], "z", interpolation="akima")
    restored = Function.from_dict(original.to_dict())
    assert restored.is_regular_grid
    assert restored.get_interpolation_method() == "pchip"
    assert restored(0.5, 1.5) == pytest.approx(original(0.5, 1.5))


def test_a_unit_conversion_keeps_the_grid(plane_points):
    grid = Function(plane_points, ["x (m)", "y"], "z", interpolation="akima")
    converted = convert_units(grid, "m", "ft", axis=0)
    assert converted.is_regular_grid
    assert converted.get_interpolation_method() == "pchip"


def test_deep_copy_evaluates_its_own_grid(plane_points):
    from copy import deepcopy  # pylint: disable=import-outside-toplevel

    original = Function(plane_points, ["x", "y"], "z")
    copy = deepcopy(original)
    copy.set_extrapolation("zero")
    assert copy.get_value_opt(9.0, 9.0) == 0.0
    assert original.get_value_opt(9.0, 9.0) == pytest.approx(99.0)


# ------------------------------------------------------------------ deprecations


def test_axes_and_values_with_the_old_flag_still_work():
    axis = np.array([0.0, 1.0, 2.0])
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    with pytest.warns(DeprecationWarning, match="regular_grid"):
        function = Function(
            ([axis, axis], 2 * x_grid + 3 * y_grid),
            ["x", "y"],
            "z",
            interpolation="regular_grid",
        )
    assert function.is_regular_grid
    assert function.get_interpolation_method() == "linear"
    assert function(0.5, 1.5) == pytest.approx(5.5)


def test_from_regular_grid_csv_is_deprecated_but_works(plane_points, tmp_path):
    grid_file = tmp_path / "grid.csv"
    np.savetxt(grid_file, plane_points, delimiter=",", header="x,y,z", comments="")
    scattered_file = tmp_path / "scattered.csv"
    np.savetxt(
        scattered_file, plane_points[:-1], delimiter=",", header="x,y,z", comments=""
    )

    with pytest.warns(DeprecationWarning):
        function = Function.from_regular_grid_csv(
            str(grid_file), ["x", "y"], "z", "constant"
        )
    assert function.is_regular_grid
    assert function(1.5, 0.25) == pytest.approx(4.0)

    with pytest.warns(DeprecationWarning):
        assert (
            Function.from_regular_grid_csv(
                str(scattered_file), ["x", "y"], "z", "constant"
            )
            is None
        )


@pytest.mark.parametrize("as_pair", [True, False])
def test_files_saved_with_the_old_flag_still_load(plane_points, as_pair):
    axis = np.linspace(0.0, 3.0, 4)
    source = (
        [[axis.tolist(), axis.tolist()], plane_points[:, -1].reshape(4, 4).tolist()]
        if as_pair
        else plane_points.tolist()
    )
    saved = {
        "source": source,
        "title": None,
        "inputs": ["x", "y"],
        "outputs": ["z"],
        "interpolation": "regular_grid",
        "extrapolation": "natural",
        "grid_method": "pchip",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        function = Function.from_dict(saved)
    assert function.is_regular_grid
    assert function.get_interpolation_method() == "pchip"
    assert function(1.5, 0.25) == pytest.approx(4.0)
