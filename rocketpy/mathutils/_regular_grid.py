import warnings
from bisect import bisect_right

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# The methods of SciPy's ``RegularGridInterpolator``, under the names a
# ``Function`` accepts. ``spline``, ``akima`` and ``polynomial`` are written for
# a single variable, so they are read as their closest counterparts on a grid.
GRID_METHODS = {
    "linear": "linear",
    "nearest": "nearest",
    "slinear": "slinear",
    "cubic": "cubic",
    "quintic": "quintic",
    "pchip": "pchip",
    "spline": "cubic",
    "akima": "pchip",
    "polynomial": "cubic",
}
# Points each method needs along every axis. SciPy raises on a coarser grid.
GRID_MIN_POINTS = {
    "nearest": 1,
    "linear": 2,
    "slinear": 2,
    "pchip": 4,
    "cubic": 4,
    "quintic": 6,
}


class _RegularGrid:
    """Values on a regular grid, and their interpolation.

    A table of points over two or more variables covers a regular grid when it
    holds every combination of the values taken by each variable, exactly once.
    Such a table can be interpolated one variable at a time, which is both
    faster and more accurate than treating its points as scattered. The
    ``Function`` class uses a ``_RegularGrid`` whenever its data allows it.

    Attributes
    ----------
    axes : list of numpy.ndarray
        Values taken by each variable, in increasing order.
    values : numpy.ndarray
        Value at each node, with one dimension per variable: ``values[i, j]``
        is the value at ``axes[0][i]`` and ``axes[1][j]``.
    method : str
        How values are interpolated between the nodes: ``"linear"``,
        ``"nearest"``, ``"slinear"``, ``"cubic"``, ``"quintic"`` or ``"pchip"``.
    extrapolation : str
        What is returned outside the grid: ``"natural"`` continues the
        interpolation, ``"constant"`` holds the value at the nearest edge and
        ``"zero"`` returns 0. It may be changed at any time.
    """

    def __init__(self, axes, values, method=None, extrapolation="natural", name=None):
        """Store the grid and prepare its interpolation.

        Parameters
        ----------
        axes : list of array_like
            Values taken by each variable, in increasing order, one sequence
            per variable. Each needs at least two values.
        values : array_like
            Value at each node, with one dimension per variable:
            ``values[i, j]`` is the value at ``axes[0][i]`` and ``axes[1][j]``.
        method : str, optional
            Interpolation asked for, under any name a ``Function`` accepts:
            ``"linear"``, ``"nearest"``, ``"slinear"``, ``"cubic"``,
            ``"quintic"``, ``"pchip"``, or ``"spline"``, ``"akima"`` and
            ``"polynomial"``, which are read as ``"cubic"``, ``"pchip"`` and
            ``"cubic"``. An unknown name, or a method needing more points
            along an axis than the grid has (4 for ``"cubic"`` and
            ``"pchip"``, 6 for ``"quintic"``), gives ``"linear"`` with a
            warning. Default is ``"linear"``.
        extrapolation : str, optional
            ``"natural"``, ``"constant"`` or ``"zero"``, see the class
            attributes. Default is ``"natural"``.
        name : str, optional
            What the data is called, used in the warnings only.
        """
        self.axes = [np.asarray(axis, dtype=np.float64) for axis in axes]
        self.values = np.asarray(values, dtype=np.float64)
        self.extrapolation = extrapolation
        self._scipy_interpolator = None  # built when first needed

        of_name = f" of '{name}'" if isinstance(name, str) else ""
        self.method = GRID_METHODS.get((method or "linear").lower())
        if self.method is None:
            warnings.warn(
                f"Interpolation method set to 'linear' because the {method} "
                f"method is not supported for the regular grid{of_name}. The "
                "supported methods are 'linear', 'nearest', 'slinear', 'cubic', "
                "'quintic' and 'pchip'."
            )
            self.method = "linear"
        smallest_axis = min(len(axis) for axis in self.axes)
        if smallest_axis < GRID_MIN_POINTS[self.method]:
            warnings.warn(
                f"Interpolation method '{self.method}' needs at least "
                f"{GRID_MIN_POINTS[self.method]} points per axis, but the coarsest "
                f"axis{of_name} has {smallest_axis}; falling back to 'linear'."
            )
            self.method = "linear"

        # A simulation asks for one point at a time, which SciPy is slow at.
        # For that case keep plain lists, faster to index one number at a time
        # than arrays: the axes, the values in a flat list, the step in that
        # list between neighbors along each axis, and, for each corner of a
        # cell, whether it lies on the lower (0) or upper (1) grid line of
        # each variable together with its offset in the flat list.
        strides = [int(np.prod(self.values.shape[i + 1 :])) for i in range(len(axes))]
        corners = [
            (corner, sum(bit * stride for bit, stride in zip(corner, strides)))
            for corner in np.ndindex(*([2] * len(axes)))
        ]
        self._lookup = (
            [axis.tolist() for axis in self.axes],
            strides,
            corners,
            self.values.ravel().tolist(),
        )

    @classmethod
    def from_points(cls, points, method=None, extrapolation="natural", name=None):
        """Read a table of points as a regular grid, if it is one.

        Parameters
        ----------
        points : numpy.ndarray
            One row per point: the value of each variable, then the value of
            the data there. The rows may come in any order.
        method, extrapolation, name
            See ``__init__``.

        Returns
        -------
        _RegularGrid or None
            The grid, or ``None`` when the points are not to be read as one:
            fewer than two variables, fewer than two values of any of them, a
            combination missing or repeated, or a ``method`` written for
            scattered points (``"shepard"`` or ``"rbf"``).
        """
        if method is not None and method.lower() in ("shepard", "rbf"):
            return None
        coordinates = points[:, :-1]
        if coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
            return None
        # The values each variable takes, and where each point sits along them
        axes, indices = zip(
            *(np.unique(column, return_inverse=True) for column in coordinates.T)
        )
        shape = tuple(len(axis) for axis in axes)
        if min(shape) < 2 or np.prod(shape) != len(points):
            return None
        # There are as many points as nodes, so a node left empty means that
        # another one is repeated
        values = np.empty(shape)
        filled = np.zeros(shape, dtype=bool)
        values[indices] = points[:, -1]
        filled[indices] = True
        if not filled.all():
            return None
        return cls(axes, values, method, extrapolation, name)

    @staticmethod
    def is_axes_and_values(source):
        """Check whether ``source`` is given as a grid, ``(axes, values)``.

        That is one sequence of values for each variable, then the data on the
        grid they span. A table of points never matches, since its rows hold
        numbers where the axes hold sequences.

        Parameters
        ----------
        source : object
            The data to check.

        Returns
        -------
        bool
            True if ``source`` is an ``(axes, values)`` pair.
        """
        return (
            isinstance(source, (tuple, list))
            and len(source) == 2
            and isinstance(source[0], (tuple, list))
            and len(source[0]) > 0
            and all(np.ndim(axis) == 1 for axis in source[0])
        )

    @staticmethod
    def points_from_axes(axes, values):
        """Write values given on a grid as a table of points.

        Parameters
        ----------
        axes : sequence of array_like
            Values taken by each variable, in any order, without repetition.
        values : array_like
            Value at each node, with one dimension per variable.

        Returns
        -------
        numpy.ndarray
            One row per node: the value of each variable, then the value of
            the data there.
        """
        axes = [np.asarray(axis, dtype=np.float64) for axis in axes]
        values = np.asarray(values, dtype=np.float64)
        if len(axes) != values.ndim:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match grid_data dimensions "
                f"({values.ndim})."
            )
        for i, axis in enumerate(axes):
            if axis.size != values.shape[i]:
                raise ValueError(
                    f"Axis {i} has {axis.size} points but grid dimension {i} has "
                    f"{values.shape[i]} points."
                )
            if len(np.unique(axis)) != axis.size:
                raise ValueError(f"Axis {i} has repeated coordinates.")
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh] + [values.ravel()])

    def evaluate(self, *args):
        """Evaluate the grid at one point or at several.

        Parameters
        ----------
        args : float or array_like
            Value of each variable: numbers for a single point, or sequences
            of the same length for several.

        Returns
        -------
        float or numpy.ndarray
            A number for a single point, an array otherwise.
        """
        extrapolation = self.extrapolation
        single_point = not isinstance(args[0], (np.ndarray, list, tuple))

        if single_point and self.method == "linear":
            # Find the cell around the point and blend the values at its
            # corners, which gives the values of SciPy's interpolator
            axes, strides, corners, values = self._lookup
            base = 0
            fractions = []
            for x, axis, stride in zip(args, axes, strides):
                i = min(max(bisect_right(axis, x) - 1, 0), len(axis) - 2)
                fraction = (x - axis[i]) / (axis[i + 1] - axis[i])
                if fraction < 0 or fraction > 1:  # outside the grid
                    if extrapolation == "zero":
                        return 0.0
                    if extrapolation == "constant":
                        fraction = 0.0 if fraction < 0 else 1.0
                base += i * stride
                fractions.append(fraction)
            result = 0.0
            for corner, offset in corners:
                weight = 1.0
                for bit, fraction in zip(corner, fractions):
                    weight *= fraction if bit else 1.0 - fraction
                result += weight * values[base + offset]
            return result

        if self._scipy_interpolator is None:
            self._scipy_interpolator = RegularGridInterpolator(
                self.axes,
                self.values,
                method=self.method,
                bounds_error=False,
                fill_value=None,  # continue the interpolation outside the grid
            )
        points = np.column_stack(args).astype(np.float64)
        lower = [axis[0] for axis in self.axes]
        upper = [axis[-1] for axis in self.axes]
        if extrapolation == "natural":
            result = self._scipy_interpolator(points)
        else:
            result = self._scipy_interpolator(np.clip(points, lower, upper))
            if extrapolation == "zero":
                result[((points < lower) | (points > upper)).any(axis=1)] = 0.0
        return float(result[0]) if len(result) == 1 else result
