"""ND interpolation and extrapolation strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
from scipy.spatial.distance import cdist

from rocketpy.mathutils._calc.polation_base import PolationBase


class LinearNDPolation(PolationBase):
    """Linear interpolation for N-dimensional scattered data."""

    __slots__ = ("_interpolator", "_triangulation")

    def __init__(self, domain: NDArray[np.float64], image: NDArray[np.float64]) -> None:
        """Initialize the LinearNDPolation.

        Parameters
        ----------
        domain : np.ndarray
            The domain coordinates of shape (n_samples, n_dimensions).
        image : np.ndarray
            The function values at the domain coordinates of shape
            (n_samples,).
        """
        self._triangulation = Delaunay(domain)
        self._interpolator = LinearNDInterpolator(self._triangulation, image)

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the linear interpolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The interpolated values at the given coordinates.
        """
        return self._interpolator(x)

    def extrapolation_mask(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return True for points outside the Delaunay triangulation.

        ``LinearNDInterpolator`` returns NaN outside the convex hull even when
        a point is inside the axis-aligned bounds. Pre-checking the simplex
        lets the evaluator route those points directly to extrapolation.
        """
        points = np.asarray(x, dtype=float)
        return self._triangulation.find_simplex(points) < 0


class RbfNDPolation(PolationBase):
    """Radial Basis Function (RBF) interpolation for N-dimensional scattered data."""

    __slots__ = ("_interpolator",)

    def __init__(
        self,
        domain: NDArray[np.float64],
        image: NDArray[np.float64],
        neighbors: int = 100,
    ) -> None:
        """Initialize the RbfNDPolation.

        Parameters
        ----------
        domain : np.ndarray
            The domain coordinates of shape (n_samples, n_dimensions).
        image : np.ndarray
            The function values at the domain coordinates of shape (n_samples,).
        neighbors : int, optional
            Number of nearest neighbors to use for interpolation. Default is 100.
        """
        self._interpolator = RBFInterpolator(domain, image, neighbors=neighbors)

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the RBF interpolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The interpolated values at the given coordinates.
        """
        return self._interpolator(x)


class ShepardNDPolation(PolationBase):
    """Shepard (IDW) interpolation for scattered ND data."""

    __slots__ = ("_domain", "_image")

    def __init__(self, domain: NDArray[np.float64], image: NDArray[np.float64]) -> None:
        """Initialize the ShepardNDPolation.

        Parameters
        ----------
        domain : np.ndarray
            The domain coordinates of shape (n_samples, n_dimensions).
        image : np.ndarray
            The function values at the domain coordinates of shape
            (n_samples,).
        """
        self._domain = np.asarray(domain, dtype=float)
        self._image = np.asarray(image, dtype=float)

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the Shepard interpolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The interpolated values at the given coordinates.
        """
        points = np.asarray(x, dtype=float)
        arg_qty = points.shape[0]

        # Vectorized path
        distances_sq = cdist(points, self._domain, metric="sqeuclidean")
        zero_mask = distances_sq == 0
        exact_match_rows = np.any(zero_mask, axis=1)

        with np.errstate(divide="ignore"):
            weights = distances_sq ** (-1.5)
        weights[exact_match_rows] = 0.0

        numerator = np.sum(self._image * weights, axis=1)
        denominator = np.sum(weights, axis=1)

        result = np.empty(arg_qty, dtype=float)
        valid = ~exact_match_rows
        result[valid] = numerator[valid] / denominator[valid]

        if exact_match_rows.any():
            match_indices = np.argmax(zero_mask[exact_match_rows], axis=1)
            result[exact_match_rows] = self._image[match_indices]

        return result


class ConstantNDExtrapolation(PolationBase):
    """Constant extrapolation for N-dimensional data using nearest neighbors."""

    __slots__ = ("_interpolator",)

    def __init__(self, domain: NDArray[np.float64], image: NDArray[np.float64]) -> None:
        """Initialize the ConstantNDExtrapolation.

        Parameters
        ----------
        domain : np.ndarray
            The domain coordinates of shape (n_samples, n_dimensions).
        image : np.ndarray
            The function values at the domain coordinates of shape
            (n_samples,).
        """
        self._interpolator = NearestNDInterpolator(domain, image)

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the constant extrapolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The extrapolated values at the given coordinates.
        """
        return self._interpolator(x)


class ZeroNDExtrapolation(PolationBase):
    """Zero extrapolation for N-dimensional data, returning zero
    for all points.
    """

    __slots__ = ()

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the zero extrapolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            An array of zeros matching the number of input points.
        """
        points = np.asarray(x)
        return np.zeros(points.shape[0], dtype=float)


class RbfNaturalNDExtrapolation(PolationBase):
    """Natural RBF extrapolation for N-dimensional data."""

    __slots__ = ("_interpolator",)

    def __init__(self, domain: NDArray[np.float64], image: NDArray[np.float64]) -> None:
        """Initialize the RbfNaturalNDExtrapolation.

        Parameters
        ----------
        domain : np.ndarray
            The domain coordinates of shape (n_samples, n_dimensions).
        image : np.ndarray
            The function values at the domain coordinates of shape
            (n_samples,).
        """
        self._interpolator = RBFInterpolator(domain, image)

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the natural RBF extrapolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The extrapolated values at the given coordinates.
        """
        return self._interpolator(x)


class RegularGridInterpolation(PolationBase):
    """Interpolation for N-dimensional data defined on a regular grid."""

    __slots__ = ("_grid_axes", "_interpolator")

    def __init__(
        self, grid_axes: list[NDArray[np.float64]], grid_data: NDArray[np.float64]
    ) -> None:
        """Initialize the RegularGridInterpolation.

        Parameters
        ----------
        grid_axes : list of np.ndarray
            A list containing 1D arrays defining the grid coordinates
            for each dimension.
        grid_data : np.ndarray
            The N-dimensional array containing the function values at
            the grid points.
        """
        self._interpolator = RegularGridInterpolator(
            grid_axes, grid_data, bounds_error=True
        )

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the regular grid interpolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The interpolated values at the given coordinates.
        """
        return self._interpolator(x)


class RegularGridNaturalExtrapolation(PolationBase):
    """Natural extrapolation for N-dimensional data defined on a regular grid."""

    __slots__ = ("_grid_axes", "_interpolator")

    def __init__(
        self, grid_axes: list[NDArray[np.float64]], grid_data: NDArray[np.float64]
    ) -> None:
        """Initialize the RegularGridNaturalExtrapolation.

        Parameters
        ----------
        grid_axes : list of np.ndarray
            A list containing 1D arrays defining the grid coordinates
            for each dimension.
        grid_data : np.ndarray
            The N-dimensional array containing the function values at
            the grid points.
        """
        self._interpolator = RegularGridInterpolator(
            grid_axes, grid_data, bounds_error=False, fill_value=None
        )

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the natural regular grid extrapolation at
        the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated, of shape
            (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The extrapolated values at the given coordinates.
        """
        return self._interpolator(x)


class RegularGridConstantExtrapolation(PolationBase):
    """Constant extrapolation for N-dimensional data defined on a
    regular grid by clamping coordinates.
    """

    __slots__ = ("_grid_axes", "_interpolator")

    def __init__(
        self, grid_axes: list[NDArray[np.float64]], grid_data: NDArray[np.float64]
    ) -> None:
        """Initialize the RegularGridConstantExtrapolation.

        Parameters
        ----------
        grid_axes : list of np.ndarray
            A list containing 1D arrays defining the grid coordinates
            for each dimension.
        grid_data : np.ndarray
            The N-dimensional array containing the function values
            at the grid points.
        """
        self._grid_axes = grid_axes
        self._interpolator = RegularGridInterpolator(
            grid_axes, grid_data, bounds_error=True
        )

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the constant regular grid extrapolation
        at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated,
            of shape (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            The extrapolated values at the clamped coordinates.
        """
        x_clamped = np.array(x, copy=True)
        for i, axis in enumerate(self._grid_axes):
            x_clamped[:, i] = np.clip(x_clamped[:, i], axis[0], axis[-1])
        return self._interpolator(x_clamped)


class RegularGridZeroExtrapolation(PolationBase):
    """Regular grid extrapolation using zeros."""

    __slots__ = ()

    def evaluate(
        self, x: NDArray[np.float64], _is_iterable: bool | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the zero extrapolation at the given coordinates.

        Parameters
        ----------
        x : np.ndarray
            The points where the function is to be evaluated,
            of shape (n_points, n_dimensions).
        _is_iterable : bool, optional
            Whether the input represents a batch of points. Default is None.

        Returns
        -------
        np.ndarray
            An array of zeros matching the number of input points.
        """
        return np.zeros(len(x))
