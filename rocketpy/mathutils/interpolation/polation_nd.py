"""ND interpolation and extrapolation strategies."""

from __future__ import annotations
import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial.distance import cdist
from rocketpy.mathutils.interpolation.polation_base import PolationBase


class LinearNDPolation(PolationBase):
    def __init__(self, domain, image):
        self._interpolator = LinearNDInterpolator(domain, image)

    def evaluate(self, x):
        return self._interpolator(x)


class RbfNDPolation(PolationBase):
    def __init__(self, domain, image, neighbors=100):
        self._interpolator = RBFInterpolator(domain, image, neighbors=neighbors)

    def evaluate(self, x):
        return self._interpolator(x)


class ShepardNDPolation(PolationBase):
    """Shepard (IDW) interpolation for scattered ND data."""

    def __init__(self, domain, image):
        self._domain = np.asarray(domain, dtype=float)
        self._image = np.asarray(image, dtype=float)

    def evaluate(self, x):
        points = np.asarray(x, dtype=float)
        arg_qty = points.shape[0]

        # Hot path: Single point query
        if arg_qty == 1:
            distances_sq = np.sum((self._domain - points[0]) ** 2, axis=1)
            if np.any(distances_sq == 0):
                return np.array([self._image[np.argmin(distances_sq)]], dtype=float)
            weights = distances_sq ** (-1.5)
            return np.array(
                [np.sum(self._image * weights) / np.sum(weights)], dtype=float
            )

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
    def __init__(self, domain, image):
        self._interpolator = NearestNDInterpolator(domain, image)

    def evaluate(self, x):
        return self._interpolator(x)


class ZeroNDExtrapolation(PolationBase):
    def evaluate(self, x):
        points = np.asarray(x)
        return np.zeros(points.shape[0], dtype=float)


class RbfNaturalNDExtrapolation(PolationBase):
    def __init__(self, domain, image):
        self._interpolator = RBFInterpolator(domain, image)

    def evaluate(self, x):
        return self._interpolator(x)


class RegularGridInterpolation(PolationBase):
    def __init__(self, grid_axes, grid_data):
        self._interpolator = RegularGridInterpolator(
            grid_axes, grid_data, bounds_error=True
        )

    def evaluate(self, x):
        return self._interpolator(x)


class RegularGridNaturalExtrapolation(PolationBase):
    def __init__(self, grid_axes, grid_data):
        self._interpolator = RegularGridInterpolator(
            grid_axes, grid_data, bounds_error=False, fill_value=None
        )

    def evaluate(self, x):
        return self._interpolator(x)


class RegularGridConstantExtrapolation(PolationBase):
    def __init__(self, grid_axes, grid_data):
        self._grid_axes = grid_axes
        self._interpolator = RegularGridInterpolator(
            grid_axes, grid_data, bounds_error=True
        )

    def evaluate(self, x):
        x_clamped = np.array(x, copy=True)
        for i, axis in enumerate(self._grid_axes):
            x_clamped[:, i] = np.clip(x_clamped[:, i], axis[0], axis[-1])
        return self._interpolator(x_clamped)


class RegularGridZeroExtrapolation(PolationBase):
    """Regular grid extrapolation using zeros."""

    def evaluate(self, x):

        return np.zeros(len(x))
