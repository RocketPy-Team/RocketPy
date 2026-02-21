"""1D interpolation and extrapolation strategies."""

from __future__ import annotations
from bisect import bisect_left
import numpy as np

from rocketpy.mathutils.interpolation._fitting import (
    fit_akima,
    fit_pchip,
    fit_polynomial,
    fit_spline,
    precompute_cubic_cumulative_integrals,
    precompute_linear_deriv_integral,
)
from rocketpy.mathutils.interpolation.polation_base import PolationBase


def _find_index(x_arr, xq, n):
    xq_type = type(xq)
    if xq_type is float or xq_type is int:
        idx = bisect_left(x_arr, xq)
        # Inline ternary is faster than max/min
        return 1 if idx < 1 else (idx if idx < n else n - 1)

    if xq_type is np.ndarray:
        idx = np.searchsorted(x_arr, xq, side="left")
        return np.clip(idx, 1, n - 1)

    idx = bisect_left(x_arr, xq.real)
    return 1 if idx < 1 else (idx if idx < n else n - 1)


def _cubic_eval_vec(t, a, b, c, d):
    return a + t * (b + t * (c + t * d))


class Linear1DPolation(PolationBase):
    def __init__(self, x, y):
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._n = self._x.size
        self._slopes, self._cum_int = precompute_linear_deriv_integral(self._x, self._y)

    def evaluate(self, x):
        x_arr = self._x
        i = _find_index(x_arr, x, self._n) - 1
        return self._y[i] + self._slopes[i] * (x - x_arr[i])

    def derivative(self, x):
        i = _find_index(self._x, x, self._n) - 1
        return self._slopes[i]

    def second_derivative(self, _x):
        return np.zeros_like(_x, dtype=float) if isinstance(_x, np.ndarray) else 0.0

    def integral(self, x):
        i = _find_index(self._x, x, self._n) - 1
        return (
            self._cum_int[i]
            + self._y[i] * (x - self._x[i])
            + self._slopes[i] * (x - self._x[i]) ** 2 / 2
        )


class Polynomial1DPolation(PolationBase):
    def __init__(self, x, y):
        coeffs = fit_polynomial(x, y)
        self._coeffs = np.asarray(coeffs, dtype=float)

        # Calculate derived coefficients
        d_coeffs = (
            self._coeffs[1:] * np.arange(1, len(self._coeffs))
            if len(self._coeffs) > 1
            else np.array([0.0])
        )
        d2_coeffs = (
            d_coeffs[1:] * np.arange(1, len(d_coeffs))
            if len(d_coeffs) > 1
            else np.array([0.0])
        )
        i_coeffs = np.empty(len(self._coeffs) + 1)
        i_coeffs[0] = 0.0
        i_coeffs[1:] = self._coeffs / np.arange(1, len(self._coeffs) + 1)

        # Pre-slice lists for high-speed Horner evaluation
        self._c_desc = self._coeffs[::-1].copy()
        c_list = self._c_desc.tolist()
        self._c_first, self._c_rest = c_list[0], c_list[1:]

        self._d_desc = d_coeffs[::-1].copy()
        d_list = self._d_desc.tolist()
        self._d_first, self._d_rest = d_list[0], d_list[1:]

        self._d2_desc = d2_coeffs[::-1].copy()
        d2_list = self._d2_desc.tolist()
        self._d2_first, self._d2_rest = d2_list[0], d2_list[1:]

        self._i_desc = i_coeffs[::-1].copy()
        i_list = self._i_desc.tolist()
        self._i_first, self._i_rest = i_list[0], i_list[1:]

    def _horner(self, xq, first, rest, desc):
        """Unified evaluation logic."""
        xq_type = type(xq)
        if xq_type in (float, int):
            r = first
            for c in rest:
                r = r * xq + c
            return r

        if xq_type is np.ndarray and xq.ndim == 0:
            val = xq.item()
            if type(val) in (float, int):
                r = first
                for c in rest:
                    r = r * val + c
                return r

        # Complex numbers, N-D arrays, and 0-D complex arrays safely fall through here
        return np.polyval(desc, xq)

    def evaluate(self, x):
        return self._horner(x, self._c_first, self._c_rest, self._c_desc)

    def derivative(self, x):
        return self._horner(x, self._d_first, self._d_rest, self._d_desc)

    def second_derivative(self, x):
        return self._horner(x, self._d2_first, self._d2_rest, self._d2_desc)

    def integral(self, x):
        return self._horner(x, self._i_first, self._i_rest, self._i_desc)

    def coefficients(self):
        return self._coeffs


class Cubic1DPolation(PolationBase):
    def __init__(self, x, coeffs):
        self._x = np.asarray(x, dtype=float)
        self._n = self._x.size
        self._a, self._b, self._c, self._d = coeffs
        self._cum_int = precompute_cubic_cumulative_integrals(
            self._x, (self._a, self._b, self._c, self._d)
        )

    def evaluate(self, x):
        i = _find_index(self._x, x, self._n) - 1
        t = x - self._x[i]
        return _cubic_eval_vec(t, self._a[i], self._b[i], self._c[i], self._d[i])

    def derivative(self, x):
        i = _find_index(self._x, x, self._n) - 1
        t = x - self._x[i]
        return self._b[i] + 2 * self._c[i] * t + 3 * self._d[i] * t**2

    def second_derivative(self, x):
        i = _find_index(self._x, x, self._n) - 1
        t = x - self._x[i]
        return 2 * self._c[i] + 6 * self._d[i] * t

    def integral(self, x):
        i = _find_index(self._x, x, self._n) - 1
        t = x - self._x[i]
        return (
            self._cum_int[i]
            + self._a[i] * t
            + self._b[i] * t**2 / 2
            + self._c[i] * t**3 / 3
            + self._d[i] * t**4 / 4
        )

    def coefficients(self):
        return [self._a, self._b, self._c, self._d]


class Spline1DPolation(Cubic1DPolation):
    def __init__(self, x, y):
        super().__init__(x, fit_spline(x, y))


class Akima1DPolation(Cubic1DPolation):
    def __init__(self, x, y):
        super().__init__(x, fit_akima(x, y))


class Pchip1DPolation(Cubic1DPolation):
    def __init__(self, x, y):
        super().__init__(x, fit_pchip(x, y))


class Constant1DExtrapolation(PolationBase):
    def __init__(self, x, y):
        self._x_min = float(x[0])
        self._x_max = float(x[-1])
        self._y_min = float(y[0])
        self._y_max = float(y[-1])

    def evaluate(self, x):
        if isinstance(x, np.ndarray):
            x_real = x.real
            result = np.empty_like(x_real, dtype=float)
            lower = x_real < self._x_min
            upper = x_real > self._x_max
            inside = ~(lower | upper)

            result[lower] = self._y_min
            result[upper] = self._y_max
            result[inside] = np.nan
            return result

        x_real = x.real
        if x_real < self._x_min:
            return self._y_min
        if x_real > self._x_max:
            return self._y_max
        return np.nan

    def definite_integral(self, a, b):
        return self.evaluate((a + b) / 2.0) * (b - a)

    def derivative(self, x):
        return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0

    def second_derivative(self, x):
        return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0


class Zero1DExtrapolation(PolationBase):
    def evaluate(self, x):
        return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0

    def definite_integral(self, a, b):
        return 0.0

    def derivative(self, x):
        return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0

    def second_derivative(self, x):
        return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
