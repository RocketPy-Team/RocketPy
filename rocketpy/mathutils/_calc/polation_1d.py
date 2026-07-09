"""1D interpolation and extrapolation strategies."""

from __future__ import annotations

from bisect import bisect_left

import numpy as np
from numpy.typing import NDArray

from rocketpy.mathutils._calc._fitting import (
    fit_akima,
    fit_pchip,
    fit_polynomial,
    fit_spline,
    precompute_cubic_cumulative_integrals,
    precompute_linear_deriv_integral,
)
from rocketpy.mathutils._calc.polation_base import PolationBase


def _find_index(
    x_arr: NDArray[np.float64],
    xq: float | NDArray[np.float64] | complex,
    n: int,
    _is_iterable: bool | None = None,
) -> int | NDArray[np.int_]:
    """Find the index in a sorted 1D array corresponding to a query value.

    Parameters
    ----------
    x_arr : np.ndarray
        Sorted 1D array of x-coordinates.
    xq : float or np.ndarray or complex
        The query coordinate(s).
    n : int
        The size of x_arr.
    _is_iterable : bool, optional
        Whether the query value represents a batch of points. Default is None.

    Returns
    -------
    int or np.ndarray
        The index or indices representing the interval.
    """
    if _is_iterable is None:
        _is_iterable = hasattr(xq, "__iter__") and np.ndim(xq) > 0

    if not _is_iterable:
        # Both real and complex scalars have the .real property
        idx = bisect_left(x_arr, xq.real)
        return 1 if idx < 1 else (idx if idx < n else n - 1)
    else:
        idx = np.searchsorted(x_arr, xq, side="left")
        return np.clip(idx, 1, n - 1)


def _cubic_eval_vec(
    t: float | NDArray[np.float64],
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    c: float | NDArray[np.float64],
    d: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Evaluate a cubic polynomial: a + t * (b + t * (c + t * d)).

    Parameters
    ----------
    t : float or np.ndarray
        The relative parameter values.
    a : float or np.ndarray
        Constant coefficients.
    b : float or np.ndarray
        Linear coefficients.
    c : float or np.ndarray
        Quadratic coefficients.
    d : float or np.ndarray
        Cubic coefficients.

    Returns
    -------
    float or np.ndarray
        The evaluated cubic polynomial value(s).
    """
    return a + b * t + c * (t**2) + d * (t**3)


class Linear1DPolation(PolationBase):
    """Linear 1D interpolation and extrapolation."""

    __slots__ = ("_x", "_y", "_n", "_slopes", "_cum_int")

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the Linear1DPolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        """
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._n = self._x.size
        self._slopes = np.diff(self._y) / np.diff(self._x)
        self._cum_int = None

    def _slopes_at(self, i):
        """Return cached or on-demand linear slopes for interval
        index/indices.
        """
        return self._slopes[i]

    def _ensure_integral_cache(self):
        """Populate linear slopes and cumulative integrals on
        first integral use.
        """
        if self._cum_int is None:
            _, self._cum_int = precompute_linear_deriv_integral(self._x, self._y)

    def evaluate(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the linear interpolation.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the function is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The interpolated values.
        """
        x_arr = self._x
        i = _find_index(x_arr, x, self._n, _is_iterable) - 1
        return self._y[i] + self._slopes_at(i) * (x - x_arr[i])

    def derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the first derivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The first derivative values.
        """
        i = _find_index(self._x, x, self._n, _is_iterable) - 1
        return self._slopes_at(i)

    def second_derivative(
        self, _x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the second derivative.

        Parameters
        ----------
        _x : float or np.ndarray
            The coordinates where the second derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The second derivative values, which are zero.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(_x, "__iter__") and np.ndim(_x) > 0
        return np.zeros_like(_x, dtype=float) if _is_iterable else 0.0

    def integral(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the antiderivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the antiderivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The antiderivative values.
        """
        self._ensure_integral_cache()
        i = _find_index(self._x, x, self._n, _is_iterable) - 1
        return (
            self._cum_int[i]
            + self._y[i] * (x - self._x[i])
            + self._slopes[i] * (x - self._x[i]) ** 2 / 2
        )


class Polynomial1DPolation(PolationBase):
    """Polynomial 1D interpolation."""

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the Polynomial1DPolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        """
        coeffs = fit_polynomial(x, y)
        self._coeffs = np.asarray(coeffs, dtype=float)

        # Pre-slice value coefficients for high-speed Horner evaluation.
        # Derivative and integral coefficients are built lazily.
        self._c_desc = self._coeffs[::-1].copy()
        c_list = self._c_desc.tolist()
        self._c_first, self._c_rest = c_list[0], c_list[1:]

        self._d_desc = None
        self._d_first = None
        self._d_rest = None
        self._d2_desc = None
        self._d2_first = None
        self._d2_rest = None
        self._i_desc = None
        self._i_first = None
        self._i_rest = None

    def _ensure_derivative_coefficients(self):
        """Build first-derivative Horner coefficients on first derivative use."""
        if self._d_desc is not None:
            return
        d_coeffs = (
            self._coeffs[1:] * np.arange(1, len(self._coeffs))
            if len(self._coeffs) > 1
            else np.array([0.0])
        )
        self._d_desc = d_coeffs[::-1].copy()
        d_list = self._d_desc.tolist()
        self._d_first, self._d_rest = d_list[0], d_list[1:]

    def _ensure_second_derivative_coefficients(self):
        """Build second-derivative Horner coefficients on first use."""
        if self._d2_desc is not None:
            return
        self._ensure_derivative_coefficients()
        d_asc = self._d_desc[::-1]
        d2_coeffs = (
            d_asc[1:] * np.arange(1, len(d_asc)) if len(d_asc) > 1 else np.array([0.0])
        )
        self._d2_desc = d2_coeffs[::-1].copy()
        d2_list = self._d2_desc.tolist()
        self._d2_first, self._d2_rest = d2_list[0], d2_list[1:]

    def _ensure_integral_coefficients(self):
        """Build antiderivative Horner coefficients on first integral use."""
        if self._i_desc is not None:
            return
        i_coeffs = np.empty(len(self._coeffs) + 1)
        i_coeffs[0] = 0.0
        i_coeffs[1:] = self._coeffs / np.arange(1, len(self._coeffs) + 1)
        self._i_desc = i_coeffs[::-1].copy()
        i_list = self._i_desc.tolist()
        self._i_first, self._i_rest = i_list[0], i_list[1:]

    def _horner(
        self,
        xq: float | NDArray[np.float64],
        first: float,
        rest: list[float],
        desc: NDArray[np.float64],
        _is_iterable: bool | None = None,
    ) -> float | NDArray[np.float64]:
        """Unified Horner evaluation logic.

        Parameters
        ----------
        xq : float or np.ndarray
            The coordinates to evaluate.
        first : float
            The first coefficient of the polynomial.
        rest : list of float
            The remaining coefficients of the polynomial.
        desc : np.ndarray
            The descending coefficients.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The evaluated polynomial value.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(xq, "__iter__") and np.ndim(xq) > 0

        if not _is_iterable:
            r = first
            for c in rest:
                r = r * xq + c
            return r
        else:
            return np.polyval(desc, xq)

    def evaluate(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the polynomial.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the polynomial is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The polynomial values.
        """
        return self._horner(x, self._c_first, self._c_rest, self._c_desc, _is_iterable)

    def derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the first derivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The first derivative values.
        """
        self._ensure_derivative_coefficients()
        return self._horner(x, self._d_first, self._d_rest, self._d_desc, _is_iterable)

    def second_derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the second derivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the second derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The second derivative values.
        """
        self._ensure_second_derivative_coefficients()
        return self._horner(
            x, self._d2_first, self._d2_rest, self._d2_desc, _is_iterable
        )

    def integral(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the antiderivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the antiderivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The antiderivative values.
        """
        self._ensure_integral_coefficients()
        return self._horner(x, self._i_first, self._i_rest, self._i_desc, _is_iterable)

    def coefficients(self) -> NDArray[np.float64]:
        """Get the polynomial coefficients.

        Returns
        -------
        np.ndarray
            The polynomial coefficients.
        """
        return self._coeffs


class Cubic1DPolation(PolationBase):
    """Cubic piecewise 1D interpolation."""

    __slots__ = ("_x", "_n", "_a", "_b", "_c", "_d", "_cum_int")

    def __init__(
        self,
        x: NDArray[np.float64],
        coeffs: tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        """Initialize the Cubic1DPolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        coeffs : tuple of np.ndarray
            A tuple (a, b, c, d) of numpy arrays representing the
            piecewise cubic coefficients.
        """
        self._x = np.asarray(x, dtype=float)
        self._n = self._x.size
        self._a, self._b, self._c, self._d = coeffs
        self._cum_int = None

    def evaluate(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the cubic piecewise interpolation.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the function is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The interpolated values.
        """
        i = _find_index(self._x, x, self._n, _is_iterable) - 1
        t = x - self._x[i]
        return _cubic_eval_vec(t, self._a[i], self._b[i], self._c[i], self._d[i])

    def derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the first derivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The first derivative values.
        """
        i = _find_index(self._x, x, self._n, _is_iterable) - 1
        t = x - self._x[i]
        return self._b[i] + 2 * self._c[i] * t + 3 * self._d[i] * t**2

    def second_derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the second derivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the second derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The second derivative values.
        """
        i = _find_index(self._x, x, self._n, _is_iterable) - 1
        t = x - self._x[i]
        return 2 * self._c[i] + 6 * self._d[i] * t

    def integral(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the antiderivative.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the antiderivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The antiderivative values.
        """
        if self._cum_int is None:
            self._cum_int = precompute_cubic_cumulative_integrals(
                self._x, (self._a, self._b, self._c, self._d)
            )

        i = _find_index(self._x, x, self._n, _is_iterable) - 1
        t = x - self._x[i]
        return (
            self._cum_int[i]
            + self._a[i] * t
            + self._b[i] * t**2 / 2
            + self._c[i] * t**3 / 3
            + self._d[i] * t**4 / 4
        )

    def coefficients(self) -> list[NDArray[np.float64]]:
        """Get the cubic coefficients.

        Returns
        -------
        list of np.ndarray
            A list [a, b, c, d] of the cubic coefficients.
        """
        return [self._a, self._b, self._c, self._d]


class Spline1DPolation(Cubic1DPolation):
    """Spline 1D interpolation."""

    __slots__ = ()

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the Spline1DPolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        """
        super().__init__(x, fit_spline(x, y))


class Akima1DPolation(Cubic1DPolation):
    """Akima 1D interpolation."""

    __slots__ = ()

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the Akima1DPolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        """
        super().__init__(x, fit_akima(x, y))


class Pchip1DPolation(Cubic1DPolation):
    """PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 1D interpolation."""

    __slots__ = ()

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the Pchip1DPolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        """
        super().__init__(x, fit_pchip(x, y))


class Constant1DExtrapolation(PolationBase):
    """Constant 1D extrapolation."""

    __slots__ = ("_x_min", "_x_max", "_y_min", "_y_max")

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the Constant1DExtrapolation.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.
        """
        self._x_min = float(x[0])
        self._x_max = float(x[-1])
        self._y_min = float(y[0])
        self._y_max = float(y[-1])

    def evaluate(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the constant extrapolation.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates to extrapolate.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            The extrapolated constant values at boundary exceedances, or NaN.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0
        if _is_iterable:
            x = np.asarray(x, dtype=float)
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

    def definite_integral(self, a: float, b: float) -> float:
        """Evaluate the definite integral over a constant extrapolation range.

        Parameters
        ----------
        a : float
            Lower bound of integration.
        b : float
            Upper bound of integration.

        Returns
        -------
        float
            The definite integral value.
        """
        midpoint = (a + b) / 2.0
        is_iterable = hasattr(midpoint, "__iter__") and np.ndim(midpoint) > 0
        return self.evaluate(midpoint, _is_iterable=is_iterable) * (b - a)

    def derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the derivative, which is zero.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            An array of zeros or a scalar zero.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0
        return np.zeros_like(x, dtype=float) if _is_iterable else 0.0

    def second_derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the second derivative, which is zero.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the second derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            An array of zeros or a scalar zero.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0
        return np.zeros_like(x, dtype=float) if _is_iterable else 0.0


class Zero1DExtrapolation(PolationBase):
    """Zero 1D extrapolation."""

    __slots__ = ()

    def evaluate(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the zero extrapolation.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates to extrapolate.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            An array of zeros or a scalar zero.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0
        return np.zeros_like(x, dtype=float) if _is_iterable else 0.0

    def definite_integral(self, a: float, b: float) -> float:
        """Evaluate the definite integral, which is zero.

        Parameters
        ----------
        a : float
            Lower bound of integration.
        b : float
            Upper bound of integration.

        Returns
        -------
        float
            Scalar zero.
        """
        return 0.0

    def derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the derivative, which is zero.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            An array of zeros or a scalar zero.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0
        return np.zeros_like(x, dtype=float) if _is_iterable else 0.0

    def second_derivative(
        self, x: float | NDArray[np.float64], _is_iterable: bool | None = None
    ) -> float | NDArray[np.float64]:
        """Evaluate the second derivative, which is zero.

        Parameters
        ----------
        x : float or np.ndarray
            The coordinates where the second derivative is to be evaluated.
        _is_iterable : bool, optional
            Whether the input is iterable. Default is None.

        Returns
        -------
        float or np.ndarray
            An array of zeros or a scalar zero.
        """
        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0
        return np.zeros_like(x, dtype=float) if _is_iterable else 0.0
