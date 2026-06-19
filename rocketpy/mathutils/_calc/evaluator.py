"""Routers that choose interpolation or extrapolation strategies."""

from __future__ import annotations

import numpy as np

from rocketpy.mathutils._calc.polation_base import PolationBase


class PolationEvaluator1D(PolationBase):
    """Route 1D evaluation to interpolation or extrapolation."""

    __slots__ = (
        "_interpolator",
        "_extrapolator",
        "_x_min",
        "_x_max",
        "_exposed_fn",
        "_scalar_fn",
        "_vector_fn",
    )

    def __init__(self, interpolator, extrapolator, x):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        self._x_min = float(x[0])
        self._x_max = float(x[-1])

        # Expose the evaluator to shorten the call stack
        self._scalar_fn = self.expose_scalar()
        self._vector_fn = self.expose_vector()
        self._exposed_fn = self.expose()

    def evaluate(self, x, _is_iterable=None):
        return self._exposed_fn(x, _is_iterable=_is_iterable)

    def expose(self):
        """Flattens the evaluator into a fast closure for the
        simulation loop.
        """
        scalar_eval = self._scalar_fn
        vector_eval = self._vector_fn

        def _eval(x, _is_iterable=None):
            if _is_iterable is None:
                _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0

            if not _is_iterable:
                return scalar_eval(x)
            return vector_eval(x)

        return _eval

    def expose_scalar(self):
        """Expose a scalar-only evaluator with no iterable checks."""
        x_min = self._x_min
        x_max = self._x_max
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate

        if self._interpolator is self._extrapolator:

            def _scalar_same(x):
                return interp_eval(x, _is_iterable=False)

            return _scalar_same

        def _scalar(x):
            if x_min <= x.real <= x_max:
                return interp_eval(x, _is_iterable=False)
            return extrap_eval(x, _is_iterable=False)

        return _scalar

    def expose_vector(self):
        """Expose a vector-only evaluator with no iterable checks."""
        x_min = self._x_min
        x_max = self._x_max
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate

        if self._interpolator is self._extrapolator:

            def _vector_same(x):
                return interp_eval(x, _is_iterable=True)

            return _vector_same

        def _vector(x):
            out_dtype = complex if np.iscomplexobj(x) else float
            result = np.empty_like(x, dtype=out_dtype)
            x_real = x.real
            inside = (x_real >= x_min) & (x_real <= x_max)
            outside = ~inside
            if inside.any():
                result[inside] = interp_eval(x[inside], _is_iterable=True)
            if outside.any():
                result[outside] = extrap_eval(x[outside], _is_iterable=True)
            return result

        return _vector

    def coefficients(self):
        return self._interpolator.coefficients()

    def derivative(self, x, _is_iterable=None):
        """Route 1st derivative to interpolation or extrapolation."""
        interp_deriv = self._interpolator.derivative
        extrap_deriv = self._extrapolator.derivative

        if self._interpolator is self._extrapolator:
            return interp_deriv(x, _is_iterable=_is_iterable)

        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0

        if not _is_iterable:
            if self._x_min <= x.real <= self._x_max:
                return interp_deriv(x, _is_iterable=False)
            return extrap_deriv(x, _is_iterable=False)
        else:
            out_dtype = complex if np.iscomplexobj(x) else float
            result = np.empty_like(x, dtype=out_dtype)
            x_real = x.real
            inside = (x_real >= self._x_min) & (x_real <= self._x_max)
            outside = ~inside
            if inside.any():
                result[inside] = interp_deriv(x[inside], _is_iterable=True)
            if outside.any():
                result[outside] = extrap_deriv(x[outside], _is_iterable=True)
            return result

    def second_derivative(self, x, _is_iterable=None):
        """Route 2nd derivative to interpolation or extrapolation."""
        interp_deriv2 = self._interpolator.second_derivative
        extrap_deriv2 = self._extrapolator.second_derivative

        if self._interpolator is self._extrapolator:
            return interp_deriv2(x, _is_iterable=_is_iterable)

        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0

        if not _is_iterable:
            if self._x_min <= x.real <= self._x_max:
                return interp_deriv2(x, _is_iterable=False)
            return extrap_deriv2(x, _is_iterable=False)
        else:
            out_dtype = complex if np.iscomplexobj(x) else float
            result = np.empty_like(x, dtype=out_dtype)
            x_real = x.real
            inside = (x_real >= self._x_min) & (x_real <= self._x_max)
            outside = ~inside
            if inside.any():
                result[inside] = interp_deriv2(x[inside], _is_iterable=True)
            if outside.any():
                result[outside] = extrap_deriv2(x[outside], _is_iterable=True)
            return result

    def integral(self, x, _is_iterable=None):
        """Calculates the continuous antiderivative F(x) anchored at x_min.
        Fully vectorized.
        """
        if self._interpolator is self._extrapolator:
            return self._interpolator.integral(x, _is_iterable=_is_iterable)

        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__") and np.ndim(x) > 0

        if not _is_iterable:
            x_val = float(x)
            if x_val < self._x_min:
                return -self._extrapolator.definite_integral(x_val, self._x_min)
            elif x_val <= self._x_max:
                return self._interpolator.definite_integral(self._x_min, x_val)
            else:
                base_area = self._interpolator.definite_integral(
                    self._x_min, self._x_max
                )
                return base_area + self._extrapolator.definite_integral(
                    self._x_max, x_val
                )

        x_arr = np.asarray(x, dtype=float)
        result = np.zeros_like(x_arr)

        # Precalculate the total area of the core domain
        base_area = self._interpolator.definite_integral(self._x_min, self._x_max)

        # 1. Left of domain (Negative area)
        lower = x_arr < self._x_min
        if lower.any():
            result[lower] = -self._extrapolator.definite_integral(
                x_arr[lower], self._x_min
            )

        # 2. Inside domain
        inside = (x_arr >= self._x_min) & (x_arr <= self._x_max)
        if inside.any():
            result[inside] = self._interpolator.definite_integral(
                self._x_min, x_arr[inside]
            )

        # 3. Right of domain (Core area + new area)
        upper = x_arr > self._x_max
        if upper.any():
            result[upper] = base_area + self._extrapolator.definite_integral(
                self._x_max, x_arr[upper]
            )

        return result


class PolationEvaluatorND(PolationBase):
    """Route ND evaluation based on bounding box and NaN detection."""

    __slots__ = (
        "_interpolator",
        "_extrapolator",
        "_min_domain",
        "_max_domain",
        "_extrapolation_mask",
        "_exposed_fn",
        "_scalar_fn",
        "_vector_fn",
    )

    def __init__(self, interpolator, extrapolator, domain):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        domain = np.asarray(domain, dtype=float)
        self._min_domain = np.min(domain, axis=0)
        self._max_domain = np.max(domain, axis=0)
        self._extrapolation_mask = getattr(interpolator, "extrapolation_mask", None)

        self._scalar_fn = self.expose_scalar()
        self._vector_fn = self.expose_vector()
        self._exposed_fn = self.expose()

    def evaluate(self, *args, _is_iterable=None):
        return self._exposed_fn(*args, _is_iterable=_is_iterable)

    def expose(self):
        scalar_eval = self._scalar_fn
        vector_eval = self._vector_fn

        def _eval(*args, _is_iterable=None):
            if _is_iterable is None:
                _is_iterable = any(
                    hasattr(arg, "__iter__") and np.ndim(arg) > 0 for arg in args
                )
            if not _is_iterable:
                return scalar_eval(*args)
            return vector_eval(*args)

        return _eval

    def expose_scalar(self):
        min_domain = self._min_domain
        max_domain = self._max_domain
        extrapolation_mask = self._extrapolation_mask
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        def _scalar(*args):
            points = np_local.array([args], dtype=float)
            point = points[0]
            outside_bounds = ((point < min_domain) | (point > max_domain)).any()
            outside_hull = (
                extrapolation_mask is not None and extrapolation_mask(points)[0]
            )
            if outside_bounds or outside_hull:
                res_val = extrap_eval(points)[0]
            else:
                res_val = interp_eval(points)[0]
                if np_local.isnan(res_val):
                    res_val = extrap_eval(points)[0]
            return (
                complex(res_val) if np_local.iscomplexobj(res_val) else float(res_val)
            )

        return _scalar

    def expose_vector(self):  # pylint: disable=too-many-statements
        min_domain = self._min_domain
        max_domain = self._max_domain
        extrapolation_mask = self._extrapolation_mask
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        def _vector(*args):
            points = np_local.column_stack(args)
            out_dtype = complex if np_local.iscomplexobj(points) else float
            result = np_local.empty(len(points), dtype=out_dtype)
            points_real = points.real
            lower = points_real < min_domain
            upper = points_real > max_domain
            extrap_mask = lower.any(axis=1) | upper.any(axis=1)
            if extrapolation_mask is not None:
                interp_candidates = ~extrap_mask
                if interp_candidates.any():
                    extrap_mask[interp_candidates] = extrapolation_mask(
                        points[interp_candidates]
                    )
            interp_mask = ~extrap_mask

            if interp_mask.any():
                inside_points = points[interp_mask]
                interp_values = interp_eval(inside_points)

                if np_local.any(np_local.isnan(interp_values)):
                    interp_values = np_local.asarray(interp_values, dtype=float)
                    nan_mask = np_local.isnan(interp_values)
                    if nan_mask.any():
                        interp_values[nan_mask] = extrap_eval(inside_points[nan_mask])

                result[interp_mask] = interp_values

            if extrap_mask.any():
                result[extrap_mask] = extrap_eval(points[extrap_mask])

            return result

        return _vector


class RegularGridEvaluator(PolationBase):
    """Route regular grid evaluation based on axes bounds."""

    __slots__ = (
        "_interpolator",
        "_extrapolator",
        "_min_domain",
        "_max_domain",
        "_exposed_fn",
        "_scalar_fn",
        "_vector_fn",
    )

    def __init__(self, interpolator, extrapolator, grid_axes):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        self._min_domain = np.array([axis[0] for axis in grid_axes], dtype=float)
        self._max_domain = np.array([axis[-1] for axis in grid_axes], dtype=float)
        self._scalar_fn = self.expose_scalar()
        self._vector_fn = self.expose_vector()
        self._exposed_fn = self.expose()

    def evaluate(self, *args, _is_iterable=None):
        return self._exposed_fn(*args, _is_iterable=_is_iterable)

    def expose(self):
        scalar_eval = self._scalar_fn
        vector_eval = self._vector_fn

        def _eval(*args, _is_iterable=None):
            if _is_iterable is None:
                _is_iterable = any(
                    hasattr(arg, "__iter__") and np.ndim(arg) > 0 for arg in args
                )
            if not _is_iterable:
                return scalar_eval(*args)
            return vector_eval(*args)

        return _eval

    def expose_scalar(self):
        min_domain = self._min_domain
        max_domain = self._max_domain
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        if self._interpolator is self._extrapolator:

            def _scalar_same(*args):
                points = np_local.array([args], dtype=float)
                res = interp_eval(points)
                return float(res[0])

            return _scalar_same

        def _scalar(*args):
            points = np_local.array([args], dtype=float)
            point = points[0]
            if ((point < min_domain) | (point > max_domain)).any():
                return float(extrap_eval(points)[0])
            return float(interp_eval(points)[0])

        return _scalar

    def expose_vector(self):
        min_domain = self._min_domain
        max_domain = self._max_domain
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        if self._interpolator is self._extrapolator:

            def _vector_same(*args):
                points = np_local.column_stack(np_local.broadcast_arrays(*args))
                return interp_eval(points)

            return _vector_same

        def _vector(*args):
            points = np_local.column_stack(np_local.broadcast_arrays(*args))
            result = np_local.empty(len(points), dtype=float)
            lower = points < min_domain
            upper = points > max_domain
            extrap_mask = lower.any(axis=1) | upper.any(axis=1)
            interp_mask = ~extrap_mask
            if interp_mask.any():
                result[interp_mask] = interp_eval(points[interp_mask])
            if extrap_mask.any():
                result[extrap_mask] = extrap_eval(points[extrap_mask])
            return result

        return _vector
