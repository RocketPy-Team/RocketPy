"""Routers that choose interpolation or extrapolation strategies."""

from __future__ import annotations

import numpy as np

from rocketpy.mathutils._calc.polation_base import PolationBase


class PolationEvaluator1D(PolationBase):
    """Route 1D evaluation to interpolation or extrapolation."""

    def __init__(self, interpolator, extrapolator, x):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        self._x_min = float(x[0])
        self._x_max = float(x[-1])

        # Pre-compile the evaluator so the test suite gets the same speedup
        self._exposed_fn = self.expose()

    def evaluate(self, x, _is_iterable=None):
        return self._exposed_fn(x, _is_iterable=_is_iterable)

    def expose(self):
        """Flattens the evaluator into a fast closure for the simulation loop."""
        # Localize variables to strip out `self.` dictionary lookups
        x_min = self._x_min
        x_max = self._x_max
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate

        # Fast path if interpolation and extrapolation are the same object
        if self._interpolator is self._extrapolator:
            return interp_eval

        def _eval(x, _is_iterable=None):
            if _is_iterable is None:
                _is_iterable = hasattr(x, "__iter__")

            if not _is_iterable:
                if x_min <= x.real <= x_max:
                    return interp_eval(x, _is_iterable=False)
                return extrap_eval(x, _is_iterable=False)
            else:
                x = np.asarray(x, dtype=float)
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

        return _eval

    def coefficients(self):
        return self._interpolator.coefficients()

    def derivative(self, x, _is_iterable=None):
        """Route 1st derivative to interpolation or extrapolation."""
        interp_deriv = self._interpolator.derivative
        extrap_deriv = self._extrapolator.derivative

        if self._interpolator is self._extrapolator:
            return interp_deriv(x, _is_iterable=_is_iterable)

        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__")

        if not _is_iterable:
            if self._x_min <= x.real <= self._x_max:
                return interp_deriv(x, _is_iterable=False)
            return extrap_deriv(x, _is_iterable=False)
        else:
            x = np.asarray(x, dtype=float)
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
            _is_iterable = hasattr(x, "__iter__")

        if not _is_iterable:
            if self._x_min <= x.real <= self._x_max:
                return interp_deriv2(x, _is_iterable=False)
            return extrap_deriv2(x, _is_iterable=False)
        else:
            x = np.asarray(x, dtype=float)
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
        """Calculates the continuous antiderivative F(x) anchored at x_min. Fully vectorized."""
        if self._interpolator is self._extrapolator:
            return self._interpolator.integral(x, _is_iterable=_is_iterable)

        if _is_iterable is None:
            _is_iterable = hasattr(x, "__iter__")

        if not _is_iterable:
            # Scalar path
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

    def __init__(self, interpolator, extrapolator, domain):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        domain = np.asarray(domain, dtype=float)
        self._min_domain = np.min(domain, axis=0)
        self._max_domain = np.max(domain, axis=0)

        self._exposed_fn = self.expose()

    def evaluate(self, *args):
        return self._exposed_fn(*args)

    def expose(self):
        # pylint: disable=too-many-statements
        min_domain = self._min_domain
        max_domain = self._max_domain
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        def _eval(*args, _is_iterable=None):
            points = np_local.column_stack(args)
            arg_qty = len(points)

            out_dtype = complex if np_local.iscomplexobj(points) else float
            result = np_local.empty(arg_qty, dtype=out_dtype)

            points_real = points.real
            lower = points_real < min_domain
            upper = points_real > max_domain
            extrap_mask = lower.any(axis=1) | upper.any(axis=1)
            interp_mask = ~extrap_mask

            if interp_mask.any():
                inside_points = points[interp_mask]
                interp_values = interp_eval(inside_points)

                # LinearNDInterpolator NaN fix
                if np_local.any(np_local.isnan(interp_values)):
                    interp_values = np_local.asarray(interp_values, dtype=float)
                    nan_mask = np_local.isnan(interp_values)
                    if nan_mask.any():
                        interp_values[nan_mask] = extrap_eval(inside_points[nan_mask])

                result[interp_mask] = interp_values

            if extrap_mask.any():
                result[extrap_mask] = extrap_eval(points[extrap_mask])

            if arg_qty == 1:
                # Return the type safely (complex if it has an imaginary part, otherwise float)
                res_val = result[0]
                return (
                    complex(res_val)
                    if np_local.iscomplexobj(res_val)
                    else float(res_val)
                )
            return result

        return _eval


class RegularGridEvaluator(PolationBase):
    """Route regular grid evaluation based on axes bounds."""

    def __init__(self, interpolator, extrapolator, grid_axes):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        self._min_domain = np.array([axis[0] for axis in grid_axes], dtype=float)
        self._max_domain = np.array([axis[-1] for axis in grid_axes], dtype=float)
        self._exposed_fn = self.expose()

    def evaluate(self, *args):
        return self._exposed_fn(*args)

    def expose(self):
        # pylint: disable=too-many-statements
        min_domain = self._min_domain
        max_domain = self._max_domain
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        if self._interpolator is self._extrapolator:

            def _eval_same(*args, _is_iterable=None):
                if _is_iterable is None:
                    _is_iterable = hasattr(args[0], "__iter__")
                if not _is_iterable:
                    points = np_local.array([args], dtype=float)
                    arg_qty = 1
                else:
                    points = np_local.column_stack(args)
                    arg_qty = len(points)
                res = interp_eval(points)
                return float(res[0]) if arg_qty == 1 else res

            return _eval_same

        def _eval(*args, _is_iterable=None):
            if _is_iterable is None:
                _is_iterable = hasattr(args[0], "__iter__")

            if not _is_iterable:
                points = np_local.array([args], dtype=float)
                arg_qty = 1
            else:
                points = np_local.column_stack(args)
                arg_qty = len(points)

            result = np_local.empty(arg_qty, dtype=float)

            lower = points < min_domain
            upper = points > max_domain
            extrap_mask = lower.any(axis=1) | upper.any(axis=1)
            interp_mask = ~extrap_mask
            if interp_mask.any():
                result[interp_mask] = interp_eval(points[interp_mask])
            if extrap_mask.any():
                result[extrap_mask] = extrap_eval(points[extrap_mask])
            if arg_qty == 1:
                return float(result[0])
            return result

        return _eval
