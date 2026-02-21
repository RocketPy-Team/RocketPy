"""Routers that choose interpolation or extrapolation strategies."""

from __future__ import annotations
import numpy as np
from rocketpy.mathutils.interpolation.polation_base import PolationBase


class PolationEvaluator1D(PolationBase):
    """Route 1D evaluation to interpolation or extrapolation."""

    def __init__(self, interpolator, extrapolator, x):
        self._interpolator = interpolator
        self._extrapolator = extrapolator
        self._x_min = float(x[0])
        self._x_max = float(x[-1])

        # Pre-compile the evaluator so the test suite gets the same speedup
        self._exposed_fn = self.expose()

    def evaluate(self, x):
        return self._exposed_fn(x)

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

        def _eval(x):
            x_type = type(x)

            # 1. Primary Hot Path: plain floats and ints
            if x_type is float or x_type is int:
                if x_min <= x <= x_max:
                    return interp_eval(x)
                return extrap_eval(x)

            # 2. Secondary Hot Path: NumPy scalars (np.float64, np.int32)
            if isinstance(x, (np.floating, np.integer)):
                x_val = float(x)
                if x_min <= x_val <= x_max:
                    return interp_eval(x_val)
                return extrap_eval(x_val)

            # 3. Vectorized Path & 0-D Arrays
            if x_type is np.ndarray:
                if x.ndim == 0:
                    val = x.item()
                    if type(val) is complex:
                        pass  # Fall through to complex step differentiation
                    else:
                        if x_min <= val <= x_max:
                            return interp_eval(val)
                        return extrap_eval(val)
                else:
                    # Standard vectorized array
                    out_dtype = complex if np.iscomplexobj(x) else float
                    result = np.empty_like(x, dtype=out_dtype)

                    x_real = x.real
                    inside = (x_real >= x_min) & (x_real <= x_max)
                    outside = ~inside

                    if inside.any():
                        result[inside] = interp_eval(x[inside])
                    if outside.any():
                        result[outside] = extrap_eval(x[outside])
                    return result

            # 4. Complex Step Differentiation / Fallback
            x_check = x.real
            if x_min <= x_check <= x_max:
                return interp_eval(x)
            return extrap_eval(x)

        return _eval

    def coefficients(self):
        return self._interpolator.coefficients()

    def derivative(self, x):
        """Route 1st derivative to interpolation or extrapolation."""
        interp_deriv = self._interpolator.derivative
        extrap_deriv = self._extrapolator.derivative

        if self._interpolator is self._extrapolator:
            return interp_deriv(x)

        x_type = type(x)
        if (x_type is float or x_type is int) or isinstance(
            x, (np.floating, np.integer)
        ):
            return (
                interp_deriv(x) if self._x_min <= x <= self._x_max else extrap_deriv(x)
            )

        if isinstance(x, np.ndarray):
            out_dtype = complex if np.iscomplexobj(x) else float
            result = np.empty_like(x, dtype=out_dtype)
            x_real = x.real
            inside = (x_real >= self._x_min) & (x_real <= self._x_max)
            outside = ~inside
            if inside.any():
                result[inside] = interp_deriv(x[inside])
            if outside.any():
                result[outside] = extrap_deriv(x[outside])
            return result

        x_check = x.real
        return (
            interp_deriv(x)
            if self._x_min <= x_check <= self._x_max
            else extrap_deriv(x)
        )

    def second_derivative(self, x):
        """Route 2nd derivative to interpolation or extrapolation."""
        interp_deriv2 = self._interpolator.second_derivative
        extrap_deriv2 = self._extrapolator.second_derivative

        if self._interpolator is self._extrapolator:
            return interp_deriv2(x)

        x_type = type(x)
        if (x_type is float or x_type is int) or isinstance(
            x, (np.floating, np.integer)
        ):
            return (
                interp_deriv2(x)
                if self._x_min <= x <= self._x_max
                else extrap_deriv2(x)
            )

        if isinstance(x, np.ndarray):
            out_dtype = complex if np.iscomplexobj(x) else float
            result = np.empty_like(x, dtype=out_dtype)
            x_real = x.real
            inside = (x_real >= self._x_min) & (x_real <= self._x_max)
            outside = ~inside
            if inside.any():
                result[inside] = interp_deriv2(x[inside])
            if outside.any():
                result[outside] = extrap_deriv2(x[outside])
            return result

        x_check = x.real
        return (
            interp_deriv2(x)
            if self._x_min <= x_check <= self._x_max
            else extrap_deriv2(x)
        )

    def integral(self, x):
        """Calculates the continuous antiderivative F(x) anchored at x_min. Fully vectorized."""
        if self._interpolator is self._extrapolator:
            return self._interpolator.integral(x)

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

        return result if isinstance(x, np.ndarray) else result.item()


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
        min_domain = self._min_domain
        max_domain = self._max_domain
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        if self._interpolator is self._extrapolator:

            def _eval_same(*args):
                arg0 = args[0]
                if type(arg0) in (float, int) or isinstance(
                    arg0, (np_local.floating, np_local.integer)
                ):
                    points = np_local.array([args], dtype=float)
                    arg_qty = 1
                else:
                    points = np_local.column_stack(args)
                    arg_qty = len(points)
                res = interp_eval(points)
                return float(res[0]) if arg_qty == 1 else res

            return _eval_same

        def _eval(*args):
            # 1. Hot Path: Single N-D point (skips heavy np.column_stack overhead)
            arg0 = args[0]
            if type(arg0) in (float, int) or isinstance(
                arg0, (np_local.floating, np_local.integer)
            ):
                points = np_local.array([args], dtype=float)
                arg_qty = 1
            else:
                # 2. Vectorized Path
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
        min_domain = self._min_domain
        max_domain = self._max_domain
        interp_eval = self._interpolator.evaluate
        extrap_eval = self._extrapolator.evaluate
        np_local = np

        if self._interpolator is self._extrapolator:

            def _eval_same(*args):
                arg0 = args[0]
                if type(arg0) in (float, int) or isinstance(
                    arg0, (np_local.floating, np_local.integer)
                ):
                    points = np_local.array([args], dtype=float)
                    arg_qty = 1
                else:
                    points = np_local.column_stack(args)
                    arg_qty = len(points)
                res = interp_eval(points)
                return float(res[0]) if arg_qty == 1 else res

            return _eval_same

        def _eval(*args):
            arg0 = args[0]
            if type(arg0) in (float, int) or isinstance(
                arg0, (np_local.floating, np_local.integer)
            ):
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
