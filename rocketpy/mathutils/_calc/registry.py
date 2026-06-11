"""Registry and factory dispatch for interpolation/extrapolation.

The main entry-point is :func:`build_interpolation_evaluator`, which
returns a router strategy that selects interpolation or extrapolation
based on the input bounds.
"""

from rocketpy.mathutils._calc.evaluator import (
    PolationEvaluator1D,
    PolationEvaluatorND,
    RegularGridEvaluator,
)
from rocketpy.mathutils._calc.polation_1d import (
    Akima1DPolation,
    Constant1DExtrapolation,
    Linear1DPolation,
    Pchip1DPolation,
    Polynomial1DPolation,
    Spline1DPolation,
    Zero1DExtrapolation,
)
from rocketpy.mathutils._calc.polation_nd import (
    ConstantNDExtrapolation,
    LinearNDPolation,
    RbfNaturalNDExtrapolation,
    RbfNDPolation,
    RegularGridConstantExtrapolation,
    RegularGridInterpolation,
    RegularGridNaturalExtrapolation,
    RegularGridZeroExtrapolation,
    ShepardNDPolation,
    ZeroNDExtrapolation,
)

_INTERP_1D = {
    "linear": Linear1DPolation,
    "polynomial": Polynomial1DPolation,
    "akima": Akima1DPolation,
    "pchip": Pchip1DPolation,
    "spline": Spline1DPolation,
}

_INTERP_ND = {
    "linear": LinearNDPolation,
    "rbf": RbfNDPolation,
    "shepard": ShepardNDPolation,
}

_EXTRAP_1D = {
    "zero": lambda interp, x, y: Zero1DExtrapolation(),
    "constant": lambda interp, x, y: Constant1DExtrapolation(x, y),
    # Natural 1D uses the interpolator, or falls back to Constant
    "natural": lambda interp, x, y: interp or Constant1DExtrapolation(x, y),
}


def _build_natural_nd(interp, method, domain, image):
    """Handles the specific quirks of ND Natural extrapolation."""
    if interp and method in ("rbf", "shepard"):
        return interp
    if method == "linear":
        return RbfNaturalNDExtrapolation(domain, image)
    return interp or ShepardNDPolation(domain, image)


_EXTRAP_ND = {
    "zero": lambda interp, method, domain, image: ZeroNDExtrapolation(),
    "constant": lambda interp, method, domain, image: ConstantNDExtrapolation(
        domain, image
    ),
    "natural": _build_natural_nd,
}

_EXTRAP_GRID = {
    "zero": lambda interp, axes, data: RegularGridZeroExtrapolation(),
    "constant": lambda interp, axes, data: RegularGridConstantExtrapolation(axes, data),
    "natural": lambda interp, axes, data: RegularGridNaturalExtrapolation(axes, data),
}


def build_interpolation_evaluator(
    method,
    extrap_method,
    dom_dim,
    x=None,
    y=None,
    domain=None,
    image=None,
    grid_axes=None,
    grid_data=None,
):
    """Build the evaluator router for interpolation and extrapolation.

    Routes to the correct 1D, ND, or Regular Grid construction path early
    to avoid passing unused arguments to the underlying classes.
    """
    if method == "regular_grid":
        if grid_axes is None or grid_data is None:
            raise ValueError("Regular grid requires both grid_axes and grid_data.")

        interp = RegularGridInterpolation(grid_axes, grid_data)
        extrap_cls = _EXTRAP_GRID.get(extrap_method, _EXTRAP_GRID["constant"])
        extrap = extrap_cls(interp, grid_axes, grid_data)

        return RegularGridEvaluator(interp, extrap, grid_axes)

    if dom_dim == 1:
        interp_cls = _INTERP_1D.get(method, Spline1DPolation)
        interp = interp_cls(x, y)

        extrap_cls = _EXTRAP_1D.get(extrap_method, _EXTRAP_1D["constant"])
        extrap = extrap_cls(interp, x, y)

        return PolationEvaluator1D(interp, extrap, x)

    interp_cls = _INTERP_ND.get(method, ShepardNDPolation)
    interp = interp_cls(domain, image)

    extrap_cls = _EXTRAP_ND.get(extrap_method, _EXTRAP_ND["constant"])
    extrap = extrap_cls(interp, method, domain, image)

    return PolationEvaluatorND(interp, extrap, domain)
