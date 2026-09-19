from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

from ...tools import find_root_linear_interpolation

_NO_CROSSING = "the value does not cross its target during the step"


def _same_side(y0, y1):
    """Whether two values are both above zero or both below it."""
    return (y0 > 0 and y1 > 0) or (y0 < 0 and y1 < 0)


def solve_linear(value_at, t0, t1):
    """Find the crossing of a straight line through the two ends of the step.

    Parameters
    ----------
    value_at : callable
        ``value_at(t) -> float``, zero at the event.
    t0, t1 : float
        Start and end of the step, in seconds.

    Returns
    -------
    float
        The event time, in seconds.

    Raises
    ------
    ValueError
        If the value does not cross zero inside the step, or is zero at both
        ends of it.
    """
    y0 = value_at(t0)
    y1 = value_at(t1)
    # Without a sign change the line crosses zero outside the step, and the
    # event would fire at a time the step never reached
    if _same_side(y0, y1):
        raise ValueError(_NO_CROSSING)
    if y0 == y1:
        raise ValueError(
            "the value is exactly at its target at both ends of the step, so "
            "the crossing could be anywhere in it"
        )
    return find_root_linear_interpolation(t0, t1, y0, y1, 0.0)


def solve_brentq(value_at, t0, t1, xtol=1e-12, rtol=1e-8, maxiter=100):
    """Find the crossing with Brent's method, evaluating inside the step.

    Parameters
    ----------
    value_at : callable
        ``value_at(t) -> float``, zero at the event. It must change sign
        between ``t0`` and ``t1``.
    t0, t1 : float
        Start and end of the step, in seconds.
    xtol, rtol : float, optional
        Absolute and relative time tolerances passed to
        ``scipy.optimize.brentq``.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    float
        The event time, in seconds.
    """
    try:
        return brentq(value_at, t0, t1, xtol=xtol, rtol=rtol, maxiter=maxiter)
    except ValueError as error:
        # scipy's own message ("f(a) and f(b) must have different signs")
        # means nothing to a user, so say what it means for the event
        if _same_side(value_at(t0), value_at(t1)):
            raise ValueError(_NO_CROSSING) from error
        raise ValueError(f"Brent's method failed: {error}") from error
    except RuntimeError as error:
        raise ValueError(f"Brent's method failed: {error}") from error


def solve_cubic_hermite(value_at, t0, t1, rate_at):
    """Find the crossing of a cubic fitted to the values and rates at the ends.

    Parameters
    ----------
    value_at : callable
        ``value_at(t) -> float``, zero at the event.
    t0, t1 : float
        Start and end of the step, in seconds.
    rate_at : callable
        ``rate_at(t) -> float``, the time derivative of ``value_at``.

    Returns
    -------
    float
        The event time, in seconds.

    Raises
    ------
    ValueError
        If the fitted cubic does not reach zero exactly once inside the step.
    """
    # Values and rates are read at one end, then the other, so the event's
    # context is only worked out once per end.
    y0, rate0 = value_at(t0), rate_at(t0)
    y1, rate1 = value_at(t1), rate_at(t1)
    fitted = CubicHermiteSpline([t0, t1], [y0, y1], [rate0, rate1])
    crossings = [t for t in fitted.roots(extrapolate=False) if t0 < t < t1]
    if not crossings:
        raise ValueError(_NO_CROSSING)
    if len(crossings) > 1:
        raise ValueError(
            f"the cubic fitted over the step reaches the target {len(crossings)} "
            "times, so it is unclear which one is the event"
        )
    return crossings[0]


# Solver name -> (solver, the exact_time_config keys it accepts besides
# "solver" and "target", the keys it requires)
SOLVERS = {
    "linear": (solve_linear, frozenset(), frozenset()),
    "brentq": (solve_brentq, frozenset({"xtol", "rtol", "maxiter"}), frozenset()),
    "cubic_hermite": (
        solve_cubic_hermite,
        frozenset({"derivative_function"}),
        frozenset({"derivative_function"}),
    ),
}
