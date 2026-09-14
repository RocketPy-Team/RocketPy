"""Root finders that place an event at its exact time inside a solver step.

Each solver looks for the moment in ``[t0, t1]`` where ``value_at`` crosses
zero and returns that time, raising ``ValueError`` when it cannot find one.
What the value means, and how the flight is read at a given moment, is up to
the :class:`Event` calling it.
"""

from scipy.optimize import brentq

from ...tools import (
    calculate_cubic_hermite_coefficients,
    find_root_linear_interpolation,
    find_roots_cubic_function,
)


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
    """
    y0 = value_at(t0)
    y1 = value_at(t1)
    if y0 == y1:
        raise ValueError("the value is the same at both ends of the step")
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
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"Brent's method found no crossing: {error}") from error


def solve_cubic_hermite(value_at, t0, t1, rate_at, max_abs_imag=1e-3):
    """Find the crossing of a cubic fitted to the values and rates at the ends.

    Parameters
    ----------
    value_at : callable
        ``value_at(t) -> float``, zero at the event.
    t0, t1 : float
        Start and end of the step, in seconds.
    rate_at : callable
        ``rate_at(t) -> float``, the time derivative of ``value_at``.
    max_abs_imag : float, optional
        Largest imaginary part a root may have and still count as real.

    Returns
    -------
    float
        The event time, in seconds.
    """
    duration = t1 - t0
    coefficients = calculate_cubic_hermite_coefficients(
        0.0, duration, value_at(t0), rate_at(t0), value_at(t1), rate_at(t1)
    )
    roots = [
        root.real
        for root in find_roots_cubic_function(*coefficients)
        if 0.0 < root.real < duration and abs(root.imag) < max_abs_imag
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one crossing inside the step, found {len(roots)}")
    return t0 + roots[0]


# Solver name -> (solver, the exact_time_config keys it accepts besides
# "solver" and "target", the keys it requires)
SOLVERS = {
    "linear": (solve_linear, frozenset(), frozenset()),
    "brentq": (solve_brentq, frozenset({"xtol", "rtol", "maxiter"}), frozenset()),
    "cubic_hermite": (
        solve_cubic_hermite,
        frozenset({"derivative_function", "max_abs_imag"}),
        frozenset({"derivative_function"}),
    ),
}
