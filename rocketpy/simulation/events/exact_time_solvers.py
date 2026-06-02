"""Utilities for solving exact event times on integration steps.

This module centralizes root-solving logic used by simulation event handlers,
keeping numerical primitives in ``rocketpy.tools`` while exposing simulation-
level policies for valid-root selection.
"""

from ...tools import (
    calculate_cubic_hermite_coefficients,
    find_root_linear_interpolation,
    find_roots_cubic_function,
)
from scipy.optimize import brentq


def filter_roots_by_policy(
    roots,
    lower_bound,
    upper_bound,
    max_abs_imag=1e-3,
    strict_interval=True,
):
    """Filter complex roots into valid real-valued candidates.

    Parameters
    ----------
    roots : iterable[complex]
        Candidate roots, usually from a polynomial solver.
    lower_bound : float
        Lower bound in seconds for valid roots.
    upper_bound : float
        Upper bound in seconds for valid roots.
    max_abs_imag : float, optional
        Maximum allowed absolute imaginary part for roots.
    strict_interval : bool, optional
        If True, enforce ``lower_bound < t < upper_bound``. If False,
        enforce ``lower_bound <= t <= upper_bound``.

    Returns
    -------
    list[float]
        Real parts of roots that satisfy the selection policy.
    """

    valid_roots = []
    for root in roots:
        root_real = root.real
        if strict_interval:
            in_interval = lower_bound < root_real < upper_bound
        else:
            in_interval = lower_bound <= root_real <= upper_bound

        if in_interval and abs(root.imag) < max_abs_imag:
            valid_roots.append(root_real)

    return valid_roots


def solve_cubic_hermite_step_roots(
    step_end,
    y0,
    yp0,
    y1,
    yp1,
    lower_bound,
    upper_bound,
    max_abs_imag=1e-3,
    strict_interval=True,
):
    """Solve cubic-Hermite roots and apply a selection policy.

    Parameters
    ----------
    step_end : float
        End of interpolation interval in seconds for the cubic-Hermite fit.
        The fit start is always 0 seconds in this helper.
    y0, yp0, y1, yp1 : float
        Scalar values and derivatives at the start/end of the interval.
    lower_bound : float
        Lower bound in seconds for valid roots.
    upper_bound : float
        Upper bound in seconds for valid roots.
    max_abs_imag : float, optional
        Maximum allowed absolute imaginary part for roots.
    strict_interval : bool, optional
        If True, enforce ``lower_bound < t < upper_bound``. If False,
        enforce ``lower_bound <= t <= upper_bound``.

    Returns
    -------
    list[float]
        Valid relative event times in seconds inside the selected interval.
    """

    a, b, c, d = calculate_cubic_hermite_coefficients(0.0, step_end, y0, yp0, y1, yp1)
    roots = find_roots_cubic_function(a, b, c, d)
    return filter_roots_by_policy(
        roots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_abs_imag=max_abs_imag,
        strict_interval=strict_interval,
    )


def solve_exact_time_linear(
    previous_state,
    current_state,
    interpolator,
    event_function,
    no_root_error_message,
    target=0.0,
    **context,
):
    """Solve exact event time/state by linear interpolation of event values.

    Parameters
    ----------
    previous_state, current_state : array_like
        Consecutive sampled states, each formatted as
        [t, x, y, z, vx, vy, vz, ...].
    interpolator : callable
        Dense-output interpolator mapping absolute time to state.
    event_function : callable
        Callable with signature ``event_function(state, **context) -> float``,
        where ``state`` is the state vector without time.
    target : float, optional
        Target scalar value to solve for, default is 0.
    no_root_error_message : str, optional
        Error message when no valid root is found.
    **context : dict
        Additional context passed to event_function.

    Returns
    -------
    dict
        Dictionary with ``event_time`` in seconds and ``event_state``.
    """
    # Pop state to avoid passing two state arguments to event_function
    function_context = dict(context)
    function_context.pop("state", None)

    t0 = previous_state[0]
    t1 = current_state[0]
    y0 = event_function(previous_state[1:], **function_context)
    y1 = event_function(current_state[1:], **function_context)

    if y0 == y1:
        raise ValueError(no_root_error_message)

    event_time = find_root_linear_interpolation(t0, t1, y0, y1, target)
    return {"event_time": event_time, "event_state": interpolator(event_time)}


def solve_exact_time_brentq(
    previous_state,
    current_state,
    interpolator,
    event_function,
    no_root_error_message,
    target=0.0,
    xtol=1e-12,
    rtol=1e-8,
    maxiter=100,
    **context,
):
    """Solve exact event time using Brent's method on the interpolated state.

    This tries a robust root find on the scalar event function evaluated on the
    dense-output interpolator. It requires the event to change sign over the
    interval; otherwise a ValueError is raised.
    """
    function_context = dict(context)
    function_context.pop("state", None)

    t0 = previous_state[0]
    t1 = current_state[0]

    def f(t):
        return event_function(interpolator(t), **function_context) - target

    y0 = f(t0)
    y1 = f(t1)

    if y0 == y1:
        raise ValueError(no_root_error_message)

    try:
        event_time = brentq(f, t0, t1, xtol=xtol, rtol=rtol, maxiter=maxiter)
    except Exception as e:
        raise ValueError(no_root_error_message) from e

    return {"event_time": event_time, "event_state": interpolator(event_time)}


def solve_exact_time_cubic_hermite(
    previous_state,
    current_state,
    interpolator,
    event_function,
    derivative_function,
    no_root_error_message,
    target=0.0,
    step_end_function=None,
    max_abs_imag=1e-3,
    **context,
):
    """Solve exact event time/state via cubic-Hermite root interpolation.

    Parameters
    ----------
    previous_state, current_state : array_like
        Consecutive sampled states, each formatted as
        [t, x, y, z, vx, vy, vz, ...].
    interpolator : callable
        Dense-output interpolator mapping absolute time to state.
    event_function : callable
        Callable with signature ``event_function(state, **context) -> float``,
        where ``state`` is the state vector without time.
    derivative_function : callable
        Callable with signature ``derivative_function(state, **context) -> float``,
        returning the derivative of ``event_function`` with respect to time.
    target : float, optional
        Target scalar value to solve for. The solver finds the time when
        ``event_function(state, **kwargs) == target``. Defaults to 0.0.
    step_end_function : callable, optional
        Callable receiving ``previous_state``, ``current_state`` and
        ``**context`` and returning the step interval endpoint (seconds)
        used to build the cubic-Hermite polynomial. If None, uses the step
        duration ``current_state[0] - previous_state[0]``.
    max_abs_imag : float, optional
        Maximum allowed absolute imaginary part for roots.
    no_root_error_message : str, optional
        Error message when no valid root is found.
    **context : dict
        Additional context passed to event and derivative callables.

    Returns
    -------
    dict
        Dictionary with ``event_time`` in seconds and ``event_state``.
    """
    function_context = dict(context)
    function_context.pop("state", None)

    step_duration = current_state[0] - previous_state[0]

    step_end = (
        step_end_function(
            previous_state=previous_state,
            current_state=current_state,
            **function_context,
        )
        if step_end_function is not None
        else step_duration
    )

    y0 = event_function(previous_state[1:], **function_context) - target
    y1 = event_function(current_state[1:], **function_context) - target
    yp0 = derivative_function(previous_state[1:], **function_context)
    yp1 = derivative_function(current_state[1:], **function_context)

    valid_t_roots = solve_cubic_hermite_step_roots(
        step_end=step_end,
        y0=y0,
        yp0=yp0,
        y1=y1,
        yp1=yp1,
        lower_bound=0.0,
        upper_bound=step_duration,
        max_abs_imag=max_abs_imag,
        strict_interval=True,
    )

    if len(valid_t_roots) > 1:
        raise ValueError(no_root_error_message)
    if len(valid_t_roots) == 0:
        raise ValueError(no_root_error_message)

    event_time = previous_state[0] + valid_t_roots[0]
    return {"event_time": event_time, "event_state": interpolator(event_time)}
