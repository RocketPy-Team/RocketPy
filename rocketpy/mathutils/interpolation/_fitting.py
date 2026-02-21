"""Coefficient fitting routines for 1-D interpolation methods.

Every public function in this module takes sorted ``x`` and ``y`` arrays and
returns a data structure (NumPy array / tuple) that the corresponding
evaluation factory can close over.
"""

import warnings

import numpy as np
from scipy import linalg

# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


def fit_polynomial(x, y):
    """Fit a single polynomial of degree ``len(x) - 1`` through all points.

    Parameters
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Returns
    -------
    coeffs : ndarray, shape (n,)
        Coefficients in *ascending* power order: ``c[0] + c[1]*x + …``.
    """
    degree = len(x) - 1
    if np.amax(np.abs(x)) ** degree > 1e308:
        warnings.warn(
            "Polynomial interpolation of too many points can't be done. "
            "Once the degree is too high, numbers get too large. "
            "Falling back to spline coefficients.",
            stacklevel=3,
        )
        return None  # caller should fall back to spline
    V = np.vander(x, increasing=True)
    return np.linalg.solve(V, y)


# ---------------------------------------------------------------------------
# Natural cubic spline
# ---------------------------------------------------------------------------


def fit_spline(x, y):
    r"""Fit a natural cubic spline in local coordinates.

    Returns coefficients ``[a, b, c, d]`` for each interval stored as a
    ``(4, n-1)`` array, where the polynomial for segment *i* is

    .. math::
        p_i(t) = a_i + b_i t + c_i t^2 + d_i t^3, \quad t = x - x_i

    Parameters
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Returns
    -------
    coeffs : ndarray, shape (4, n-1)
        Rows are ``[a, b, c, d]``.
    """
    n = len(x)
    h = np.diff(x)

    # Build tridiagonal system for the c coefficients (natural BC: c[0]=c[-1]=0)
    banded = np.zeros((3, n))
    banded[1, 0] = banded[1, -1] = 1.0
    banded[2, :-2] = h[:-1]
    banded[1, 1:-1] = 2.0 * (h[:-1] + h[1:])
    banded[0, 2:] = h[1:]

    rhs = np.zeros(n)
    rhs[1:-1] = 3.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    c = linalg.solve_banded((1, 1), banded, rhs, overwrite_ab=True, overwrite_b=True)

    b = (y[1:] - y[:-1]) / h - h * (2.0 * c[:-1] + c[1:]) / 3.0
    d = (c[1:] - c[:-1]) / (3.0 * h)

    return np.vstack([y[:-1], b, c[:-1], d])


# ---------------------------------------------------------------------------
# Akima  (vectorised – no Python loop, no linalg.solve per interval)
# ---------------------------------------------------------------------------


def fit_akima(x, y):
    r"""Fit an Akima spline in local coordinates.

    The slopes at each knot are computed using the original Akima weighting
    of adjacent finite differences, extended at the boundaries by reflected
    values.  This is fully vectorised — no Python loop.

    Returns coefficients ``[a, b, c, d]`` for each interval in a ``(4, n-1)``
    array identical in layout to :func:`fit_spline`.

    Parameters
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Returns
    -------
    coeffs : ndarray, shape (4, n-1)
    """
    n = len(x)
    h = np.diff(x)
    m = np.diff(y) / h  # slopes of segments, shape (n-1,)

    # Extend m with 2 ghost values on each side (Akima boundary reflection)
    # m_ext indices: 0..n+2  (n-1 interior + 2 left + 2 right)
    m_ext = np.empty(n + 3)
    m_ext[2:-2] = m
    m_ext[1] = 2.0 * m[0] - m[1]
    m_ext[0] = 2.0 * m_ext[1] - m[0]
    m_ext[-2] = 2.0 * m[-1] - m[-2]
    m_ext[-1] = 2.0 * m_ext[-2] - m[-1]

    # Weights: |m_{i+1} - m_i|
    dm = np.abs(np.diff(m_ext))  # shape (n+2,)

    # Akima slope at each knot: weighted average of adjacent segment slopes.
    # w1 = |m_{i+1} - m_i|, w2 = |m_{i-1} - m_{i-2}|
    # t_i = (w1 * m_{i-1} + w2 * m_i) / (w1 + w2)
    # When w1 + w2 == 0, fall back to simple average.
    w1 = dm[2:]  # |m_{i+1} - m_i|  for i in 0..n-1
    w2 = dm[:-2]  # |m_{i-1} - m_{i-2}| for i in 0..n-1
    m_left = m_ext[1:-2]  # m_{i-1}
    m_right = m_ext[2:-1]  # m_i

    wsum = w1 + w2
    t = np.where(
        wsum > 0,
        (w1 * m_left + w2 * m_right) / wsum,
        0.5 * (m_left + m_right),
    )

    # Hermite-to-local coefficients:
    #   a = y_i
    #   b = t_i
    #   c = (3*m_i - 2*t_i - t_{i+1}) / h_i
    #   d = (t_i + t_{i+1} - 2*m_i) / h_i^2
    a = y[:-1]
    b = t[:-1]
    c_coeff = (3.0 * m - 2.0 * t[:-1] - t[1:]) / h
    d_coeff = (t[:-1] + t[1:] - 2.0 * m) / (h * h)

    return np.vstack([a, b, c_coeff, d_coeff])


# ---------------------------------------------------------------------------
# PCHIP  (Fritsch–Carlson monotone-preserving Hermite)
# ---------------------------------------------------------------------------


def fit_pchip(x, y):
    r"""Fit a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).

    Uses the Fritsch–Carlson method to compute monotone-preserving slopes
    and converts them to local-coordinate cubic coefficients identical in
    layout to :func:`fit_spline`.

    Parameters
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Returns
    -------
    coeffs : ndarray, shape (4, n-1)
    """
    n = len(x)
    h = np.diff(x)
    m = np.diff(y) / h

    # Compute slopes at each knot
    t = np.zeros(n)

    # Interior knots
    if n > 2:
        # Harmonic mean weighted by segment lengths
        w1 = 2.0 * h[1:] + h[:-1]
        w2 = h[1:] + 2.0 * h[:-1]

        # Where signs differ or either is zero, slope is zero (monotonicity)
        sign_change = (m[:-1] * m[1:]) <= 0
        pos_mask = ~sign_change

        # Safe harmonic mean
        t[1:-1] = np.where(
            pos_mask,
            (w1 + w2) / (w1 / m[:-1] + w2 / m[1:]),
            0.0,
        )

    # Boundary slopes: one-sided, clamped for monotonicity
    t[0] = _pchip_edge_slope(
        h[0], h[1] if n > 2 else h[0], m[0], m[1] if n > 2 else m[0]
    )
    t[-1] = _pchip_edge_slope(
        h[-1], h[-2] if n > 2 else h[-1], m[-1], m[-2] if n > 2 else m[-1]
    )

    # Convert to local Hermite coefficients (same as akima)
    a = y[:-1]
    b = t[:-1]
    c_coeff = (3.0 * m - 2.0 * t[:-1] - t[1:]) / h
    d_coeff = (t[:-1] + t[1:] - 2.0 * m) / (h * h)

    return np.vstack([a, b, c_coeff, d_coeff])


def _pchip_edge_slope(h0, h1, m0, m1):
    """Compute boundary slope for PCHIP (non-centred 3-point formula,
    clamped to preserve monotonicity)."""
    t = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(t) != np.sign(m0):
        t = 0.0
    elif np.sign(m0) != np.sign(m1) and np.abs(t) > 3.0 * np.abs(m0):
        t = 3.0 * m0
    return t


# ---------------------------------------------------------------------------
# Pre-computed data for analytical derivatives and integrals
# ---------------------------------------------------------------------------


def precompute_linear_deriv_integral(x, y):
    """Pre-compute data for analytical linear derivative and integral.

    Parameters
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Returns
    -------
    slopes : ndarray, shape (n-1,)
        Piecewise-constant derivative values.
    cum_integrals : ndarray, shape (n,)
        Cumulative trapezoid integral from x[0] to x[i].
    """
    h = np.diff(x)
    slopes = np.diff(y) / h
    # Cumulative trapezoid: area of each trapezoid is 0.5*(y[i]+y[i+1])*h[i]
    trap_areas = 0.5 * (y[:-1] + y[1:]) * h
    cum_integrals = np.empty(len(x))
    cum_integrals[0] = 0.0
    np.cumsum(trap_areas, out=cum_integrals[1:])
    return slopes, cum_integrals


def precompute_cubic_cumulative_integrals(x, coeffs):
    """Pre-compute cumulative integrals over full intervals for cubic methods.

    For piece *i* with local polynomial ``a + b*t + c*t² + d*t³``,
    the integral over the full interval ``[0, h_i]`` is
    ``a*h + b*h²/2 + c*h³/3 + d*h⁴/4``.

    Parameters
    ----------
    x : ndarray, shape (n,)
    coeffs : ndarray, shape (4, n-1)

    Returns
    -------
    cum : ndarray, shape (n,)
        ``cum[0] = 0``; ``cum[i] = ∫_{x[0]}^{x[i]} p(t) dt``.
    """
    h = np.diff(x)
    a, b, c, d = coeffs
    h2 = h * h
    h3 = h2 * h
    h4 = h3 * h
    piece_integrals = a * h + b * h2 / 2.0 + c * h3 / 3.0 + d * h4 / 4.0
    cum = np.empty(len(x))
    cum[0] = 0.0
    np.cumsum(piece_integrals, out=cum[1:])
    return cum
