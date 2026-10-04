"""Functions that support the aerodynamic surface classes.

They build the :class:`Function` objects the surfaces store, describe the
direction of the flow from the angle of attack and the sideslip angle (see
:func:`_wind_axes`), and convert coefficients between the wind frame, the body
frame and a surface's own turned frame.
"""

import inspect
import math

import numpy as np

from rocketpy.mathutils import Function

# Below this size the crossflow is rounding noise, and its direction means nothing
_NO_CROSSFLOW = 1e-12


def _as_function(func, variables, name):
    """Wrap a callable that takes ``*args`` as a :class:`Function`.

    ``Function`` counts a callable's parameters to know how many inputs it has,
    so ``func`` is given a signature with one parameter per variable. The
    signature is set on ``func`` itself.

    Parameters
    ----------
    func : callable
        Callable taking one value per variable of ``variables``.
    variables : sequence of str
        Names of the inputs, in the order ``func`` takes them.
    name : str
        Name of the output.

    Returns
    -------
    Function
        ``func`` as a Function of ``variables``.
    """
    func.__signature__ = inspect.Signature(
        inspect.Parameter(var, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for var in variables
    )
    return Function(func, list(variables), [name])


def _computed_together(compute, variables, names, reuse=True):
    """Build the Functions of several coefficients that are computed together.

    The flight asks for all the coefficients of a surface at the same state one
    after the other, so by default ``compute`` runs once per state and its
    result is kept for the other coefficients. Array inputs are evaluated one
    element at a time.

    Parameters
    ----------
    compute : callable
        Takes the tuple of input values and returns one value per name.
    variables : sequence of str
        Names of the inputs, in the order ``compute`` reads them.
    names : sequence of str
        Names of the coefficients, in the order ``compute`` returns them.
    reuse : bool, optional
        Whether to keep the last result. Only safe when ``compute`` depends on
        its inputs alone. Default True.

    Returns
    -------
    tuple of Function
        One Function of ``variables`` per name.
    """
    last = [None, None]

    def values(args):
        try:
            if reuse and args == last[0]:
                return last[1]
            result = compute(args)
        except (TypeError, ValueError):
            if not any(np.ndim(arg) for arg in args):
                raise
            return _element_by_element(values, args, len(names))
        last[0], last[1] = args, result
        return result

    return tuple(
        _as_function(lambda *args, i=i: values(args)[i], variables, name)
        for i, name in enumerate(names)
    )


def _element_by_element(values, args, count):
    """Evaluate ``values`` at each point of array inputs, one array per output."""
    arrays = np.broadcast_arrays(*(np.asarray(arg, dtype=float) for arg in args))
    result = np.empty((count, *arrays[0].shape))
    for point in np.ndindex(arrays[0].shape):
        result[(slice(None), *point)] = values(
            tuple(float(array[point]) for array in arrays)
        )
    return result


def _wind_axes(alpha, beta):
    """Compute the unit vectors of the wind frame, written in the body frame.

    Drag acts along ``-u``, lift along ``(0, -u_z, u_y) / h`` (perpendicular to
    ``u``, in the body y-z plane) and the side force along
    ``(h, -u_x * u_y / h, -u_x * u_z / h)``, which completes the right-handed
    set. With ``beta = 0`` this is the plain angle-of-attack rotation.

    The two partial angles are not the angles of a rotation sequence, so the
    direction is rebuilt from them instead of chaining an ``alpha`` and a
    ``beta`` rotation, which would only be exact with one of them at zero. With
    the flow exactly sideways both angles are 90 degrees and no longer say how
    the crossflow splits between x and y; it then comes out split evenly.

    Parameters
    ----------
    alpha : float
        Angle of attack, ``arctan2(v_y, v_z)``, in radians, where ``v`` is the
        rocket's velocity relative to the air in the body frame.
    beta : float
        Sideslip angle, ``arctan2(v_x, v_z)``, in radians.

    Returns
    -------
    tuple of float
        ``(u_x, u_y, u_z, h)``: the direction ``u`` of the rocket's velocity
        relative to the air, and ``h = hypot(u_y, u_z)``.
    """
    sin_alpha, cos_alpha = math.sin(alpha), math.cos(alpha)
    sin_beta, cos_beta = math.sin(beta), math.cos(beta)
    # Parallel to v scaled by v_z; the sign of cos(alpha) restores tail-first flow
    u_x, u_y, u_z = sin_beta * cos_alpha, sin_alpha * cos_beta, cos_alpha * cos_beta
    # Never zero: the cosine of a float is never exactly zero
    norm = math.copysign(math.sqrt(u_x**2 + u_y**2 + u_z**2), cos_alpha)
    u_x, u_y, u_z = u_x / norm, u_y / norm, u_z / norm
    return u_x, u_y, u_z, math.hypot(u_y, u_z)


def total_angle_and_roll(alpha, beta):
    """Compute the total angle of attack and the roll angle of the wind.

    Parameters
    ----------
    alpha : float or array
        Angle of attack, in radians.
    beta : float or array
        Sideslip angle, in radians.

    Returns
    -------
    alpha_total : float or array
        Angle between the rocket's axis and its velocity relative to the air, in
        radians: 0 flying straight into the air, ``pi / 2`` sideways and ``pi``
        tail first.
    phi : float or array
        Direction around the body the crossflow comes from, in radians:
        ``pi / 2`` for a pure angle of attack (``alpha > 0``) and 0 for a pure
        sideslip (``beta > 0``). It is 0 when there is no crossflow.
    """
    try:
        u_x, u_y, u_z, _ = _wind_axes(alpha, beta)
    except TypeError:
        if not (np.ndim(alpha) or np.ndim(beta)):
            raise
        return np.vectorize(total_angle_and_roll)(alpha, beta)
    crossflow = math.hypot(u_x, u_y)
    roll = math.atan2(u_y, u_x) if crossflow > _NO_CROSSFLOW else 0.0
    return math.atan2(crossflow, u_z), roll


def _wind_to_body_coefficients(c_lift, c_drag, c_side, variables):
    """Rotate wind-frame force coefficients into the body frame.

    See :func:`_wind_axes` for the directions of lift, drag and side force.

    Parameters
    ----------
    c_lift, c_drag, c_side : AeroCoefficient
        Lift, drag and side-force coefficients.
    variables : sequence of str
        Inputs of the returned Functions: the angle of attack and the sideslip
        angle first, then any other variable the three coefficients depend on.

    Returns
    -------
    tuple of Function
        Body-frame normal, side and axial force coefficients ``(cN, cY, cA)``,
        as Functions of ``variables``.
    """
    lift, drag, side = (c.evaluator(variables) for c in (c_lift, c_drag, c_side))

    def compute(args):
        u_x, u_y, u_z, h = _wind_axes(args[0], args[1])
        c_l, c_d, c_s = lift(*args), drag(*args), side(*args)
        return (
            (c_s * u_x * u_y + c_l * u_z) / h + c_d * u_y,
            c_s * h - c_d * u_x,
            (c_s * u_x * u_z - c_l * u_y) / h + c_d * u_z,
        )

    return _computed_together(compute, variables, ("cN", "cY", "cA"))


def _body_to_wind_coefficients(c_normal, c_side, c_axial, variables):
    """Rotate body-frame force coefficients into the wind frame.

    Inverse of :func:`_wind_to_body_coefficients`.

    Parameters
    ----------
    c_normal, c_side, c_axial : AeroCoefficient
        Body-frame normal, side and axial force coefficients.
    variables : sequence of str
        Every variable of the surface, in the order its coefficients are called
        with (the angle of attack and the sideslip angle first).

    Returns
    -------
    tuple of Function
        Wind-frame lift, drag and side-force coefficients ``(cL, cD, cQ)``, as
        Functions of ``variables``.
    """
    normal, side, axial = (c.get_value_opt for c in (c_normal, c_side, c_axial))

    def compute(args):
        u_x, u_y, u_z, h = _wind_axes(args[0], args[1])
        c_n, c_y, c_a = normal(*args), side(*args), axial(*args)
        return (
            (c_n * u_z - c_a * u_y) / h,
            -c_y * u_x + c_n * u_y + c_a * u_z,
            c_y * h + (c_n * u_y + c_a * u_z) * u_x / h,
        )

    # Not kept: a fin's coefficients also read its cant angle, which a
    # controller may change between two calls with the same inputs
    return _computed_together(compute, variables, ("cL", "cD", "cQ"), reuse=False)


def _wind_plane_lift_to_body_coefficients(lift, drag, variables):
    """Convert a lift given against the total angle of attack, and the drag
    that goes with it, into the body-frame ``cN``, ``cY`` and ``cA``.

    Such a lift acts in the plane that holds the rocket's axis and the wind,
    perpendicular to the wind. With the total angle ``a`` it gives the normal
    force in that plane and the axial force::

        cN_plane = cL * cos(a) + cD * sin(a)
        cA = cD * cos(a) - cL * sin(a)

    and the normal force is split between the pitch and yaw planes as in
    :func:`_total_angle_to_body_coefficients`.

    Parameters
    ----------
    lift, drag : AeroCoefficient
        Lift and drag coefficients.
    variables : sequence of str
        Inputs of the returned Functions: the angle of attack and the sideslip
        angle first, then any other variable the two depend on.

    Returns
    -------
    tuple of Function
        ``cN``, ``cY`` and ``cA``, as Functions of ``variables``.
    """
    read_lift, read_drag = lift.evaluator(variables), drag.evaluator(variables)

    def compute(args):
        total, phi = total_angle_and_roll(args[0], args[1])
        c_l, c_d = read_lift(*args), read_drag(*args)
        sin, cos = math.sin(total), math.cos(total)
        normal = c_l * cos + c_d * sin
        return normal * math.sin(phi), -normal * math.cos(phi), c_d * cos - c_l * sin

    return _computed_together(compute, variables, ("cN", "cY", "cA"))


def _total_angle_to_body_coefficients(coefficient, variables, names):
    """Split a total-angle coefficient between the pitch and yaw planes.

    The coefficient acts in the plane that holds the rocket's axis and the wind,
    and is split along the crossflow, whose direction is the roll angle of the
    wind ``phi`` (see :func:`total_angle_and_roll`)::

        cN = coefficient * sin(phi)      (or cm, for a moment)
        cY = -coefficient * cos(phi)     (or cn)

    Parameters
    ----------
    coefficient : AeroCoefficient
        Normal force or pitch moment coefficient, against the total angle of
        attack.
    variables : sequence of str
        Inputs of the returned Functions: the angle of attack and the sideslip
        angle first, then any other variable the coefficient depends on.
    names : tuple of str
        Names of the two parts, such as ``("cN", "cY")``.

    Returns
    -------
    tuple of Function
        The pitch-plane and yaw-plane parts, as Functions of ``variables``.
    """
    evaluate = coefficient.evaluator(variables)

    def compute(args):
        _, phi = total_angle_and_roll(args[0], args[1])
        value = evaluate(*args)
        return value * math.sin(phi), -value * math.cos(phi)

    return _computed_together(compute, variables, names)
