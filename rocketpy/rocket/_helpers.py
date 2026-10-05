"""Functions that support the :class:`rocketpy.Rocket` class.

They are kept out of ``rocket.py`` so that the class stays simple. Most take
the rocket as their first argument: the whole-rocket sums, the linearity and
symmetry checks, the neutral point, and the rocket as one set of coefficients.
"""

import cmath
import math
import re
import warnings

import numpy as np

from rocketpy.mathutils._regular_grid import _RegularGrid
from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.aero_surface._barrowman_surface import _BarrowmanSurface
from rocketpy.rocket.aero_surface._helpers import _as_function
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket.aero_surface.linear_generic_surface import LinearGenericSurface

# The surfaces the stability analysis looks at


def stability_surfaces(rocket, phase=None):
    """The surfaces of ``rocket`` the stability analysis sums: those active
    during ``phase`` (``"power_on"`` or ``"power_off"``; the rocket's
    ``stability_phase`` when ``None``), as ``(surface, position)`` pairs.

    A surface is left out when its ``active_during`` names the other phase,
    or when it starts the flight switched off (``active=False``).
    """
    phase = rocket.stability_phase if phase is None else phase
    if phase not in ("power_on", "power_off"):
        raise ValueError(
            f"stability_phase must be 'power_on' or 'power_off', got {phase!r}."
        )
    other = "power_on" if phase == "power_off" else "power_off"
    return [
        (surface, position)
        for surface, position in rocket.aerodynamic_surfaces
        if getattr(surface, "active_during", "always") != other
        and getattr(surface, "active", True)
    ]


# Forces and moments of the whole rocket

# The atmosphere the whole-rocket sums are taken in: unit density and the
# vanishing-Reynolds limit
_UNIT_DENSITY = Function(1.0)
_NO_VISCOSITY = Function(1e30)


class _UniformValue:
    """A quantity that is the same at every altitude, read like a Function."""

    def __init__(self, value):
        self.value = value

    def get_value_opt(self, *_):
        return self.value


def summed_force_and_moment(
    rocket,
    alpha,
    beta,
    mach,
    omega,
    speed=1.0,
    reference_z=0.0,
    phase=None,
    reynolds=None,
):
    """Total body-frame force ``(R1, R2, R3)`` and moment ``(M1, M2, M3)``,
    summed over the aerodynamic surfaces of ``rocket`` active during the motor
    ``phase`` (``"power_on"`` or ``"power_off"``; the rocket's
    ``stability_phase`` when ``None``), at a flow state and set of body rates.

    Mirrors the per-surface computation the flight integrator performs: each
    surface is fed its own local stream velocity, which includes the
    ``omega x cp`` lever-arm term, so the sum captures the pitch and yaw damping
    the distributed surfaces produce through their fore-and-aft positions.
    Evaluated at unit air density; the result scales out of any dimensionless
    coefficient, and the chosen ``speed`` cancels from every coefficient built
    from it. ``omega`` is the body angular rate in rad/s. The stream is given at,
    the rates are about, and the moment is taken about the point ``reference_z``
    meters ahead of the center of dry mass along the body axis (the center of
    dry mass itself by default). ``reynolds`` is the Reynolds number of the
    rocket, based on its diameter and ``speed``; each surface then sees its own,
    scaled by its own length. ``None`` is the vanishing-Reynolds limit.
    """
    if reynolds is None or reynolds <= 0:
        viscosity = _NO_VISCOSITY
    else:
        # Unit density: Re = speed * diameter / viscosity
        viscosity = _UniformValue(speed * 2 * rocket.radius / reynolds)
    stream_direction = Vector([-math.tan(beta), -math.tan(alpha), -1.0])
    stream_at_reference = stream_direction / abs(stream_direction) * speed
    body_rates = Vector(list(omega))
    reference = Vector([0.0, 0.0, reference_z])
    speed_of_sound = speed / mach if mach > 0 else 1e30
    totals = np.zeros(6)
    for surface, _ in stability_surfaces(rocket, phase):
        cp = rocket.surfaces_cp_to_cdm[surface] - reference
        comp_stream = stream_at_reference - (body_rates ^ cp)
        comp_speed = abs(comp_stream)
        forces = surface.compute_forces_and_moments(
            comp_stream,
            comp_speed,
            comp_speed / speed_of_sound,
            1.0,
            cp,
            body_rates,
            _UNIT_DENSITY,
            viscosity,
            0.0,
        )
        totals += np.array(forces)
    return totals


# Linearity in the angle of attack

# Angles and Mach numbers at which the coefficients are read to tell whether a
# rocket is linear in the angle of attack. The angles cover what a rocket sees
# while its stability matters; beyond them it is tumbling.
_PROBE_ANGLES = np.radians([-15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0])
_PROBE_MACHS = (0.3, 0.9, 2.0)


def _is_linear_by_construction(surface):
    """Whether the forces of ``surface`` are linear in the flow angles whatever
    its data: the Barrowman surfaces, and a linear generic surface whose
    derivatives do not themselves depend on the angles."""
    if isinstance(surface, _BarrowmanSurface):
        return True
    if isinstance(surface, LinearGenericSurface):
        # pylint: disable-next=protected-access
        names = surface._get_default_coefficients()
        return not any(
            {"alpha", "beta"} & set(getattr(surface, name).depends_on) for name in names
        )
    return False


def _varies_with(coefficient, variable):
    """Whether a coefficient may change with ``variable``. A plain Function
    does not say what it depends on, so it is taken to."""
    if isinstance(coefficient, AeroCoefficient):
        return not coefficient.is_zero and variable in coefficient.depends_on
    return True


def uses_rate_coefficients(rocket):
    """Tell whether a surface of ``rocket`` has a coefficient that depends on a
    body rate (``pitch_rate``, ``yaw_rate`` or ``roll_rate``). The built-in
    surfaces are not counted: their damping comes from their position, not from
    a rate coefficient."""
    # pylint: disable=protected-access
    rates = ("pitch_rate", "yaw_rate", "roll_rate")
    for surface, _ in stability_surfaces(rocket):
        if isinstance(surface, _BarrowmanSurface):
            continue
        for name in surface._get_default_coefficients():
            coefficient = getattr(surface, name)
            # A linear surface gives its rate dependence as a derivative
            # (cm_q, cl_p, ...); a generic one through what a coefficient reads
            if isinstance(surface, LinearGenericSurface):
                if name[-2:] in ("_p", "_q", "_r") and not coefficient.is_zero:
                    return True
            elif any(_varies_with(coefficient, rate) for rate in rates):
                return True
    return False


def is_incidence_linear(rocket):
    """Tell whether the neutral point of ``rocket`` stays where it is as the
    angle of attack (or the sideslip angle) changes.

    It does for a rocket made of surfaces that are linear in the angle, such as
    the nose cone, the fins and the tail. It does not when a surface gives a
    force that is nonlinear in the angle and sits away from the rest of the lift,
    for example a body-lift term or a table with a stall, given as a
    :class:`rocketpy.GenericSurface`.

    Only the coefficients of the surfaces that can be nonlinear are read, at a
    few angles up to 15 degrees and at Mach 0.3, 0.9 and 2, so a rocket made of
    the built-in surfaces costs nothing. The neutral point between two
    neighboring angles is ``sum(x_i * slope_i) / sum(slope_i)``, with
    ``slope_i`` the change of the force of surface ``i`` between them; the
    rocket is linear when that point is the same between every pair of angles.

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket to check.

    Returns
    -------
    bool
        ``True`` when the neutral point moves by less than a micrometer.
    """
    # pylint: disable=protected-access
    surfaces = [surface for surface, _ in stability_surfaces(rocket)]
    nonlinear_candidates = [
        surface for surface in surfaces if not _is_linear_by_construction(surface)
    ]
    intervals = len(_PROBE_ANGLES) - 1

    def slopes(coefficient, surface, angle_index, mach):
        """Change of a coefficient per radian between neighboring angles."""
        if not _varies_with(coefficient, ("alpha", "beta")[angle_index]):
            return np.zeros(intervals)
        args = [0.0] * len(surface.independent_vars)
        args[2] = mach
        values = []
        for angle in _PROBE_ANGLES:
            args[angle_index] = angle
            values.append(coefficient(*args))
        return np.diff(values) / np.diff(_PROBE_ANGLES)

    planes = (
        ("cN", "cm", "cN_alpha", "cm_alpha", 0),
        ("cY", "cn", "cY_beta", "cn_beta", 1),
    )
    for force, moment, force_slope, moment_slope, angle_index in planes:
        angle_name = ("alpha", "beta")[angle_index]
        probed = [
            surface
            for surface in nonlinear_candidates
            if _varies_with(getattr(surface, force), angle_name)
            or _varies_with(getattr(surface, moment), angle_name)
        ]
        if not probed:
            continue
        # An axial force that changes with the angle, acting off the axis, also
        # moves the neutral point. It is rare, so it is not followed further.
        for surface in probed:
            point = surface.force_application_point
            if (point[0] or point[1]) and _varies_with(surface.cA, angle_name):
                return False
        any_mach = any(
            _varies_with(getattr(surface, name), "mach")
            for surface in probed
            for name in (force, moment)
        )
        for mach in _PROBE_MACHS if any_mach else _PROBE_MACHS[:1]:
            numerator = np.zeros(intervals)
            denominator = np.zeros(intervals)
            for surface, position in stability_surfaces(rocket):
                if surface in probed:
                    forces = slopes(getattr(surface, force), surface, angle_index, mach)
                    moments = slopes(
                        getattr(surface, moment), surface, angle_index, mach
                    )
                else:
                    args = [0.0] * len(surface.independent_vars)
                    args[2] = mach
                    forces = getattr(surface, force_slope)(*args)
                    moments = getattr(surface, moment_slope)(*args)
                weight = surface.reference_area / rocket.area
                application = (
                    surface._rotation_surface_to_body @ surface.force_application_point
                )[2]
                numerator += weight * (
                    forces * (position.z + rocket._csys * application)
                    + rocket._csys * surface.reference_length * moments
                )
                denominator += weight * forces
            lifting = np.abs(denominator) > 1e-12
            if not lifting.any():
                continue
            if not lifting.all():
                return False  # the lift appears or vanishes with the angle
            if np.ptp(numerator / denominator) > 1e-6:
                return False
    return True


# Axisymmetry

# Turning an axisymmetric rocket about its axis changes nothing, so the linear
# map from the crossflow (or from the lateral rates) to the lateral force and
# moment is the same in every plane. Written with the body-frame signs of
# ``cN``, ``cY``, ``cm`` and ``cn``, that is one relation per pair below: the
# yaw plane mirrors the pitch plane ...
_SYMMETRIC_PAIRS = (
    ("cN_alpha", "cY_beta", -1.0),
    ("cm_alpha", "cn_beta", -1.0),
    ("cN_q", "cY_r", 1.0),
    ("cm_q", "cn_r", 1.0),
    # ... and any coupling between the planes is the same seen from either.
    # Canted fins have such a coupling, through the axial part of their force,
    # and are still axisymmetric; a lone fin off the body axes is not.
    ("cN_beta", "cY_alpha", 1.0),
    ("cm_beta", "cn_alpha", 1.0),
    ("cN_r", "cY_q", -1.0),
    ("cm_r", "cn_q", -1.0),
)
# Nothing pushes sideways at zero angle, and the roll rate, which has no
# direction across the axis, gives no lateral force or moment. The roll terms
# themselves are left out: canted fins roll the rocket without breaking the
# symmetry of the pitch and yaw planes.
_ZERO_WHEN_SYMMETRIC = (
    "cN_0",
    "cY_0",
    "cm_0",
    "cn_0",
    "cN_p",
    "cY_p",
    "cm_p",
    "cn_p",
)
_SYMMETRY_MACHS = np.linspace(0.0, 3.0, 16)


def is_axisymmetric(rocket, machs=_SYMMETRY_MACHS, tolerance=1e-6):
    """Tell whether ``rocket`` behaves the same in every plane through its
    axis, as far as its linear coefficients go.

    The whole rocket's small-angle derivatives are read at ``machs`` (the
    surfaces of the rocket's ``stability_phase``, its drag left out since it is
    axial) and checked against what turning the rocket about its axis must
    leave unchanged: the yaw plane mirrors the pitch plane (``cY_beta =
    -cN_alpha``, ``cn_beta = -cm_alpha``, ``cY_r = cN_q``, ``cn_r = cm_q``),
    any coupling between the planes is the same seen from either (``cY_alpha
    = cN_beta``, ``cn_alpha = cm_beta``, ``cY_q = -cN_r``, ``cn_q = -cm_r``),
    and nothing acts sideways at zero angle or from the roll rate. Each
    relation must hold to ``tolerance`` times the largest lateral derivative
    at that Mach. Roll terms are not looked at.

    An asymmetry that only appears at large angles is not seen, since the
    derivatives are taken at zero angle.
    """
    rocket.evaluate_surfaces_cp_to_cdm()
    machs = np.asarray(machs, dtype=float)
    values = _linear_lumping(
        _surfaces_evaluator(rocket, rocket.stability_phase),
        (len(machs),),
        lambda index: (machs[index[0]], None),
        rates=True,
    )
    lateral = [name for name in values if name[:2] in ("cN", "cY", "cm", "cn")]
    scale = np.max(np.abs(np.array([values[name] for name in lateral])), axis=0)
    allowed = tolerance * scale + 1e-15
    for name in _ZERO_WHEN_SYMMETRIC:
        if np.any(np.abs(values[name]) > allowed):
            return False
    for pitch, yaw, sign in _SYMMETRIC_PAIRS:
        if np.any(np.abs(values[pitch] - sign * values[yaw]) > allowed):
            return False
    return True


# Neutral point and stability


def axial_force_slope(surface, cp, mach, plane="pitch", step=1e-4):
    """How fast one surface's axial force changes with the angle of attack
    (``plane="pitch"``) or the sideslip angle (``"yaw"``), at zero angles, unit
    speed and unit density, in newtons per radian.

    Taken by central difference of the surface's own
    ``compute_forces_and_moments``, as the flight computes it. A surface whose
    axial force is even in the angle, or absent, gives exactly zero.

    Parameters
    ----------
    surface : GenericSurface
        The surface.
    cp : Vector
        Its force application point relative to the rocket's center of dry
        mass, in the body frame.
    mach : float
        Free-stream Mach number.
    plane : str, optional
        ``"pitch"`` or ``"yaw"``. Default ``"pitch"``.
    step : float, optional
        Half-step, in radians, of the central difference. Default ``1e-4``.

    Returns
    -------
    float
        ``dR3/dalpha`` (pitch) or ``dR3/dbeta`` (yaw).
    """
    no_rotation = Vector([0.0, 0.0, 0.0])

    def axial_force(alpha, beta):
        stream = Vector([-math.tan(beta), -math.tan(alpha), -1.0])
        stream = stream / abs(stream)
        return surface.compute_forces_and_moments(
            stream, 1.0, mach, 1.0, cp, no_rotation, _UNIT_DENSITY, _NO_VISCOSITY, 0.0
        )[2]

    if plane == "yaw":
        return (axial_force(0.0, step) - axial_force(0.0, -step)) / (2 * step)
    return (axial_force(step, 0.0) - axial_force(-step, 0.0)) / (2 * step)


def moment_slopes_left_out(rocket, surface, position, lift_slope, side_slope):
    """Parts of one surface's pitch and yaw moment slopes that the rocket's
    aerodynamic center leaves out.

    The aerodynamic center averages the surfaces' own centers, each weighted
    by its force slope, so it reads a surface's whole moment as its force slope
    times an arm along the rocket's axis. Three parts of the moment do not fit
    that reading:

    - the moment of a surface with no force slope (a pure couple), whose
      weight is zero;
    - the axial force of a surface that changes with the angle, acting at a
      sideways offset from the center of dry mass (a canted fin);
    - the shift of an individual fin's position along the axis caused by its
      cant angle.

    Each part is exactly zero for a surface without it, so adding them leaves
    every other rocket's result unchanged.

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket the surface is on. Its ``surfaces_cp_to_cdm`` must be
        current.
    surface : GenericSurface
        The surface.
    position : Vector
        Where the surface was added to the rocket, in the user frame.
    lift_slope, side_slope : Function
        The surface's pitch force slope ``cN_alpha`` and signed yaw force slope
        ``-cY_beta``, as functions of Mach, without the reference-area factor.

    Returns
    -------
    tuple
        ``(pitch, yaw)``: each a :class:`Function` of Mach giving the missing
        moment slope in meters times the rocket's force coefficient (the units
        of the weighted sum behind the aerodynamic center), or ``None`` when the
        surface has nothing missing in that plane.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from rocketpy.rocket.aero_surface.fins.fin import Fin

    def _is_uncanted_fin(surface):
        # An uncanted fin pushes straight across the axis, with no axial force.
        # Read on each call, as a controller may change the cant.
        return isinstance(surface, Fin) and surface.cant_angle_rad == 0

    csys = rocket._csys
    ref_factor = surface.reference_area / rocket.area
    force_scale = 0.5 * rocket.area  # unit speed and density
    pitch_terms, yaw_terms = [], []

    if isinstance(surface, Fin):
        # The cant moves the fin's origin along the axis; read on each call so
        # a cant changed by a controller is followed
        def shift():
            return rocket._surface_origin(surface, position).z - position.z

        pitch_terms.append(lambda mach: ref_factor * lift_slope(mach) * shift())
        yaw_terms.append(lambda mach: ref_factor * side_slope(mach) * shift())

    if not isinstance(surface, _BarrowmanSurface):
        # Barrowman surfaces carry their moment in the application point, so
        # their moment slopes are zero
        couple = csys * ref_factor * surface.reference_length

        def pitch_couple(mach):
            args = (0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0)
            if surface.cN_alpha.get_value_opt(*args) != 0:
                return 0.0
            return couple * surface.cm_alpha.get_value_opt(*args)

        def yaw_couple(mach):
            args = (0.0, 0.0, mach, 0.0, 0.0, 0.0, 0.0)
            if surface.cY_beta.get_value_opt(*args) != 0:
                return 0.0
            return -couple * surface.cn_beta.get_value_opt(*args)

        pitch_terms.append(pitch_couple)
        yaw_terms.append(yaw_couple)

    # The axial force acts at the application point's sideways offset. The
    # offset is read on each call, as the flight reads it.
    offset = rocket.surfaces_cp_to_cdm[surface]
    if isinstance(surface, Fin) or offset[1] != 0:

        def pitch_axial(mach):
            if _is_uncanted_fin(surface):
                return 0.0
            cp = rocket.surfaces_cp_to_cdm[surface]
            return csys * cp[1] * axial_force_slope(surface, cp, mach) / force_scale

        pitch_terms.append(pitch_axial)
    if isinstance(surface, Fin) or offset[0] != 0:

        def yaw_axial(mach):
            if _is_uncanted_fin(surface):
                return 0.0
            cp = rocket.surfaces_cp_to_cdm[surface]
            slope = axial_force_slope(surface, cp, mach, "yaw")
            return csys * cp[0] * slope / force_scale

        yaw_terms.append(yaw_axial)

    def combine(terms, label):
        if not terms:
            return None
        return Function(lambda mach: sum(term(mach) for term in terms), "Mach", label)

    return (
        combine(pitch_terms, "Moment slope left out (m)"),
        combine(yaw_terms, "Moment slope left out - Yaw (m)"),
    )


def neutral_point_and_slope(rocket, alpha, beta, mach, plane="pitch", step=1e-4):
    """Local (tangent) neutral point and force-curve slope at a finite incidence.

    Generalizes the aerodynamic center to a non-zero angle of attack. The neutral
    point is the point about which the aerodynamic moment does not change for a
    *small* perturbation of the incidence angle around the given ``(alpha, beta)``
    state, i.e. the tangent of the moment-versus-force curve at that state. It is
    obtained by central-differencing the rocket's summed body-frame force and
    moment (about the center of dry mass) with respect to the plane's incidence
    angle, then forming ``x_cdm + csys * L_ref * (dCm/da) / (dCN/da)``.

    For a rocket whose surfaces are all linear in incidence (the built-in
    Barrowman surfaces) the result is independent of ``alpha``/``beta`` and equals
    :attr:`rocketpy.Rocket.aerodynamic_center`. It moves with incidence only when
    a surface's normal-force coefficient is nonlinear in the incidence angle (for
    example a Galejs ``sin**2(alpha)`` body-lift term added as a
    :class:`rocketpy.GenericSurface`).

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket to evaluate.
    alpha, beta : float
        Angle of attack and sideslip angle, in radians, defining the state the
        neutral point is taken about.
    mach : float
        Free-stream Mach number.
    plane : str, optional
        ``"pitch"`` (perturb ``alpha``, use the normal force and pitch moment) or
        ``"yaw"`` (perturb ``beta``, use the side force and yaw moment). Default
        ``"pitch"``.
    step : float, optional
        Half-step, in radians, of the central difference. Default ``1e-4``.

    Returns
    -------
    tuple of float
        ``(neutral_point, slope)``: the neutral-point axial position in the
        user-defined rocket frame, and the force-curve slope ``dCN/da`` (pitch)
        or ``dCY/db`` (yaw) at the state. When the slope vanishes (no lift at all)
        the neutral point falls back to the zero-incidence aerodynamic center.
    """
    rocket.evaluate_surfaces_cp_to_cdm()
    reference_length = 2 * rocket.radius
    dynamic_pressure_area = 0.5 * rocket.area  # unit speed, unit density
    dynamic_pressure_area_length = dynamic_pressure_area * reference_length

    def coefficients(a, b):
        r1, r2, _, m1, m2, _ = summed_force_and_moment(
            rocket, a, b, mach, (0.0, 0.0, 0.0)
        )
        if plane == "yaw":
            return r1 / dynamic_pressure_area, m2 / dynamic_pressure_area_length
        return -r2 / dynamic_pressure_area, m1 / dynamic_pressure_area_length

    if plane == "yaw":
        force_high, moment_high = coefficients(alpha, beta + step)
        force_low, moment_low = coefficients(alpha, beta - step)
    else:
        force_high, moment_high = coefficients(alpha + step, beta)
        force_low, moment_low = coefficients(alpha - step, beta)

    force_slope = (force_high - force_low) / (2 * step)
    moment_slope = (moment_high - moment_low) / (2 * step)
    if force_slope == 0:
        center = (
            rocket.aerodynamic_center_yaw
            if plane == "yaw"
            else rocket.aerodynamic_center
        )
        return center.get_value_opt(mach), 0.0
    neutral_point = rocket.center_of_dry_mass_position + (
        rocket._csys * reference_length * moment_slope / force_slope
    )
    return neutral_point, force_slope


def stability_margin_and_slope(rocket, alpha, beta, mach, time, plane="pitch"):
    """Stability margin (in calibers) and local restoring-force slope of one
    plane of ``rocket`` at a single flow state, the shared computation behind
    ``Rocket.stability_margin`` and the flight dynamic-stability oscillator.

    The state is the full pair ``(alpha, beta)``. When a surface is
    nonlinear in incidence the linearization of one plane depends on the
    angle in the other: an axisymmetric rocket read in the plane of the
    wind, at ``(theta, 0)``, gives the in-plane (tangent) values from the
    pitch plane and the across-the-wind (secant, ``Cm/CN``) values from the
    yaw plane, and they differ.

    For a rocket that is linear in incidence the neutral point is the
    zero-incidence :attr:`rocketpy.Rocket.aerodynamic_center`, both angles are
    ignored, and the fast analytic path is kept. Otherwise the neutral point
    and slope are found at the state by :func:`neutral_point_and_slope`.

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket to evaluate.
    alpha, beta : float
        Angle of attack and sideslip angle, in radians, of the state.
    mach : float
        Free-stream Mach number.
    time : float
        Flight time, in seconds, at which the center of mass is taken.
    plane : str, optional
        ``"pitch"`` or ``"yaw"``. Default ``"pitch"``.

    Returns
    -------
    tuple of float
        ``(margin, slope)``: the stability margin in calibers and the local
        slope of the restoring force, ``dCN/dalpha`` for pitch and
        ``-dCY/dbeta`` for yaw. Both are positive for a surface that pushes
        the rocket back, so the corrective moment ``q A slope margin d`` is
        positive for a stable rocket in either plane.
    """
    # pylint: disable=protected-access
    rocket._refresh_margins()
    if plane == "yaw":
        center = rocket.aerodynamic_center_yaw
        slope_curve = rocket.total_side_coeff_der
    else:
        center = rocket.aerodynamic_center
        slope_curve = rocket.total_lift_coeff_der

    if rocket._is_incidence_linear:
        neutral_point = center.get_value_opt(mach)
        slope = slope_curve.get_value_opt(mach)
    else:
        neutral_point, slope = neutral_point_and_slope(rocket, alpha, beta, mach, plane)
        if plane == "yaw":
            # A restoring side force has a negative ``dCY/dbeta``. The
            # slope is given the sign of the pitch plane, as
            # ``total_side_coeff_der`` is, so that it is positive for a
            # surface that pushes the rocket back.
            slope = -slope

    margin = (
        rocket._csys
        * (rocket.center_of_mass.get_value_opt(time) - neutral_point)
        / (2 * rocket.radius)
    )
    return margin, slope


def center_of_pressure_position(rocket, alpha, beta, mach, plane="pitch"):
    """Point where the resultant aerodynamic force of one plane acts at a flow
    state: the moment about the center of dry mass divided by the force,
    ``x_cdm + csys * L_ref * Cm / CN`` for pitch (``Cn / CY`` for yaw).

    Unlike :func:`neutral_point_and_slope`, which weights each surface by how
    fast its force grows with the angle, this weights each surface by its force
    itself. The two agree for a rocket linear in the angle with no force at zero
    angle.

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket to evaluate.
    alpha, beta : float
        Angle of attack and sideslip angle, in radians, of the flow state.
    mach : float
        Free-stream Mach number.
    plane : str, optional
        ``"pitch"`` (normal force and pitch moment) or ``"yaw"`` (side force
        and yaw moment). Default ``"pitch"``.

    Returns
    -------
    float
        Axial position of the center of pressure in the user-defined rocket
        frame, in meters. Where the plane's force vanishes (zero angle on a
        symmetric rocket) the point is undefined and the zero-incidence
        aerodynamic center, its limit for such a rocket, is returned.
    """
    rocket.evaluate_surfaces_cp_to_cdm()
    r1, r2, _, m1, m2, _ = summed_force_and_moment(
        rocket, alpha, beta, mach, (0.0, 0.0, 0.0)
    )
    force, moment = (r1, m2) if plane == "yaw" else (-r2, m1)
    # Unit speed and density: the force coefficient is force / (0.5 * area)
    if abs(force) <= _ZERO * 0.5 * rocket.area:
        center = (
            rocket.aerodynamic_center_yaw
            if plane == "yaw"
            else rocket.aerodynamic_center
        )
        return center.get_value_opt(mach)
    return rocket.center_of_dry_mass_position + rocket._csys * moment / force


def damping_derivative(
    rocket, alpha, beta, mach, plane="pitch", reference_z=0.0, step=1e-3
):
    """Aerodynamic damping of ``rocket`` in one plane at a flow state, per unit
    air density and airspeed.

    It is ``-dM/domega``, the change of the summed aerodynamic moment for a small
    body rate about the reference point, taken at unit density and speed by a
    central difference of :func:`summed_force_and_moment` around zero rate.
    Multiplied by ``rho * V`` it is the damping moment coefficient ``C2`` of the
    flight (N m s/rad). It includes whatever the surfaces feel from a rate: the
    lever arm of each surface with its force slope at ``(alpha, beta)``, and any
    coefficient that depends on a rate.

    Parameters
    ----------
    rocket : rocketpy.Rocket
        The rocket to evaluate.
    alpha, beta : float
        Angle of attack and sideslip angle, in radians, of the state the damping
        is taken about.
    mach : float
        Free-stream Mach number.
    plane : str, optional
        ``"pitch"`` (rate and moment about the body x axis) or ``"yaw"`` (about
        the body y axis). Default ``"pitch"``.
    reference_z : float, optional
        Position of the point the rocket rotates about, in meters ahead of the
        center of dry mass along the body axis. Default ``0.0``.
    step : float, optional
        Half-step of the central difference, in rad/s at unit speed. Default
        ``1e-3``.

    Returns
    -------
    float
        ``-dM/domega`` at unit density and speed, in m**4 (positive when the
        moment opposes the rate).
    """
    rocket.evaluate_surfaces_cp_to_cdm()
    axis = 1 if plane == "yaw" else 0

    def moment(rate):
        omega = [0.0, 0.0, 0.0]
        omega[axis] = rate
        return summed_force_and_moment(
            rocket, alpha, beta, mach, omega, reference_z=reference_z
        )[3 + axis]

    return -(moment(step) - moment(-step)) / (2 * step)


def aerodynamic_damping(rocket, alpha, beta, mach, time, plane="pitch"):
    """Aerodynamic damping of ``rocket`` in the ``"pitch"`` or ``"yaw"`` plane,
    per unit air density and airspeed, at a state ``(alpha, beta)`` (rad), Mach
    and time. Times ``rho * V`` it is the aerodynamic part of the flight's
    damping moment coefficient ``C2``.

    For a rocket linear in the angle and without rate coefficients it is
    ``0.5 A sum((A_i / A) slope_i arm_i**2)`` over the surfaces of the
    stability phase, with the zero-angle slopes and aerodynamic centers (a
    surface with a negative slope, such as a boat tail, takes damping
    away). Otherwise the summed moment is differenced with respect to the
    body rate at the state, about the center of mass (see
    :func:`damping_derivative`).
    """
    # pylint: disable=protected-access
    rocket._refresh_margins()
    center_of_mass = rocket.center_of_mass.get_value_opt(time)
    if rocket._is_incidence_linear and not rocket._uses_rate_coefficients:
        damping = 0.0
        for surface, position in stability_surfaces(rocket):
            if plane == "yaw":
                # A restoring side force has a negative dCY/dbeta
                slope = -surface.cY_beta.get_value_opt(0, 0, mach, 0, 0, 0, 0)
                center = surface.aerodynamic_center_yaw
            else:
                slope = surface.cN_alpha.get_value_opt(0, 0, mach, 0, 0, 0, 0)
                center = surface.aerodynamic_center
            arm = (
                position.z + rocket._csys * center.get_value_opt(mach) - center_of_mass
            )
            damping += surface.reference_area / rocket.area * slope * arm**2
        return 0.5 * rocket.area * damping
    # Body-axis position of the center of mass ahead of the center of dry mass
    reference_z = -rocket.com_to_cdm_function.get_value_opt(time)
    return damping_derivative(rocket, alpha, beta, mach, plane, reference_z)


# The attitude oscillator: I_L theta'' + C2 theta' + C1 theta = 0


def lateral_inertia_and_rate(rocket, inertia_about_cdm, time):
    """Lateral moment of inertia of ``rocket`` about its center of mass at
    ``time`` and its rate of change, from ``inertia_about_cdm`` (the rocket's
    ``I_11`` or ``I_22``, a Function of time about the center of dry mass).
    The rate is negative while propellant is consumed and enters the jet
    damping."""
    offset = rocket.com_to_cdm_function
    total_mass = rocket.total_mass.get_value_opt(time)
    mass_rate = rocket.total_mass_flow_rate.get_value_opt(time)
    cm_to_cdm = offset.get_value_opt(time)
    cm_to_cdm_rate = offset.differentiate_complex_step(time)
    # Parallel axis from the center of dry mass to the center of mass, and its
    # time derivative
    inertia = inertia_about_cdm.get_value_opt(time) - total_mass * cm_to_cdm**2
    rate = (
        inertia_about_cdm.differentiate_complex_step(time)
        - mass_rate * cm_to_cdm**2
        - 2 * total_mass * cm_to_cdm * cm_to_cdm_rate
    )
    return inertia, rate


def corrective_and_damping_moments(
    rocket,
    alpha,
    beta,
    mach,
    time,
    speed,
    density,
    dynamic_pressure,
    inertia_rate,
    plane="pitch",
):
    """Corrective moment coefficient ``C1`` (N m/rad) and damping moment
    coefficient ``C2`` (N m s/rad) of one plane of ``rocket`` at a flow state
    ``(alpha, beta)`` (rad), Mach number, time, airspeed (m/s), air density
    (kg/m^3) and dynamic pressure (Pa). ``inertia_rate`` is the rate of change
    of the lateral inertia (see :func:`lateral_inertia_and_rate`)."""
    margin, force_slope = stability_margin_and_slope(
        rocket, alpha, beta, mach, time, plane
    )
    damping_aero = (
        density * speed * aerodynamic_damping(rocket, alpha, beta, mach, time, plane)
    )

    # Corrective moment per radian: q A C_Nalpha (z_cm - z_np).
    corrective = (
        dynamic_pressure * rocket.area * force_slope * margin * (2 * rocket.radius)
    )

    # Jet (propulsive) damping. The exhaust leaves the nozzle moving sideways
    # with the rocket and carries angular momentum away,
    # |mdot| (z_nozzle - z_cm)^2 per unit rate; the lateral inertia lost with
    # the consumed propellant gives part of it back, dI/dt (negative).
    # Together they are Thomson's mdot (l_n^2 - l_p^2).
    center_of_mass = rocket.center_of_mass.get_value_opt(time)
    damping_jet = (
        abs(rocket.motor.total_mass_flow_rate.get_value_opt(time))
        * (rocket.nozzle_position - center_of_mass) ** 2
        + inertia_rate
    )
    return corrective, damping_aero + damping_jet


def disturbance_response(
    corrective, damping, inertia, disturbance, duration=None, samples=500
):
    """Angle of a rocket after a sudden disturbance, as a Function of the time
    since it: the solution of ``I_L theta'' + C2 theta' + C1 theta = 0`` that
    starts at the angle ``disturbance`` with no rotation rate, with the three
    coefficients held fixed.

    The solution is written from the two roots of ``I_L s^2 + C2 s + C1``, so
    it also covers a rocket with no restoring moment (the angle then grows).
    ``duration`` defaults to the time the response takes to settle: four times
    the slowest decay time, kept between 3 and 15 oscillation periods.
    """
    if inertia <= 0:
        raise ValueError(
            "The rocket has no lateral inertia (a point mass rocket), so it "
            "has no attitude to disturb."
        )
    discriminant = complex(damping**2 - 4 * inertia * corrective)
    root_1 = (-damping + cmath.sqrt(discriminant)) / (2 * inertia)
    root_2 = (-damping - cmath.sqrt(discriminant)) / (2 * inertia)
    decay = max(root_1.real, root_2.real)  # the slowest root; negative: settles
    frequency = abs(root_1.imag)

    if duration is None:
        if decay < 0:
            duration = 4 / -decay
            if frequency > 0:
                period = 2 * math.pi / frequency
                duration = min(max(duration, 3 * period), 15 * period)
        elif decay > 0:
            duration = math.log(10) / decay  # until the angle grows tenfold
        else:
            duration = 10 * 2 * math.pi / frequency if frequency > 0 else 5.0

    time = np.linspace(0, duration, samples)
    if abs(root_1 - root_2) > 1e-12 * max(abs(root_1), abs(root_2), 1e-300):
        angle = (root_2 * np.exp(root_1 * time) - root_1 * np.exp(root_2 * time)) / (
            root_2 - root_1
        )
    else:  # a repeated root: critically damped
        angle = np.exp(root_1 * time) * (1 - root_1 * time)
    angle = disturbance * np.real(angle)

    if corrective > 0:
        natural_frequency = math.sqrt(corrective / inertia)
        damping_ratio = damping / (2 * math.sqrt(corrective * inertia))
        title = (
            f"Response to a {disturbance:g}° disturbance (natural frequency "
            f"{natural_frequency:.2f} rad/s, damping ratio {damping_ratio:.3f})"
        )
    else:
        title = f"Response to a {disturbance:g}° disturbance (no restoring moment)"
    return Function(
        np.column_stack((time, angle)),
        "Time after the disturbance (s)",
        "Angle (°)",
        interpolation="linear",
        extrapolation="constant",
        title=title,
    )


# The rocket as one set of coefficients (``to_coefficients`` / ``to_surface``)

_LUMPED_COEFFICIENTS = ("cN", "cY", "cA", "cm", "cn", "cl")
# Derivative suffix -> the state variable it perturbs
_ANGLE_TERMS = {"alpha": "alpha", "beta": "beta"}
_RATE_TERMS = {"p": "roll", "q": "pitch", "r": "yaw"}
_RATE_NAMES = {"p": "roll_rate", "q": "pitch_rate", "r": "yaw_rate"}
_SLOPE_STEP = 1e-5
_ZERO = 1e-12


def _surfaces_evaluator(rocket, phase):
    """A function giving the six body-frame coefficients ``(cN, cY, cA, cm,
    cn, cl)`` of the surfaces active during ``phase`` at a state: Mach
    number, flow angles, reduced rates and Reynolds number (of the rocket,
    based on its diameter; ``None`` for the vanishing-Reynolds limit)."""
    reference_length = 2 * rocket.radius
    dynamic_pressure_area = 0.5 * rocket.area  # unit speed, unit density
    dynamic_pressure_area_length = dynamic_pressure_area * reference_length
    rate_factor = 2.0 / reference_length  # reduced rate -> rate at unit speed

    def coefficients_at(
        mach, alpha=0.0, beta=0.0, pitch=0.0, yaw=0.0, roll=0.0, reynolds=None
    ):
        omega = (pitch * rate_factor, yaw * rate_factor, roll * rate_factor)
        r1, r2, r3, m1, m2, m3 = summed_force_and_moment(
            rocket, alpha, beta, mach, omega, phase=phase, reynolds=reynolds
        )
        return np.array(
            [
                -r2 / dynamic_pressure_area,
                r1 / dynamic_pressure_area,
                -r3 / dynamic_pressure_area,
                m1 / dynamic_pressure_area_length,
                m2 / dynamic_pressure_area_length,
                m3 / dynamic_pressure_area_length,
            ]
        )

    return coefficients_at


def _drag_evaluator(rocket, phase):
    """Like :func:`_surfaces_evaluator`, for the rocket's own drag coefficient
    of ``phase``, which is an axial force."""
    drag = getattr(rocket, f"{phase}_drag_7d").get_value_opt

    def coefficients_at(
        mach, alpha=0.0, beta=0.0, pitch=0.0, yaw=0.0, roll=0.0, reynolds=None
    ):
        axial = drag(alpha, beta, mach, reynolds or 0.0, pitch, yaw, roll)
        return np.array([0.0, 0.0, axial, 0.0, 0.0, 0.0])

    return coefficients_at


def _slopes_at(evaluate, mach, reynolds, terms, **state):
    """The slope of the six coefficients with each variable in ``terms``
    (suffix to state name) at ``state`` (the flow angles; the rates are zero),
    as one array of six per suffix."""
    slopes = {}
    for suffix, variable in terms.items():
        base = state.get(variable, 0.0)
        high = evaluate(
            mach, reynolds=reynolds, **{**state, variable: base + _SLOPE_STEP}
        )
        low = evaluate(
            mach, reynolds=reynolds, **{**state, variable: base - _SLOPE_STEP}
        )
        slopes[suffix] = (high - low) / (2 * _SLOPE_STEP)
    return slopes


# The lumping runs over a grid of conditions: the Mach numbers and, when asked
# for, the Reynolds numbers and the values of each control. ``shape`` is the
# shape of that grid and ``condition(index)`` moves the rocket's controls to
# the condition at ``index`` and returns its ``(mach, reynolds)``.


def _linear_lumping(evaluate, shape, condition, rates):
    """The derivatives of the linear model on the grid of conditions: the
    value at zero (``_0``), the angle slopes and, when ``rates``, the rate
    slopes."""
    terms = {**_ANGLE_TERMS, **(_RATE_TERMS if rates else {})}
    values = {
        f"{coefficient}_{suffix}": np.empty(shape)
        for coefficient in _LUMPED_COEFFICIENTS
        for suffix in ("0", *terms)
    }
    for index in np.ndindex(*shape):
        mach, reynolds = condition(index)
        at_zero = evaluate(mach, reynolds=reynolds)
        for coefficient, value in zip(_LUMPED_COEFFICIENTS, at_zero):
            values[f"{coefficient}_0"][index] = value
        for suffix, slopes in _slopes_at(evaluate, mach, reynolds, terms).items():
            for coefficient, slope in zip(_LUMPED_COEFFICIENTS, slopes):
                values[f"{coefficient}_{suffix}"][index] = slope
    return values


def _angle_axes(angles, axisymmetric):
    """The angle axes of the table model and their names. An axisymmetric
    rocket is swept in one plane, over the total angle of attack."""
    if axisymmetric:
        return [np.unique(np.abs(angles))], ["alpha_total"]
    return [angles, angles], ["alpha", "beta"]


def _table_lumping(evaluate, angle_axes, shape, condition, rates):
    """The six coefficients on the grid of the angles and the conditions, read
    at zero rates, keyed by name, plus the rate slopes when ``rates``: on the
    grid of conditions at zero angle, or, for ``rates="at_each_angle"``, on
    the whole grid."""
    angle_shape = tuple(len(axis) for axis in angle_axes)
    at_each_angle = rates == "at_each_angle"
    values = np.empty(angle_shape + tuple(shape) + (6,))
    rate_shape = values.shape if at_each_angle else tuple(shape) + (6,)
    rate_values = {suffix: np.empty(rate_shape) for suffix in _RATE_TERMS}
    for index in np.ndindex(*values.shape[:-1]):
        mach, reynolds = condition(index[len(angle_shape) :])
        angles_at = (axis[i] for axis, i in zip(angle_axes, index))
        state = dict(zip(("alpha", "beta"), angles_at))
        values[index] = evaluate(mach, reynolds=reynolds, **state)
        if at_each_angle:
            slopes = _slopes_at(evaluate, mach, reynolds, _RATE_TERMS, **state)
            for suffix, slope in slopes.items():
                rate_values[suffix][index] = slope
    if rates and not at_each_angle:
        for index in np.ndindex(*shape):
            mach, reynolds = condition(index)
            slopes = _slopes_at(evaluate, mach, reynolds, _RATE_TERMS)
            for suffix, slope in slopes.items():
                rate_values[suffix][index] = slope
    tables = {
        coefficient: values[..., i]
        for i, coefficient in enumerate(_LUMPED_COEFFICIENTS)
    }
    rate_tables = {}
    if rates:
        rate_tables = {
            f"{coefficient}_{suffix}": rate_values[suffix][..., i]
            for i, coefficient in enumerate(_LUMPED_COEFFICIENTS)
            for suffix in _RATE_TERMS
        }
    return tables, rate_tables


def _mach_curve(machs, values, name):
    """A Function of Mach through the tabulated ``values``: a constant for a
    single Mach number, linear between two, Akima beyond."""
    if len(machs) == 1:
        return Function(float(values[0]), "Mach", name)
    return Function(
        np.column_stack([machs, values]),
        "Mach",
        name,
        interpolation="akima" if len(machs) > 2 else "linear",
        extrapolation="constant",
    )


def _grid_table(axes, names, values, name):
    """A Function interpolating linearly on the regular grid ``axes`` (one
    array per variable, named after ``names``), holding its edge values
    outside it."""
    return Function(
        _RegularGrid.points_from_axes(axes, values),
        list(names),
        [name],
        interpolation="linear",
        extrapolation="constant",
    )


def _condition_curve(axes, names, values, name):
    """A Function over the grid of conditions: a curve over Mach when Mach is
    the only one, a table otherwise."""
    if len(axes) == 1:
        return _mach_curve(axes[0], values, name)
    return _grid_table(axes, names, values, name)


def _control_axes(rocket, controls):
    """Match the ``controls`` asked for (name to values) with the controls of
    the rocket's surfaces.

    Returns one ``(name, targets, values)`` per control axis of the lumped
    coefficients, ``targets`` being the ``(surface, control)`` pairs it moves.
    A control name used by one surface keeps its name. A name used by several
    surfaces gives one axis per surface, named ``<surface>_<control>``; asking
    for the shared name sweeps each of them over the same values, with a
    warning.
    """
    owners = {}
    for surface, _ in rocket.aerodynamic_surfaces:
        for control in getattr(surface, "control_variables", ()):
            owners.setdefault(control, []).append(surface)

    def slug(text):
        return re.sub(r"\W+", "_", str(text)).strip("_").lower()

    # The name of each control once lumped
    lumped = {}
    for control, surfaces in owners.items():
        if len(surfaces) == 1:
            lumped[control] = [(surfaces[0], control)]
            continue
        slugs = [slug(surface.name) for surface in surfaces]
        for i, (surface, prefix) in enumerate(zip(surfaces, slugs)):
            if slugs.count(prefix) > 1:
                prefix = f"{prefix}_{i + 1}"
            lumped[f"{prefix}_{control}"] = [(surface, control)]

    axes = []
    for name, values in controls.items():
        values = np.unique(np.asarray(values, dtype=float))
        if len(values) < 2:
            raise ValueError(
                f"Give at least two different values for the control '{name}'."
            )
        if name in lumped:
            axes.append((name, lumped[name], values))
        elif name in owners:
            separate = [
                key
                for key, targets in lumped.items()
                if targets[0][1] == name and targets[0][0] in owners[name]
            ]
            warnings.warn(
                f"The control '{name}' is used by {len(separate)} surfaces. Each "
                "is kept as a separate control of the result, swept over the "
                f"same values: {', '.join(separate)}. Give these names in "
                "`controls` to choose the values of each.",
                UserWarning,
                stacklevel=4,
            )
            axes.extend((key, lumped[key], values) for key in separate)
        else:
            raise ValueError(
                f"The rocket has no control named '{name}'. Its controls are: "
                f"{sorted(lumped) or 'none'}."
            )
    return axes


def full_body_coefficients(  # pylint: disable=too-many-locals
    rocket,
    machs=None,
    force_convention="body",
    model="linear",
    angles=None,
    rates=True,
    reynolds=None,
    controls=None,
):
    """Compute the rocket's lumped coefficient set, split by motor phase. Backs
    :meth:`rocketpy.Rocket.to_coefficients`; see that method for the full
    description of the two models and their limitations.
    """
    # pylint: disable=too-many-statements
    if force_convention not in ("body", "wind"):
        raise ValueError(
            f"force_convention must be 'body' or 'wind', got {force_convention!r}."
        )
    if model not in ("linear", "table"):
        raise ValueError(f"model must be 'linear' or 'table', got {model!r}.")
    if model == "table" and force_convention == "wind":
        raise ValueError(
            "The table model gives the body-frame coefficients only; use "
            "force_convention='body'."
        )
    if not (rates is True or rates is False or rates == "at_each_angle"):
        raise ValueError(
            f"rates must be True, False or 'at_each_angle', got {rates!r}."
        )
    if rates == "at_each_angle" and model != "table":
        raise ValueError("rates='at_each_angle' needs model='table'.")
    if machs is None:
        machs = np.arange(0.0, 3.01, 0.02 if model == "linear" else 0.05)
    machs = np.asarray(machs, dtype=float)
    if angles is None:
        angles = np.radians(np.arange(-30.0, 31.0, 2.0))
    angles = np.asarray(angles, dtype=float)

    # The conditions the coefficients are read at, besides the angles
    fixed_reynolds = None
    condition_axes, condition_names = [machs], ["mach"]
    if reynolds is not None:
        reynolds = np.unique(np.atleast_1d(np.asarray(reynolds, dtype=float)))
        if len(reynolds) == 1:
            fixed_reynolds = float(reynolds[0])
        else:
            condition_axes.append(reynolds)
            condition_names.append("reynolds")
    sweeps_reynolds = len(condition_axes) == 2
    control_axes = _control_axes(rocket, controls or {})
    for name, _, values in control_axes:
        condition_axes.append(values)
        condition_names.append(name)
    if len(machs) < 2 and (model == "table" or len(condition_axes) > 1):
        raise ValueError(
            "The coefficients are interpolated over Mach, so `machs` needs at "
            "least two values."
        )
    shape = tuple(len(axis) for axis in condition_axes)

    def condition(index):
        mach_index, *others = index
        reynolds_at = fixed_reynolds
        if sweeps_reynolds:
            reynolds_at = reynolds[others.pop(0)]
        for (_, targets, values), i in zip(control_axes, others):
            for surface, control in targets:
                surface.set_control(control, values[i])
        return machs[mach_index], reynolds_at

    # Make sure each surface's center-of-pressure offset is current.
    rocket.evaluate_surfaces_cp_to_cdm()
    # A deflected control or a rate at an angle breaks the single-plane sweep
    axisymmetric = (
        model == "table"
        and rocket.is_axisymmetric
        and not control_axes
        and rates != "at_each_angle"
    )
    angle_axes, angle_names = _angle_axes(angles, axisymmetric)

    def lump(evaluate):
        if model == "linear":
            return _linear_lumping(evaluate, shape, condition, rates), {}
        return _table_lumping(evaluate, angle_axes, shape, condition, rates)

    def package(values, rate_values):
        if model == "linear":
            if force_convention == "wind":
                values = body_derivatives_to_wind(values)
            return {
                name: _condition_curve(condition_axes, condition_names, array, name)
                for name, array in values.items()
                if np.any(np.abs(array) > _ZERO)
            }
        axes, names = angle_axes + condition_axes, angle_names + condition_names
        kept = ("cN", "cA", "cm", "cl") if axisymmetric else _LUMPED_COEFFICIENTS
        packaged = {
            name: _grid_table(axes, names, array, name)
            for name, array in values.items()
            if name in kept
        }
        for name, array in rate_values.items():
            if not np.any(np.abs(array) > _ZERO):
                continue
            if rates == "at_each_angle":
                packaged[name] = _grid_table(axes, names, array, name)
            else:
                packaged[name] = _condition_curve(
                    condition_axes, condition_names, array, name
                )
        return packaged

    # The sweep moves the controls of the rocket's surfaces; put them back
    moved = {
        (surface, control): surface.control_state[control]
        for _, targets, _ in control_axes
        for surface, control in targets
    }
    # The rocket's own drag coefficient differs by phase; the surfaces do so
    # only when one is gated to a phase, so their sweep is shared otherwise
    by_surfaces = {}
    result = {}
    try:
        for phase in ("power_off", "power_on"):
            surfaces = tuple(
                surface for surface, _ in stability_surfaces(rocket, phase)
            )
            if surfaces not in by_surfaces:
                by_surfaces[surfaces] = lump(_surfaces_evaluator(rocket, phase))
            drag = lump(_drag_evaluator(rocket, phase))
            values, rate_values = (
                {name: array + of_drag[name] for name, array in of_surfaces.items()}
                for of_surfaces, of_drag in zip(by_surfaces[surfaces], drag)
            )
            result[phase] = package(values, rate_values)
    finally:
        for (surface, control), value in moved.items():
            surface.control_state[control] = value
    return result


_FLOW_INPUTS = (
    "alpha",
    "beta",
    "alpha_total",
    "phi",
    "mach",
    "reynolds",
    *_RATE_NAMES.values(),
)


def _input_names(function):
    """The names of the inputs of a lumped coefficient, in order."""
    return [str(name).lower() for name in function.__inputs__]


def lumped_control_names(coefficients):
    """The control inputs of a coefficient set of the table model, in order."""
    return [
        name for name in _input_names(coefficients["cN"]) if name not in _FLOW_INPUTS
    ]


def lumped_surface_coefficients(coefficients):
    """The coefficient input of the surface that carries one phase of
    :func:`full_body_coefficients` with ``model="table"``: each of the six
    coefficients is its table plus its rate terms times the reduced rates.
    Every coefficient takes the inputs of its table and of its rate terms
    (angles, Mach and, when present, Reynolds number and controls), then the
    reduced rates.

    An axisymmetric set (tables over the total angle of attack) is split
    between the pitch and yaw planes with the roll angle of the wind, as the
    generic surface does against the total angle: ``cN = table * sin(phi)``,
    ``cY = -table * cos(phi)``, and likewise ``cm``/``cn``.
    """
    axisymmetric = "cY" not in coefficients
    # Along-the-crossflow coefficients of an axisymmetric set read the total
    # angle and the roll angle of the wind
    source_of = {"cY": "cN", "cn": "cm"} if axisymmetric else {}
    direction = {}
    if axisymmetric:
        direction = {
            "cN": math.sin,
            "cm": math.sin,
            "cY": lambda phi: -math.cos(phi),
            "cn": lambda phi: -math.cos(phi),
        }

    def build(coefficient):
        table = coefficients[source_of.get(coefficient, coefficient)]
        turn = direction.get(coefficient)
        terms = [
            (_RATE_NAMES[suffix], coefficients[f"{coefficient}_{suffix}"])
            for suffix in _RATE_TERMS
            if f"{coefficient}_{suffix}" in coefficients
        ]
        if turn is None and not terms:
            return table
        table_names = _input_names(table)
        names = list(table_names)
        if turn is not None:
            names.insert(1, "phi")
        for _, term in terms:
            names.extend(name for name in _input_names(term) if name not in names)
        names.extend(rate for rate, _ in terms)
        read_table = table.get_value_opt
        table_at = [names.index(name) for name in table_names]
        phi_at = names.index("phi") if turn is not None else None
        terms_at = [
            (
                term.get_value_opt,
                [names.index(name) for name in _input_names(term)],
                names.index(rate),
            )
            for rate, term in terms
        ]

        def value(*args):
            total = read_table(*[args[i] for i in table_at])
            if turn is not None:
                total *= turn(args[phi_at])
            for read_term, term_at, rate_at in terms_at:
                total += read_term(*[args[i] for i in term_at]) * args[rate_at]
            return total

        return _as_function(value, names, coefficient)

    return {coefficient: build(coefficient) for coefficient in _LUMPED_COEFFICIENTS}


def body_derivatives_to_wind(body):
    """Express a body-frame derivative set (``cN_*``/``cY_*``/``cA_*``) in the
    wind-frame names (``cL_*``/``cQ_*``/``cD_*``); the moment derivatives are
    frame-shared. Four cross terms fold the zero-angle forces in at incidence,
    ``cL_alpha = cN_alpha - cA_0``, ``cQ_beta = cY_beta + cA_0``,
    ``cD_alpha = cA_alpha + cN_0`` and ``cD_beta = cA_beta - cY_0`` -- the
    linear inverse of the wind-to-body rotation :class:`LinearGenericSurface`
    applies to a wind-frame input, so feeding the result back with
    ``force_convention="wind"`` recovers the same body-frame surface. Operates on
    the tabulated derivative values (arrays over the Mach grid).
    """
    rename = {"cN": "cL", "cY": "cQ", "cA": "cD"}
    wind = {}
    for key, value in body.items():
        prefix, sep, suffix = key.partition("_")
        wind[f"{rename.get(prefix, prefix)}{sep}{suffix}"] = value
    cross_terms = (
        ("cL_alpha", "cN_alpha", "cA_0", -1.0),
        ("cQ_beta", "cY_beta", "cA_0", 1.0),
        ("cD_alpha", "cA_alpha", "cN_0", 1.0),
        ("cD_beta", "cA_beta", "cY_0", -1.0),
    )
    for name, first, second, sign in cross_terms:
        if first in body or second in body:
            wind[name] = body.get(first, 0.0) + sign * body.get(second, 0.0)
    return wind
