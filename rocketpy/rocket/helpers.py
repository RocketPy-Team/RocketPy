"""Helper functions backing some :class:`rocketpy.Rocket` methods.

These carry the heavier computations behind the rocket's full-body aerodynamic
reduction (``to_coefficients`` / ``to_surface``), kept out of ``rocket.py`` so
that module stays focused on the rocket's public interface. Each function takes
the rocket it operates on as its first argument.
"""

import math

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient


def zero_drag(rocket, which):
    """Set one of the rocket's built-in drag curves to zero. ``which`` is
    ``"power_on"`` or ``"power_off"``. Rebuilds the ``*_drag_7d``,
    ``*_drag_by_mach`` and the public ``*_drag`` alias to match, mirroring how
    they are built in ``Rocket.__init__``.
    """
    label = "Power On" if which == "power_on" else "Power Off"
    setattr(
        rocket,
        f"{which}_drag_7d",
        AeroCoefficient(
            0,
            name=f"Drag Coefficient with {label}",
            extrapolation="constant",
            single_var="mach",
        ),
    )
    by_mach = Function(
        lambda mach: 0.0,
        inputs="Mach Number",
        outputs=f"Drag Coefficient with {label}",
        interpolation="linear",
        extrapolation="constant",
    )
    setattr(rocket, f"{which}_drag_by_mach", by_mach)
    setattr(rocket, f"{which}_drag", by_mach)
    setattr(rocket, f"_{which}_drag_input", 0)


def summed_force_and_moment(rocket, alpha, beta, mach, omega, speed=1.0):
    """Total body-frame force ``(R1, R2, R3)`` and moment ``(M1, M2, M3)`` about
    the center of dry mass, summed over every aerodynamic surface of ``rocket``
    at a flow state and set of body rates.

    Mirrors the per-surface computation the flight integrator performs: each
    surface is fed its own local stream velocity, which includes the
    ``omega x cp`` lever-arm term, so the sum captures the pitch and yaw damping
    the distributed surfaces produce through their fore-and-aft positions.
    Evaluated at unit air density; the result scales out of any dimensionless
    coefficient, and the chosen ``speed`` cancels from every coefficient built
    from it. ``omega`` is the body angular rate in rad/s.
    """
    stream_direction = Vector([-math.tan(beta), -math.tan(alpha), -1.0])
    stream_at_cdm = stream_direction / abs(stream_direction) * speed
    body_rates = Vector(list(omega))
    density = Function(1.0)
    dynamic_viscosity = Function(1e30)  # vanishing-Reynolds limit
    speed_of_sound = speed / mach if mach > 0 else 1e30
    totals = np.zeros(6)
    for surface, _ in rocket.aerodynamic_surfaces:
        cp = rocket.surfaces_cp_to_cdm[surface]
        comp_stream = stream_at_cdm - (body_rates ^ cp)
        comp_speed = abs(comp_stream)
        forces = surface.compute_forces_and_moments(
            comp_stream,
            comp_speed,
            comp_speed / speed_of_sound,
            1.0,
            cp,
            body_rates,
            density,
            dynamic_viscosity,
            0.0,
        )
        totals += np.array(forces)
    return totals


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


def full_body_coefficients(rocket, machs=None, force_convention="body"):
    """Compute the rocket's lumped stability-derivative set, split by motor
    phase. Backs :meth:`rocketpy.Rocket.to_coefficients`; see that method for the
    full description of the returned coefficient sets and their limitations.
    """
    if force_convention not in ("body", "wind"):
        raise ValueError(
            f"force_convention must be 'body' or 'wind', got {force_convention!r}."
        )
    if machs is None:
        machs = np.arange(0.0, 3.01, 0.02)
    machs = np.asarray(machs, dtype=float)
    # Make sure each surface's center-of-pressure offset is current.
    rocket.evaluate_surfaces_cp_to_cdm()

    reference_length = 2 * rocket.radius
    dynamic_pressure_area = 0.5 * rocket.area  # unit speed, unit density
    dynamic_pressure_area_length = dynamic_pressure_area * reference_length

    def coefficients_at(alpha, beta, red_pitch, red_yaw, red_roll, mach):
        # reduced rate -> body rate at unit speed: omega = rate * 2 V / L_ref
        rate_factor = 2.0 / reference_length
        omega = (
            red_pitch * rate_factor,
            red_yaw * rate_factor,
            red_roll * rate_factor,
        )
        r1, r2, _, m1, m2, m3 = summed_force_and_moment(
            rocket, alpha, beta, mach, omega
        )
        return {
            "cN": -r2 / dynamic_pressure_area,
            "cY": r1 / dynamic_pressure_area,
            "cm": m1 / dynamic_pressure_area_length,
            "cn": m2 / dynamic_pressure_area_length,
            "cl": m3 / dynamic_pressure_area_length,
        }

    step = 1e-5

    def slope(field, coeff):
        values = []
        for mach in machs:
            state = {
                "alpha": 0.0,
                "beta": 0.0,
                "red_pitch": 0.0,
                "red_yaw": 0.0,
                "red_roll": 0.0,
            }
            high = coefficients_at(mach=mach, **{**state, field: step})
            low = coefficients_at(mach=mach, **{**state, field: -step})
            values.append((high[coeff] - low[coeff]) / (2 * step))
        return np.array(values)

    # Motor-independent derivative values on the Mach grid, in the body frame.
    # The rocket's shape does not change with the motor, so these are shared by
    # both phases; only the drag below differs.
    derivatives = {
        "cN_alpha": slope("alpha", "cN"),
        "cm_alpha": slope("alpha", "cm"),
        "cN_q": slope("red_pitch", "cN"),
        "cm_q": slope("red_pitch", "cm"),
        "cY_beta": slope("beta", "cY"),
        "cn_beta": slope("beta", "cn"),
        "cY_r": slope("red_yaw", "cY"),
        "cn_r": slope("red_yaw", "cn"),
        "cl_p": slope("red_roll", "cl"),
    }
    # The drag is a Mach curve at zero incidence (the axial coefficient's
    # constant term), and it is the one term that differs by motor phase.
    drag_by_phase = {
        "power_off": rocket.power_off_drag_by_mach,
        "power_on": rocket.power_on_drag_by_mach,
    }

    result = {}
    for phase, drag_curve in drag_by_phase.items():
        values = {
            **derivatives,
            "cA_0": np.array([drag_curve.get_value_opt(m) for m in machs]),
        }
        if force_convention == "wind":
            values = body_derivatives_to_wind(values)
        result[phase] = {
            coeff_name: Function(
                np.column_stack([machs, curve]),
                "Mach",
                coeff_name,
                interpolation="akima",
                extrapolation="constant",
            )
            for coeff_name, curve in values.items()
        }
    return result


def body_derivatives_to_wind(body):
    """Express a body-frame derivative set (``cN_*``/``cY_*``/``cA_*``) in the
    wind-frame names (``cL_*``/``cQ_*``/``cD_*``); the moment derivatives are
    frame-shared. Two cross terms fold the axial force in at incidence,
    ``cL_alpha = cN_alpha - cA_0`` and ``cQ_beta = cY_beta + cA_0`` -- the linear
    inverse of the wind-to-body rotation :class:`LinearGenericSurface` applies to
    a wind-frame input, so feeding the result back with
    ``force_convention="wind"`` recovers the same body-frame surface. Operates on
    the tabulated derivative values (arrays over the Mach grid).
    """
    drag = body.get("cA_0", 0.0)
    rename = {"cN": "cL", "cY": "cQ", "cA": "cD"}
    wind = {}
    for key, value in body.items():
        prefix, sep, suffix = key.partition("_")
        wind[f"{rename.get(prefix, prefix)}{sep}{suffix}"] = value
    if "cN_alpha" in body:
        wind["cL_alpha"] = body["cN_alpha"] - drag
    if "cY_beta" in body:
        wind["cQ_beta"] = body["cY_beta"] + drag
    return wind
