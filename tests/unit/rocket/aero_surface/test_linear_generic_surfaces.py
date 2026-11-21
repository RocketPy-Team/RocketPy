import math

import pytest

from rocketpy import Function, GenericSurface, LinearGenericSurface
from rocketpy.mathutils import Vector

REFERENCE_AREA = 1
REFERENCE_LENGTH = 1

# (alpha, beta, mach, reynolds, pitch_rate, yaw_rate, roll_rate)
_ARGS = (0.0, 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "coefficients",
    [
        "cN_0",
        {"invalid_name": 0},
        {"cN_0": "inexistent_file.csv"},
        {"cN_0": Function(lambda x1, x2, x3, x4, x5, x6: 0)},
        {"cN_0": {}},
    ],
)
def test_invalid_initialization(coefficients):
    """Checks if linear generic surface raises errors in initialization
    when coefficient argument is invalid"""

    with pytest.raises((ValueError, TypeError)):
        LinearGenericSurface(
            reference_area=REFERENCE_AREA,
            reference_length=REFERENCE_LENGTH,
            coefficients=coefficients,
        )


def test_a_one_input_derivative_is_against_mach():
    """A derivative name already fixes the angle or rate, so a one-argument
    function or one-column table is read against Mach."""
    surface = LinearGenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN_alpha": lambda x1: 2 + x1}
    )
    assert surface.cN_alpha(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(2.5)


def test_invalid_initialization_from_csv(filename_invalid_coeff_linear_generic_surface):
    """Checks if linear generic surfaces raises errors when initialized incorrectly
    from a csv file"""
    with pytest.raises(ValueError):
        LinearGenericSurface(
            reference_area=REFERENCE_AREA,
            reference_length=REFERENCE_LENGTH,
            coefficients={"cN_0": str(filename_invalid_coeff_linear_generic_surface)},
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {"cN_0": 0},
        {
            "cN_0": 0,
            "cY_0": Function(lambda x1, x2, x3, x4, x5, x6, x7: 0),
            "cA_0": lambda x1, x2, x3, x4, x5, x6, x7: 0,
        },
    ],
)
def test_valid_initialization(coefficients):
    """Checks if linear generic surface initializes correctly when coefficient
    argument is valid"""

    LinearGenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients=coefficients,
    )


def test_valid_initialization_from_csv(filename_valid_coeff_linear_generic_surface):
    """Checks if linear generic surfaces initializes correctly when
    coefficients is set from a csv file"""
    LinearGenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN_0": str(filename_valid_coeff_linear_generic_surface)},
    )


def test_compute_forces_and_moments():
    """Checks if there are not logical errors in
    compute forces and moments"""

    lgs_object = LinearGenericSurface(REFERENCE_AREA, REFERENCE_LENGTH, {})
    forces_and_moments = lgs_object.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, 0)),
        stream_speed=1,
        stream_mach=0,
        rho=0,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, 0),
        density=Function(1.0),
        dynamic_viscosity=Function(1.0),
        z=0,
    )
    assert forces_and_moments == (0, 0, 0, 0, 0, 0)


def test_roll_damping_uses_reduced_rate():
    """The roll-damping derivative cl_p is applied to the reduced roll rate
    p* = p L_ref / (2 V). This must equal the previous raw-rate scaling
    (0.5 rho V A L^2 / 2) * cl_p * p, confirming the change is result-identical
    for the linear model."""
    ref_area, ref_length, cl_p = 2.0, 0.5, 3.0
    lgs = LinearGenericSurface(ref_area, ref_length, {"cl_p": cl_p})

    rho, speed, raw_roll = 1.2, 10.0, 4.0
    *_, roll_moment = lgs.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, -speed)),  # along centerline -> alpha=beta=0
        stream_speed=speed,
        stream_mach=0,
        rho=rho,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, raw_roll),  # raw body roll rate p, rad/s
        density=Function(1.0),
        dynamic_viscosity=Function(1.0),
        z=0,
    )

    reduced_roll = raw_roll * ref_length / (2 * speed)
    dyn_pressure_area_length = 0.5 * rho * speed**2 * ref_area * ref_length
    # New (reduced-rate) formulation:
    assert roll_moment == pytest.approx(dyn_pressure_area_length * cl_p * reduced_roll)
    # Old (raw-rate) formulation -- identical value:
    old_damping_scaling = 0.5 * rho * speed * ref_area * ref_length**2 / 2
    assert roll_moment == pytest.approx(old_damping_scaling * cl_p * raw_roll)


def test_force_convention_inference():
    """The input frame is inferred from the derivative names when
    ``force_convention`` is not given, defaulting to body when there are no
    force derivatives to infer from."""
    assert LinearGenericSurface(1, 1, {"cN_alpha": 2.0}).force_convention == "body"
    assert LinearGenericSurface(1, 1, {"cL_alpha": 2.0}).force_convention == "wind"
    # Moment-only / empty input carries no force frame -> body (canonical).
    assert LinearGenericSurface(1, 1, {"cm_alpha": 1.0}).force_convention == "body"
    assert LinearGenericSurface(1, 1, {}).force_convention == "body"


def test_mixed_frame_input_raises():
    """Supplying both wind (cL_*) and body (cN_*) force derivatives without
    declaring the frame is rejected."""
    with pytest.raises(ValueError, match="[Mm]ixed"):
        LinearGenericSurface(1, 1, {"cL_alpha": 1.0, "cN_0": 1.0})


def test_invalid_wind_coefficient_name_raises():
    """A wind-frame input with an unknown derivative name is rejected."""
    with pytest.raises(ValueError, match="Invalid coefficient name"):
        LinearGenericSurface(1, 1, {"cL_gamma": 1.0}, force_convention="wind")


def test_wind_derivatives_give_the_body_slopes_at_zero_angle():
    """The body-frame derivatives of a surface given in the wind frame are its
    slopes at zero angle, which the stability analysis reads: the four cross
    terms plus straight renames. A nonzero base drag ``cD_0`` couples into
    the normal- and axial-force slopes."""
    surface = LinearGenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={
            "cL_0": 0.1,
            "cL_alpha": 5.0,
            "cD_0": 0.3,
            "cD_alpha": 0.2,
            "cQ_beta": -4.0,
            "cQ_0": 0.05,
            "cm_alpha": -2.0,  # frame-shared moment, must pass through unchanged
        },
        force_convention="wind",
    )
    assert surface.force_convention == "wind"
    assert surface.cN_0.get_value_opt(*_ARGS) == pytest.approx(0.1)  # cL_0
    assert surface.cN_alpha.get_value_opt(*_ARGS) == pytest.approx(5.3)  # cL_alpha+cD_0
    assert surface.cY_beta.get_value_opt(*_ARGS) == pytest.approx(-4.3)  # cQ_beta-cD_0
    assert surface.cA_0.get_value_opt(*_ARGS) == pytest.approx(0.3)  # cD_0
    assert surface.cA_alpha.get_value_opt(*_ARGS) == pytest.approx(0.1)  # cD_alpha-cL_0
    assert surface.cA_beta.get_value_opt(*_ARGS) == pytest.approx(0.05)  # cD_beta+cQ_0
    assert surface.cm_alpha.get_value_opt(*_ARGS) == pytest.approx(-2.0)  # passthrough

    # and they are the slopes of the forces the surface produces
    step = 1e-6
    for name, angle, expected in (("cN", 0, 5.3), ("cY", 1, -4.3), ("cA", 0, 0.1)):
        at = [0.0, 0.0, 0.5, 1e6, 0.0, 0.0, 0.0]
        below, above = list(at), list(at)
        below[angle], above[angle] = -step, step
        slope = (getattr(surface, name)(*above) - getattr(surface, name)(*below)) / (
            2 * step
        )
        assert slope == pytest.approx(expected, abs=1e-6)


_WIND_DERIVATIVES = {
    "cL_0": 0.1,
    "cL_alpha": 5.0,
    "cL_q": 0.4,
    "cD_0": 0.3,
    "cD_alpha": 0.2,
    "cQ_beta": -4.0,
}


def _as_given(alpha, beta, pitch_rate):
    """Lift, drag and side force of ``_WIND_DERIVATIVES``."""
    return (
        0.1 + 5.0 * alpha + 0.4 * pitch_rate,
        0.3 + 0.2 * alpha,
        -4.0 * beta,
    )


@pytest.mark.parametrize("alpha", [0.0, 0.02, 0.1, math.radians(15), math.radians(30)])
@pytest.mark.parametrize("beta", [0.0, 0.01, 0.2])
def test_a_surface_given_in_the_wind_frame_produces_the_forces_given(alpha, beta):
    """A surface given in the wind frame is linear in that frame: its lift,
    drag and side force are the ones given at any angle, and its normal, side
    and axial force are those turned into the body frame, as for a
    ``GenericSurface`` given the same lift, drag and side force."""
    linear = LinearGenericSurface(0.01, 0.1, _WIND_DERIVATIVES)
    generic = GenericSurface(
        0.01,
        0.1,
        coefficients={
            "cL": lambda alpha, pitch_rate: 0.1 + 5.0 * alpha + 0.4 * pitch_rate,
            "cD": lambda alpha: 0.3 + 0.2 * alpha,
            "cQ": lambda beta: -4.0 * beta,
        },
    )
    args = (alpha, beta, 0.5, 1e6, 0.03, 0.0, 0.0)

    lift, drag, side = _as_given(alpha, beta, 0.03)
    assert linear.cL(*args) == pytest.approx(lift, abs=1e-12)
    assert linear.cD(*args) == pytest.approx(drag, abs=1e-12)
    assert linear.cQ(*args) == pytest.approx(side, abs=1e-12)
    for name in ("cN", "cY", "cA"):
        assert getattr(linear, name).get_value_opt(*args) == pytest.approx(
            getattr(generic, name).get_value_opt(*args), abs=1e-12
        )


def test_a_surface_given_in_the_wind_frame_keeps_its_derivatives():
    """The derivatives are read back as given, with the forcing and damping
    parts of lift, drag and side force. A surface given in the body frame has
    those of the normal, side and axial force instead."""
    wind = LinearGenericSurface(0.01, 0.1, _WIND_DERIVATIVES)
    args = (0.1, 0.05, 0.5, 1e6, 0.03, 0.0, 0.0)

    for name, value in _WIND_DERIVATIVES.items():
        assert getattr(wind, name).get_value_opt(*args) == pytest.approx(value)
    assert wind.cL_beta.is_zero and wind.cQ_r.is_zero
    assert wind.cLf(*args) == pytest.approx(0.1 + 5.0 * 0.1)
    assert wind.cLd(*args) == pytest.approx(0.4 * 0.03)
    assert wind.cDf(*args) == pytest.approx(0.3 + 0.2 * 0.1)
    assert wind.cQf(*args) == pytest.approx(-4.0 * 0.05)
    assert wind.cL(*args) == pytest.approx(wind.cLf(*args) + wind.cLd(*args))
    assert not hasattr(wind, "cNf") and not hasattr(wind, "cAd")

    body = LinearGenericSurface(0.01, 0.1, {"cN_alpha": 5.3, "cA_0": 0.3})
    assert not hasattr(body, "cL_alpha") and not hasattr(body, "cLf")
    assert body.cN(*args) == pytest.approx(body.cNf(*args) + body.cNd(*args))
    assert body.cN(*args) == pytest.approx(5.3 * 0.1)


def test_wind_and_body_input_agree_near_zero_angle_only():
    """A surface given in the wind frame and the body-frame surface with the
    same slopes at zero angle are two different linear models. Their forces
    agree to first order in the angles and drift apart beyond that."""
    wind = LinearGenericSurface(1, 1, {"cL_alpha": 2.0, "cD_0": 0.3})
    body = LinearGenericSurface(1, 1, {"cN_alpha": 2.3, "cA_0": 0.3})

    small = (1e-4, 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)
    large = (math.radians(15), 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)
    for name in ("cN", "cA"):
        assert getattr(wind, name)(*small) == pytest.approx(
            getattr(body, name)(*small), abs=1e-7
        )
    # at 15 degrees the body-frame surface has half as much drag again
    assert wind.cD(*large) == pytest.approx(0.3)
    assert body.cD(*large) == pytest.approx(0.4456, abs=1e-4)


def test_an_axisymmetric_surface_given_in_the_wind_frame_fills_its_side_force():
    """With ``axisymmetric=True`` the side force derivatives come from the
    lift ones, so the surface pushes back the same in every plane."""
    surface = LinearGenericSurface(
        1, 1, {"cL_alpha": 2.0, "cL_q": 0.4, "cD_0": 0.3}, axisymmetric=True
    )
    args = (0.0, 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)

    assert surface.cQ_beta.get_value_opt(*args) == pytest.approx(-2.0)
    assert surface.cQ_r.get_value_opt(*args) == pytest.approx(0.4)
    assert surface.cY_beta.get_value_opt(*args) == pytest.approx(-2.3)
    # the same push for the same angle in either plane
    pitch = surface.cN(0.1, 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)
    yaw = surface.cY(0.0, 0.1, 0.5, 1e6, 0.0, 0.0, 0.0)
    assert yaw == pytest.approx(-pitch, rel=1e-12)


def test_a_surface_given_in_the_wind_frame_is_saved_as_given():
    surface = LinearGenericSurface(0.01, 0.1, _WIND_DERIVATIVES)
    loaded = LinearGenericSurface.from_dict(surface.to_dict())
    args = (0.2, 0.05, 0.5, 1e6, 0.03, 0.0, 0.0)

    assert loaded.force_convention == "wind"
    for name in ("cL", "cD", "cQ", "cN", "cY", "cA", "cL_alpha", "cD_0"):
        assert getattr(loaded, name)(*args) == pytest.approx(
            getattr(surface, name)(*args)
        )


def test_the_moment_follows_the_force_of_a_surface_given_in_the_wind_frame():
    """With a center of pressure that varies with Mach, the moment of the
    normal force about it uses the force the surface produces, not its slope
    at zero angle."""
    arm = Function(lambda mach: 0.2 + 0.1 * mach, "Mach", "z (m)")
    surface = LinearGenericSurface(
        1, 2, {"cL_alpha": 2.0, "cD_0": 0.3}, center_of_pressure=(0, 0, arm)
    )
    args = (math.radians(15), 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)

    assert surface.cm(*args) == pytest.approx(surface.cN(*args) * 0.25 / 2)


def test_public_coefficient_includes_the_damping_terms():
    """``surface.cm`` is the whole coefficient, so it follows the rotation rates
    as well as the angles; ``cmf`` and ``cmd`` are its two parts."""
    surface = LinearGenericSurface(
        reference_area=1.3,
        reference_length=0.7,
        coefficients={"cm_alpha": -3.0, "cm_q": -50.0, "cm_p": lambda mach: mach},
    )
    alpha, mach, pitch_rate, roll_rate = 0.1, 0.5, 0.02, 0.03
    args = (alpha, 0.0, mach, 0.0, pitch_rate, 0.0, roll_rate)

    forcing = -3.0 * alpha
    damping = -50.0 * pitch_rate + mach * roll_rate
    assert surface.cmf(*args) == pytest.approx(forcing)
    assert surface.cmd(*args) == pytest.approx(damping)
    assert surface.cm(*args) == pytest.approx(forcing + damping)
    # A coefficient with no derivatives given is zero
    assert surface.cn(*args) == 0.0


def test_public_coefficients_give_the_simulated_forces_and_moments():
    """The forces and moments used in the simulation are the public coefficients
    times the dynamic pressure, the reference area and the reference length."""
    names = [
        f"{coefficient}_{suffix}"
        for coefficient in ("cN", "cY", "cA", "cm", "cn", "cl")
        for suffix in ("0", "alpha", "beta", "p", "q", "r")
    ]
    surface = LinearGenericSurface(
        reference_area=1.3,
        reference_length=0.7,
        coefficients={name: 0.05 * i - 0.4 for i, name in enumerate(names)},
    )
    rho, speed, mach = 1.1, 80.0, 0.4
    alpha, beta, rates = 0.05, -0.02, (0.01, -0.02, 0.03)
    args = (alpha, beta, mach, 0, *rates)
    force_scale = 0.5 * rho * speed**2 * 1.3
    moment_scale = force_scale * 0.7
    # The flow and the body rates that give these angles and reduced rates
    direction = Vector([math.tan(beta), math.tan(alpha), 1.0])
    stream_velocity = direction * (-speed / abs(direction))
    omega = [rate * 2 * speed / 0.7 for rate in rates]

    r1, r2, r3, pitch, yaw, roll = surface.compute_forces_and_moments(
        stream_velocity,
        speed,
        mach,
        rho,
        Vector([0, 0, 0]),
        omega,
        Function(1.2),
        Function(1.8e-5),
        0.0,
    )

    assert r1 == pytest.approx(force_scale * surface.cY(*args))
    assert r2 == pytest.approx(-force_scale * surface.cN(*args))
    assert r3 == pytest.approx(-force_scale * surface.cA(*args))
    assert pitch == pytest.approx(moment_scale * surface.cm(*args))
    assert yaw == pytest.approx(moment_scale * surface.cn(*args))
    assert roll == pytest.approx(moment_scale * surface.cl(*args))


def test_linear_surface_from_csv(tmp_path):
    """The coefficient derivatives of a linear surface load from one file."""
    csv_file = tmp_path / "derivatives.csv"
    csv_file.write_text(
        "mach,cN_alpha,cm_alpha,cm_q\n0,2,-3,-50\n1,3,-4,-50\n2,4,-5,-50\n"
    )
    surface = LinearGenericSurface.from_csv(str(csv_file), 1.0, 1.0)
    assert surface.cN_alpha.depends_on == ("mach",)
    assert surface.cN(0.1, 0.0, 1.0, 0, 0, 0, 0) == pytest.approx(0.3)
    assert surface.cm(0.1, 0.0, 1.0, 0, 0.01, 0, 0) == pytest.approx(-0.4 - 0.5)
