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
        {"cN_0": lambda x1: 0},
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


def test_wind_derivatives_convert_to_body_first_order():
    """Wind-frame derivatives are converted to the body-frame set by linearizing
    the wind/body rotation about zero: the four cross terms plus straight
    renames. A nonzero base drag ``cD_0`` couples into the normal- and
    axial-force slopes."""
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


def test_wind_input_matches_body_input():
    """A wind-frame surface equals the body-frame surface built from the
    hand-converted derivatives, at an arbitrary angle."""
    wind = LinearGenericSurface(
        1,
        1,
        coefficients={"cL_0": 0.1, "cL_alpha": 5.0, "cD_0": 0.3, "cQ_beta": -4.0},
        force_convention="wind",
    )
    body = LinearGenericSurface(
        1,
        1,
        coefficients={
            "cN_0": 0.1,
            "cN_alpha": 5.3,
            "cA_0": 0.3,
            "cA_alpha": -0.1,  # cD_alpha - cL_0
            "cY_beta": -4.3,
        },
    )
    args = (0.02, 0.01, 0.5, 1e6, 0.0, 0.0, 0.0)
    for coeff in ("cN", "cY", "cA"):
        assert getattr(wind, coeff).get_value_opt(*args) == pytest.approx(
            getattr(body, coeff).get_value_opt(*args)
        )


def test_wind_linear_matches_generic_surface_to_first_order():
    """The body-frame forces of a wind-input linear surface agree with the
    exact rotation used by GenericSurface to first order in the flow angles."""
    coeffs = {"cL_0": 0.1, "cL_alpha": 5.0, "cD_0": 0.3, "cQ_beta": -4.0}
    linear = LinearGenericSurface(0.01, 0.1, coeffs, force_convention="wind")
    generic = GenericSurface(
        0.01,
        0.1,
        coefficients={
            "cL": lambda a, b, m, re, p, q, r: 0.1 + 5.0 * a,
            "cD": lambda a, b, m, re, p, q, r: 0.3,
            "cQ": lambda a, b, m, re, p, q, r: -4.0 * b,
        },
        force_convention="wind",
    )
    eps = 1e-3
    for alpha, beta in [(eps, 0.0), (0.0, eps), (eps, eps)]:
        args = (alpha, beta, 0.5, 1e6, 0.0, 0.0, 0.0)
        for coeff in ("cN", "cY", "cA"):
            # Difference is second order in the angle (~1e-6 at eps=1e-3).
            assert getattr(linear, coeff).get_value_opt(*args) == pytest.approx(
                getattr(generic, coeff).get_value_opt(*args), abs=1e-5
            )
