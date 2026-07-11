import pytest

from rocketpy import Function, LinearGenericSurface
from rocketpy.mathutils import Vector

REFERENCE_AREA = 1
REFERENCE_LENGTH = 1


@pytest.mark.parametrize(
    "coefficients",
    [
        "cL_0",
        {"invalid_name": 0},
        {"cL_0": "inexistent_file.csv"},
        {"cL_0": Function(lambda x1, x2, x3, x4, x5, x6: 0)},
        {"cL_0": lambda x1: 0},
        {"cL_0": {}},
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
            coefficients={"cL_0": str(filename_invalid_coeff_linear_generic_surface)},
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {"cL_0": 0},
        {
            "cL_0": 0,
            "cQ_0": Function(lambda x1, x2, x3, x4, x5, x6, x7: 0),
            "cD_0": lambda x1, x2, x3, x4, x5, x6, x7: 0,
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
        coefficients={"cL_0": str(filename_valid_coeff_linear_generic_surface)},
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
