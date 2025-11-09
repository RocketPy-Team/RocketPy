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
        reynolds=0,
    )
    assert forces_and_moments == (0, 0, 0, 0, 0, 0)
