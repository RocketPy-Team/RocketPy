import pytest

from rocketpy import Function, GenericSurface
from rocketpy.mathutils import Vector

REFERENCE_AREA = 1
REFERENCE_LENGTH = 1


@pytest.mark.parametrize(
    "coefficients",
    [
        "cL",
        {"invalid_name": 0},
        {"cL": "inexistent_file.csv"},
        {"cL": Function(lambda x1, x2, x3, x4, x5, x6: 0)},
        {"cL": lambda x1: 0},
        {"cL": {}},
    ],
)
def test_invalid_initialization(coefficients):
    """Checks if generic surface raises errors in initialization
    when coefficient argument is invalid"""

    with pytest.raises((ValueError, TypeError)):
        GenericSurface(
            reference_area=REFERENCE_AREA,
            reference_length=REFERENCE_LENGTH,
            coefficients=coefficients,
        )


def test_invalid_initialization_from_csv(filename_invalid_coeff):
    """Checks if generic surfaces raises errors when initialized incorrectly
    from a csv file"""
    with pytest.raises(ValueError):
        GenericSurface(
            reference_area=REFERENCE_AREA,
            reference_length=REFERENCE_LENGTH,
            coefficients={"cL": str(filename_invalid_coeff)},
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {"cL": 0},
        {
            "cL": 0,
            "cQ": Function(lambda x1, x2, x3, x4, x5, x6, x7: 0),
            "cD": lambda x1, x2, x3, x4, x5, x6, x7: 0,
        },
    ],
)
def test_valid_initialization(coefficients):
    """Checks if generic surface initializes correctly when coefficient
    argument is valid"""

    GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients=coefficients,
    )


def test_valid_initialization_from_csv(filename_valid_coeff):
    """Checks if generic surfaces initializes correctly when
    coefficients is set from a csv file"""
    GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cL": str(filename_valid_coeff)},
    )


def test_compute_forces_and_moments():
    """Checks if there are not logical errors in
    compute forces and moments"""

    gs_object = GenericSurface(REFERENCE_AREA, REFERENCE_LENGTH, {})
    forces_and_moments = gs_object.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, 0)),
        stream_speed=0,
        stream_mach=0,
        rho=0,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, 0),
        reynolds=0,
    )
    assert forces_and_moments == (0, 0, 0, 0, 0, 0)
