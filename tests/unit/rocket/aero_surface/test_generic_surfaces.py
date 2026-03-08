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


def test_csv_independent_variables_accept_any_order(tmp_path):
    """Checks if GenericSurface correctly maps CSV columns by header names,
    regardless of independent variable column order."""
    filename = tmp_path / "valid_coefficients_shuffled_order.csv"
    filename.write_text(
        "mach,alpha,cL\n0,0,0\n0,1,10\n2,0,2\n2,1,12\n",
        encoding="utf-8",
    )

    generic_surface = GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cL": str(filename)},
    )

    closure = generic_surface.cL.source.__closure__
    csv_function = next(
        cell.cell_contents
        for cell in closure
        if isinstance(cell.cell_contents, Function)
    )

    assert generic_surface.cL(1, 0, 2, 0, 0, 0, 0) == pytest.approx(12)
    assert csv_function.get_interpolation_method() == "regular_grid"


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
