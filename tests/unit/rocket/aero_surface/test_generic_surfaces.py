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

    # The coefficient is stored at minimal dimension over its CSV columns, in
    # header order; AeroCoefficient maps the full argument tuple onto them.
    assert generic_surface.cL.depends_on == ("mach", "alpha")
    csv_function = generic_surface.cL.function

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
        density=Function(1.0),
        dynamic_viscosity=Function(1.0),
        z=0,
    )
    assert forces_and_moments == (0, 0, 0, 0, 0, 0)


def test_angular_rates_are_non_dimensionalized():
    """Coefficients receive the conventional reduced rate q* = q L_ref / (2 V),
    not the raw body rate in rad/s."""
    ref_area, ref_length = 2.0, 0.5
    # Roll-moment coefficient that simply returns the roll rate it is given, so
    # the resulting roll moment exposes which rate value reached the coefficient.
    gs = GenericSurface(ref_area, ref_length, {"cl": lambda roll_rate: roll_rate})

    rho, speed, raw_roll = 1.2, 10.0, 4.0
    *_, roll_moment = gs.compute_forces_and_moments(
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
    # The coefficient saw the reduced rate, ...
    assert roll_moment == pytest.approx(dyn_pressure_area_length * reduced_roll)
    # ... not the raw rad/s rate.
    assert roll_moment != pytest.approx(dyn_pressure_area_length * raw_roll)
