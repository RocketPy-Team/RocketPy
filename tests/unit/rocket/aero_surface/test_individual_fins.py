"""Unit tests for individual fin classes."""

from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import (
    EllipticalFin,
    FreeFormFin,
    Rocket,
    TrapezoidalFin,
    TrapezoidalFins,
)


@pytest.mark.parametrize(
    "fixture_name,expected_class",
    [
        ("calisto_trapezoidal_fin", TrapezoidalFin),
        ("calisto_elliptical_fin", EllipticalFin),
        ("calisto_free_form_fin", FreeFormFin),
    ],
)
def test_individual_fin_info_returns_none(request, fixture_name, expected_class):
    """Ensure info() executes for all individual fin classes."""
    # Arrange
    fin = request.getfixturevalue(fixture_name)

    # Act
    result = fin.info()

    # Assert
    assert isinstance(fin, expected_class)
    assert result is None


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize(
    "fixture_name",
    [
        "calisto_trapezoidal_fin",
        "calisto_elliptical_fin",
        "calisto_free_form_fin",
    ],
)
def test_individual_fin_draw_returns_none(mock_show, request, fixture_name):  # pylint: disable=unused-argument
    """Ensure draw() executes for all individual fin classes."""
    # Arrange
    fin = request.getfixturevalue(fixture_name)

    # Act
    result = fin.draw(filename=None)

    # Assert
    assert result is None


@pytest.mark.parametrize(
    "fixture_name",
    [
        "calisto_trapezoidal_fin",
        "calisto_elliptical_fin",
        "calisto_free_form_fin",
    ],
)
def test_individual_fin_angular_position_updates_radians(request, fixture_name):
    """Ensure angular_position setter updates angular_position_rad."""
    # Arrange
    fin = request.getfixturevalue(fixture_name)

    # Act
    fin.angular_position = 45

    # Assert
    assert fin.angular_position == 45
    assert fin.angular_position_rad == pytest.approx(np.pi / 4)


def test_trapezoidal_fin_setters_update_geometry(calisto_trapezoidal_fin):
    """Ensure trapezoidal fin geometry setters update exposed values."""
    # Arrange
    fin = calisto_trapezoidal_fin

    # Act
    fin.tip_chord = 0.05
    fin.sweep_angle = 12.0
    fin.sweep_length = 0.03

    # Assert
    assert fin.tip_chord == pytest.approx(0.05)
    assert fin.sweep_angle == pytest.approx(12.0)
    assert fin.sweep_length == pytest.approx(0.03)


def test_trapezoidal_fin_rejects_inconsistent_sweep_inputs():
    """Ensure trapezoidal fin rejects sweep_length with sweep_angle together."""
    # Arrange / Act / Assert
    with pytest.raises(
        ValueError, match="Cannot use sweep_length and sweep_angle together"
    ):
        TrapezoidalFin(
            angular_position=0,
            root_chord=0.12,
            tip_chord=0.04,
            span=0.1,
            rocket_radius=0.0635,
            sweep_length=0.02,
            sweep_angle=10.0,
        )


def test_free_form_fin_shape_points_property(calisto_free_form_fin):
    """Ensure free-form fin exposes the original shape points."""
    # Arrange
    fin = calisto_free_form_fin

    # Act
    shape_points = fin.shape_points

    # Assert
    assert shape_points == [(0, 0), (0.08, 0.1), (0.12, 0.1), (0.12, 0)]


@pytest.mark.parametrize(
    "fixture_name,required_keys",
    [
        (
            "calisto_trapezoidal_fin",
            {
                "angular_position",
                "root_chord",
                "span",
                "rocket_radius",
                "cant_angle",
                "airfoil",
                "name",
                "tip_chord",
                "sweep_length",
                "sweep_angle",
            },
        ),
        (
            "calisto_elliptical_fin",
            {
                "angular_position",
                "root_chord",
                "span",
                "rocket_radius",
                "cant_angle",
                "airfoil",
                "name",
            },
        ),
        (
            "calisto_free_form_fin",
            {
                "angular_position",
                "rocket_radius",
                "cant_angle",
                "airfoil",
                "name",
                "shape_points",
            },
        ),
    ],
)
def test_individual_fin_to_dict_contains_expected_keys(
    request, fixture_name, required_keys
):
    """Ensure to_dict for each individual fin exposes expected input keys."""
    # Arrange
    fin = request.getfixturevalue(fixture_name)

    # Act
    data = fin.to_dict()

    # Assert
    assert required_keys.issubset(data.keys())


@pytest.mark.parametrize(
    "fixture_name,fin_class,comparisons",
    [
        (
            "calisto_trapezoidal_fin",
            TrapezoidalFin,
            ["angular_position", "root_chord", "tip_chord", "span", "rocket_radius"],
        ),
        (
            "calisto_elliptical_fin",
            EllipticalFin,
            ["angular_position", "root_chord", "span", "rocket_radius"],
        ),
        (
            "calisto_free_form_fin",
            FreeFormFin,
            ["angular_position", "rocket_radius"],
        ),
    ],
)
def test_individual_fin_from_dict_roundtrip(
    request, fixture_name, fin_class, comparisons
):
    """Ensure each individual fin can be reconstructed with from_dict."""
    # Arrange
    fin = request.getfixturevalue(fixture_name)
    data = fin.to_dict()

    # Act
    reconstructed = fin_class.from_dict(data)

    # Assert
    assert isinstance(reconstructed, fin_class)
    for field in comparisons:
        assert getattr(reconstructed, field) == pytest.approx(getattr(fin, field))

    if fin_class is FreeFormFin:
        assert reconstructed.shape_points == fin.shape_points


def test_calisto_finset_vs_four_individual_fins_close():
    """Ensure a 4-fin set and 4 individual fins produce close aerodynamics.

    Notes
    -----
    A fin set model includes finite-set lift correction for the number of fins.
    For 4 fins, this correction is equivalent to scaling the sum of 4
    individual-fin lift derivatives by 1/2.
    """
    # Arrange
    finset_rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    finset_rocket.add_surfaces(
        TrapezoidalFins(
            n=4,
            span=0.100,
            root_chord=0.120,
            tip_chord=0.040,
            rocket_radius=0.0635,
            name="calisto_trapezoidal_fins",
            cant_angle=0,
            sweep_length=None,
            sweep_angle=None,
            airfoil=None,
        ),
        -1.168,
    )

    individual_fins_rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    individual_fins = [
        TrapezoidalFin(
            angular_position=angle,
            root_chord=0.120,
            tip_chord=0.040,
            span=0.100,
            rocket_radius=0.0635,
            name=f"calisto_trapezoidal_fin_{i}",
            cant_angle=0,
            sweep_length=None,
            sweep_angle=None,
            airfoil=None,
        )
        for i, angle in enumerate((0, 90, 180, 270), start=1)
    ]
    individual_fins_rocket.add_surfaces(individual_fins, [-1.168] * 4)

    mach_grid = np.linspace(0, 2, 21)

    # Act
    cp_finset = finset_rocket.cp_position(mach_grid)
    cp_individual = individual_fins_rocket.cp_position(mach_grid)
    clalpha_finset = finset_rocket.total_lift_coeff_der(mach_grid)
    clalpha_individual = individual_fins_rocket.total_lift_coeff_der(mach_grid)
    lift_correction = TrapezoidalFins.fin_num_correction(4) / 4
    clalpha_individual_corrected = np.array(clalpha_individual) * lift_correction

    # Assert
    np.testing.assert_allclose(cp_individual, cp_finset, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(clalpha_individual_corrected, clalpha_finset)
