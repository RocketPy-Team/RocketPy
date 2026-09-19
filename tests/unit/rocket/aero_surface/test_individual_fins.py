"""Unit tests for individual fin classes."""

from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import (
    EllipticalFin,
    EllipticalFins,
    FreeFormFin,
    FreeFormFins,
    Rocket,
    TrapezoidalFin,
    TrapezoidalFins,
)
from rocketpy.mathutils.vector_matrix import Vector


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
    np.testing.assert_allclose(fin.angular_position_rad, np.pi / 4)


def test_trapezoidal_fin_setters_update_geometry(calisto_trapezoidal_fin):
    """Ensure trapezoidal fin geometry setters update exposed values."""
    # Arrange
    fin = calisto_trapezoidal_fin

    # Act
    fin.tip_chord = 0.05
    fin.sweep_angle = 12.0
    fin.sweep_length = 0.03

    # Assert
    np.testing.assert_allclose(fin.tip_chord, 0.05)
    np.testing.assert_allclose(fin.sweep_angle, 12.0)
    np.testing.assert_allclose(fin.sweep_length, 0.03)


def test_individual_fin_rocket_diameter_aliases_are_kept_in_sync(
    calisto_trapezoidal_fin,
):
    """Ensure rocket_diameter is canonical and old aliases remain compatible."""
    # Arrange
    fin = calisto_trapezoidal_fin

    # Act
    fin.rocket_diameter = 0.15

    # Assert
    np.testing.assert_allclose(fin.rocket_diameter, 0.15)
    np.testing.assert_allclose(fin.diameter, 0.15)
    np.testing.assert_allclose(fin.d, 0.15)
    np.testing.assert_allclose(fin.rocket_radius, 0.075)
    np.testing.assert_allclose(fin.reference_length, 0.15)

    # Act
    fin.d = 0.20

    # Assert
    np.testing.assert_allclose(fin.rocket_diameter, 0.20)
    np.testing.assert_allclose(fin.diameter, 0.20)
    np.testing.assert_allclose(fin.d, 0.20)
    np.testing.assert_allclose(fin.rocket_radius, 0.10)
    np.testing.assert_allclose(fin.reference_length, 0.20)


def test_individual_fin_reference_area_and_ref_area_alias_are_kept_in_sync(
    calisto_trapezoidal_fin,
):
    """Ensure reference_area is canonical and ref_area remains compatible."""
    # Arrange
    fin = calisto_trapezoidal_fin

    # Act
    fin.reference_area = 0.123

    # Assert
    np.testing.assert_allclose(fin.reference_area, 0.123)
    np.testing.assert_allclose(fin.ref_area, 0.123)

    # Act
    fin.ref_area = 0.456

    # Assert
    np.testing.assert_allclose(fin.reference_area, 0.456)
    np.testing.assert_allclose(fin.ref_area, 0.456)


def test_individual_fin_to_dict_include_outputs_exposes_diameter_aliases(
    calisto_trapezoidal_fin,
):
    """Ensure output serialization exposes canonical and alias diameter keys."""
    # Arrange
    fin = calisto_trapezoidal_fin

    # Act
    data = fin.to_dict(include_outputs=True)

    # Assert
    np.testing.assert_allclose(data["rocket_diameter"], fin.rocket_diameter)
    np.testing.assert_allclose(data["diameter"], fin.rocket_diameter)
    np.testing.assert_allclose(data["d"], fin.rocket_diameter)
    np.testing.assert_allclose(data["reference_area"], fin.reference_area)
    np.testing.assert_allclose(data["ref_area"], fin.reference_area)


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
        np.testing.assert_allclose(getattr(reconstructed, field), getattr(fin, field))

    if fin_class is FreeFormFin:
        assert reconstructed.shape_points == fin.shape_points


def test_trapezoidal_fin_from_dict_roundtrip_preserves_sweep_length():
    """Ensure TrapezoidalFin round-trip preserves non-default sweep geometry."""
    # Arrange
    original = TrapezoidalFin(
        angular_position=0,
        root_chord=0.12,
        tip_chord=0.04,
        span=0.1,
        rocket_radius=0.0635,
        cant_angle=0,
        sweep_angle=15.0,
        name="roundtrip_trapezoidal_fin",
    )
    data = original.to_dict()

    # Act
    reconstructed = TrapezoidalFin.from_dict(data)

    # Assert
    np.testing.assert_allclose(reconstructed.sweep_length, original.sweep_length)


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
    cp_finset = finset_rocket.aerodynamic_center(mach_grid)
    cp_individual = individual_fins_rocket.aerodynamic_center(mach_grid)
    clalpha_finset = finset_rocket.total_lift_coeff_der(mach_grid)
    clalpha_individual = individual_fins_rocket.total_lift_coeff_der(mach_grid)

    # Assert. Each individual fin projects its lift slope onto the pitch plane by
    # sin(phi)**2, so an evenly spaced set of 4 sums to fin_num_correction(4) = 2
    # in the plane -- matching the fin set directly, with no extra correction.
    np.testing.assert_allclose(cp_individual, cp_finset, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(clalpha_individual, clalpha_finset)


@pytest.mark.parametrize(
    "fin_cls, geometry",
    [
        (
            TrapezoidalFin,
            dict(root_chord=0.120, tip_chord=0.040, span=0.100, rocket_radius=0.0635),
        ),
        (EllipticalFin, dict(root_chord=0.120, span=0.100, rocket_radius=0.0635)),
        (
            FreeFormFin,
            dict(shape_points=[(0, 0), (0.06, 0.1), (0.12, 0.0)], rocket_radius=0.0635),
        ),
    ],
)
def test_canted_individual_fin_builds_and_places(fin_cls, geometry):
    """A canted individual fin of any shape must build its body<->fin rotation
    matrices at construction, so it can be placed on a rocket. Regression test
    for a crash where non-trapezoidal individual fins lacked
    ``_rotation_fin_to_body_uncanted`` and failed in
    ``_compute_leading_edge_position``."""
    fin = fin_cls(angular_position=30, cant_angle=2.0, **geometry)
    assert hasattr(fin, "_rotation_fin_to_body_uncanted")
    position = fin._compute_leading_edge_position(-1.168, 1)
    assert position is not None


@pytest.mark.parametrize(
    "fin_cls, geometry",
    [
        (
            TrapezoidalFin,
            dict(root_chord=0.120, tip_chord=0.040, span=0.100, rocket_radius=0.0635),
        ),
        (EllipticalFin, dict(root_chord=0.120, span=0.100, rocket_radius=0.0635)),
        (
            FreeFormFin,
            dict(shape_points=[(0, 0), (0.06, 0.1), (0.12, 0.0)], rocket_radius=0.0635),
        ),
    ],
)
def test_individual_fin_roll_moment_independent_of_angular_position(fin_cls, geometry):
    """A canted individual fin's roll moment must be the same at any angular
    position (rotational symmetry about the roll axis). Regression test for a bug
    where the fin's center of pressure was not rotated to its azimuth (the
    rotation matrix was left as the identity), making the roll moment vary with
    angular position."""
    stream_velocity = Vector([0, 0, -1.0])
    omega = Vector([0, 0, 0])

    roll_moments = []
    for angle in (0, 90, 180, 270):
        rocket = Rocket(
            radius=0.0635,
            mass=14.426,
            inertia=(6.321, 6.321, 0.034),
            power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
            power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
            center_of_mass_without_motor=0,
            coordinate_system_orientation="tail_to_nose",
        )
        fin = fin_cls(angular_position=angle, cant_angle=2.0, **geometry)
        rocket.add_surfaces(fin, -1.168)
        cp = rocket.surfaces_cp_to_cdm[fin]
        roll = fin.compute_forces_and_moments(
            stream_velocity, 1.0, 0.3, 1.0, cp, omega
        )[5]
        roll_moments.append(roll)

    np.testing.assert_allclose(roll_moments, roll_moments[0], rtol=1e-9)
    assert abs(roll_moments[0]) > 0


@pytest.mark.parametrize(
    "set_cls, fin_cls, geometry",
    [
        (
            TrapezoidalFins,
            TrapezoidalFin,
            dict(root_chord=0.120, tip_chord=0.040, span=0.100, rocket_radius=0.0635),
        ),
        (
            EllipticalFins,
            EllipticalFin,
            dict(root_chord=0.120, span=0.100, rocket_radius=0.0635),
        ),
        (
            FreeFormFins,
            FreeFormFin,
            dict(shape_points=[(0, 0), (0.06, 0.1), (0.12, 0.0)], rocket_radius=0.0635),
        ),
    ],
)
def test_finset_roll_forcing_equals_n_single_fins(set_cls, fin_cls, geometry):
    """A fin set's roll forcing coefficient must scale with the full fin count
    ``n`` (every identically-canted fin adds the same roll moment), so it equals
    ``n`` times a single fin's roll forcing -- for every fin shape. Regression
    test for a bug where the set used the normal-force ``fin_num_correction(n)``
    (~n/2), halving the roll forcing (and roll rate) of a fin set."""
    n = 4
    finset = set_cls(n=n, cant_angle=2.0, **geometry)
    single = fin_cls(angular_position=0, cant_angle=2.0, **geometry)

    mach_grid = np.linspace(0, 2, 11)
    clf_finset = finset.roll_parameters[0](mach_grid)
    clf_single = single.roll_parameters[0](mach_grid)
    np.testing.assert_allclose(clf_finset, n * np.array(clf_single), rtol=1e-6)


@pytest.mark.parametrize(
    "position_input",
    [
        (0.02, -0.01, -1.2),
        Vector([0.02, -0.01, -1.2]),
    ],
)
def test_add_individual_fin_accepts_full_3d_position(position_input):
    """Ensure individual fins accept full (x, y, z) position inputs."""
    # Arrange
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    fin = TrapezoidalFin(
        angular_position=30,
        root_chord=0.120,
        tip_chord=0.040,
        span=0.100,
        rocket_radius=0.0635,
        cant_angle=0,
        name="position_test_fin",
    )

    # Act
    rocket.add_surfaces(fin, position_input)
    stored_position = rocket.aerodynamic_surfaces[0].position

    # Assert
    assert stored_position == Vector([0.02, -0.01, -1.2])
