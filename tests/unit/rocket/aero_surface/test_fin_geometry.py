"""Unit tests for fin geometry strategy classes."""

import numpy as np
import pytest

from rocketpy.rocket.aero_surface.fins._geometry import (
    _EllipticalGeometry,
    _FreeFormGeometry,
    _TrapezoidalGeometry,
)


def test_trapezoidal_geometry_evaluate_geometrical_parameters(
    calisto_trapezoidal_fin,
):
    """Ensure trapezoidal geometry populates the derived fin parameters."""
    # Arrange
    geometry = calisto_trapezoidal_fin.geometry

    # Act
    geometry.evaluate_geometrical_parameters()

    # Assert
    owner = calisto_trapezoidal_fin
    expected_area = (owner.root_chord + owner.tip_chord) * owner.span / 2
    expected_aspect_ratio = 2 * owner.span**2 / expected_area
    expected_gamma_c = np.arctan(
        (geometry.sweep_length + 0.5 * owner.tip_chord - 0.5 * owner.root_chord)
        / owner.span
    )
    expected_mid_aerodynamic_span = (
        owner.span
        / 3
        * (owner.root_chord + 2 * owner.tip_chord)
        / (owner.root_chord + owner.tip_chord)
    )
    tau = (owner.span + owner.rocket_radius) / owner.rocket_radius
    lambda_ = owner.tip_chord / owner.root_chord
    expected_roll_constant = (
        (owner.root_chord + 3 * owner.tip_chord) * owner.span**3
        + 4
        * (owner.root_chord + 2 * owner.tip_chord)
        * owner.rocket_radius
        * owner.span**2
        + 6 * (owner.root_chord + owner.tip_chord) * owner.span * owner.rocket_radius**2
    ) / 12
    expected_lift_interference_factor = 1 + 1 / tau
    expected_roll_damping_factor = 1 + (
        ((tau - lambda_) / tau) - ((1 - lambda_) / (tau - 1)) * np.log(tau)
    ) / (
        ((tau + 1) * (tau - lambda_)) / 2
        - ((1 - lambda_) * (tau**3 - 1)) / (3 * (tau - 1))
    )
    expected_roll_forcing_factor = (1 / np.pi**2) * (
        (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
        + (np.pi * (tau**2 + 1) ** 2 / (tau**2 * (tau - 1) ** 2))
        * np.arcsin((tau**2 - 1) / (tau**2 + 1))
        - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
        + ((tau**2 + 1) ** 2 / (tau**2 * (tau - 1) ** 2))
        * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
        - (4 * (tau + 1)) / (tau * (tau - 1)) * np.arcsin((tau**2 - 1) / (tau**2 + 1))
        + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
    )

    assert isinstance(geometry, _TrapezoidalGeometry)
    assert owner.Af == pytest.approx(expected_area)
    assert owner.AR == pytest.approx(expected_aspect_ratio)
    assert owner.gamma_c == pytest.approx(expected_gamma_c)
    assert owner.Yma == pytest.approx(expected_mid_aerodynamic_span)
    assert owner.roll_geometrical_constant == pytest.approx(expected_roll_constant)
    assert owner.tau == pytest.approx(tau)
    assert owner.lift_interference_factor == pytest.approx(
        expected_lift_interference_factor
    )
    assert owner.roll_damping_interference_factor == pytest.approx(
        expected_roll_damping_factor
    )
    assert owner.roll_forcing_interference_factor == pytest.approx(
        expected_roll_forcing_factor
    )


def test_trapezoidal_geometry_evaluate_shape_sets_expected_points(
    calisto_trapezoidal_fin,
):
    """Ensure trapezoidal geometry shape points match the configured sweep."""
    # Arrange
    geometry = calisto_trapezoidal_fin.geometry

    # Act
    geometry.evaluate_shape()

    # Assert
    np.testing.assert_allclose(
        calisto_trapezoidal_fin.shape_vec[0],
        np.array([0.0, 0.08, 0.12, 0.12]),
    )
    np.testing.assert_allclose(
        calisto_trapezoidal_fin.shape_vec[1],
        np.array([0.0, 0.1, 0.1, 0.0]),
    )


def test_trapezoidal_geometry_get_data_returns_inputs_and_outputs(
    calisto_trapezoidal_fin,
):
    """Ensure trapezoidal geometry serialization includes optional outputs."""
    # Arrange
    geometry = calisto_trapezoidal_fin.geometry

    # Act
    geometry.evaluate_geometrical_parameters()
    data_without_outputs = geometry.get_data()
    data_with_outputs = geometry.get_data(include_outputs=True)

    # Assert
    assert data_without_outputs["tip_chord"] == pytest.approx(
        calisto_trapezoidal_fin.tip_chord
    )
    assert data_without_outputs["sweep_length"] == pytest.approx(
        calisto_trapezoidal_fin.sweep_length
    )
    assert data_without_outputs["sweep_angle"] is None
    assert set(data_with_outputs) >= {
        "tip_chord",
        "sweep_length",
        "sweep_angle",
        "shape_vec",
        "Af",
        "AR",
        "gamma_c",
        "Yma",
        "roll_geometrical_constant",
        "tau",
        "lift_interference_factor",
        "roll_damping_interference_factor",
        "roll_forcing_interference_factor",
    }
    np.testing.assert_allclose(
        data_with_outputs["shape_vec"][0], calisto_trapezoidal_fin.shape_vec[0]
    )


def test_elliptical_geometry_evaluate_geometrical_parameters(
    calisto_elliptical_fin,
):
    """Ensure elliptical geometry populates the derived fin parameters."""
    # Arrange
    geometry = calisto_elliptical_fin.geometry

    # Act
    geometry.evaluate_geometrical_parameters()

    # Assert
    owner = calisto_elliptical_fin
    expected_area = np.pi * owner.root_chord / 2 * owner.span / 2
    expected_aspect_ratio = 2 * owner.span**2 / expected_area
    expected_mid_aerodynamic_span = (
        owner.span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
    )
    expected_roll_constant = (
        owner.root_chord
        * owner.span
        * (
            3 * np.pi * owner.span**2
            + 32 * owner.rocket_radius * owner.span
            + 12 * np.pi * owner.rocket_radius**2
        )
        / 48
    )
    tau = (owner.span + owner.rocket_radius) / owner.rocket_radius
    expected_lift_interference_factor = 1 + 1 / tau

    assert isinstance(geometry, _EllipticalGeometry)
    assert owner.Af == pytest.approx(expected_area)
    assert owner.AR == pytest.approx(expected_aspect_ratio)
    assert owner.gamma_c == pytest.approx(0)
    assert owner.Yma == pytest.approx(expected_mid_aerodynamic_span)
    assert owner.roll_geometrical_constant == pytest.approx(expected_roll_constant)
    assert owner.tau == pytest.approx(tau)
    assert owner.lift_interference_factor == pytest.approx(
        expected_lift_interference_factor
    )
    assert owner.roll_damping_interference_factor > 0
    assert owner.roll_forcing_interference_factor > 0


def test_elliptical_geometry_evaluate_shape_sets_expected_points(
    calisto_elliptical_fin,
):
    """Ensure elliptical geometry evaluates the expected semi-ellipse shape."""
    # Arrange
    geometry = calisto_elliptical_fin.geometry

    # Act
    geometry.evaluate_shape()

    # Assert
    angles = np.arange(0, 180, 5)
    expected_x = calisto_elliptical_fin.root_chord / 2 + (
        calisto_elliptical_fin.root_chord / 2 * np.cos(np.radians(angles))
    )
    expected_y = calisto_elliptical_fin.span * np.sin(np.radians(angles))
    np.testing.assert_allclose(calisto_elliptical_fin.shape_vec[0], expected_x)
    np.testing.assert_allclose(calisto_elliptical_fin.shape_vec[1], expected_y)


def test_elliptical_geometry_get_data_returns_expected_outputs(
    calisto_elliptical_fin,
):
    """Ensure elliptical geometry serialization includes optional outputs."""
    # Arrange
    geometry = calisto_elliptical_fin.geometry

    # Act
    geometry.evaluate_geometrical_parameters()
    data_without_outputs = geometry.get_data()
    data_with_outputs = geometry.get_data(include_outputs=True)

    # Assert
    assert data_without_outputs == {}
    assert data_with_outputs["Af"] == pytest.approx(calisto_elliptical_fin.Af)
    assert data_with_outputs["AR"] == pytest.approx(calisto_elliptical_fin.AR)
    assert data_with_outputs["gamma_c"] == pytest.approx(calisto_elliptical_fin.gamma_c)
    assert data_with_outputs["Yma"] == pytest.approx(calisto_elliptical_fin.Yma)
    assert data_with_outputs["roll_geometrical_constant"] == pytest.approx(
        calisto_elliptical_fin.roll_geometrical_constant
    )
    assert data_with_outputs["tau"] == pytest.approx(calisto_elliptical_fin.tau)


def test_free_form_geometry_infer_dimensions_warns_on_jagged_shape():
    """Ensure jagged free-form fins emit a warning while dimensions are inferred."""
    # Arrange
    shape_points = [(0, 0), (0.05, 0.1), (0.06, 0.05), (0.09, 0.07), (0.12, 0)]

    # Act
    with pytest.warns(UserWarning, match="Jagged fin shape detected"):
        root_chord, span = _FreeFormGeometry.infer_dimensions(shape_points)

    # Assert
    assert root_chord == pytest.approx(0.12)
    assert span == pytest.approx(0.1)


def test_free_form_geometry_evaluate_geometrical_parameters(calisto_free_form_fin):
    """Ensure free-form geometry populates the derived fin parameters."""
    # Arrange
    geometry = calisto_free_form_fin.geometry

    # Act
    geometry.evaluate_geometrical_parameters()

    # Assert
    owner = calisto_free_form_fin
    assert isinstance(geometry, _FreeFormGeometry)
    assert owner.Af == pytest.approx(0.008)
    assert owner.AR == pytest.approx(2.5)
    assert owner.gamma_c > 0
    assert owner.Yma > 0
    assert owner.mac_length > 0
    assert owner.mac_lead >= 0
    assert owner.roll_geometrical_constant > 0
    assert owner.tau > 0
    assert owner.lift_interference_factor > 1
    assert owner.roll_damping_interference_factor > 1
    assert owner.roll_forcing_interference_factor > 0


def test_free_form_geometry_evaluate_shape_sets_expected_points(
    calisto_free_form_fin,
):
    """Ensure free-form geometry exposes the configured shape points."""
    # Arrange
    geometry = calisto_free_form_fin.geometry

    # Act
    geometry.evaluate_shape()

    # Assert
    np.testing.assert_allclose(
        calisto_free_form_fin.shape_vec[0],
        np.array([0.0, 0.08, 0.12, 0.12]),
    )
    np.testing.assert_allclose(
        calisto_free_form_fin.shape_vec[1],
        np.array([0.0, 0.1, 0.1, 0.0]),
    )


def test_free_form_geometry_get_data_returns_expected_outputs(
    calisto_free_form_fin,
):
    """Ensure free-form geometry serialization includes optional outputs."""
    # Arrange
    geometry = calisto_free_form_fin.geometry

    # Act
    geometry.evaluate_geometrical_parameters()
    data_without_outputs = geometry.get_data()
    data_with_outputs = geometry.get_data(include_outputs=True)

    # Assert
    assert data_without_outputs == {
        "shape_points": [(0, 0), (0.08, 0.1), (0.12, 0.1), (0.12, 0)],
    }
    assert data_with_outputs["shape_points"] == calisto_free_form_fin.shape_points
    assert data_with_outputs["Af"] == pytest.approx(calisto_free_form_fin.Af)
    assert data_with_outputs["AR"] == pytest.approx(calisto_free_form_fin.AR)
    assert data_with_outputs["gamma_c"] == pytest.approx(calisto_free_form_fin.gamma_c)
    assert data_with_outputs["Yma"] == pytest.approx(calisto_free_form_fin.Yma)
    assert data_with_outputs["mac_length"] == pytest.approx(
        calisto_free_form_fin.mac_length
    )
    assert data_with_outputs["mac_lead"] == pytest.approx(
        calisto_free_form_fin.mac_lead
    )
