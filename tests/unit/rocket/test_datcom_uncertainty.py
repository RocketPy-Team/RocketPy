"""Paper-table provenance, coefficient scenarios and honest scalar envelopes."""

import numpy as np
import pytest

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.datcom import DatcomSurface
from rocketpy.rocket.datcom_uncertainty import DatcomUncertaintySurface, Wu2019Errors
from rocketpy.simulation.datcom_uncertainty import DatcomUncertaintyReport


@pytest.mark.parametrize(
    "mach,alpha,errors",
    [
        (0.8, -6, (17.727, 20.943, -6.56)),
        (1.2, -2, (-48.274, -25.887, -41.49)),
        (3.02, 6, (-17.933, -2.769, -26.82)),
    ],
)
def test_wu2019_native_relative_denominator(mach, alpha, errors):
    """Published percentages refer to measurements, not to native predictions."""
    # Arrange
    profile = Wu2019Errors()
    expected = np.abs(np.asarray(errors) / (100 + np.asarray(errors)))
    # Act
    actual = profile.fractions(mach, alpha)
    # Assert
    assert actual == pytest.approx(expected)
    assert max(actual) < 1


def test_wu2019_interpolation_and_boundary_assumption():
    """Interpolate fractional magnitudes; hold, rather than extrapolate, edges."""
    # Arrange
    profile = Wu2019Errors()
    low = np.asarray(profile.fractions(0.8, 0))
    high = np.asarray(profile.fractions(0.8, 2))
    # Act
    middle = profile.fractions(0.8, 1)
    outside = profile.fractions(0.1, -20)
    # Assert
    assert middle == pytest.approx((low + high) / 2)
    assert outside == pytest.approx(profile.fractions(0.8, -6))
    assert profile.fractions(0.95, 0)[2] == pytest.approx(81.47 / 181.47)


def test_wu2019_zero_alpha_uses_other_angles_mean():
    """Stability avoids the paper's near-zero normal-force denominator artifact."""
    # Arrange
    profile = Wu2019Errors()
    percentages = profile.cn_error_percent[0, profile.alpha != 0]
    expected = np.mean(np.abs(percentages / (100 + percentages)))
    # Act
    _, normal, moment = profile.fractions(0.8, 0)
    # Assert
    assert normal == pytest.approx(expected)
    assert normal != pytest.approx(645.672 / 745.672)
    assert moment == pytest.approx(6.56 / 93.44)


@pytest.mark.parametrize(
    "mach,alpha,expected",
    [
        (0.6, 0, 0.2),
        (0.8, 0, 0.2),
        (1.2, -2, 48.274 / 51.726),
        (3.1, 0, 0.2),
        (1.2, 7, 0.2),
    ],
)
def test_datcom_drag_floor_and_larger_published_errors(
    mach, alpha, expected, datcom_rocket, datcom_table_factory
):
    """CD has a 20% floor; larger Wu discrepancies apply only inside its data."""
    # Arrange
    table = datcom_table_factory()
    native = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    model = DatcomUncertaintySurface.from_model(native, 1, nose_position=1.2)
    # Act / Assert
    assert model.drag_fraction(mach, alpha) == pytest.approx(expected)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("beta", [0, 3])
def test_datcom_scenarios_scale_drag_preserve_lift_and_native_tables(
    sign, beta, datcom_rocket, datcom_table_factory
):
    """At finite alpha/beta the airflow-parallel force changes by exactly 20%."""
    # Arrange
    table = datcom_table_factory(
        {
            "CA": 0.5,
            "CN": 0.2,
            "CYB": -0.02,
            "CNA": 0.1,
            "CMA": -0.2,
            "CNQ": 2,
            "CMQ": -4,
        }
    )
    native = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0.2, False: 0}
    )
    model = DatcomUncertaintySurface.from_model(native, sign, nose_position=1.2)
    alpha, beta_rad = np.radians([2, beta])
    before = native.coefficients(False, 0.6, alpha, beta_rad).copy()
    direction = np.array([np.tan(beta_rad), np.tan(alpha), 1.0])
    direction /= np.linalg.norm(direction)
    original_force = np.array([before[3] + before[6] * beta, -before[0], -before[2]])
    original_drag = -original_force @ direction
    # Act
    values = model.coefficients(False, 0.6, alpha, beta_rad)
    force = np.array([values[3] + values[6] * beta, -values[0], -values[2]])
    drag = -force @ direction
    # Assert
    assert drag == pytest.approx(original_drag * (1 + sign * 0.2))
    assert values[-1] == pytest.approx(drag)
    np.testing.assert_allclose(
        force + drag * direction, original_force + original_drag * direction, atol=1e-12
    )
    np.testing.assert_array_equal(values[6:20], before[6:20])
    np.testing.assert_array_equal(
        native.coefficients(False, 0.6, alpha, beta_rad), before
    )
    assert values[5] == before[5]  # roll moment unchanged


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("powered", [True, False])
def test_datcom_scenario_cp_matches_varied_slopes(
    direction, sign, powered, datcom_rocket, datcom_table_factory
):
    """A 2% nose-to-CP shift agrees with the force/moment slopes in both frames."""
    # Arrange
    table = datcom_table_factory({"CA": 0.5, "CNA": 0.1, "CMA": -0.2})
    table.body["CN"][:] = 0.05 + 0.1 * table[5]
    table.body["CM"][:] = 0.03 - 0.2 * table[5]
    native = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0.2 * direction, False: 0}
    )
    native.csys = direction
    nose = direction * 1.2
    model = DatcomUncertaintySurface.from_model(native, sign, nose_position=nose)
    step = 1e-4  # degrees
    cp_before = model._native_cp(powered, 0.6)
    cp_expected = cp_before - sign * 0.02 * (nose - cp_before)
    # Act
    upper = model.coefficients(powered, 0.6, np.radians(step))
    lower = model.coefficients(powered, 0.6, -np.radians(step))
    slopes = (upper - lower) / (2 * step)
    cp_from_moments = (
        native.references[powered]
        + direction * model.reference_length * slopes[1] / slopes[0]
    )
    # Assert
    assert cp_from_moments == pytest.approx(cp_expected, rel=1e-8)
    if not powered:
        assert model.small_angle_cp(0.6) == pytest.approx(cp_expected)
    assert (nose - cp_expected) / (nose - cp_before) == pytest.approx(1 + sign * 0.02)


def test_datcom_scenario_preserves_dynamic_load_increments(
    datcom_rocket, datcom_table_factory
):
    """The static drag/CP scenario does not rescale native DAMP contributions."""
    # Arrange
    table = datcom_table_factory(
        {"CA": 0.5, "CNA": 0.1, "CMA": -0.2, "CNQ": 2, "CMQ": -4, "CAQ": 1, "CYR": 3}
    )
    native = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    model = DatcomUncertaintySurface.from_model(native, 1, nose_position=1.2)

    def loads(surface, omega):
        return np.array(
            surface.compute_forces_and_moments_at_time(
                0, 0, Vector([0, -5, -100]), 100, 0.6, 1.2, Vector.zeros(), omega, 1e6
            )
        )

    # Act
    rates = Vector([0.1, 0.2, 0.3])
    increment = loads(model, rates) - loads(model, Vector.zeros())
    expected = loads(native, rates) - loads(native, Vector.zeros())
    # Assert
    np.testing.assert_allclose(increment, expected, rtol=1e-10, atol=1e-10)


def test_datcom_scenario_undefined_cp(datcom_rocket, datcom_table_factory):
    """Unknown CP stays unavailable while the drag scenario remains finite."""
    # Arrange
    table = datcom_table_factory({"CA": 0.5})
    native = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    model = DatcomUncertaintySurface.from_model(native, 1, nose_position=1.2)
    # Act / Assert
    assert np.isnan(model.small_angle_cp(0.6))
    assert model.coefficients(False, 0.6, 0)[-1] == pytest.approx(0.6)


def test_datcom_report_asymmetric_and_incomplete_envelope():
    """± is the largest nominal departure; an absent case is never zero error."""
    # Arrange
    report = DatcomUncertaintyReport(
        cases={
            "nominal": {"apogee": 1000},
            "upper": {"apogee": 900},
            "lower": {"apogee": 1200},
        }
    )
    # Act / Assert
    assert report.uncertainty("apogee") == pytest.approx(200)
    assert report.uncertainty("unavailable") is None
    report.cases.pop("lower")
    assert report.uncertainty("apogee") is None
