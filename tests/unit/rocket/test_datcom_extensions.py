"""DATCOM spanwise geometry, section definitions and asymmetric layout behavior."""

import json
import re
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rocketpy import DatcomFins, DatcomFreeFormFins, Rocket
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.datcom import (
    DatcomResult,
    DatcomSurface,
    datcom_decks,
    parse_datcom_output,
)


def _values(deck, variable):
    text = re.search(rf"\b{variable}\s*=\s*([\dEe+.,\s-]+)", deck)[1]
    return [float(v) for v in re.split(r"[\s,]+", text.strip(" ,\n"))]


def test_datcom_default_alpha_dense_near_zero():
    """Default incidence spans ±20 degrees with most points within ±2 degrees."""
    # Arrange
    from rocketpy.rocket.datcom import _flight_conditions

    # Act
    _, alpha, _, _ = _flight_conditions({})
    # Assert
    assert alpha[[0, -1]] == pytest.approx([-20, 20])
    assert len(alpha) <= 20
    assert np.count_nonzero(abs(alpha) <= 2) > len(alpha) / 2
    assert np.count_nonzero(alpha == 0) == 1
    assert np.diff(alpha).min() > 0


@pytest.mark.parametrize("profile,card", [("conical", "CONE"), ("ogive", "OGIVE")])
def test_datcom_tail_kind_export(datcom_rocket, profile, card):
    """The assembled tail's actual profile selects the DATCOM afterbody card."""
    # Arrange
    from rocketpy import Tail

    tail = next(
        s for s, _, _ in datcom_rocket.aerodynamic_surfaces if isinstance(s, Tail)
    )
    if profile == "ogive":
        datcom_rocket.aerodynamic_surfaces.remove(tail)
        tail = datcom_rocket.add_tail(
            tail.top_radius, tail.bottom_radius, tail.length, -1.2, kind="ogive"
        )
    # Act
    decks, _, _, _ = datcom_decks(datcom_rocket)
    # Assert
    assert all(f"TAFT = {card}" in deck for deck in decks.values())


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_datcom_fins_panel_count_and_triangular_tip(count, tmp_path):
    """DATCOM accepts 1..8 panels and a zero tip chord without inherited loads."""
    # Arrange / Act
    fins = DatcomFins(count, 0.3, 0, 0.1, 0.06, path=tmp_path / "datcom")
    # Assert
    assert fins.n == count
    assert fins.tip_chord == 0
    assert fins.panel_angles == pytest.approx(np.arange(count) * 360 / count)
    assert np.isfinite(fins.shape_vec).all()


def test_datcom_multi_station_geometry_and_serialization(tmp_path):
    """The drawn multi-segment outline and serialized section stations agree."""
    # Arrange
    fins = DatcomFreeFormFins(
        4,
        [0, 0.06, 0.12],
        [0.3, 0.22, 0.04],
        [0, 0.02, 0.1],
        0.06,
        path=tmp_path / "datcom",
        thickness_to_chord=[0.02, 0.025, 0.03],
        panel_angles=[10, 100, 190, 280],
        dihedral=[1, 2, 3, 4],
    )
    # Act
    restored = json.loads(json.dumps(fins, cls=RocketPyEncoder), cls=RocketPyDecoder)
    # Assert
    assert isinstance(restored, DatcomFreeFormFins)
    assert restored.to_dict() == fins.to_dict()
    assert fins.datcom_span_stations == pytest.approx([0, 0.06, 0.12])
    np.testing.assert_allclose(fins.shape_vec[0], [0, 0.02, 0.1, 0.14, 0.24, 0.3])
    assert fins.upper_thickness_to_chord == pytest.approx([0.01, 0.0125, 0.015])


@pytest.mark.parametrize(
    "section,options",
    [
        ("NACA", {"naca": "0012"}),
        ("NACA", {"naca": "6-64-005"}),
        (
            "USER",
            {
                "airfoil_coordinates": {
                    "x": [0, 0.5, 1],
                    "upper": [0, 0.03, 0],
                    "lower": [0, -0.02, 0],
                }
            },
        ),
        (
            "USER",
            {
                "airfoil_coordinates": {
                    "x": [0, 0.5, 1],
                    "mean": [0, 0.005, 0],
                    "thickness": [0, 0.05, 0],
                }
            },
        ),
    ],
)
def test_datcom_airfoil_sections_generate_decks(section, options, datcom_rocket):
    """NACA control cards and USER coordinate namelists use the requested section."""
    # Arrange
    old = datcom_rocket.fins[0]
    datcom_rocket.aerodynamic_surfaces.remove(old)
    datcom_rocket.add_datcom_trapezoidal_fins(
        4,
        0.25,
        0.1,
        0.1,
        -0.9,
        path=old.datcom_path,
        section_type=section,
        datcom_options=old.datcom_options,
        **options,
    )
    # Act
    deck = datcom_decks(datcom_rocket)[0][True]
    # Assert
    assert f"SECTYP = {section}" in deck
    if section == "NACA":
        assert "NACA-1-" in deck
        assert "ZUPPER" not in deck
    else:
        assert _values(deck, "XCORD") == pytest.approx([0, 0.5, 1])
        assert ("YUPPER" in deck) == ("upper" in options["airfoil_coordinates"])


def test_datcom_asymmetric_hex_and_cant_sign(datcom_rocket):
    """Separate upper/lower geometry and RocketPy-positive cant survive the deck."""
    # Arrange
    old = datcom_rocket.fins[0]
    datcom_rocket.aerodynamic_surfaces.remove(old)
    fins = datcom_rocket.add_datcom_hex_fins(
        4,
        0.25,
        0.1,
        0.1,
        -0.9,
        path=old.datcom_path,
        datcom_options=old.datcom_options,
        upper_thickness_to_chord=[0.01, 0.02],
        lower_thickness_to_chord=[0, 0.015],
        lower_max_thickness_position=0.2,
        lower_flat_length=0.4,
        cant_angle=0.5,
    )
    # Act
    deck = datcom_decks(datcom_rocket)[0][True]
    # Assert
    assert _values(deck, "ZUPPER") == pytest.approx([0.01, 0.02])
    assert _values(deck, "ZLOWER") == pytest.approx([0, 0.015])
    assert _values(deck, "LMAXL") == pytest.approx([0.2, 0.2])
    assert _values(deck, "DELTA1") == pytest.approx([-0.5] * 4)
    assert fins.thickness_to_chord == pytest.approx([0.01, 0.035])
    assert deck.count("NEXT CASE") == 3


def test_datcom_fin_sets_sorted_with_matching_naca_and_deflection(datcom_rocket):
    """Sorting moves NACA cards and DELTA arrays along with their fin geometry."""
    # Arrange
    old = datcom_rocket.fins[0]
    datcom_rocket.add_datcom_trapezoidal_fins(
        4,
        0.12,
        0.06,
        0.06,
        0.4,
        path=old.datcom_path,
        datcom_options=old.datcom_options,
        section_type="NACA",
        naca="0012",
        incidence=2,
    )
    # Act
    deck = datcom_decks(datcom_rocket)[0][True]
    first = deck.split("$FINSET1", 1)[1].split("$FINSET2", 1)[0]
    # Assert
    assert _values(first, "XLE")[0] == pytest.approx(0.8)
    assert "NACA-1-4-0012" in deck
    assert _values(deck, "DELTA1") == pytest.approx([2] * 4)
    assert _values(deck, "DELTA2") == pytest.approx([0] * 4)


def test_datcom_fin_sets_overlap_rejected(datcom_rocket):
    """Axially overlapping planform envelopes fail before executing DATCOM."""
    # Arrange
    old = datcom_rocket.fins[0]
    datcom_rocket.add_datcom_hex_fins(
        4, 0.25, 0.1, 0.1, -0.8, path=old.datcom_path, datcom_options=old.datcom_options
    )
    # Act / Assert
    with pytest.raises(ValueError, match="overlap"):
        datcom_decks(datcom_rocket)


@pytest.mark.parametrize("profile", ["conical", "ogive"])
def test_datcom_fins_automatic_boattail_attachment(datcom_rocket, profile):
    """SSPAN=0 attaches to the interpolated body radius at root mid-chord."""
    # Arrange
    from rocketpy import Tail

    tail = next(
        s for s, _, _ in datcom_rocket.aerodynamic_surfaces if isinstance(s, Tail)
    )
    if profile == "ogive":
        datcom_rocket.aerodynamic_surfaces.remove(tail)
        tail = datcom_rocket.add_tail(
            tail.top_radius, tail.bottom_radius, tail.length, -1.2, kind="ogive"
        )
    old = datcom_rocket.fins[0]
    datcom_rocket.aerodynamic_surfaces.remove(old)
    fins = datcom_rocket.add_datcom_arc_fins(
        4,
        0.04,
        0.02,
        0.05,
        -1.21,
        path=old.datcom_path,
        datcom_options=old.datcom_options,
    )
    # Act
    deck = datcom_decks(datcom_rocket)[0][True]
    # Assert
    assert _values(deck, "SSPAN") == pytest.approx([0, 0.05])
    expected = (
        float(tail.radius_at(0.03))
        if profile == "ogive"
        else 0.0635 + 0.3 * (0.045 - 0.0635)
    )
    assert fins._datcom_attachment_radius == pytest.approx(expected)


def test_datcom_projection_respects_panel_angles_and_dihedral(tmp_path):
    """The rocket drawing projects actual PHIF/GAM positions instead of equal spacing."""
    # Arrange
    fins = DatcomFins(
        2,
        0.3,
        0.1,
        0.1,
        0.06,
        path=tmp_path / "datcom",
        panel_angles=[0, 90],
        dihedral=[10, 0],
    )
    fig, ax = plt.subplots()
    records = []
    # Act
    fins.draw_on_rocket(ax, 0, records, {"fins": "black", "line_width": 1}, "xz", 1)
    # Assert
    assert ax.lines[0].get_ydata()[1] == pytest.approx(0.1 * np.sin(np.radians(10)))
    assert ax.lines[1].get_ydata()[0] == pytest.approx(0.06)
    assert len(records) == 1
    plt.close(fig)


@pytest.mark.parametrize(
    "options",
    [
        {"section_type": "NACA"},
        {"section_type": "NACA", "naca": "0012\nDAMP"},
        {
            "section_type": "USER",
            "airfoil_coordinates": {
                "x": [0, 0.5, 0.4],
                "upper": [0, 0.1, 0],
                "lower": [0, 0, 0],
            },
        },
        {"panel_angles": [0, 0, 90, 180]},
        {"dihedral": 90},
        {"incidence": [0, 1]},
        {"incidence": 1, "cant_angle": 1},
    ],
)
def test_datcom_invalid_section_or_panel_definition(options, tmp_path):
    """Unsupported or inconsistent definitions are rejected without a native run."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        DatcomFins(4, 0.3, 0.1, 0.1, 0.06, path=tmp_path / "datcom", **options)


def test_datcom_full_beta_interpolation_does_not_add_derivatives_twice(
    datcom_rocket, datcom_table_factory
):
    """Full beta data already contains the side force and must bypass CYB expansion."""
    # Arrange
    tables = {b: datcom_table_factory({"CY": b, "CYB": 100}) for b in [-5, 0, 5]}
    zero = tables[0]
    grid = DatcomResult(
        tuple(zero),
        zero.body,
        zero.derivatives,
        zero.damping,
        beta=[-5, 0, 5],
        sideslip=tables,
    )
    model = DatcomSurface(datcom_rocket, {True: grid, False: grid}, {True: 0, False: 0})
    velocity = Vector([100 * np.tan(np.radians(2.5)), 0, 100])
    speed = abs(velocity)
    # Act
    loads = model.compute_forces_and_moments_at_time(
        0.5, 0, -velocity, speed, 0.4, 1.2, Vector.zeros(), Vector.zeros(), 0
    )
    # Assert
    assert loads[0] == pytest.approx(0.5 * 1.2 * speed**2 * model.reference_area * 2.5)


def test_datcom_separate_power_altitude_schedules(datcom_rocket):
    """Powered and coast decks retain different altitude schedules at the same Mach."""
    # Arrange
    datcom_rocket.fins[0].datcom_options["altitude"] = {
        "power_on": [100, 200],
        "power_off": [900, 500],
    }
    # Act
    decks = datcom_decks(datcom_rocket)[0]
    # Assert
    assert _values(decks[True], "ALT") == pytest.approx([100, 200])
    assert _values(decks[False], "ALT") == pytest.approx([900, 500])


def test_datcom_incidence_rate_uses_body_projected_alpha(
    datcom_rocket, datcom_table_factory
):
    """Local alpha-dot follows atan(v_down/v_forward), even with lateral velocity."""
    # Arrange
    from rocketpy import Environment, Flight

    rocket = datcom_rocket
    table = datcom_table_factory()
    model = DatcomSurface(
        rocket,
        {True: table, False: table},
        {
            True: rocket.center_of_dry_mass_position,
            False: rocket.center_of_dry_mass_position,
        },
    )
    rocket._datcom_model = model
    flight = object.__new__(Flight)
    flight.rocket, flight.env = rocket, Environment()
    velocity = Vector([7, 8, 100])
    omega = Vector([0.2, 0.3, 0.1])
    acceleration = Vector([1, 2, 3])
    state = [0, 0, 1000, *velocity, 1, 0, 0, 0, *omega]
    derivative = [*velocity, *acceleration, 0, 0, 0, 0, 0, 0, 0]
    body_acceleration = acceleration - (omega ^ velocity)
    expected = (velocity.z * body_acceleration.y - velocity.y * body_acceleration.z) / (
        velocity.z**2 + velocity.y**2
    )
    # Act
    rate = flight._datcom_incidence_rate(0.5, state, derivative)
    # Assert
    assert rate == pytest.approx(expected, rel=1e-12)


def test_datcom_zero_normal_slope_has_undefined_cp(datcom_rocket, datcom_table_factory):
    """A zero normal-force slope has no CP and must not fabricate static stability."""
    # Arrange
    table = datcom_table_factory({"CNA": 0, "CMA": 0})
    model = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    # Act
    cp = model.small_angle_cp(0.4)
    datcom_rocket._datcom_model = model
    datcom_rocket.evaluate_center_of_pressure()
    # Assert
    assert np.isnan(cp)
    assert np.isnan(datcom_rocket.static_margin(0))


def test_datcom_boattail_fin_drawing_preserves_body_anchors(datcom_rocket):
    """Fins on a boattail remain overlays and do not create cylindrical body segments."""
    # Arrange
    old = datcom_rocket.fins[0]
    datcom_rocket.aerodynamic_surfaces.remove(old)
    fins = datcom_rocket.add_datcom_arc_fins(
        4,
        0.04,
        0.02,
        0.05,
        -1.21,
        path=old.datcom_path,
        datcom_options=old.datcom_options,
    )
    datcom_decks(datcom_rocket)
    nose, tail = datcom_rocket.nosecones[0], datcom_rocket.tails[0]
    records = [(nose, 0.65, 0.0635, 0.65), (tail, -1.2, 0.045, -1.3)]
    original = list(records)
    fig, ax = plt.subplots()
    # Act
    fins.draw_on_rocket(ax, -1.21, records, {"fins": "black", "line_width": 1}, "xz", 1)
    # Assert
    assert records == original
    assert len(ax.lines) == 4
    plt.close(fig)
