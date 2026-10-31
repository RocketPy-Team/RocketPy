"""DATCOM geometry, interpolation, units and reference-frame regressions."""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import DatcomFins, EmptyMotor, Environment, Flight, Function, Rocket
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.datcom import (
    DAMPING_KEYS,
    DatcomSurface,
    datcom_decks,
    parse_datcom_output,
)


def _deck_value(deck, name):
    return float(re.search(rf"\b{name}\s*=\s*([-+\d.eE]+)", deck)[1])


def test_datcom_fin_trailing_edge_can_extend_past_base(datcom_rocket):
    """Root-centre attachment permits aft fin overhang without extending the body."""
    # Arrange
    rocket = datcom_rocket
    original_body = _deck_value(datcom_decks(rocket)[0][False], "LCENTR")
    fin, position, _ = next(
        s for s in rocket.aerodynamic_surfaces if s[0] is rocket.fins[0]
    )
    fin.root_chord = 0.45
    # Act
    decks, _, _, _ = datcom_decks(rocket)
    # Assert
    assert _deck_value(decks[False], "LCENTR") == pytest.approx(original_body)
    assert _deck_value(decks[False], "CHORD") == pytest.approx(0.45)


def test_datcom_fin_root_center_cannot_be_beyond_body(datcom_rocket):
    """A fin with its root centre behind the body has no valid attachment."""
    # Arrange
    datcom_rocket.fins[0].root_chord = 1.0
    # Act / Assert
    with pytest.raises(ValueError, match="root chord centers"):
        datcom_decks(datcom_rocket)


@pytest.mark.parametrize("mode", ["none", "nozzle", "full"])
def test_datcom_powered_base_drag_policy(datcom_rocket, mode):
    """Nozzle mode includes the real base annulus without enabling plume models."""
    # Arrange
    datcom_rocket.fins[0].datcom_options["powered_base_drag"] = mode
    # Act
    decks, _, _, _ = datcom_decks(datcom_rocket)
    # Assert
    assert _deck_value(decks[False], "DEXIT") == 0
    assert "BASE = .FALSE." in decks[True]
    if mode == "none":
        assert "DEXIT" not in decks[True]
    else:
        expected = 2 * datcom_rocket.motor.nozzle_radius if mode == "nozzle" else 0
        assert _deck_value(decks[True], "DEXIT") == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["bad", True])
def test_datcom_powered_base_drag_rejects_invalid_policy(datcom_rocket, mode):
    """Ambiguous policy values cannot silently change the base-drag model."""
    # Arrange
    datcom_rocket.fins[0].datcom_options["powered_base_drag"] = mode
    # Act / Assert
    with pytest.raises(ValueError, match="powered_base_drag"):
        datcom_decks(datcom_rocket)


def test_datcom_powered_nozzle_must_fit_base(datcom_rocket):
    """A nozzle outside the body cannot produce a valid annular base area."""
    # Arrange
    datcom_rocket.fins[0].datcom_options["powered_base_drag"] = "nozzle"
    datcom_rocket.motor.nozzle_radius = 1
    # Act / Assert
    with pytest.raises(ValueError, match="nozzle diameter"):
        datcom_decks(datcom_rocket)


@pytest.mark.parametrize(
    "method,section",
    [
        ("add_datcom_trapezoidal_fins", "HEX"),
        ("add_datcom_arc_fins", "ARC"),
        ("add_datcom_hex_fins", "HEX"),
    ],
)
def test_add_datcom_fins_retains_geometry(method, section, calisto_motorless, tmp_path):
    """The public methods preserve planform dimensions without executing DATCOM."""
    # Arrange
    rocket = calisto_motorless
    # Act
    with patch("rocketpy.rocket.datcom.run_datcom") as run:
        fins = getattr(rocket, method)(
            4,
            0.3,
            0.1,
            0.12,
            position=-1,
            path=tmp_path / "datcom",
            sweep_angle=30,
            thickness_to_chord=(0.02, 0.03),
        )
    # Assert
    run.assert_not_called()
    assert isinstance(fins, DatcomFins)
    assert fins.section_type == section
    assert fins.sweep_length == pytest.approx(0.12 * np.tan(np.radians(30)))
    assert fins.thickness_to_chord == pytest.approx((0.02, 0.03))
    assert rocket.aerodynamic_surfaces[0].component is fins
    assert np.isnan(rocket.cp_position(0))


@pytest.mark.parametrize("orientation", ["tail_to_nose", "nose_to_tail"])
def test_datcom_decks_cg_geometry_and_base_drag(orientation, cesaroni_m1670, tmp_path):
    """Both rocket orientations produce the same nose-to-tail DATCOM geometry."""
    # Arrange
    sign = 1 if orientation == "tail_to_nose" else -1
    rocket = Rocket(
        0.0635, 14, (6, 6, 0.04), 0.9, 0.8, 0, coordinate_system_orientation=orientation
    )
    rocket.add_motor(cesaroni_m1670, -1.373 * sign)
    rocket.add_nose(0.55, "von karman", 1.2 * sign)
    rocket.add_datcom_hex_fins(
        4, 0.25, 0.1, 0.1, -0.9 * sign, sweep_length=0.1, path=tmp_path / "datcom"
    )
    rocket.add_tail(0.0635, 0.045, 0.1, -1.2 * sign)
    # Act
    decks, references, _, _ = datcom_decks(rocket)
    # Assert
    for powered, deck in decks.items():
        assert _deck_value(deck, "LNOSE") == pytest.approx(0.55)
        assert _deck_value(deck, "LCENTR") == pytest.approx(1.85)
        assert _deck_value(deck, "XLE") == pytest.approx(2.1)
        assert _deck_value(deck, "XCG") == pytest.approx(
            (1.2 * sign - references[powered]) * sign
        )
        assert "DAMP\n" in deck
        assert "PROTUB" not in deck
        assert "BASE = .FALSE." in deck
        assert max(map(len, deck.splitlines())) < 80
    assert "DEXIT" not in decks[True]
    assert _deck_value(decks[False], "DEXIT") == 0
    assert references[True] == pytest.approx(
        rocket.center_of_mass(rocket.motor.burn_start_time)
    )
    assert references[False] == pytest.approx(rocket.center_of_dry_mass_position)
    assert references[True] != pytest.approx(references[False])


def test_datcom_decks_filled_tank_cg(calisto_liquid_modded, tmp_path):
    """The ignition reference follows existing liquid tank masses and positions."""
    # Arrange
    rocket = calisto_liquid_modded
    rocket.add_nose(0.55, "ogive", 3)
    rocket.add_datcom_arc_fins(4, 0.25, 0.1, 0.1, -2, path=tmp_path / "datcom")
    rocket.add_tail(rocket.radius, 0.04, 0.1, -3)
    time = rocket.motor.burn_start_time
    mass = rocket.motor.propellant_mass(time)
    expected = (
        rocket.dry_mass * rocket.center_of_dry_mass_position
        + mass * rocket.center_of_propellant_position(time)
    ) / (rocket.dry_mass + mass)
    # Act
    _, references, _, _ = datcom_decks(rocket)
    # Assert
    assert references[True] == pytest.approx(expected, rel=1e-10)
    assert references[True] != pytest.approx(references[False])


def test_prepare_datcom_disables_component_models_and_rebuilds(
    datcom_rocket, datcom_table_factory
):
    """One complete model replaces component loads and geometry edits regenerate cases."""
    # Arrange
    rocket = datcom_rocket
    original = tuple(rocket.aerodynamic_surfaces)
    # Act
    with patch(
        "rocketpy.rocket.datcom.run_datcom", return_value=datcom_table_factory()
    ) as run:
        model = rocket.prepare_datcom()
        same = rocket.prepare_datcom()
        assert run.call_count == 2
        original[1].component.tip_chord = 0.11
        rebuilt = rocket.prepare_datcom()
    # Assert
    assert same is model
    assert rebuilt is not model
    assert run.call_count == 4
    assert tuple(rocket.aerodynamic_surfaces) == original
    assert [s for s, _, _ in rocket._flight_aerodynamic_surfaces] == [rebuilt]
    assert model not in rocket.surfaces_cp_to_cdm
    assert rocket.power_on_drag(0.3) == pytest.approx(0.4)


def test_prepare_datcom_failed_run_is_atomic(datcom_rocket, datcom_table_factory):
    """A failed coast run never installs the powered half of a database."""
    # Arrange
    rocket = datcom_rocket
    original_drag = rocket.power_on_drag_7d
    # Act / Assert
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        side_effect=[datcom_table_factory(), RuntimeError("failed")],
    ):
        with pytest.raises(RuntimeError, match="failed"):
            rocket.prepare_datcom()
    assert rocket._datcom_model is None
    assert rocket.power_on_drag_7d is original_drag


@pytest.mark.parametrize(
    "options,message",
    [
        ({"mach": [0.5, 0.2]}, "increasing"),
        ({"alpha": [0, 1, 2]}, "negative"),
        ({"altitude": [0]}, "one value per Mach"),
        ({"timeout": 0}, "timeout"),
        ({"unknown": 1}, "Unknown"),
    ],
)
def test_datcom_decks_invalid_conditions(options, message, datcom_rocket):
    """Bad grids fail before launching an external process."""
    # Arrange
    datcom_rocket.fins[0].datcom_options.update(options)
    # Act / Assert
    with pytest.raises(ValueError, match=message):
        datcom_decks(datcom_rocket)


@pytest.mark.parametrize(
    "field,value",
    [
        ("thickness_to_chord", 0),
        ("flat_length", 0.9),
        ("leading_edge_radius", -1),
        ("thickness_to_chord", [1, 2, 3]),
    ],
)
def test_datcom_fins_invalid_section(field, value, tmp_path):
    """Invalid or mismatched section stations are rejected."""
    # Arrange
    kwargs = {field: value}
    # Act / Assert
    with pytest.raises(ValueError):
        DatcomFins(4, 0.3, 0.1, 0.1, 0.06, path=tmp_path / "datcom", **kwargs)


def test_datcom_fins_serialization(datcom_rocket):
    """Saved rockets retain section dimensions and deferred execution settings."""
    # Arrange
    fins = datcom_rocket.fins[0]
    # Act
    restored = json.loads(json.dumps(fins, cls=RocketPyEncoder), cls=RocketPyDecoder)
    # Assert
    assert isinstance(restored, DatcomFins)
    assert restored.to_dict() == fins.to_dict()


def test_datcom_fins_direct_inclusion(datcom_rocket):
    """Adding a constructed DATCOM surface also selects the whole-vehicle model."""
    # Arrange
    fins = DatcomFins.from_dict(datcom_rocket.fins[0].to_dict())
    rocket = Rocket(0.0635, 14, (6, 6, 0.04), 0, 0, 0)
    # Act
    rocket.add_surfaces(fins, positions=-0.9)
    # Assert
    assert rocket._has_datcom_fins()
    assert rocket.fins[0] is fins


def test_datcom_prepared_rocket_serialization(datcom_rocket, datcom_table_factory):
    """Rocket export preserves inputs and geometry without embedding live model closures."""
    # Arrange
    with patch(
        "rocketpy.rocket.datcom.run_datcom", return_value=datcom_table_factory()
    ):
        datcom_rocket.prepare_datcom()
    # Act
    restored = json.loads(
        json.dumps(datcom_rocket, cls=RocketPyEncoder), cls=RocketPyDecoder
    )
    # Assert
    assert isinstance(restored, Rocket)
    assert restored._has_datcom_fins()
    assert restored._datcom_model is None
    assert restored.fins[0].to_dict() == datcom_rocket.fins[0].to_dict()


@pytest.mark.parametrize(
    "key,axis,index,sign",
    [
        ("CNQ", 0, 1, -1),
        ("CMQ", 0, 3, 1),
        ("CAQ", 0, 2, -1),
        ("CNAD", None, 1, -1),
        ("CMAD", None, 3, 1),
        ("CYR", 1, 0, 1),
        ("CLNR", 1, 4, 1),
        ("CLLR", 1, 5, 1),
        ("CYP", 2, 0, 1),
        ("CLNP", 2, 4, 1),
        ("CLLP", 2, 5, 1),
    ],
)
@pytest.mark.parametrize("direction", [-1, 1])
def test_datcom_damping_units_axes(
    key, axis, index, sign, direction, datcom_rocket, datcom_table_factory
):
    """Every DAMP channel has the correct per-degree rate scaling and body axis."""
    # Arrange
    table = datcom_table_factory({key: 2})
    model = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    omega = [0, 0, 0]
    if axis is not None:
        omega[axis] = direction
    expected = np.zeros(6)
    scale = 0.5 * 1.2 * 100**2 * model.reference_area
    if index >= 3:
        scale *= model.reference_length
    expected[index] = (
        sign * 2 * direction * np.degrees(model.reference_length / 200) * scale
    )
    # Act
    result = model.compute_forces_and_moments_at_time(
        0.5,
        direction if axis is None else 0,
        Vector([0, 0, -100]),
        100,
        0.3,
        1.2,
        Vector.zeros(),
        Vector(omega),
        1e6,
    )
    # Assert
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_datcom_interpolation_and_power_boundaries(datcom_rocket, datcom_table_factory):
    """Mach/alpha interpolate, while base-drag state switches at ignition/burnout."""
    # Arrange
    on = datcom_table_factory({"CA": 1})
    on.body["CN"][:] = [[0, 2, 4], [4, 6, 8]]
    off = datcom_table_factory({"CA": 2})
    model = DatcomSurface(datcom_rocket, {True: on, False: off}, {True: 0, False: 0})
    model.burn_start, model.burn_out = 1, 3
    # Act
    values = model.coefficients(True, 0.4, np.radians(-1))
    states = [model.is_powered(t) for t in [0, 1, 2, 3, 4]]
    # Assert
    assert values[0] == pytest.approx(3)
    assert states == [False, True, True, False, False]
    assert model.coefficients(False, 0.4, 0)[2] == 2
    assert model.coefficients(True, 5, 1)[0] == 8  # bounded envelope


@pytest.mark.parametrize("speed", [0, 1e-12, 0.5, 100, -100])
def test_datcom_low_speed_and_reverse_flow(speed, datcom_rocket, datcom_table_factory):
    """Loads remain finite near rest and axial drag opposes reverse motion."""
    # Arrange
    table = datcom_table_factory({"CA": 1, "CMQ": -2})
    model = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    # Act
    loads = model.compute_forces_and_moments_at_time(
        0.5,
        0,
        Vector([0, 0, -speed]),
        abs(speed),
        0.3,
        1.2,
        Vector.zeros(),
        Vector([1, 0, 0]),
        0,
    )
    # Assert
    assert np.isfinite(loads).all()
    assert loads[2] * speed <= 0


def test_datcom_moment_reference_transfer(datcom_rocket, datcom_table_factory):
    """Moments are transferred from the case reference to dry CG exactly once."""
    # Arrange
    table = datcom_table_factory({"CN": 1, "CM": 2})
    model = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0.5, False: 0}
    )
    cp = Vector([0, 0, 0.7])
    scale = 0.5 * 1.2 * 100**2 * model.reference_area
    # Act
    loads = model.compute_forces_and_moments_at_time(
        0.5,
        0,
        Vector([0, 0, -100]),
        100,
        0.3,
        1.2,
        cp,
        Vector.zeros(),
        0,
    )
    # Assert
    assert loads[1] == pytest.approx(-scale)
    assert loads[3] == pytest.approx(scale * (2 * model.reference_length + 0.7))


def test_datcom_axial_coefficient_is_not_projected_twice(
    datcom_rocket, datcom_table_factory
):
    """CA already acts along the body axis, including at nonzero incidence."""
    # Arrange
    table = datcom_table_factory({"CA": 0.4, "CYB": -0.1})
    model = DatcomSurface(
        datcom_rocket, {True: table, False: table}, {True: 0, False: 0}
    )
    velocity = Vector([1, 2, 100])
    speed = abs(velocity)
    scale = 0.5 * 1.2 * speed**2 * model.reference_area
    beta = np.arctan2(1, 100)
    # Act
    loads = model.compute_forces_and_moments_at_time(
        0.5,
        0,
        -velocity,
        speed,
        0.3,
        1.2,
        Vector.zeros(),
        Vector.zeros(),
        0,
    )
    # Assert
    assert loads[2] == pytest.approx(-0.4 * scale)
    assert loads[0] == pytest.approx(-0.1 * np.degrees(beta) * scale)


def test_datcom_rail_uses_body_axial_coefficient(datcom_rocket, datcom_table_factory):
    """The constrained rail phase uses CA rather than the reporting CD curve."""
    # Arrange
    rocket = datcom_rocket
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        return_value=datcom_table_factory({"CA": 0.4, "CD": 0.8}),
    ):
        rocket.prepare_datcom()
    flight = object.__new__(Flight)
    flight.rocket, flight.env = rocket, Environment()
    state = [0, 0, 1000, 0, 0, 20, 1, 0, 0, 0, 0, 0, 0]
    time = 0.5
    thrust = max(
        rocket.motor.thrust(time)
        + rocket.motor.pressure_thrust(flight.env.pressure(1000)),
        0,
    )
    force = -0.5 * flight.env.density(1000) * 20**2 * rocket.area * 0.4
    expected = (thrust + force) / rocket.total_mass(time) - flight.env.gravity(1000)
    # Act
    derivative = flight.udot_rail1(time, state)
    # Assert
    assert expected > 0
    assert derivative[5] == pytest.approx(expected)


@pytest.mark.parametrize("time", [0.2, 1.0, 2.0])
@pytest.mark.parametrize("rocket_fixture", ["calisto", "calisto_nose_to_tail"])
def test_datcom_generalized_torque_uses_current_cg(
    time, rocket_fixture, request, tmp_path, datcom_table_factory
):
    """A fixed DATCOM reference transfers to the actual CG throughout the burn."""
    # Arrange
    rocket = request.getfixturevalue(rocket_fixture)
    sign = rocket._csys
    path = tmp_path / "datcom"
    path.touch()
    rocket.add_nose(0.55, "von karman", 1.2 * sign)
    rocket.add_datcom_hex_fins(
        4,
        0.25,
        0.1,
        0.1,
        -0.9 * sign,
        path=path,
        datcom_options={"mach": [0.2, 0.6], "alpha": [-2, 0, 2]},
    )
    rocket.add_tail(rocket.radius, 0.045, 0.1, -1.2 * sign)
    table = datcom_table_factory({"CN": 1, "CM": 0.5})
    with patch("rocketpy.rocket.datcom.run_datcom", return_value=table):
        model = rocket.prepare_datcom()
    flight = object.__new__(Flight)
    flight.rocket, flight.env = rocket, Environment()
    state = [0, 0, 1000, 0, 0, 100, 1, 0, 0, 0, 0, 0, 0]
    cg = rocket.center_of_mass(time)
    offset = (model.references[model.is_powered(time)] - cg) * rocket._csys
    q_area = 0.5 * flight.env.density(1000) * 100**2 * rocket.area
    inertia = (
        rocket.I_11(time)
        - rocket.total_mass(time) * rocket.com_to_cdm_function(time) ** 2
    )
    expected = q_area * (0.5 * model.reference_length + offset) / inertia
    # Act
    derivative = flight.u_dot_generalized(time, state)
    # Assert
    assert derivative[10] == pytest.approx(expected, rel=1e-9)


def test_datcom_unsteady_coupling_is_history_independent(
    datcom_rocket, datcom_table_factory
):
    """CNAD feeds back through acceleration; repeated RHS evaluations are deterministic."""
    # Arrange
    rocket = datcom_rocket
    rocket.add_motor(EmptyMotor(), position=0)
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        return_value=datcom_table_factory({"CNAD": 1}),
    ):
        model = rocket.prepare_datcom()
    # Put both case references at dry CG to isolate the analytic CNAD feedback.
    model.references = {True: model.dry_reference, False: model.dry_reference}
    flight = object.__new__(Flight)
    flight.rocket, flight.env = rocket, Environment()
    state = [0, 0, 1000, 0, 0, 100, 1, 0, 0, 0, 0.2, 0, 0]
    time = 0.5
    q_area = 0.5 * flight.env.density(1000) * 100**2 * rocket.area
    expected = 0.2 / (
        1
        + q_area
        * np.degrees(model.reference_length / 200)
        / (rocket.total_mass(time) * 100)
    )
    # Act
    first = flight.u_dot_generalized(time, state)
    flight.u_dot_generalized(0.1, state)
    repeated = flight.u_dot_generalized(time, state)
    # Assert
    assert flight._datcom_incidence_rate(time, state, first) == pytest.approx(
        expected, rel=1e-8
    )
    np.testing.assert_allclose(first, repeated, atol=1e-12)
    assert flight._datcom_alpha_dot == 0
    assert flight._datcom_evaluating is False


@pytest.mark.parametrize(
    "override,expected_cd", [(None, 0.4), (False, 0.6), (True, 0.2)]
)
def test_datcom_no_duplicate_drag_or_component_loads(
    override, expected_cd, datcom_rocket, datcom_table_factory
):
    """Full-vehicle axial force is applied once despite nonzero original rocket drag."""
    # Arrange
    rocket = datcom_rocket
    if override is not None:
        rocket.air_brakes.append(
            SimpleNamespace(
                deployment_level=1,
                override_rocket_drag=override,
                drag_coefficient=Function(lambda level, mach: 0.2),
                reference_area=rocket.area,
            )
        )
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        return_value=datcom_table_factory({"CA": 0.4, "CD": 0.4}),
    ):
        rocket.prepare_datcom()
    flight = object.__new__(Flight)
    flight.rocket, flight.env = rocket, Environment()
    flight._Flight__post_processed_variables = []
    state = [0, 0, 1000, 0, 0, 100, 1, 0, 0, 0, 0, 0, 0]
    expected = -0.5 * flight.env.density(1000) * 100**2 * rocket.area * expected_cd
    # Act
    with patch.object(
        rocket.nosecones[0],
        "compute_forces_and_moments",
        side_effect=AssertionError("component loads used"),
    ):
        flight.u_dot_generalized(0.5, state, post_processing=True)
    # Assert
    assert len(flight._Flight__post_processed_variables) == 1
    assert flight._Flight__post_processed_variables[0][9] == pytest.approx(expected)


def test_datcom_namelist_scientific_notation_uppercase():
    """Legacy CONERR rejects lowercase exponents for small physical inputs."""
    # Arrange
    from rocketpy.rocket.datcom import _namelist

    # Act
    text = _namelist("FINSET1", [("LER", [1e-5, 1.25e-5])])
    # Assert
    assert "1.E-05" in text
    assert "1.25E-05" in text
    assert "e" not in text


def test_datcom_exit_diameter_override_preserves_motor(datcom_rocket):
    """Measured aerodynamic exit diameter can correct a legacy motor placeholder."""
    # Arrange
    fins = datcom_rocket.fins[0]
    original = datcom_rocket.motor.nozzle_radius
    fins.datcom_options["nozzle_exit_diameter"] = 0.03
    # Act
    decks, _, _, _ = datcom_decks(datcom_rocket)
    # Assert
    assert _deck_value(decks[True], "DEXIT") == pytest.approx(0.03)
    assert _deck_value(decks[False], "DEXIT") == 0
    assert datcom_rocket.motor.nozzle_radius == original


@pytest.mark.parametrize("diameter", [-0.01, 0, float("nan"), 0.5, None, [0.02, 0.03]])
def test_datcom_exit_diameter_override_invalid(datcom_rocket, diameter):
    """A nozzle override must be finite, positive and fit the body base."""
    # Arrange
    datcom_rocket.fins[0].datcom_options["nozzle_exit_diameter"] = diameter
    # Act / Assert
    with pytest.raises(ValueError, match="Powered nozzle diameter"):
        datcom_decks(datcom_rocket)


@pytest.mark.parametrize("mode", ["none", "full"])
def test_datcom_exit_diameter_override_conflicting_mode(datcom_rocket, mode):
    """An explicit nozzle opening cannot be silently ignored by another mode."""
    # Arrange
    datcom_rocket.fins[0].datcom_options.update(
        nozzle_exit_diameter=0.03, powered_base_drag=mode
    )
    # Act / Assert
    with pytest.raises(ValueError, match="requires powered_base_drag='nozzle'"):
        datcom_decks(datcom_rocket)


def test_datcom_partial_component_tables_cannot_replace_total_damping(tmp_path):
    """Component DAMP rows printed by PART precede the total, which must win."""
    # Arrange
    sample = Path(__file__).parents[2] / "fixtures/surfaces/data/datcom/for006.dat"
    raw = sample.read_text()
    expected = parse_datcom_output(tmp_path, _output=raw)
    rows = []
    for line in raw.splitlines():
        columns = line.split()
        if columns in (
            ["ALPHA", "CNQ", "CMQ", "CAQ", "CNAD", "CMAD"],
            ["ALPHA", "CYR", "CLNR", "CLLR", "CYP", "CLNP", "CLLP"],
        ):
            rows.append(line)
            rows.extend(
                f"{alpha} " + " ".join(["999"] * (len(columns) - 1))
                for alpha in [-2, 0, 2]
            )
            rows.append("")
        rows.append(line)
    # Act
    actual = parse_datcom_output(tmp_path, _output="\n".join(rows))
    # Assert
    for key in DAMPING_KEYS:
        assert actual.damping[key] == pytest.approx(expected.damping[key])
