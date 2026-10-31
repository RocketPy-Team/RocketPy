"""DATCOM text parsing, isolated executable lifecycle and flight integration."""

import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import DatcomFins, Environment, Flight, Rocket
from rocketpy.rocket.datcom import DAMPING_KEYS, parse_datcom_output, run_datcom

SAMPLE = Path(__file__).parents[1] / "fixtures/surfaces/data/datcom/for006.dat"


@pytest.mark.slow
def test_datcom_native_fin_overhang(datcom_rocket):
    """The native executable accepts aft overhang with an attached root centre."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    datcom_rocket.fins[0].datcom_path = executable
    datcom_rocket.fins[0].root_chord = 0.45
    # Act
    model = datcom_rocket.prepare_datcom()
    # Assert
    for state in (True, False):
        assert model.coefficients(state, 0.6, 0)[2] > 0
        assert set(model.tables[state].damping) == set(DAMPING_KEYS)


@pytest.mark.slow
def test_datcom_native_wind_tunnel_fit(datcom_rocket):
    """A drag-only synthetic tunnel sweep recovers native coefficients."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    rocket = datcom_rocket
    rocket.fins[0].datcom_path = executable
    rocket.fins[0].datcom_options["reynolds"] = [2e7, 4e7]
    rocket.add_datcom_calibration(roughness_height=30.48e-6)
    known = rocket.prepare_datcom()
    mach = [0.2, 0.2, 0.2, 0.6, 0.6, 0.6]
    alpha = [-2, 0, 2, -2, 0, 2]
    drag = []
    for m, a in zip(mach, alpha):
        radians = np.radians(a)
        values = known.coefficients(False, m, radians)
        drag.append(values[2] * np.cos(radians) + values[0] * np.sin(radians))
    rocket.add_datcom_calibration()
    del rocket.fins[0].datcom_options["reynolds"]
    # Act
    report = rocket.fit_datcom_calibration(
        {
            "mach": mach,
            "alpha": alpha,
            "CD": drag,
            "reynolds_per_meter": [2e7] * 3 + [4e7] * 3,
        },
        bounds={"roughness_height": (0, 100e-6)},
    )
    model = rocket.prepare_datcom()
    # Assert
    assert report["weighted_rmse_fitted"] < 0.001
    assert report["weighted_rmse_fitted"] < report["weighted_rmse_initial"]
    assert report["calibration"]["roughness_height"] == pytest.approx(
        30.48e-6, abs=5e-6
    )
    assert set(model.tables[False].damping) == set(DAMPING_KEYS)
    assert "reynolds" not in rocket.fins[0].datcom_options


@pytest.mark.slow
def test_datcom_native_calibration_roughness(datcom_rocket):
    """Public calibration changes native drag and reset restores the baseline."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    datcom_rocket.fins[0].datcom_path = executable
    baseline = datcom_rocket.prepare_datcom()
    # Act
    datcom_rocket.add_datcom_calibration(roughness_height=30.48e-6)
    calibrated = datcom_rocket.prepare_datcom()
    datcom_rocket.add_datcom_calibration()
    reset = datcom_rocket.prepare_datcom()
    # Assert
    for powered in (True, False):
        assert (
            calibrated.coefficients(powered, 0.6, 0)[2]
            > baseline.coefficients(powered, 0.6, 0)[2]
        )
        np.testing.assert_array_equal(
            reset.tables[powered].body["CA"], baseline.tables[powered].body["CA"]
        )


def test_parse_datcom_output_complete_signed_tables():
    """A real for006 sample supplies static, lateral and all dynamic channels."""
    # Arrange
    directory = SAMPLE.parent
    # Act
    result = parse_datcom_output(directory)
    # Assert
    np.testing.assert_allclose(result[4], [0.2, 0.6])
    np.testing.assert_allclose(result[5], [-2, 0, 2])
    assert result[0].shape == (2, 3)
    assert set(result.damping) == set(DAMPING_KEYS)
    assert result.body["CA"][0, 1] > 0
    assert result.damping["CMQ"][0, 1] < 0
    assert result.damping["CLLP"][0, 1] < 0
    for values in result.damping.values():
        assert values.shape == (2, 3)
        assert np.isfinite(values).all()


def test_parse_datcom_output_joined_fixed_width_columns(tmp_path):
    """An overflowing negative CP field must not truncate a valid alpha table."""
    # Arrange
    lines = SAMPLE.read_text().splitlines()
    header = next(
        i
        for i, line in enumerate(lines)
        if line.split() == ["ALPHA", "CL", "CD", "CL/CD", "X-C.P."]
    )
    row = next(i for i in range(header + 1, len(lines)) if len(lines[i].split()) == 5)
    fields = lines[row].split()
    fields[-1] = "-28096.705"
    lines[row] = " ".join(fields[:-1]) + fields[-1]
    (tmp_path / "for006.dat").write_text("\n".join(lines))
    # Act
    result = parse_datcom_output(tmp_path)
    # Assert
    assert result[3][0, 0] == pytest.approx(-28096.705)
    assert result[5] == pytest.approx([-2, 0, 2])


@pytest.mark.parametrize(
    "replacement,message",
    [
        (("DYNAMIC DERIVATIVES", "MISSING DERIVATIVES"), "lacks DAMP"),
        (("ALPHA       CNQ", "ALPHA       BAD"), "CNQ"),
        (("NALPHA = 3.", "NALPHA = 4."), "angle count"),
    ],
)
def test_parse_datcom_output_rejects_incomplete_data(replacement, message, tmp_path):
    """Missing damping or truncated grids fail instead of silently returning partial data."""
    # Arrange
    text = SAMPLE.read_text()
    old, new = replacement
    assert old in text
    (tmp_path / "for006.dat").write_text(text.replace(old, new))
    # Act / Assert
    with pytest.raises(RuntimeError, match=message):
        parse_datcom_output(tmp_path)


@pytest.mark.parametrize(
    "failure", [None, "exit", "timeout", "missing", "input", "grid", "airfoil"]
)
def test_run_datcom_temp_directory_cleanup(failure, tmp_path):
    """Executable/deck copies and all outputs are removed on success and every failure."""
    # Arrange
    source = tmp_path / "native datcom"
    source.write_bytes(b"executable placeholder")
    source.chmod(0o755)
    working_directories = []
    deck = "DIM M\nDAMP\nNEXT CASE\n"
    if failure == "grid":
        deck = "DIM M\n $FLTCON MACH = .2,.7,$\nDAMP\nNEXT CASE\n"

    def execute(argv, cwd, **kwargs):
        directory = Path(cwd)
        working_directories.append(directory)
        assert Path(argv[0]).read_bytes() == source.read_bytes()
        assert Path(argv[0]).stat().st_mode & 0o111
        assert directory != source.parent
        assert (directory / "for005.dat").read_text() == deck
        assert kwargs["timeout"] == 2
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 2)
        if failure != "missing":
            shutil.copyfile(SAMPLE, directory / "for006.dat")
        if failure == "input":
            (directory / "for006.dat").write_text("** ERROR ** 1*F")
        if failure == "airfoil":
            with (directory / "for006.dat").open("a") as stream:
                stream.write("\n*** ANALYSIS FOR THIS AIRFOIL IMPOSSIBLE.\n")
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0, stderr="failure"
        )

    # Act
    with patch("rocketpy.rocket.datcom.subprocess.run", side_effect=execute):
        if failure and failure != "airfoil":
            with pytest.raises(RuntimeError):
                run_datcom(source, deck, timeout=2)
        else:
            parsed = run_datcom(source, deck, timeout=2)
            assert parsed[0].shape == (2, 3)
            if failure == "airfoil":
                assert any(
                    "AIRFOIL IMPOSSIBLE" in message for message in parsed.diagnostics
                )
    # Assert
    assert len(working_directories) == 1
    assert not working_directories[0].exists()
    assert source.read_bytes() == b"executable placeholder"
    assert sorted(p.name for p in tmp_path.iterdir()) == [source.name]


@pytest.mark.parametrize(
    "mode,equations",
    [
        ("6 DOF", "standard"),
        ("6 DOF", "solid_propulsion"),
        ("3 DOF", "standard"),
    ],
)
def test_datcom_flight_uses_full_model(mode, equations, datcom_rocket):
    """Each supported free-flight solver consumes DATCOM and produces finite states."""
    # Arrange
    rocket = datcom_rocket
    tables = parse_datcom_output(SAMPLE.parent)
    # Act
    with patch("rocketpy.rocket.datcom.run_datcom", return_value=tables) as run:
        flight = Flight(
            rocket,
            Environment(),
            rail_length=5,
            initial_solution=[0, 0, 0, 1000, 0, 0, 100, 1, 0, 0, 0, 0.05, 0, 0],
            max_time=0.02,
            max_time_step=0.01,
            simulation_mode=mode,
            equations_of_motion=equations,
        )
    # Assert
    assert run.call_count == 2
    assert np.isfinite(np.asarray(flight.solution)).all()
    assert flight.solution[-1][0] == pytest.approx(0.02)
    assert len(rocket.aerodynamic_surfaces) == 3


@pytest.mark.parametrize(
    "mode,equations",
    [("6 DOF", "standard"), ("6 DOF", "solid_propulsion"), ("3 DOF", "standard")],
)
def test_datcom_uncertainty_flight_replays_and_prints(
    mode, equations, datcom_rocket, capsys
):
    """Three independent ascents report numeric ± without changing nominal loads."""
    # Arrange
    rocket = datcom_rocket
    tables = parse_datcom_output(SAMPLE.parent)
    environment = Environment(elevation=100)
    options = dict(
        rail_length=5,
        inclination=90,
        terminate_on_apogee=True,
        max_time=60,
        max_time_step=0.2,
        simulation_mode=mode,
        equations_of_motion=equations,
    )
    # Act
    with patch("rocketpy.rocket.datcom.run_datcom", return_value=tables) as run:
        baseline = Flight(rocket, environment, datcom_uncertainty=False, **options)
        flight = Flight(rocket, environment, **options)
    report = flight.datcom_uncertainty
    flight.prints.apogee_conditions()
    flight.prints.stability_margin()
    output = capsys.readouterr().out
    # Assert
    assert report is not None
    assert not report.errors
    assert set(report.cases) == {"nominal", "upper", "lower"}
    assert run.call_count == 2  # replays reuse the native tables
    assert flight.apogee == pytest.approx(baseline.apogee, abs=1e-8)
    np.testing.assert_allclose(flight.solution, baseline.solution, atol=1e-8, rtol=0)
    assert (
        report.cases["upper"]["apogee"]
        < flight.apogee
        < report.cases["lower"]["apogee"]
    )
    assert flight.datcom_apogee_uncertainty > 0
    assert report.uncertainty("initial_stability_margin") > 0
    assert report.cases["nominal"]["apogee_agl"] == pytest.approx(flight.apogee - 100)
    for case in ("upper", "lower"):
        replay = report.flights[case]
        assert replay.rocket is not rocket
        assert replay.datcom_uncertainty is None
        assert replay._datcom_load_model is not replay.rocket._datcom_model
        assert replay.terminate_on_apogee
    assert flight._datcom_load_model is rocket._datcom_model
    assert "±" in output
    assert "DATCOM coefficient upper: apogee" in output
    assert "DATCOM coefficient lower stability" in output
    assert "not a confidence interval" in output
    assert "outside Wu" in output


def test_datcom_uncertainty_incomplete_ascent(datcom_rocket, capsys):
    """A cutoff before apogee cannot silently produce a zero-width uncertainty."""
    # Arrange
    tables = parse_datcom_output(SAMPLE.parent)
    # Act
    with patch("rocketpy.rocket.datcom.run_datcom", return_value=tables):
        flight = Flight(datcom_rocket, Environment(), rail_length=5, max_time=0.02)
    flight.prints.apogee_conditions()
    output = capsys.readouterr().out
    # Assert
    assert flight.datcom_uncertainty is None
    assert flight.datcom_apogee_uncertainty is None
    assert "did not reach apogee" in output
    assert "±" not in output


def test_datcom_uncertainty_replays_restore_random_states(datcom_rocket):
    """Replays do not consume the caller's random sequence beyond nominal flight."""
    # Arrange
    tables = parse_datcom_output(SAMPLE.parent)
    options = dict(
        rail_length=5,
        inclination=90,
        terminate_on_apogee=True,
        max_time=60,
        max_time_step=0.5,
    )

    def callback(_flight):
        np.random.random()
        random.random()

    original_numpy_state, original_python_state = (
        np.random.get_state(),
        random.getstate(),
    )
    # Act
    try:
        with patch("rocketpy.rocket.datcom.run_datcom", return_value=tables):
            np.random.seed(123)
            random.seed(123)
            Flight(
                datcom_rocket,
                Environment(),
                post_step_callback=callback,
                datcom_uncertainty=False,
                **options,
            )
            expected = (np.random.random(), random.random())
            np.random.seed(123)
            random.seed(123)
            flight = Flight(
                datcom_rocket, Environment(), post_step_callback=callback, **options
            )
            actual = (np.random.random(), random.random())
    finally:
        np.random.set_state(original_numpy_state)
        random.setstate(original_python_state)
    # Assert
    assert actual == expected
    assert not flight.datcom_uncertainty.errors


def test_datcom_uncertainty_ordinary_rocket_unchanged(calisto, capsys):
    """The default introduces no replay or DATCOM output for ordinary rockets."""
    # Arrange
    environment = Environment()
    options = dict(rail_length=5, max_time=0.1)
    # Act
    baseline = Flight(calisto, environment, datcom_uncertainty=False, **options)
    flight = Flight(calisto, environment, **options)
    flight.prints.apogee_conditions()
    output = capsys.readouterr().out
    # Assert
    np.testing.assert_array_equal(flight.solution, baseline.solution)
    assert flight.datcom_uncertainty is None
    assert flight.datcom_apogee_uncertainty is None
    assert "DATCOM" not in output
    assert "±" not in output


@pytest.mark.parametrize("setting", [True, False, "upper", "lower"])
def test_datcom_uncertainty_setting_serialization(setting, datcom_rocket):
    """Flight dictionary round trips preserve opt-out and individual scenarios."""
    # Arrange
    tables = parse_datcom_output(SAMPLE.parent)
    # Act
    with patch("rocketpy.rocket.datcom.run_datcom", return_value=tables):
        flight = Flight(
            datcom_rocket,
            Environment(),
            rail_length=5,
            max_time=0.02,
            datcom_uncertainty=setting,
        )
        data = flight.to_dict()
        restored = Flight.from_dict(data)
    # Assert
    assert data["datcom_uncertainty"] == setting
    assert restored._datcom_uncertainty_setting == setting
    assert restored.apogee == flight.apogee


@pytest.mark.slow
@pytest.mark.parametrize("powered_base_drag", ["none", "nozzle"])
@pytest.mark.parametrize(
    "method",
    [
        "add_datcom_trapezoidal_fins",
        "add_datcom_arc_fins",
        "add_datcom_hex_fins",
    ],
)
def test_datcom_native_executable(method, powered_base_drag, calisto):
    """Optional real executable test, enabled with DATCOM_EXECUTABLE and --runslow."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to a native DATCOM executable")
    rocket = calisto
    rocket.add_nose(0.55, "von karman", 1.2)
    fins = getattr(rocket, method)(
        4,
        0.25,
        0.1,
        0.1,
        -0.9,
        path=executable,
        datcom_options={"powered_base_drag": powered_base_drag},
    )
    rocket.add_tail(rocket.radius, 0.045, 0.1, -1.2)
    # Act
    model = rocket.prepare_datcom()
    flight = Flight(
        rocket,
        Environment(),
        rail_length=5,
        max_time=30,
        max_time_step=0.1,
        terminate_on_apogee=True,
    )
    # Assert
    assert isinstance(fins, DatcomFins)
    assert model.tables[True].damping["CMQ"][0, 3] < 0
    assert rocket.power_off_drag(0.3) > rocket.power_on_drag(0.3)
    assert model.references[True] != pytest.approx(model.references[False])
    assert np.isfinite(np.asarray(flight.solution)).all()
    assert flight.t_final > rocket.motor.burn_out_time
    assert flight.apogee > 100


def test_parse_datcom_multiple_sideslip_cases(tmp_path):
    """Repeated Mach grids at different sideslip angles form one complete database."""
    # Arrange
    text = SAMPLE.read_text()
    assert "SIDESLIP =       0.00" in text
    blocks = [
        text.replace("SIDESLIP =       0.00", f"SIDESLIP =       {b:.2f}")
        for b in [-5, 0, 5]
    ]
    (tmp_path / "for006.dat").write_text("\n".join(blocks))
    # Act
    parsed = parse_datcom_output(tmp_path)
    # Assert
    assert parsed.beta == pytest.approx([-5, 0, 5])
    assert list(parsed.sideslip) == [-5, 0, 5]
    for table in parsed.sideslip.values():
        assert table[0].shape == (2, 3)
        assert set(table.damping) == set(DAMPING_KEYS)


@pytest.mark.slow
@pytest.mark.parametrize(
    "feature",
    [
        "multi_hex",
        "arc_asymmetric",
        "naca4",
        "naca6",
        "naca1",
        "naca5",
        "naca_modified",
        "naca_supersonic",
        "fifty_point_user",
        "rocketpy_cant",
        "user_surfaces",
        "user_mean",
        "one_panel_triangle",
        "dihedral_incidence",
        "eight_panels_ten_stations",
        "boattail",
    ],
)
def test_datcom_native_extended_fin_definitions(feature, calisto):
    """The installed native executable accepts every supported geometry/section family."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to a native DATCOM executable")
    rocket = calisto
    rocket.add_nose(0.55, "von karman", 1.2)
    rocket.add_tail(rocket.radius, 0.045, 0.1, -1.2)
    kwargs = dict(
        n=4,
        position=-0.9,
        span_stations=[0, 0.05, 0.1],
        chords=[0.25, 0.2, 0.1],
        leading_edges=[0, 0.04, 0.1],
        path=executable,
        datcom_options={"mach": [0.2, 0.6], "alpha": [-2, 0, 2]},
    )
    if feature == "arc_asymmetric":
        kwargs.update(
            section_type="ARC",
            upper_thickness_to_chord=[0.01, 0.02, 0.025],
            lower_thickness_to_chord=[0, 0.01, 0.02],
        )
    elif feature.startswith("naca"):
        designations = {
            "naca4": "0012",
            "naca6": "6-64-005",
            "naca1": "1-16-212.25",
            "naca5": "5-23012",
            "naca_modified": "4-0012-62",
            "naca_supersonic": "S-3-25-5-50",
        }
        kwargs.update(section_type="NACA", naca=designations[feature])
    elif feature == "fifty_point_user":
        x = np.linspace(0, 1, 50)
        upper = 0.03 * np.sin(np.pi * x)
        upper[[0, -1]] = 0
        kwargs.update(
            section_type="USER",
            airfoil_coordinates={"x": x, "upper": upper, "lower": -upper},
        )
    elif feature == "rocketpy_cant":
        kwargs.update(cant_angle=0.5)
    elif feature == "user_surfaces":
        kwargs.update(
            section_type="USER",
            airfoil_coordinates={
                "x": [0, 0.25, 0.5, 0.75, 1],
                "upper": [0, 0.03, 0.04, 0.03, 0],
                "lower": [0, -0.02, -0.03, -0.02, 0],
            },
        )
    elif feature == "user_mean":
        kwargs.update(
            section_type="USER",
            airfoil_coordinates={
                "x": [0, 0.25, 0.5, 0.75, 1],
                "mean": [0, 0.005, 0.005, 0.005, 0],
                "thickness": [0, 0.05, 0.07, 0.05, 0],
            },
        )
    elif feature == "one_panel_triangle":
        kwargs.update(n=1, chords=[0.25, 0.15, 0])
    elif feature == "dihedral_incidence":
        kwargs.update(n=2, panel_angles=[20, 200], dihedral=[5, -5], incidence=[1, -1])
    elif feature == "eight_panels_ten_stations":
        kwargs.update(
            n=8,
            span_stations=np.linspace(0, 0.1, 10),
            chords=np.linspace(0.25, 0.1, 10),
            leading_edges=np.linspace(0, 0.1, 10),
        )
    elif feature == "boattail":
        kwargs.update(
            position=-1.21,
            span_stations=[0, 0.025, 0.05],
            chords=[0.04, 0.03, 0.02],
            leading_edges=[0, 0.005, 0.01],
        )
    fins = rocket.add_datcom_fins(**kwargs)
    # Act
    model = rocket.prepare_datcom()
    # Assert
    for powered in (True, False):
        table = model.tables[powered]
        assert table[0].shape == (2, 3)
        assert np.isfinite(table[1]).all()
        assert set(table.damping) == set(DAMPING_KEYS)
    if feature in ("one_panel_triangle", "dihedral_incidence"):
        assert len(model.tables[True].beta) == 3
        b = model.tables[True].sideslip[5]
        assert model.coefficients(True, 0.2, 0, np.radians(5))[3] == pytest.approx(
            b.body["CY"][0, 1]
        )
    if feature == "rocketpy_cant":
        assert model.coefficients(True, 0.2, 0)[5] > 0
    assert len(fins.datcom_span_stations) <= 10


@pytest.mark.slow
def test_datcom_native_body_axis_angle_convention(calisto):
    """The target 3/99 executable uses projected atan/atan body incidence angles."""
    # Arrange
    import re

    from rocketpy.rocket.datcom import datcom_decks

    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    calisto.add_nose(0.55, "von karman", 1.2)
    calisto.add_datcom_hex_fins(
        4,
        0.25,
        0.1,
        0.1,
        -0.9,
        path=executable,
        datcom_options={
            "mach": [0.3, 0.6],
            "alpha": [-20, 0, 20],
            "beta": [-30, 0, 30],
        },
    )
    calisto.add_tail(calisto.radius, 0.045, 0.1, -1.2)
    deck = datcom_decks(calisto)[0][True]
    # Isolate the axisymmetric body; BUILD enables body-only damping output.
    deck = re.sub(r"\s*\$FINSET1.*?\$", "", deck, flags=re.S)
    deck = re.sub(r"\s*\$DEFLCT.*?\$", "", deck, flags=re.S)
    deck = deck.replace("DAMP", "BUILD\nDAMP")
    # Act
    table = run_datcom(executable, deck).sideslip[30]
    ratio = abs(table.body["CY"][0, 2] / table.body["CN"][0, 2])
    # Assert
    expected = np.tan(np.radians(30)) / np.tan(np.radians(20))
    assert ratio == pytest.approx(expected, rel=1e-3)


def test_datcom_tipoff_alpha_dot_is_self_consistent(
    datcom_rocket, datcom_table_factory
):
    """The implicit solve includes the rail constraint rather than the free-flight acceleration."""
    # Arrange
    rocket = datcom_rocket
    rocket.set_rail_buttons(
        upper_button_position=0.1, lower_button_position=-0.6, angular_position=0
    )
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        return_value=datcom_table_factory({"CNAD": 0.5, "CMAD": -0.2, "CA": 0.4}),
    ):
        model = rocket.prepare_datcom()
    state = [0, 0, 1000, 0, 0, 20, 1, 0, 0, 0, 0.2, 0, 0]
    flight = Flight(
        rocket,
        Environment(),
        rail_length=5,
        inclination=90,
        initial_solution=[0, *state],
        max_time=0.001,
        max_time_step=0.001,
        use_udot_rail2=True,
    )
    # Act
    with patch.object(
        model,
        "compute_forces_and_moments_at_time",
        wraps=model.compute_forces_and_moments_at_time,
    ) as loads:
        derivative = flight.udot_rail2(0.5, state)
    # Assert
    assert loads.call_count == 3
    used_alpha_dot = loads.call_args.args[1]
    assert used_alpha_dot == pytest.approx(
        flight._datcom_incidence_rate(0.5, state, derivative), rel=1e-8, abs=1e-10
    )
    assert flight._datcom_evaluating is False
    assert np.isfinite(derivative).all()


@pytest.mark.slow
def test_datcom_native_small_leading_edge_radius(datcom_rocket):
    """A measured ten-micrometre edge radius must survive native input parsing."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    datcom_rocket.fins[0].datcom_path = executable
    datcom_rocket.add_datcom_calibration(leading_edge_radius=1e-5)
    # Act
    model = datcom_rocket.prepare_datcom()
    # Assert
    assert np.isfinite(model.coefficients(False, 0.6, 0)).all()


@pytest.mark.slow
def test_datcom_native_maximum_adaptive_grid(datcom_rocket):
    """All 20 adaptive Mach points, including the transonic cluster, survive I/O."""
    # Arrange
    from rocketpy.rocket.datcom_conditions import (
        adaptive_alpha_grid,
        adaptive_mach_grid,
    )

    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    datcom_rocket.fins[0].datcom_path = executable
    mach = adaptive_mach_grid(3)
    alpha = adaptive_alpha_grid([-35, 40])
    datcom_rocket.fins[0].datcom_options.update(mach=mach, alpha=alpha)
    # Act
    model = datcom_rocket.prepare_datcom()
    # Assert
    for state in (True, False):
        assert len(model.tables[state][4]) == 20
        assert model.tables[state][4] == pytest.approx(mach, abs=0.005)
        assert model.tables[state][5] == pytest.approx(alpha, abs=0.005)


@pytest.mark.slow
def test_datcom_partial_output_preserves_total_damping(datcom_rocket):
    """PART adds component tables but must not replace whole-vehicle DAMP data."""
    # Arrange
    from rocketpy.rocket.datcom import datcom_decks

    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    deck = datcom_decks(datcom_rocket)[0][False]
    baseline = run_datcom(executable, deck)
    # Act
    partial = run_datcom(executable, deck.replace("NEXT CASE", "PART\nNEXT CASE"))
    # Assert
    for key in DAMPING_KEYS:
        assert partial.damping[key] == pytest.approx(baseline.damping[key])


@pytest.mark.slow
def test_datcom_native_manual_worked_example():
    """The independent published example reproduces its total CN/CM/CA table."""
    # Arrange
    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    deck = (SAMPLE.parent / "manual_example_for005.dat").read_text()
    # Act
    result = run_datcom(executable, deck)
    # Assert
    plain = run_datcom(executable, deck.replace("PART\n", ""))
    for key in DAMPING_KEYS:
        assert result.damping[key] == pytest.approx(plain.damping[key])
    assert result.body["CN"][0] == pytest.approx(
        [0, 1.179, 2.476, 3.992, 5.582, 7.151, 8.601, 10.093], abs=0.002
    )
    assert result.body["CA"][0] == pytest.approx(
        [0.466, 0.465, 0.463, 0.458, 0.453, 0.445, 0.436, 0.426], abs=0.002
    )
    assert result.body["CM"][0] == pytest.approx(
        [0, -1.577, -3.194, -5.056, -6.962, -8.904, -10.792, -12.692], abs=0.002
    )


@pytest.mark.slow
def test_datcom_generated_geometry_matches_manual_example():
    """Public geometry conversion reproduces the manual with matched methods."""
    # Arrange
    from rocketpy.rocket.datcom import _namelist, datcom_decks

    executable = os.environ.get("DATCOM_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATCOM_EXECUTABLE to the target 3/99 executable")
    inch = 0.0254
    rocket = Rocket(
        radius=3.75 * inch / 2,
        mass=1,
        inertia=(1, 1, 1),
        power_off_drag=0,
        power_on_drag=0,
        center_of_mass_without_motor=18.75 * inch,
    )
    rocket.add_nose(11.25 * inch, "ogive", position=37.5 * inch)
    options = dict(
        mach=[2.35, 2.36],
        alpha=[-4, 0, 4, 8, 12, 16, 20, 24, 28],
        beta=[0],
        reynolds=3e6 / 0.3048,  # Published REN is per foot, including with DIM IN.
        body_length=26.25 * inch,
    )
    for n, xle, root, tip, span, maximum, flat, panels in [
        (2, 15.42, 6.96, 0, 3.48, 0.238, 0.524, [90, 270]),
        (4, 31.915, 5.585, 2.792, 4.385, 0.288, 0.428, [0, 90, 180, 270]),
    ]:
        rocket.add_datcom_hex_fins(
            n,
            root_chord=root * inch,
            tip_chord=tip * inch,
            span=span * inch,
            position=(37.5 - xle) * inch,
            sweep_length=(root - tip) * inch,
            path=executable,
            thickness_to_chord=0.04476,
            max_thickness_position=maximum,
            flat_length=flat,
            leading_edge_radius=0.015 * inch,
            panel_angles=panels,
            datcom_options=options,
        )
    reference = (SAMPLE.parent / "manual_example_for005.dat").read_text()
    # Native numerical derivatives depend on the ALPHA schedule. Match the
    # signed grid required by the API and retain the independent input geometry.
    reference_conditions = _namelist(
        "FLTCON",
        [
            ("NMACH", 2),
            ("MACH", options["mach"]),
            ("REN", [3e6, 3e6]),
            ("NALPHA", len(options["alpha"])),
            ("ALPHA", options["alpha"]),
            ("BETA", 0),
        ],
    )
    reference = re.sub(
        r"\$FLTCON.*?\$", lambda _: reference_conditions, reference, flags=re.S
    )
    # Act
    with pytest.warns(UserWarning, match="four exposed semispans"):
        generated = datcom_decks(rocket)[0][False]
    # Match the independently published body method. The production options
    # currently expose no SOSE selector; do not mistake this for a geometry fix.
    generated = generated.replace("NEXT CASE", "SOSE\nPART\nNEXT CASE")
    actual = run_datcom(executable, generated)
    expected = run_datcom(executable, reference)
    # Assert
    for key in ("CN", "CM", "CA"):
        assert actual.body[key][-1] == pytest.approx(expected.body[key][-1], abs=0.002)
    for key in DAMPING_KEYS:
        assert actual.damping[key][-1] == pytest.approx(
            expected.damping[key][-1], abs=0.002
        )
