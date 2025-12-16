"""Unit tests for FlightDataExporter class.

This module tests the data export functionality of the FlightDataExporter class,
which exports flight simulation data to various formats (CSV, JSON, KML).
"""

import json

import numpy as np
import pytest

from rocketpy.simulation import FlightDataExporter


def test_export_pressures_writes_csv_rows(flight_calisto_robust, tmp_path):
    """Test that export_pressures writes CSV rows with pressure data.

    Validates that the exported file contains multiple data rows in CSV format
    with 2-3 columns (time and pressure values).

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight object with parachutes configured.
    tmp_path : pathlib.Path
        Pytest fixture for temporary directories.
    """
    out = tmp_path / "pressures.csv"
    FlightDataExporter(flight_calisto_robust).export_pressures(str(out), time_step=0.2)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 5
    # Basic CSV shape "t, value" or "t, clean, noisy"
    parts = lines[0].split(",")
    assert len(parts) in (2, 3)


def test_export_sensor_data_writes_json(flight_calisto, tmp_path, monkeypatch):
    """Test that export_sensor_data writes JSON with sensor data.

    Validates that sensor data is exported as JSON with sensor names as keys
    and measurement arrays as values.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested.
    tmp_path : pathlib.Path
        Pytest fixture for temporary directories.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for modifying attributes.
    """

    class DummySensor:
        """Dummy sensor with name attribute for testing."""

        def __init__(self, name):
            self.name = name

    s1 = DummySensor("DummySensor")
    monkeypatch.setattr(
        flight_calisto, "sensor_data", {s1: [1.0, 2.0, 3.0]}, raising=False
    )
    out = tmp_path / "sensors.json"

    FlightDataExporter(flight_calisto).export_sensor_data(str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["DummySensor"] == [1.0, 2.0, 3.0]


def test_export_data_default_variables(flight_calisto, tmp_path):
    """Test export_data with default variables (full solution matrix).

    Validates that all state variables are exported correctly when no specific
    variables are requested: position (x, y, z), velocity (vx, vy, vz),
    quaternions (e0, e1, e2, e3), and angular velocities (w1, w2, w3).

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested.
    tmp_path : pathlib.Path
        Pytest fixture for temporary directories.
    """
    file_name = tmp_path / "flight_data.csv"
    FlightDataExporter(flight_calisto).export_data(str(file_name))

    test_data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    # Verify time column
    assert np.allclose(flight_calisto.x[:, 0], test_data[:, 0], atol=1e-5)

    # Verify position
    assert np.allclose(flight_calisto.x[:, 1], test_data[:, 1], atol=1e-5)
    assert np.allclose(flight_calisto.y[:, 1], test_data[:, 2], atol=1e-5)
    assert np.allclose(flight_calisto.z[:, 1], test_data[:, 3], atol=1e-5)

    # Verify velocity
    assert np.allclose(flight_calisto.vx[:, 1], test_data[:, 4], atol=1e-5)
    assert np.allclose(flight_calisto.vy[:, 1], test_data[:, 5], atol=1e-5)
    assert np.allclose(flight_calisto.vz[:, 1], test_data[:, 6], atol=1e-5)

    # Verify quaternions
    assert np.allclose(flight_calisto.e0[:, 1], test_data[:, 7], atol=1e-5)
    assert np.allclose(flight_calisto.e1[:, 1], test_data[:, 8], atol=1e-5)
    assert np.allclose(flight_calisto.e2[:, 1], test_data[:, 9], atol=1e-5)
    assert np.allclose(flight_calisto.e3[:, 1], test_data[:, 10], atol=1e-5)

    # Verify angular velocities
    assert np.allclose(flight_calisto.w1[:, 1], test_data[:, 11], atol=1e-5)
    assert np.allclose(flight_calisto.w2[:, 1], test_data[:, 12], atol=1e-5)
    assert np.allclose(flight_calisto.w3[:, 1], test_data[:, 13], atol=1e-5)


def test_export_data_custom_variables_and_time_step(flight_calisto, tmp_path):
    """Test export_data with custom variables and time step.

    Validates that specific variables can be exported with custom time intervals,
    including derived quantities like angle of attack.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested.
    tmp_path : pathlib.Path
        Pytest fixture for temporary directories.
    """
    file_name = tmp_path / "custom_flight_data.csv"
    time_step = 0.1

    FlightDataExporter(flight_calisto).export_data(
        str(file_name),
        "z",
        "vz",
        "e1",
        "w3",
        "angle_of_attack",
        time_step=time_step,
    )

    test_data = np.loadtxt(file_name, delimiter=",", skiprows=1)
    time_points = np.arange(flight_calisto.t_initial, flight_calisto.t_final, time_step)

    # Verify time column
    assert np.allclose(time_points, test_data[:, 0], atol=1e-5)

    # Verify custom variables
    assert np.allclose(flight_calisto.z(time_points), test_data[:, 1], atol=1e-5)
    assert np.allclose(flight_calisto.vz(time_points), test_data[:, 2], atol=1e-5)
    assert np.allclose(flight_calisto.e1(time_points), test_data[:, 3], atol=1e-5)
    assert np.allclose(flight_calisto.w3(time_points), test_data[:, 4], atol=1e-5)
    assert np.allclose(
        flight_calisto.angle_of_attack(time_points), test_data[:, 5], atol=1e-5
    )


def test_export_kml_trajectory(flight_calisto_robust, tmp_path):
    """Test export_kml creates valid KML file with trajectory coordinates.

    Validates that the KML export contains correct latitude, longitude, and
    altitude coordinates for the flight trajectory in absolute altitude mode.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested.
    tmp_path : pathlib.Path
        Pytest fixture for temporary directories.
    """
    file_name = tmp_path / "trajectory.kml"
    FlightDataExporter(flight_calisto_robust).export_kml(
        str(file_name), time_step=None, extrude=True, altitude_mode="absolute"
    )

    # Parse KML coordinates
    with open(file_name, "r") as kml_file:
        for row in kml_file:
            if row.strip().startswith("<coordinates>"):
                coords_str = (
                    row.strip()
                    .replace("<coordinates>", "")
                    .replace("</coordinates>", "")
                )
                coords_list = coords_str.strip().split(" ")

    # Extract lon, lat, z from coordinates
    parsed_coords = [c.split(",") for c in coords_list]
    lon = [float(point[0]) for point in parsed_coords]
    lat = [float(point[1]) for point in parsed_coords]
    z = [float(point[2]) for point in parsed_coords]

    # Verify coordinates match flight data
    assert np.allclose(flight_calisto_robust.latitude[:, 1], lat, atol=1e-3)
    assert np.allclose(flight_calisto_robust.longitude[:, 1], lon, atol=1e-3)
    assert np.allclose(flight_calisto_robust.z[:, 1], z, atol=1e-3)


def test_export_data_csv_column_names_no_leading_spaces(flight_calisto, tmp_path):
    """Test that CSV column headers have no leading spaces after commas.

    This validates that exported CSV files can be easily read with pandas
    without requiring leading spaces in column names (Issue #864).

    When reading CSVs with pandas, column names should be accessible as
    'Vz (m/s)' not ' Vz (m/s)' (with leading space).

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested.
    tmp_path : pathlib.Path
        Pytest fixture for temporary directories.
    """
    file_name = tmp_path / "flight_data_columns.csv"
    FlightDataExporter(flight_calisto).export_data(
        str(file_name), "z", "vz", "altitude"
    )

    # Read the header line directly
    with open(file_name, "r") as f:
        header_line = f.readline().strip()

    # Verify header format - should have no spaces after commas
    # Format should be: # Time (s),Z (m),Vz (m/s),Altitude AGL (m)
    assert header_line.startswith("# Time (s),")
    assert ", " not in header_line, "Header should not contain ', ' (comma-space)"

    # Verify with pandas that columns are accessible without leading spaces
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(file_name)
    columns = df.columns.tolist()

    # First column should be '# Time (s)'
    assert columns[0] == "# Time (s)"

    # Other columns should NOT have leading spaces
    for col in columns[1:]:
        assert not col.startswith(" "), f"Column '{col}' has leading space"

    # Verify columns are accessible with expected names (no leading spaces)
    assert "Z (m)" in columns
    assert "Vz (m/s)" in columns
    assert "Altitude AGL (m)" in columns

    # Verify we can access data using column names without spaces
    _ = df["# Time (s)"]
    _ = df["Z (m)"]
    _ = df["Vz (m/s)"]
    _ = df["Altitude AGL (m)"]
