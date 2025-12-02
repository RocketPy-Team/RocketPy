"""Integration tests for FlightComparator.

These tests exercise the full workflow of:
- running a Flight simulation,
- adding external data,
- generating comparison plots,
- summarizing key events.
"""

import numpy as np

from rocketpy.simulation.flight_comparator import FlightComparator
from rocketpy.simulation.flight_data_importer import FlightDataImporter


def test_full_workflow(flight_calisto):
    """Test complete workflow: add data, compare, summary, plots.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    # Simulate external data with realistic errors
    time_data = np.linspace(0, flight_calisto.t_final, 100)

    comparator.add_data(
        "OpenRocket",
        {
            "altitude": (
                time_data,
                flight_calisto.z(time_data) + np.random.normal(0, 5, 100),
            ),
            "vz": (
                time_data,
                flight_calisto.vz(time_data) + np.random.normal(0, 1, 100),
            ),
            "x": (time_data, flight_calisto.x(time_data)),
            "z": (time_data, flight_calisto.z(time_data)),
        },
    )

    # Test all methods - should run without error
    comparator.summary()
    comparator.compare("altitude")
    results = comparator.compare_key_events()
    comparator.trajectories_2d(plane="xz")

    # Verify results - compare_key_events now returns a dict
    assert isinstance(results, dict)
    assert len(results) >= 4  # At least 4 metrics
    assert "Apogee Altitude (m)" in results
    assert "Apogee Time (s)" in results
    assert "Max Velocity (m/s)" in results
    assert "Impact Velocity (m/s)" in results


def test_full_workflow_with_importer(flight_calisto, tmp_path):
    """Full workflow using FlightDataImporter as external source."""
    comparator = FlightComparator(flight_calisto)

    # Create a tiny CSV with time,z,vz columns
    csv_path = tmp_path / "flight_log.csv"
    time_data = np.linspace(0, flight_calisto.t_final, 50)
    z_data = flight_calisto.z(time_data) * 0.97
    vz_data = flight_calisto.vz(time_data)

    lines = ["time,z,vz\n"]
    for t, z, vz in zip(time_data, z_data, vz_data):
        lines.append(f"{t},{z},{vz}\n")
    csv_path.write_text("".join(lines), encoding="utf-8")

    # Build importer
    importer = FlightDataImporter(
        paths=str(csv_path),
        columns_map={"time": "time", "z": "z", "vz": "vz"},
        units=None,
    )

    # Use importer directly
    comparator.add_data("Imported Log", importer)

    comparator.summary()
    comparator.compare("z")
    results = comparator.compare_key_events()
    comparator.trajectories_2d(plane="xz")

    assert isinstance(results, dict)
    assert "Apogee Altitude (m)" in results
    assert "Impact Velocity (m/s)" in results
