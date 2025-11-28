"""Integration tests for FlightComparator and Flight.compare.

These tests exercise the full workflow of:
- running a Flight simulation,
- adding external data,
- generating comparison plots,
- summarizing key events.
"""

import numpy as np

from rocketpy.simulation.flight_comparator import FlightComparator


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


def test_flight_compare_helper(flight_calisto):
    """Test Flight.compare() convenience wrapper."""
    time_data = np.linspace(0, flight_calisto.t_final, 100)
    external = {
        "z": (time_data, flight_calisto.z(time_data) + 5.0),
        "vz": (time_data, flight_calisto.vz(time_data)),
    }

    comparator = flight_calisto.compare(external, variable="z", label="External")

    assert isinstance(comparator, FlightComparator)
    assert "External" in comparator.data_sources
    assert "z" in comparator.data_sources["External"]
