"""Tests for the FlightComparator class.

This module tests the FlightComparator class which compares RocketPy Flight
simulations against external data sources such as flight logs, OpenRocket
simulations, and RAS Aero simulations.
"""

import os

import numpy as np
import pytest

from rocketpy import Function
from rocketpy.simulation.flight_comparator import FlightComparator


# Test FlightComparator initialization
def test_flight_comparator_init(flight_calisto):
    """Test FlightComparator initialization.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    assert comparator.flight == flight_calisto
    assert isinstance(comparator.data_sources, dict)
    assert len(comparator.data_sources) == 0


# Test add_data method with different input formats
def test_add_data_with_function(flight_calisto):
    """Test adding external data using RocketPy Function objects.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    # Create mock Function object
    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data) + np.random.normal(0, 5, 100)

    alt_function = Function(
        np.column_stack((time_data, altitude_data)),
        inputs="Time (s)",
        outputs="Altitude (m)",
        interpolation="linear",
    )

    comparator.add_data("Mock Data", {"z": alt_function})

    assert "Mock Data" in comparator.data_sources
    assert "z" in comparator.data_sources["Mock Data"]
    assert isinstance(comparator.data_sources["Mock Data"]["z"], Function)


def test_add_data_with_tuple(flight_calisto):
    """Test adding external data using (time, data) tuples.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    # Create mock data as tuples
    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data) + np.random.normal(0, 5, 100)
    velocity_data = flight_calisto.vz(time_data) + np.random.normal(0, 1, 100)

    comparator.add_data(
        "External Simulator",
        {"z": (time_data, altitude_data), "vz": (time_data, velocity_data)},
    )

    assert "External Simulator" in comparator.data_sources
    assert "z" in comparator.data_sources["External Simulator"]
    assert "vz" in comparator.data_sources["External Simulator"]
    assert isinstance(comparator.data_sources["External Simulator"]["z"], Function)


def test_add_data_overwrite_warning(flight_calisto):
    """Test that adding data with same label raises warning.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data)

    comparator.add_data("Test", {"z": (time_data, altitude_data)})

    with pytest.warns(UserWarning, match="already exists"):
        comparator.add_data("Test", {"z": (time_data, altitude_data)})


def test_add_data_empty_dict_raises_error(flight_calisto):
    """Test that empty data_dict raises ValueError.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    with pytest.raises(ValueError, match="cannot be empty"):
        comparator.add_data("Empty", {})


def test_add_data_invalid_format_warning(flight_calisto):
    """Test that invalid data format raises warning and skips variable.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    with pytest.warns(UserWarning, match="Format not recognized"):
        comparator.add_data("Invalid", {"z": "invalid_string", "vz": 12345})

    # Should have added the source but with no valid variables
    assert "Invalid" in comparator.data_sources
    assert len(comparator.data_sources["Invalid"]) == 0


# Test compare method
def test_compare_basic(flight_calisto):
    """Test basic comparison plot generation.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    # Add mock external data
    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data) + 10  # 10m offset

    comparator.add_data("OpenRocket", {"z": (time_data, altitude_data)})

    # This should generate plots and print metrics without error
    comparator.compare("z")


def test_compare_with_time_range(flight_calisto):
    """Test comparison with time_range parameter.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data)

    comparator.add_data("Test", {"z": (time_data, altitude_data)})

    # Compare only first 5 seconds
    comparator.compare("z", time_range=(0, 5))


def test_compare_missing_attribute_warning(flight_calisto):
    """Test that comparing non-existent attribute raises warning.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    with pytest.warns(UserWarning, match="not found in the RocketPy Flight"):
        comparator.compare("nonexistent_attribute")


def test_compare_no_external_data_warning(flight_calisto):
    """Test warning when no external sources have the requested variable.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    # Add data but without 'z' variable
    time_data = np.linspace(0, flight_calisto.t_final, 100)
    velocity_data = flight_calisto.vz(time_data)

    comparator.add_data("Test", {"vz": (time_data, velocity_data)})

    with pytest.warns(UserWarning, match="No external sources have data"):
        comparator.compare("z")


@pytest.mark.parametrize("filename", [None, "test_comparison.png"])
def test_compare_save_file(flight_calisto, filename):
    """Test comparison plot saving functionality.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    filename : str or None
        Filename to save plot to, or None to show plot.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data)

    comparator.add_data("Test", {"z": (time_data, altitude_data)})
    comparator.compare("z", filename=filename)

    if filename:
        assert os.path.exists(filename)
        os.remove(filename)


# Test compare_key_events method
def test_compare_key_events_basic(flight_calisto):
    """Test compare_key_events returns proper dict.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    # Add mock data with slight offset
    time_data = np.linspace(0, flight_calisto.t_final, 100)
    altitude_data = flight_calisto.z(time_data) + 5
    velocity_data = flight_calisto.vz(time_data)

    comparator.add_data(
        "Simulator",
        {"altitude": (time_data, altitude_data), "vz": (time_data, velocity_data)},
    )

    results = comparator.compare_key_events()

    # Check it's a dict
    assert isinstance(results, dict)

    # Check metrics exist
    assert "Apogee Altitude (m)" in results
    assert "Apogee Time (s)" in results
    assert "Max Velocity (m/s)" in results
    assert "Impact Velocity (m/s)" in results

    # Check RocketPy values exist
    assert "RocketPy" in results["Apogee Altitude (m)"]

    # Check external source data exists
    assert "Simulator" in results["Apogee Altitude (m)"]
    assert "value" in results["Apogee Altitude (m)"]["Simulator"]
    assert "error" in results["Apogee Altitude (m)"]["Simulator"]
    assert "error_percentage" in results["Apogee Altitude (m)"]["Simulator"]


def test_compare_key_events_multiple_sources(flight_calisto):
    """Test compare_key_events with multiple data sources.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)

    # Add two different simulators
    comparator.add_data(
        "OpenRocket", {"z": (time_data, flight_calisto.z(time_data) + 10)}
    )

    comparator.add_data("RASAero", {"z": (time_data, flight_calisto.z(time_data) - 5)})

    results = comparator.compare_key_events()

    # Check both sources are in the results
    assert "OpenRocket" in results["Apogee Altitude (m)"]
    assert "RASAero" in results["Apogee Altitude (m)"]

    # Check data structure for each source
    assert "value" in results["Apogee Altitude (m)"]["OpenRocket"]
    assert "value" in results["Apogee Altitude (m)"]["RASAero"]


# Test summary method
def test_summary(flight_calisto, capsys):
    """Test summary method prints correct information.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    capsys :
        Pytest fixture to capture stdout/stderr.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)
    comparator.add_data("Test", {"z": (time_data, flight_calisto.z(time_data))})

    comparator.summary()

    captured = capsys.readouterr()
    assert "FLIGHT COMPARISON SUMMARY" in captured.out
    assert "RocketPy Simulation:" in captured.out
    assert "External Data Sources:" in captured.out
    assert "Test" in captured.out


# Test all method
def test_all_plots(flight_calisto):
    """Test that all() method generates plots for common variables.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)

    # Add multiple variables
    comparator.add_data(
        "Simulator",
        {
            "z": (time_data, flight_calisto.z(time_data)),
            "vz": (time_data, flight_calisto.vz(time_data)),
        },
    )

    # This should run without error
    comparator.all()


def test_all_no_common_variables(flight_calisto, capsys):
    """Test all() when no common variables exist.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    capsys :
        Pytest fixture to capture stdout/stderr.
    """
    comparator = FlightComparator(flight_calisto)

    # Don't add any data
    comparator.all()

    captured = capsys.readouterr()
    assert "No common variables found" in captured.out


# Test trajectories_2d method
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_trajectories_2d_planes(flight_calisto, plane):
    """Test 2D trajectory plotting in different planes.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    plane : str
        Plane to plot trajectory in ('xy', 'xz', or 'yz').
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)

    # Add trajectory data
    comparator.add_data(
        "External",
        {
            "x": (time_data, flight_calisto.x(time_data)),
            "y": (time_data, flight_calisto.y(time_data)),
            "z": (time_data, flight_calisto.z(time_data)),
        },
    )

    # Should run without error
    comparator.trajectories_2d(plane=plane)


def test_trajectories_2d_invalid_plane(flight_calisto):
    """Test that invalid plane raises ValueError.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    with pytest.raises(ValueError, match="plane must be"):
        comparator.trajectories_2d(plane="invalid")


def test_trajectories_2d_missing_data_warning(flight_calisto):
    """Test warning when external data missing required axes.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)

    # Add only 'z', missing 'x', should give a warning
    comparator.add_data("Incomplete", {"z": (time_data, flight_calisto.z(time_data))})

    with pytest.warns(UserWarning, match="No external sources have both"):
        comparator.trajectories_2d(plane="xz")


@pytest.mark.parametrize("filename", [None, "test_trajectory.png"])
def test_trajectories_2d_save(flight_calisto, filename):
    """Test trajectory plot saving.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See conftest.py for more info.
    filename : str or None
        Filename to save plot to, or None to show plot.
    """
    comparator = FlightComparator(flight_calisto)

    time_data = np.linspace(0, flight_calisto.t_final, 100)

    comparator.add_data(
        "Test",
        {
            "x": (time_data, flight_calisto.x(time_data)),
            "z": (time_data, flight_calisto.z(time_data)),
        },
    )

    comparator.trajectories_2d(plane="xz", filename=filename)

    if filename:
        assert os.path.exists(filename)
        os.remove(filename)


# Integration test
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

    # Verify results
    assert isinstance(results, dict)
    assert len(results) >= 4  # At least 4 metrics
    assert "Apogee Altitude (m)" in results
