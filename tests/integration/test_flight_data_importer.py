"""Tests the FlightDataImporter class from rocketpy.simulation module."""

import numpy as np

from rocketpy.simulation import FlightDataImporter


def test_flight_importer_bella_lui():
    """Tests the class using the Bella Lui flight data."""
    columns_map = {
        "time_aprox_(s)": "time",
        "z_(m)": "altitude",
        "v_(m/s)": "vz",
    }
    path = "data/rockets/EPFL_Bella_Lui/bella_lui_flight_data_filtered.csv"

    fd = FlightDataImporter(
        name="Bella Lui, EPFL Rocket Team, 2020",
        paths=path,
        columns_map=columns_map,
        units=None,
        interpolation="linear",
        extrapolation="zero",
        delimiter=",",
        encoding="utf-8",
    )
    assert fd.name == "Bella Lui, EPFL Rocket Team, 2020"
    assert "time" in fd._columns[path], "Can't find 'time' column in fd._columns"
    assert "altitude" in fd._columns[path], (
        "Can't find 'altitude' column in fd._columns"
    )
    assert "vz" in fd._columns[path], "Can't find 'vz' column in fd._columns"
    assert np.isclose(fd.altitude(0), 0.201, atol=1e-4)
    assert np.isclose(fd.vz(0), 5.028, atol=1e-4)


def test_flight_importer_ndrt():
    """Tests the class using the NDRT 2020 flight data."""
    columns_map = {
        "Time_(s)": "time",
        "Altitude_(Ft-AGL)": "altitude",
    }
    units = {"Altitude_(Ft-AGL)": "ft"}
    path = "data/rockets/NDRT_2020/ndrt_2020_flight_data.csv"

    fd = FlightDataImporter(
        name="NDRT Rocket team, 2020",
        paths=[path],
        columns_map=columns_map,
        units=units,
    )
    assert fd.name == "NDRT Rocket team, 2020"
    assert "time" in fd._columns[path], "Can't find 'time' column in fd._columns"
    assert "altitude" in fd._columns[path], (
        "Can't find 'altitude' column in fd._columns"
    )
    assert np.isclose(fd.altitude(0), 0)
