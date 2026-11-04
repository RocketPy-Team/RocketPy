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


def test_flight_importer_converts_imperial_and_degree_columns(tmp_path):
    """Columns logged in ft/s^2 and degrees should be converted to SI."""
    path = tmp_path / "log.csv"
    path.write_text("time,accel_ft_s2,pitch_deg\n0,32.174,90\n1,32.174,90\n")

    fd = FlightDataImporter(
        paths=str(path),
        columns_map={
            "time": "time",
            "accel_ft_s2": "az",
            "pitch_deg": "attitude_angle",
        },
        units={"accel_ft_s2": "ft/s^2", "pitch_deg": "deg"},
    )

    assert np.isclose(fd.az(0), 32.174 * 0.3048)
    assert np.isclose(fd.attitude_angle(0), np.pi / 2)
