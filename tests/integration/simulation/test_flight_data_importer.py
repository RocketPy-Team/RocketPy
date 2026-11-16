"""Tests the FlightDataImporter class from rocketpy.simulation module."""

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "unit, value_in_si",
    [
        ("g", 1e-3),
        ("mg", 1e-6),
        ("atm", 101325),
        ("grad", np.pi / 200),
        ("degC", 274.15),
    ],
)
def test_flight_importer_converts_column_to_si(tmp_path, unit, value_in_si):
    """A column given in a non-SI unit should be converted to SI, while the
    data read from the file keeps its original values."""
    path = tmp_path / "log.csv"
    path.write_text("time,value\n0,1\n1,1\n")

    fd = FlightDataImporter(
        paths=str(path),
        columns_map={"time": "time", "value": "value"},
        units={"value": unit},
    )

    assert np.isclose(fd.value(0), value_in_si)
    data = next(iter(fd._data.values()))
    np.testing.assert_array_equal(data[:, 1], [1, 1])


def test_flight_importer_unsupported_unit(tmp_path):
    """An unknown unit should raise the same error as the units module."""
    path = tmp_path / "log.csv"
    path.write_text("time,value\n0,1\n1,1\n")

    with pytest.raises(ValueError, match="Unit 'furlong' is not supported."):
        FlightDataImporter(
            paths=str(path),
            columns_map={"time": "time", "value": "value"},
            units={"value": "furlong"},
        )
