"""Unit tests for atmosphere netCDF/JSON disk caching (#654)."""

from datetime import datetime
from unittest.mock import MagicMock

import netCDF4
import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.environment import atmosphere_cache


def _write_minimal_profile_nc(path, elevation=1400.0):
    """Create a tiny valid profile cache file for apply tests."""
    height = np.array([1400.0, 5000.0, 10000.0])
    pressure = np.array([85000.0, 54000.0, 26500.0])
    temperature = np.array([288.0, 255.0, 223.0])
    wind_u = np.array([1.0, 2.0, 3.0])
    wind_v = np.array([-1.0, 0.0, 1.0])
    assert atmosphere_cache.write_profile_netcdf(
        path,
        height=height,
        pressure=pressure,
        temperature=temperature,
        wind_u=wind_u,
        wind_v=wind_v,
        elevation=elevation,
        max_expected_height=10000.0,
        kind="forecast",
    )


def test_cache_root_honors_rocketpy_cache_env(monkeypatch, tmp_path):
    """``ROCKETPY_CACHE`` redirects the atmosphere cache root."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    assert atmosphere_cache.get_cache_root() == tmp_path
    assert atmosphere_cache.get_atmosphere_cache_dir() == tmp_path / "atmosphere"


def test_profile_netcdf_roundtrip(monkeypatch, tmp_path):
    """Write and read forecast profile netCDF through the cache helpers."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("forecast_test_key", ".nc")
    _write_minimal_profile_nc(path)
    loaded = atmosphere_cache.read_profile_netcdf(path)
    assert loaded is not None
    assert loaded["elevation"] == pytest.approx(1400.0)
    np.testing.assert_allclose(loaded["height"], [1400.0, 5000.0, 10000.0])
    np.testing.assert_allclose(loaded["pressure"], [85000.0, 54000.0, 26500.0])


def test_json_cache_roundtrip(monkeypatch, tmp_path):
    """Windy-style JSON cache round-trips through disk."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("windy_test_key", ".json")
    payload = {"data": {"hours": [1, 2, 3], "temp-surface": [288]}}
    assert atmosphere_cache.save_json_cache(path, payload)
    assert atmosphere_cache.load_json_cache(path) == payload


def test_forecast_shortcut_reuses_disk_cache(monkeypatch, tmp_path):
    """Second Forecast shortcut call loads profiles from disk (no re-fetch)."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    fixture = "data/weather/SpaceportAmerica_2018_ERA-5.nc"
    fetch_calls = []

    def fake_fetch():
        fetch_calls.append(1)
        return netCDF4.Dataset(fixture)

    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    env.set_date((2018, 10, 15, 12))
    env._Environment__atm_type_file_to_function_map["forecast"]["GFS"] = fake_fetch

    env.set_atmospheric_model(
        type="Forecast",
        file="GFS",
        dictionary="ECMWF",
        pressure_conversion_factor="hPa",
    )
    assert len(fetch_calls) == 1
    pressure_first = env.pressure(env.elevation)
    cached_files = list((tmp_path / "atmosphere").glob("*.nc"))
    assert cached_files, "Expected a profile .nc cache file after first fetch"

    env2 = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    env2.set_date((2018, 10, 15, 12))
    env2._Environment__atm_type_file_to_function_map["forecast"]["GFS"] = fake_fetch
    env2.set_atmospheric_model(
        type="Forecast",
        file="GFS",
        dictionary="ECMWF",
        pressure_conversion_factor="hPa",
    )
    assert len(fetch_calls) == 1, "Second call should reuse disk cache"
    assert env2.pressure(env2.elevation) == pytest.approx(pressure_first, rel=1e-6)


def test_forecast_no_cache_bypasses_disk(monkeypatch, tmp_path):
    """``no_cache=True`` forces a re-fetch even when a cache file exists."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    fixture = "data/weather/SpaceportAmerica_2018_ERA-5.nc"
    fetch_calls = []

    def fake_fetch():
        fetch_calls.append(1)
        return netCDF4.Dataset(fixture)

    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    env.set_date((2018, 10, 15, 12))
    env._Environment__atm_type_file_to_function_map["forecast"]["GFS"] = fake_fetch

    env.set_atmospheric_model(
        type="Forecast",
        file="GFS",
        dictionary="ECMWF",
        pressure_conversion_factor="hPa",
    )
    env.set_atmospheric_model(
        type="Forecast",
        file="GFS",
        dictionary="ECMWF",
        pressure_conversion_factor="hPa",
        no_cache=True,
    )
    assert len(fetch_calls) == 2


def test_windy_json_cache_hit(monkeypatch, tmp_path):
    """Windy response is cached as JSON; second call skips the network."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))

    # Minimal Windy payload matching __parse_windy_file expectations.
    levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150]
    payload = {
        "header": {"elevation": 1234.0},
        "data": {
            "hours": [1_540_000_000_000, 1_540_003_600_000],
        },
    }
    for level in levels:
        # Geopotential heights increasing with altitude (decreasing pressure).
        payload["data"][f"gh-{level}h"] = [
            float(2000 + (1000 - level) * 10),
            float(2000 + (1000 - level) * 10),
        ]
        payload["data"][f"temp-{level}h"] = [280.0, 281.0]
        payload["data"][f"wind_u-{level}h"] = [1.0, 1.5]
        payload["data"][f"wind_v-{level}h"] = [-1.0, -0.5]

    fetch_mock = MagicMock(return_value=payload)
    monkeypatch.setattr(
        "rocketpy.environment.environment.fetch_atmospheric_data_from_windy",
        fetch_mock,
    )

    env = Environment(latitude=45.0, longitude=10.0, elevation=100)
    env.set_date(datetime(2018, 10, 15, 12))
    env.set_atmospheric_model(type="Windy", file="ECMWF")
    assert fetch_mock.call_count == 1
    assert list((tmp_path / "atmosphere").glob("*.json"))

    env2 = Environment(latitude=45.0, longitude=10.0, elevation=100)
    env2.set_date(datetime(2018, 10, 15, 12))
    env2.set_atmospheric_model(type="Windy", file="ECMWF")
    assert fetch_mock.call_count == 1

    env3 = Environment(latitude=45.0, longitude=10.0, elevation=100)
    env3.set_date(datetime(2018, 10, 15, 12))
    env3.set_atmospheric_model(type="Windy", file="ECMWF", no_cache=True)
    assert fetch_mock.call_count == 2
