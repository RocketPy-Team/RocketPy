"""Unit tests for atmosphere netCDF/JSON disk caching (#654)."""

import time
import warnings
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


# ---------------------------------------------------------------------------
# Regression tests for the cache-hit / fresh-fetch parity contract
# ---------------------------------------------------------------------------


#: Attributes ``Environment`` derives from the dataset. A cache hit must
#: reproduce every one of them, otherwise ``info()`` and ``to_dict()`` break on
#: the second run of an otherwise identical script.
DERIVED_MODEL_ATTRIBUTES = [
    "atmospheric_model_init_date",
    "atmospheric_model_end_date",
    "atmospheric_model_interval",
    "atmospheric_model_init_lat",
    "atmospheric_model_end_lat",
    "atmospheric_model_init_lon",
    "atmospheric_model_end_lon",
    "lat_array",
    "lon_array",
    "lat_index",
    "lon_index",
    "geopotentials",
    "wind_us",
    "wind_vs",
    "levels",
    "temperatures",
    "time_array",
    "height",
]


def _forecast_env(fetch):
    """Build an Environment wired to ``fetch`` and load the Forecast model."""
    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    env.set_date((2018, 10, 15, 12))
    env._Environment__atm_type_file_to_function_map["forecast"]["GFS"] = fetch
    env.set_atmospheric_model(
        type="Forecast",
        file="GFS",
        dictionary="ECMWF",
        pressure_conversion_factor="hPa",
    )
    return env


def _counting_fetch(calls, fixture="data/weather/SpaceportAmerica_2018_ERA-5.nc"):
    def fetch():
        calls.append(1)
        return netCDF4.Dataset(fixture)

    return fetch


def test_cache_hit_restores_every_derived_attribute(monkeypatch, tmp_path):
    """A cache hit must rebuild the same Environment a fresh fetch produces."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    calls = []
    fetch = _counting_fetch(calls)

    fresh = _forecast_env(fetch)
    cached = _forecast_env(fetch)
    assert len(calls) == 1, "second call should have been served from disk"

    for name in DERIVED_MODEL_ATTRIBUTES:
        expected = getattr(fresh, name)
        actual = getattr(cached, name, None)
        assert actual is not None, f"'{name}' was lost on the cached path"
        if isinstance(expected, datetime):
            assert actual == expected, name
        else:
            np.testing.assert_allclose(
                np.ma.filled(np.ma.asarray(actual, dtype=float), np.nan),
                np.ma.filled(np.ma.asarray(expected, dtype=float), np.nan),
                rtol=1e-10,
                err_msg=name,
            )


def test_cache_hit_environment_can_print_info(monkeypatch, tmp_path):
    """``info()`` used to raise AttributeError on the second (cached) run."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    calls = []
    fetch = _counting_fetch(calls)

    _forecast_env(fetch)
    cached = _forecast_env(fetch)

    assert len(calls) == 1
    cached.info()  # must not raise


def test_cache_hit_matches_fresh_profiles(monkeypatch, tmp_path):
    """Profiles served from disk are numerically identical to a fresh fetch."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    calls = []
    fetch = _counting_fetch(calls)

    fresh = _forecast_env(fetch)
    cached = _forecast_env(fetch)

    heights = np.linspace(fresh.elevation, fresh.max_expected_height, 25)
    for name in (
        "pressure",
        "temperature",
        "wind_velocity_x",
        "wind_velocity_y",
        "wind_speed",
        "wind_heading",
        "wind_direction",
    ):
        np.testing.assert_allclose(
            [getattr(cached, name)(h) for h in heights],
            [getattr(fresh, name)(h) for h in heights],
            rtol=1e-10,
            atol=1e-10,
            err_msg=name,
        )


def test_constant_wind_profile_does_not_break_caching(monkeypatch, tmp_path):
    """Saving must tolerate scalar profiles instead of raising IndexError.

    ``set_atmospheric_model`` used to index every profile as a 2-D array while
    only checking ``pressure``, so a constant wind blew up the cache write.
    """
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    env = Environment(latitude=0, longitude=0, elevation=0)
    env._Environment__atm_type_file_to_function_map = {
        "forecast": {"GFS": lambda: "fake-dataset"},
        "ensemble": {},
    }
    env.process_forecast_reanalysis = lambda dataset, dictionary, conversion_factor: (
        None
    )

    env.set_atmospheric_model(type="Forecast", file="gfs")  # must not raise

    assert not list((tmp_path / "atmosphere").glob("*.nc")), (
        "nothing worth caching should have been written"
    )


def test_cache_disabled_by_environment_variable(monkeypatch, tmp_path):
    """``ROCKETPY_CACHE=0`` turns the disk cache off entirely."""
    monkeypatch.setenv("ROCKETPY_CACHE", "0")
    assert atmosphere_cache.is_cache_enabled() is False

    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    assert atmosphere_cache.is_cache_enabled() is True


def test_disabled_cache_refetches_every_time(monkeypatch, tmp_path):
    """With caching off, a repeated request hits the network again."""
    monkeypatch.setenv("ROCKETPY_CACHE", "off")
    calls = []
    fetch = _counting_fetch(calls)

    _forecast_env(fetch)
    _forecast_env(fetch)

    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.nc"))


def test_expired_forecast_entry_is_refetched(monkeypatch, tmp_path):
    """Forecast entries older than the TTL are discarded, not served."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    monkeypatch.setenv("ROCKETPY_CACHE_TTL", "3600")
    calls = []
    fetch = _counting_fetch(calls)

    _forecast_env(fetch)
    assert len(calls) == 1

    # Pretend the entry was written two hours ago. The real clock has to be
    # sampled before patching, or the stub would call itself.
    two_hours_from_now = time.time() + 7200
    monkeypatch.setattr(atmosphere_cache.time, "time", lambda: two_hours_from_now)
    _forecast_env(fetch)
    assert len(calls) == 2, "a stale forecast must not be reused"


def test_zero_ttl_disables_expiry(monkeypatch):
    """``ROCKETPY_CACHE_TTL=0`` keeps entries forever."""
    monkeypatch.setenv("ROCKETPY_CACHE_TTL", "0")
    assert atmosphere_cache.is_entry_expired(0.0, "forecast") is False


def test_reanalysis_entries_never_expire(monkeypatch):
    """Reanalysis data is immutable, so the TTL does not apply to it."""
    monkeypatch.setenv("ROCKETPY_CACHE_TTL", "1")
    assert atmosphere_cache.is_entry_expired(0.0, "reanalysis") is False
    assert atmosphere_cache.is_entry_expired(0.0, "forecast") is True


def test_different_dictionary_uses_a_separate_entry(monkeypatch, tmp_path):
    """The same source decoded with another dictionary must not collide."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    calls = []
    fetch = _counting_fetch(calls)

    _forecast_env(fetch)

    env = Environment(
        latitude=32.990254, longitude=-106.974998, elevation=1400, datum="WGS84"
    )
    env.set_date((2018, 10, 15, 12))
    env._Environment__atm_type_file_to_function_map["forecast"]["GFS"] = fetch
    env.set_atmospheric_model(
        type="Forecast",
        file="GFS",
        dictionary="ECMWF_v0",
        pressure_conversion_factor="hPa",
    )

    assert len(calls) == 2, "a different dictionary must miss the cache"
    assert len(list((tmp_path / "atmosphere").glob("*.nc"))) == 2


def test_corrupt_cache_file_falls_back_to_fetch(monkeypatch, tmp_path):
    """A damaged cache file degrades to a re-fetch instead of raising."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    calls = []
    fetch = _counting_fetch(calls)

    _forecast_env(fetch)
    cached_file = next((tmp_path / "atmosphere").glob("*.nc"))
    cached_file.write_bytes(b"this is not a netCDF file")

    with pytest.warns(UserWarning):
        _forecast_env(fetch)

    assert len(calls) == 2


def test_clear_atmosphere_cache_removes_entries(monkeypatch, tmp_path):
    """``clear_atmosphere_cache`` empties the cache directory."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("forecast_clear_me", ".nc")
    _write_minimal_profile_nc(path)
    assert path.is_file()

    assert atmosphere_cache.clear_atmosphere_cache() is True
    assert not path.is_file()
    # Clearing an already-absent cache is not an error.
    assert atmosphere_cache.clear_atmosphere_cache() is True


def test_json_cache_respects_ttl(monkeypatch, tmp_path):
    """Windy JSON entries expire like the netCDF ones."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    monkeypatch.setenv("ROCKETPY_CACHE_TTL", "3600")
    path = atmosphere_cache.cache_path_for("windy_ttl", ".json")
    payload = {"data": {"hours": [1, 2, 3]}}

    assert atmosphere_cache.save_json_cache(path, payload)
    assert atmosphere_cache.load_json_cache(path) == payload

    two_hours_from_now = time.time() + 7200
    monkeypatch.setattr(atmosphere_cache.time, "time", lambda: two_hours_from_now)
    assert atmosphere_cache.load_json_cache(path) is None


def test_mismatched_profiles_are_skipped_without_warning(monkeypatch, tmp_path):
    """A scalar wind profile is skipped cleanly, not written and not warned about.

    The save path used to check only ``pressure`` before slicing all four
    profiles as 2-D arrays, so a constant wind raised ``IndexError``. Guarding
    only ``pressure`` is not enough either: ``np.asarray(None, dtype=float)``
    silently yields ``nan``, so a missing column would be persisted as a cache
    entry full of NaN winds. Checking every column keeps the write from being
    attempted at all, which is why this asserts on the absence of a warning and
    not just of a file.
    """
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    env = Environment(latitude=0, longitude=0, elevation=0)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=[[0.0, 101325.0], [1000.0, 89875.0]],
        temperature=[[0.0, 288.0], [1000.0, 281.0]],
        wind_u=5,
        wind_v=-3,
    )
    assert env.pressure.is_array_source()
    assert not env.wind_velocity_x.is_array_source()

    path = atmosphere_cache.cache_path_for("forecast_mismatched", ".nc")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env._Environment__save_forecast_profiles_to_cache(path)

    assert not path.exists()


# ---------------------------------------------------------------------------
# The cache swallows unusable-file errors, and nothing else
# ---------------------------------------------------------------------------


def test_unexpected_error_while_opening_is_not_swallowed(monkeypatch, tmp_path):
    """A defect inside the cache must surface, not look like a cache miss."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("forecast_boom", ".nc")
    _write_minimal_profile_nc(path)

    def explode(*_args, **_kwargs):
        raise ZeroDivisionError("a bug, not a damaged file")

    monkeypatch.setattr(atmosphere_cache.netCDF4, "Dataset", explode)

    with pytest.raises(ZeroDivisionError):
        atmosphere_cache.read_profile_netcdf(path)


def test_unexpected_error_while_reading_is_not_swallowed(monkeypatch, tmp_path):
    """Same contract for failures after the file has been opened."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("forecast_boom_read", ".nc")
    _write_minimal_profile_nc(path)

    def explode(_dataset):
        raise ZeroDivisionError("a bug, not a damaged file")

    monkeypatch.setattr(atmosphere_cache, "_read_metadata", explode)

    with pytest.raises(ZeroDivisionError):
        atmosphere_cache.read_profile_netcdf(path)


@pytest.mark.parametrize(
    "error",
    [
        OSError("damaged file"),
        RuntimeError("dataset is closed"),
        ValueError("wrong length"),
        TypeError("bad dtype"),
        KeyError("missing variable"),
        AttributeError("missing attribute"),
    ],
)
def test_unusable_cache_file_degrades_to_a_miss(monkeypatch, tmp_path, error):
    """Every way netCDF4 reports an unusable file must become a cache miss."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("forecast_unusable", ".nc")
    _write_minimal_profile_nc(path)

    def explode(_dataset):
        raise error

    monkeypatch.setattr(atmosphere_cache, "_read_metadata", explode)

    with pytest.warns(UserWarning):
        assert atmosphere_cache.read_profile_netcdf(path) is None


def test_write_failure_degrades_to_no_cache_entry(monkeypatch, tmp_path):
    """A failed write warns and reports False instead of raising."""
    monkeypatch.setenv("ROCKETPY_CACHE", str(tmp_path))
    path = atmosphere_cache.cache_path_for("forecast_writefail", ".nc")

    def explode(_dataset, _metadata):
        raise OSError("disk full")

    monkeypatch.setattr(atmosphere_cache, "_write_metadata", explode)

    with pytest.warns(UserWarning):
        written = atmosphere_cache.write_profile_netcdf(
            path,
            height=np.array([0.0, 1000.0]),
            pressure=np.array([101325.0, 89875.0]),
            temperature=np.array([288.0, 281.0]),
            wind_u=np.array([1.0, 2.0]),
            wind_v=np.array([0.0, 1.0]),
            elevation=0.0,
            max_expected_height=1000.0,
        )

    assert written is False
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp")), "temporary file was left behind"
