"""Offline unit tests for the Open-Meteo atmospheric models.

Every test here patches the fetchers, so no network requests are made. The live
API is exercised by the ``@pytest.mark.slow`` tests in
``tests/integration/environment/test_environment.py``.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.environment.fetchers import open_meteo_fetcher
from rocketpy.environment.tools import (
    calculate_wind_heading,
    calculate_wind_speed,
    convert_wind_heading_to_direction,
    convert_wind_speed_direction_to_components,
)

# Three pressure levels are enough to build a profile and to check the
# hPa -> Pa, Celsius -> Kelvin and speed/direction -> u/v conversions.
FAKE_LEVELS = {
    1000: {
        "temperature": 15.0,
        "geopotential_height": 100.0,
        "wind_speed": 10.0,
        "wind_direction": 270.0,  # from the west -> blows east -> u > 0
    },
    850: {
        "temperature": 5.0,
        "geopotential_height": 1500.0,
        "wind_speed": 20.0,
        "wind_direction": 0.0,  # from the north -> blows south -> v < 0
    },
    500: {
        "temperature": -20.0,
        "geopotential_height": 5500.0,
        "wind_speed": 30.0,
        "wind_direction": 90.0,  # from the east -> blows west -> u < 0
    },
}

# Two hourly steps, one hour apart, both in the future relative to the fixtures.
FAKE_TIMES = [1_700_000_000, 1_700_003_600]


def _build_hourly(levels=None, member_suffixes=("",), times=None, offset=0.0):
    """Builds a fake Open-Meteo ``hourly`` payload.

    Parameters
    ----------
    levels : dict, optional
        Mapping of pressure level (hPa) to its variables. Defaults to
        :data:`FAKE_LEVELS`.
    member_suffixes : tuple of str, optional
        Member suffixes to emit (``""`` for the deterministic/control run).
    times : list of int, optional
        Unix timestamps for the hourly steps.
    offset : float, optional
        Value added to every member's temperature and wind speed, multiplied by
        the member index, so members differ from one another.
    """
    levels = FAKE_LEVELS if levels is None else levels
    times = FAKE_TIMES if times is None else times
    hourly = {"time": list(times)}

    for member_index, suffix in enumerate(member_suffixes):
        shift = offset * member_index
        for level, variables in levels.items():
            for name, value in variables.items():
                if value is None:
                    values = [None] * len(times)
                elif name in ("temperature", "wind_speed"):
                    values = [value + shift] * len(times)
                else:
                    values = [value] * len(times)
                hourly[f"{name}_{level}hPa{suffix}"] = values

    return hourly


def _build_response(hourly=None, elevation=100.0):
    """Builds a fake Open-Meteo JSON response around ``hourly``."""
    return {
        "latitude": 39.4,
        "longitude": -8.3,
        "elevation": elevation,
        "hourly": _build_hourly() if hourly is None else hourly,
    }


def _patch_forecast(monkeypatch, response=None, recorder=None):
    """Replaces the Open-Meteo forecast fetcher with an offline fake."""
    payload = _build_response() if response is None else response

    def fake_fetch(latitude, longitude, model="best_match", date=None):
        if recorder is not None:
            recorder.update(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "model": model,
                    "date": date,
                }
            )
        return payload

    monkeypatch.setattr(
        "rocketpy.environment.environment.fetch_open_meteo_forecast", fake_fetch
    )


def _patch_ensemble(monkeypatch, response=None, recorder=None):
    """Replaces the Open-Meteo ensemble fetcher with an offline fake."""
    payload = _build_response() if response is None else response

    def fake_fetch(latitude, longitude, model="gfs05", date=None):  # pylint: disable=unused-argument
        if recorder is not None:
            recorder.update({"model": model, "date": date})
        return payload

    monkeypatch.setattr(
        "rocketpy.environment.environment.fetch_open_meteo_ensemble", fake_fetch
    )


class TestWindComponentConversion:
    """Tests for convert_wind_speed_direction_to_components."""

    @pytest.mark.parametrize(
        ("direction", "expected_u", "expected_v"),
        [
            (0.0, 0.0, -10.0),  # from north -> blows south
            (90.0, -10.0, 0.0),  # from east  -> blows west
            (180.0, 0.0, 10.0),  # from south -> blows north
            (270.0, 10.0, 0.0),  # from west  -> blows east
        ],
    )
    def test_cardinal_directions(self, direction, expected_u, expected_v):
        """Convert the four cardinal wind directions to u/v components."""
        u, v = convert_wind_speed_direction_to_components(10.0, direction)

        assert u == pytest.approx(expected_u, abs=1e-9)
        assert v == pytest.approx(expected_v, abs=1e-9)

    @pytest.mark.parametrize("direction", [0.0, 37.0, 135.0, 212.5, 359.0])
    def test_round_trips_back_to_direction(self, direction):
        """Recover the original speed and direction from the components."""
        u, v = convert_wind_speed_direction_to_components(12.5, direction)

        assert calculate_wind_speed(u, v) == pytest.approx(12.5)
        recovered = convert_wind_heading_to_direction(calculate_wind_heading(u, v))
        assert recovered == pytest.approx(direction, abs=1e-9)

    def test_accepts_arrays(self):
        """Convert whole profiles at once, elementwise."""
        speeds = np.array([10.0, 20.0])
        directions = np.array([270.0, 90.0])

        u, v = convert_wind_speed_direction_to_components(speeds, directions)

        assert u == pytest.approx([10.0, -20.0], abs=1e-9)
        assert v == pytest.approx([0.0, 0.0], abs=1e-9)


class TestOpenMeteoForecast:
    """Tests for the ``open_meteo`` atmospheric model."""

    def test_builds_profiles_with_unit_conversions(self, example_euroc_env):
        """Build pressure, temperature and wind profiles from Open-Meteo data.

        Pressure levels arrive in hPa and temperatures in Celsius, so the
        profiles must expose Pa and Kelvin. Heights are geopotential and are
        converted to geometric altitude, which is a sub-metre correction at
        these levels.
        """
        example_euroc_env.set_atmospheric_model(type="open_meteo")

        assert example_euroc_env.atmospheric_model_type == "open_meteo"
        # 1000 hPa -> 100 000 Pa, at ~100 m geometric altitude
        assert example_euroc_env.pressure(100.0) == pytest.approx(100_000.0, rel=1e-3)
        # 15 degC -> 288.15 K
        assert example_euroc_env.temperature(100.0) == pytest.approx(288.15, rel=1e-3)
        # 500 hPa level, at ~5500 m
        assert example_euroc_env.pressure(5500.0) == pytest.approx(50_000.0, rel=1e-3)
        assert example_euroc_env.temperature(5500.0) == pytest.approx(253.15, rel=1e-3)

    def test_converts_wind_direction_to_components(self, example_euroc_env):
        """Turn the reported speed/direction into RocketPy's u/v components."""
        example_euroc_env.set_atmospheric_model(type="open_meteo")

        # 1000 hPa: 10 m/s from the west -> blows east -> u = +10, v = 0
        assert example_euroc_env.wind_velocity_x(100.0) == pytest.approx(10.0, abs=1e-6)
        assert example_euroc_env.wind_velocity_y(100.0) == pytest.approx(0.0, abs=1e-6)
        assert example_euroc_env.wind_speed(100.0) == pytest.approx(10.0, abs=1e-6)
        # The direction is preserved (the wind still comes from the west).
        assert example_euroc_env.wind_direction(100.0) == pytest.approx(270.0, abs=1e-6)
        # And the heading points where the wind blows to.
        assert example_euroc_env.wind_heading(100.0) == pytest.approx(90.0, abs=1e-6)

    def test_forwards_model_and_date_to_fetcher(self, example_euroc_env, monkeypatch):
        """Forward the requested model and the launch date to the fetcher."""
        recorder = {}
        _patch_forecast(monkeypatch, recorder=recorder)

        example_euroc_env.set_atmospheric_model(type="open_meteo", file="ecmwf_ifs025")

        assert recorder["model"] == "ecmwf_ifs025"
        assert recorder["date"] == example_euroc_env.datetime_date
        assert recorder["latitude"] == example_euroc_env.latitude
        assert recorder["longitude"] == example_euroc_env.longitude

    def test_defaults_to_best_match_model(self, example_euroc_env, monkeypatch):
        """Query the ``best_match`` model when none is given."""
        recorder = {}
        _patch_forecast(monkeypatch, recorder=recorder)

        example_euroc_env.set_atmospheric_model(type="open_meteo")

        assert recorder["model"] == "best_match"

    def test_reads_elevation_and_metadata(self, example_euroc_env):
        """Take the launch-site elevation and the period from the response."""
        example_euroc_env.set_atmospheric_model(type="open_meteo")

        assert example_euroc_env.elevation == pytest.approx(100.0)
        assert example_euroc_env.atmospheric_model_init_lat == pytest.approx(39.4)
        assert example_euroc_env.atmospheric_model_init_lon == pytest.approx(-8.3)
        # Two steps one hour apart.
        assert example_euroc_env.atmospheric_model_interval == pytest.approx(1)
        assert example_euroc_env.max_expected_height == pytest.approx(5500.0, rel=1e-3)

    def test_selects_hour_closest_to_launch(self, example_euroc_env, monkeypatch):
        """Pick the hourly step nearest to the launch time.

        The launch date is set 40 minutes past the first step, so the first step
        is the closest one and its values must be the ones used.
        """
        launch = datetime(2024, 6, 1, 12, 40, tzinfo=timezone.utc)
        first = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
        second = datetime(2024, 6, 1, 14, tzinfo=timezone.utc)

        levels = {
            1000: {
                "temperature": 15.0,
                "geopotential_height": 100.0,
                "wind_speed": 10.0,
                "wind_direction": 270.0,
            },
            500: {
                "temperature": -20.0,
                "geopotential_height": 5500.0,
                "wind_speed": 30.0,
                "wind_direction": 90.0,
            },
        }
        hourly = _build_hourly(
            levels=levels, times=[first.timestamp(), second.timestamp()]
        )
        # Make the second step unmistakably different from the first one.
        hourly["temperature_1000hPa"] = [15.0, 99.0]
        _patch_forecast(monkeypatch, response=_build_response(hourly=hourly))

        example_euroc_env.set_date(launch, timezone="UTC")
        example_euroc_env.set_atmospheric_model(type="open_meteo")

        assert example_euroc_env.temperature(100.0) == pytest.approx(288.15, rel=1e-3)

    def test_skips_levels_without_data(self, example_euroc_env, monkeypatch):
        """Drop pressure levels the model does not resolve.

        Open-Meteo returns the same set of level keys for every model but fills
        only the levels the model actually resolves, so a ``None`` level must be
        skipped rather than poison the profile.
        """
        levels = {
            1000: FAKE_LEVELS[1000],
            850: {**FAKE_LEVELS[850], "temperature": None},
            500: FAKE_LEVELS[500],
        }
        _patch_forecast(
            monkeypatch, response=_build_response(hourly=_build_hourly(levels=levels))
        )

        example_euroc_env.set_atmospheric_model(type="open_meteo")

        # Only the 1000 and 500 hPa levels survive.
        assert len(example_euroc_env.levels) == 2
        assert example_euroc_env.levels == pytest.approx([1000.0, 500.0])

    def test_sorts_levels_by_altitude(self, example_euroc_env, monkeypatch):
        """Return a profile monotonic in altitude even if levels arrive unsorted."""
        levels = {
            500: FAKE_LEVELS[500],
            1000: FAKE_LEVELS[1000],
            850: FAKE_LEVELS[850],
        }
        _patch_forecast(
            monkeypatch, response=_build_response(hourly=_build_hourly(levels=levels))
        )

        example_euroc_env.set_atmospheric_model(type="open_meteo")

        assert np.all(np.diff(example_euroc_env.height) > 0)
        # Pressure must decrease as altitude increases.
        assert np.all(np.diff(example_euroc_env.pressure.get_source()[:, 1]) < 0)

    def test_single_usable_level_raises(self, example_euroc_env, monkeypatch):
        """Refuse a collapsed profile instead of failing later during a flight."""
        levels = {
            1000: FAKE_LEVELS[1000],
            850: {key: None for key in FAKE_LEVELS[850]},
            500: {key: None for key in FAKE_LEVELS[500]},
        }
        _patch_forecast(
            monkeypatch, response=_build_response(hourly=_build_hourly(levels=levels))
        )

        with pytest.raises(ValueError, match="fewer than two usable pressure levels"):
            example_euroc_env.set_atmospheric_model(type="open_meteo")

    def test_missing_date_raises(self, example_plain_env, monkeypatch):
        """Require a launch date, since the profile is time-dependent."""
        _patch_forecast(monkeypatch)

        with pytest.raises(ValueError, match="specify the launch date"):
            example_plain_env.set_atmospheric_model(type="open_meteo")

    def test_computes_derived_profiles(self, example_euroc_env):
        """Compute density, speed of sound and viscosity from the new profiles."""
        example_euroc_env.set_atmospheric_model(type="open_meteo")

        # rho = p / (R * T) with R = 287.05 J/(kg K)
        expected_density = 100_000.0 / (287.05 * 288.15)
        assert example_euroc_env.density(100.0) == pytest.approx(
            expected_density, rel=1e-2
        )
        assert example_euroc_env.speed_of_sound(100.0) == pytest.approx(340.3, rel=1e-2)
        assert example_euroc_env.dynamic_viscosity(100.0) > 0


class TestOpenMeteoEnsemble:
    """Tests for the ``open_meteo_ensemble`` atmospheric model."""

    @staticmethod
    def _ensemble_response(num_members=3, offset=1.0):
        """Builds a fake ensemble payload with a control run plus members."""
        suffixes = [""] + [f"_member{index + 1:02d}" for index in range(num_members)]
        return _build_response(
            hourly=_build_hourly(member_suffixes=suffixes, offset=offset)
        )

    def test_stores_every_member(self, example_euroc_env, monkeypatch):
        """Expose the control run plus every perturbed member."""
        _patch_ensemble(monkeypatch, response=self._ensemble_response(num_members=3))

        example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")

        # 3 perturbed members plus the unsuffixed control run.
        assert example_euroc_env.num_ensemble_members == 4
        assert example_euroc_env.height_ensemble.shape == (4, 3)
        assert example_euroc_env.temperature_ensemble.shape == (4, 3)

    def test_control_run_is_member_zero(self, example_euroc_env, monkeypatch):
        """Activate the unperturbed control run by default.

        The documented convention is that member 0 is the control member, so the
        unsuffixed series must come first.
        """
        _patch_ensemble(monkeypatch, response=self._ensemble_response(offset=5.0))

        example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")

        assert example_euroc_env.ensemble_member == 0
        # The control run keeps the unshifted temperature (15 degC -> 288.15 K).
        assert example_euroc_env.temperature(100.0) == pytest.approx(288.15, rel=1e-3)

    def test_select_ensemble_member_switches_profiles(
        self, example_euroc_env, monkeypatch
    ):
        """Switch the active profile when another member is selected."""
        _patch_ensemble(monkeypatch, response=self._ensemble_response(offset=5.0))

        example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")
        control_temperature = example_euroc_env.temperature(100.0)

        example_euroc_env.select_ensemble_member(2)

        assert example_euroc_env.ensemble_member == 2
        # Member 2 is shifted by 2 * 5 degC relative to the control run.
        assert example_euroc_env.temperature(100.0) == pytest.approx(
            control_temperature + 10.0, rel=1e-3
        )

    def test_out_of_range_member_raises(self, example_euroc_env, monkeypatch):
        """Reject a member index beyond the number of members available."""
        _patch_ensemble(monkeypatch, response=self._ensemble_response(num_members=3))

        example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")

        with pytest.raises(ValueError, match="Please choose member from 0 to 3"):
            example_euroc_env.select_ensemble_member(4)

    def test_levels_are_converted_to_pascal(self, example_euroc_env, monkeypatch):
        """Store ensemble pressure levels in Pa, like the netCDF ensembles."""
        _patch_ensemble(monkeypatch, response=self._ensemble_response())

        example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")

        assert example_euroc_env.level_ensemble == pytest.approx(
            [100_000.0, 85_000.0, 50_000.0]
        )

    def test_payload_without_members_raises(self, example_euroc_env, monkeypatch):
        """Fail clearly when the response carries no ensemble members at all."""
        _patch_ensemble(monkeypatch, response=_build_response())

        with pytest.raises(ValueError, match="did not contain any ensemble members"):
            example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")

    def test_forwards_model_to_fetcher(self, example_euroc_env, monkeypatch):
        """Forward the requested ensemble model, defaulting to gfs05."""
        recorder = {}
        _patch_ensemble(
            monkeypatch, response=self._ensemble_response(), recorder=recorder
        )

        example_euroc_env.set_atmospheric_model(
            type="open_meteo_ensemble", file="ecmwf_ifs025"
        )

        assert recorder["model"] == "ecmwf_ifs025"

    def test_missing_date_raises(self, example_plain_env, monkeypatch):
        """Require a launch date for the ensemble model as well."""
        _patch_ensemble(monkeypatch, response=self._ensemble_response())

        with pytest.raises(ValueError, match="specify the launch date"):
            example_plain_env.set_atmospheric_model(type="open_meteo_ensemble")


class TestOpenMeteoFetchers:
    """Tests for the Open-Meteo fetcher helpers (no network access)."""

    def test_build_hourly_variables_covers_every_level(self):
        """Request all four variables for every pressure level."""
        hourly = open_meteo_fetcher.build_hourly_variables()

        variables = hourly.split(",")
        expected_count = 4 * len(open_meteo_fetcher.OPEN_METEO_PRESSURE_LEVELS)
        assert len(variables) == expected_count
        assert "temperature_1000hPa" in variables
        assert "wind_direction_500hPa" in variables
        assert "geopotential_height_30hPa" in variables

    @pytest.mark.parametrize(
        "model",
        [
            "gfs025",  # HTTP 200 but nulls at every pressure level
            "icon_global",  # same
            "bom_access_global_ensemble",  # same
            "gem_global",  # temperature and heights, but no winds at all
        ],
    )
    def test_rejects_ensemble_models_without_complete_data(self, model):
        """Reject ensemble models that cannot produce a full profile.

        These models all answer with HTTP 200, so without an up-front check the
        failure would only surface as an opaque parsing error much later. Note
        that ``gem_global`` is the subtle one: it publishes temperature and
        geopotential height but no pressure-level winds.
        """
        with pytest.raises(ValueError, match="Invalid Open-Meteo ensemble model"):
            open_meteo_fetcher.fetch_open_meteo_ensemble(0.0, 0.0, model=model)

    def test_accepts_the_supported_ensemble_models(self, monkeypatch):
        """Accept the two ensemble models that do publish complete data."""
        monkeypatch.setattr(
            open_meteo_fetcher, "_request", lambda url, params, endpoint: {"hourly": {}}
        )

        for model in ("gfs05", "ecmwf_ifs025"):
            open_meteo_fetcher.fetch_open_meteo_ensemble(0.0, 0.0, model=model)

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            (timedelta(days=-30), True),
            (timedelta(days=-2), True),
            (timedelta(hours=-1), False),
            (timedelta(days=2), False),
        ],
    )
    def test_past_date_detection(self, delta, expected):
        """Route only genuinely past dates to the historical archive.

        The forecast endpoint keeps a couple of past days available, so a launch
        date a few hours ago must stay on the forecast endpoint.
        """
        date = datetime.now(timezone.utc) + delta

        assert open_meteo_fetcher._is_past_date(date) is expected

    def test_naive_dates_are_treated_as_utc(self):
        """Assume UTC for naive datetimes instead of raising."""
        naive_past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=5)

        assert open_meteo_fetcher._is_past_date(naive_past) is True

    def test_no_date_uses_forecast_endpoint(self):
        """Treat a missing date as a plain forecast request."""
        assert open_meteo_fetcher._is_past_date(None) is False

    def test_past_date_queries_historical_endpoint(self, monkeypatch):
        """Send past launch dates to the historical-forecast API.

        Open-Meteo's ERA5 archive endpoint serves surface variables only, so the
        historical-forecast API is the only archive that can feed a vertical
        profile.
        """
        recorder = {}

        def fake_request(url, params, endpoint):
            recorder.update({"url": url, "params": params, "endpoint": endpoint})
            return {"hourly": {}}

        monkeypatch.setattr(open_meteo_fetcher, "_request", fake_request)

        open_meteo_fetcher.fetch_open_meteo_forecast(
            39.4, -8.3, date=datetime(2024, 1, 10, 12, tzinfo=timezone.utc)
        )

        assert recorder["url"] == open_meteo_fetcher.OPEN_METEO_HISTORICAL_URL
        # A one-day pad on each side keeps the launch hour inside the window.
        assert recorder["params"]["start_date"] == "2024-01-09"
        assert recorder["params"]["end_date"] == "2024-01-11"

    def test_warns_for_dates_before_the_archive_starts(self, monkeypatch):
        """Warn when the launch date predates Open-Meteo's archive.

        Such requests answer with HTTP 200 and nulls at every level, so without
        a warning the user would only see a generic "not enough pressure levels"
        error with no hint that the date is the problem.
        """
        monkeypatch.setattr(
            open_meteo_fetcher, "_request", lambda url, params, endpoint: {"hourly": {}}
        )

        with pytest.warns(UserWarning, match="precedes Open-Meteo's"):
            open_meteo_fetcher.fetch_open_meteo_forecast(
                39.4, -8.3, date=datetime(2019, 6, 15, tzinfo=timezone.utc)
            )

    def test_does_not_warn_for_supported_past_dates(self, monkeypatch, recwarn):
        """Stay silent for past dates the archive does cover."""
        monkeypatch.setattr(
            open_meteo_fetcher, "_request", lambda url, params, endpoint: {"hourly": {}}
        )

        open_meteo_fetcher.fetch_open_meteo_forecast(
            39.4, -8.3, date=datetime(2024, 1, 10, tzinfo=timezone.utc)
        )

        assert not [w for w in recwarn if "precedes Open-Meteo" in str(w.message)]

    def test_future_date_queries_forecast_endpoint(self, monkeypatch):
        """Send future launch dates to the regular forecast API."""
        recorder = {}

        def fake_request(url, params, endpoint):  # pylint: disable=unused-argument
            recorder.update({"url": url, "params": params})
            return {"hourly": {}}

        monkeypatch.setattr(open_meteo_fetcher, "_request", fake_request)

        open_meteo_fetcher.fetch_open_meteo_forecast(
            39.4, -8.3, date=datetime.now(timezone.utc) + timedelta(days=2)
        )

        assert recorder["url"] == open_meteo_fetcher.OPEN_METEO_FORECAST_URL
        assert "start_date" not in recorder["params"]

    def test_requests_wind_in_metres_per_second(self, monkeypatch):
        """Ask for m/s so no unit conversion is needed downstream.

        Open-Meteo defaults to km/h, which would silently inflate wind speeds by
        3.6x if the parameter were dropped.
        """
        recorder = {}

        def fake_request(url, params, endpoint):  # pylint: disable=unused-argument
            recorder.update(params)
            return {"hourly": {}}

        monkeypatch.setattr(open_meteo_fetcher, "_request", fake_request)

        open_meteo_fetcher.fetch_open_meteo_forecast(39.4, -8.3)

        assert recorder["wind_speed_unit"] == "ms"
        assert recorder["timeformat"] == "unixtime"
        assert recorder["timezone"] == "UTC"

    def test_api_error_payload_raises_runtime_error(self, monkeypatch):
        """Surface Open-Meteo's own error message instead of a bare status code."""

        class FakeResponse:
            """Stands in for an Open-Meteo error response."""

            ok = False
            status_code = 400

            @staticmethod
            def json():
                return {"error": True, "reason": "Invalid time interval"}

        monkeypatch.setattr(
            open_meteo_fetcher, "_get_json", lambda url, params: FakeResponse()
        )

        with pytest.raises(RuntimeError, match="Invalid time interval"):
            open_meteo_fetcher.fetch_open_meteo_forecast(39.4, -8.3)

    def test_response_without_hourly_raises_runtime_error(self, monkeypatch):
        """Fail clearly when the response carries no hourly block."""

        class FakeResponse:
            """Stands in for a successful response missing its hourly block."""

            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"latitude": 39.4, "longitude": -8.3}

        monkeypatch.setattr(
            open_meteo_fetcher, "_get_json", lambda url, params: FakeResponse()
        )

        with pytest.raises(RuntimeError, match="did not contain any hourly data"):
            open_meteo_fetcher.fetch_open_meteo_forecast(39.4, -8.3)


class TestOpenMeteoSerialization:
    """Tests that Open-Meteo environments survive a to_dict/from_dict cycle."""

    def test_forecast_round_trip(self, example_euroc_env):
        """Preserve profiles and metadata for the deterministic model."""
        example_euroc_env.set_atmospheric_model(type="open_meteo")

        restored = Environment.from_dict(example_euroc_env.to_dict())

        assert restored.atmospheric_model_type == "open_meteo"
        assert restored.pressure(1000.0) == pytest.approx(
            example_euroc_env.pressure(1000.0)
        )
        assert restored.wind_direction(1000.0) == pytest.approx(
            example_euroc_env.wind_direction(1000.0)
        )
        assert restored.atmospheric_model_init_lat == pytest.approx(
            example_euroc_env.atmospheric_model_init_lat
        )

    def test_ensemble_round_trip(self, example_euroc_env, monkeypatch):
        """Preserve every member so selection still works after reloading."""
        _patch_ensemble(
            monkeypatch,
            response=TestOpenMeteoEnsemble._ensemble_response(
                num_members=3, offset=5.0
            ),
        )
        example_euroc_env.set_atmospheric_model(type="open_meteo_ensemble")

        restored = Environment.from_dict(example_euroc_env.to_dict())

        assert restored.num_ensemble_members == 4
        restored.select_ensemble_member(2)
        example_euroc_env.select_ensemble_member(2)
        assert restored.temperature(100.0) == pytest.approx(
            example_euroc_env.temperature(100.0)
        )


@pytest.fixture(autouse=True)
def _patch_forecast_fetcher_by_default(monkeypatch):
    """Patches the forecast fetcher for every test in this module.

    Keeps the whole module offline: tests that need a custom payload patch the
    fetcher again, which simply overrides this one.
    """
    _patch_forecast(monkeypatch)
