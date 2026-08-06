from datetime import datetime, timedelta, timezone

import pytest

from rocketpy.environment import fetchers


@pytest.mark.parametrize(
    "fetcher,expected_url",
    [
        (
            fetchers.fetch_gfs_file_return_dataset,
            "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best",
        ),
        (
            fetchers.fetch_nam_file_return_dataset,
            "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NAM/CONUS_12km/Best",
        ),
        (
            fetchers.fetch_rap_file_return_dataset,
            "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/RAP/CONUS_13km/Best",
        ),
    ],
)
def test_fetcher_returns_dataset_on_first_attempt(fetcher, expected_url, monkeypatch):
    """Return dataset immediately when the first OPeNDAP attempt succeeds."""
    # Arrange
    calls = []
    sentinel_dataset = object()

    def fake_dataset(url):
        calls.append(url)
        return sentinel_dataset

    monkeypatch.setattr(fetchers.netCDF4, "Dataset", fake_dataset)

    # Act
    dataset = fetcher(max_attempts=3, base_delay=2)

    # Assert
    assert dataset is sentinel_dataset
    assert calls == [expected_url]


def test_fetch_gfs_retries_then_succeeds(monkeypatch):
    """Retry GFS fetch after OSError and return data once endpoint responds."""
    # Arrange
    attempt_counter = {"count": 0}
    sleep_calls = []

    def fake_dataset(_):
        attempt_counter["count"] += 1
        if attempt_counter["count"] < 3:
            raise OSError("temporary failure")
        return "gfs-dataset"

    monkeypatch.setattr(fetchers.netCDF4, "Dataset", fake_dataset)
    monkeypatch.setattr(fetchers.time, "sleep", sleep_calls.append)

    # Act
    dataset = fetchers.fetch_gfs_file_return_dataset(max_attempts=3, base_delay=2)

    # Assert
    assert dataset == "gfs-dataset"
    assert sleep_calls == [2, 4]


def test_fetch_rap_raises_runtime_error_after_max_attempts(monkeypatch):
    """Raise RuntimeError when all RAP attempts fail with OSError."""
    # Arrange
    sleep_calls = []

    def always_fails(_):
        raise OSError("endpoint down")

    monkeypatch.setattr(fetchers.netCDF4, "Dataset", always_fails)
    monkeypatch.setattr(fetchers.time, "sleep", sleep_calls.append)

    # Act / Assert
    with pytest.raises(
        RuntimeError, match="Unable to load latest weather data for RAP"
    ):
        fetchers.fetch_rap_file_return_dataset(max_attempts=2, base_delay=2)

    assert sleep_calls == [2, 4]


class _FakeResponse:
    """Minimal stand-in for a ``requests.Response`` used in Meteomatics tests."""

    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fetchers.requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def _meteomatics_value_for(parameter):
    """Return a deterministic fake value for a Meteomatics parameter string."""
    if parameter.startswith("t_"):
        return 288.0
    if parameter.startswith("pressure_"):
        return 90000.0
    if parameter.startswith("wind_speed_u_"):
        return 4.0
    if parameter.startswith("wind_speed_v_"):
        return -2.0
    raise AssertionError(f"unexpected parameter requested: {parameter}")


def _make_fake_meteomatics_get(calls, extra_bad_parameter=False, data_status=200):
    """Build a fake ``requests.get`` that mimics the Meteomatics endpoints."""

    def fake_get(url, headers=None, params=None, **_kwargs):
        calls.append((url, params))
        if url == fetchers.METEOMATICS_LOGIN_URL:
            assert headers is not None and "Authorization" in headers
            return _FakeResponse({"access_token": "fake-token"})
        if data_status >= 400:
            return _FakeResponse(
                {}, status_code=data_status, text="validation error: altitude"
            )
        # Data request: parameters are the 5th path segment.
        parameters = url.split("/")[4].split(",")
        data = [
            {
                "parameter": parameter,
                "coordinates": [
                    {"dates": [{"value": _meteomatics_value_for(parameter)}]}
                ],
            }
            for parameter in parameters
        ]
        if extra_bad_parameter:
            data.append(
                {
                    "parameter": "not_a_known_parameter:xx",
                    "coordinates": [{"dates": [{"value": 1.0}]}],
                }
            )
        return _FakeResponse({"data": data})

    return fake_get


def test_fetch_meteomatics_token_success(monkeypatch):
    """Return the access token when the login service responds with one."""
    monkeypatch.setattr(
        fetchers.requests, "get", lambda *a, **k: _FakeResponse({"access_token": "tok"})
    )
    assert fetchers.fetch_meteomatics_token("user", "pass") == "tok"


def test_fetch_meteomatics_token_missing_token_raises(monkeypatch):
    """Raise when the login service returns 200 but without a token."""
    monkeypatch.setattr(fetchers.requests, "get", lambda *a, **k: _FakeResponse({}))
    with pytest.raises(RuntimeError, match="did not return an access token"):
        fetchers.fetch_meteomatics_token("user", "pass")


def test_fetch_meteomatics_token_auth_failure_not_retried(monkeypatch):
    """A 401/403 is a definitive auth failure: report clearly and do not retry."""
    calls = []

    def fake_get(*args, **_kwargs):
        calls.append(args)
        return _FakeResponse({}, status_code=401, text="unauthorized")

    # If a retry happened it would sleep; make that observable instead of slow.
    monkeypatch.setattr(
        fetchers.time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(fetchers.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="rejected the credentials"):
        fetchers.fetch_meteomatics_token("user", "pass")
    assert len(calls) == 1  # no retries


def test_fetch_meteomatics_data_groups_and_parses(monkeypatch):
    """Group parameters within the query limit and parse the profiles."""
    # Arrange
    calls = []
    monkeypatch.setattr(fetchers.requests, "get", _make_fake_meteomatics_get(calls))

    # Act: distinct wind (fine) and temperature/pressure (coarse) resolutions so
    # a fine-vs-coarse grid swap would be detectable.
    profiles = fetchers.fetch_atmospheric_data_from_meteomatics(
        username="user",
        password="pass",
        latitude=39.0,
        longitude=-8.0,
        date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
        model="mix",
        min_altitude=10,
        max_altitude=1000,
        wind_resolution=3,
        temperature_pressure_resolution=2,
        query_limit=3,
    )

    # Assert
    # 6 wind params (u,v at 3 levels) + 4 temp/pressure params (t,p at 2 levels)
    # = 10 params, grouped by 3 -> ceil(10/3) = 4 groups.
    data_calls = [c for c in calls if c[0] != fetchers.METEOMATICS_LOGIN_URL]
    assert len(calls) == 5  # 1 token + 4 data groups
    assert len(data_calls) == 4
    assert all(call[1]["access_token"] == "fake-token" for call in data_calls)
    assert all(call[1]["model"] == "mix" for call in data_calls)

    # Wind uses the fine grid (3 levels); temperature/pressure the coarse (2).
    assert profiles["temperature"] == {10: 288.0, 1000: 288.0}
    assert profiles["pressure"] == {10: 90000.0, 1000: 90000.0}
    assert profiles["wind_u"] == {10: 4.0, 505: 4.0, 1000: 4.0}
    assert profiles["wind_v"] == {10: -2.0, 505: -2.0, 1000: -2.0}


def test_fetch_meteomatics_data_unrecognized_parameter_raises(monkeypatch):
    """Raise a ValueError when the response contains an unknown parameter."""
    calls = []
    monkeypatch.setattr(
        fetchers.requests,
        "get",
        _make_fake_meteomatics_get(calls, extra_bad_parameter=True),
    )
    with pytest.raises(ValueError, match="Unrecognized Meteomatics parameter"):
        fetchers.fetch_atmospheric_data_from_meteomatics(
            username="user",
            password="pass",
            latitude=39.0,
            longitude=-8.0,
            date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            wind_resolution=2,
            temperature_pressure_resolution=2,
        )


def test_fetch_meteomatics_data_client_error_not_retried(monkeypatch):
    """A 4xx data response yields an actionable RuntimeError and is not retried."""
    calls = []
    monkeypatch.setattr(
        fetchers.time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(
        fetchers.requests, "get", _make_fake_meteomatics_get(calls, data_status=400)
    )

    with pytest.raises(RuntimeError, match="data API request failed"):
        fetchers.fetch_atmospheric_data_from_meteomatics(
            username="user",
            password="pass",
            latitude=39.0,
            longitude=-8.0,
            date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            wind_resolution=2,
            temperature_pressure_resolution=2,
        )
    # 1 token call + exactly 1 data call (the 400 was not retried).
    data_calls = [c for c in calls if c[0] != fetchers.METEOMATICS_LOGIN_URL]
    assert len(data_calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing "data"
        {"data": [{"parameter": "t_10m:K", "coordinates": []}]},  # empty coordinates
    ],
)
def test_extract_meteomatics_json_bad_structure_raises(payload):
    """Turn an unexpected 200 payload into a clear RuntimeError, not KeyError."""
    with pytest.raises(RuntimeError, match="Unexpected Meteomatics response"):
        fetchers.MeteomaticsFetcher._extract_json(payload)


@pytest.mark.parametrize(
    "altitudes",
    [
        {"min_altitude": -1, "max_altitude": 1000},  # negative floor
        {"min_altitude": 10, "max_altitude": 5},  # max below min
    ],
)
def test_fetch_meteomatics_data_invalid_altitude_range_raises(altitudes):
    """Reject invalid altitude ranges before making any request."""
    with pytest.raises(ValueError, match="altitude"):
        fetchers.fetch_atmospheric_data_from_meteomatics(
            username="user",
            password="pass",
            latitude=39.0,
            longitude=-8.0,
            date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            **altitudes,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wind_resolution": 1}, "at least"),
        ({"temperature_pressure_resolution": 0}, "at least"),
        ({"query_limit": 0}, "query_limit must be at least 1"),
    ],
)
def test_fetch_meteomatics_data_invalid_sampling_raises(kwargs, message):
    """Reject degenerate resolutions and query limits with a clear message.

    Without the up-front check these reach ``linspace``/``range`` and fail with
    an opaque low-level error (or an empty request) instead.
    """
    with pytest.raises(ValueError, match=message):
        fetchers.fetch_atmospheric_data_from_meteomatics(
            username="user",
            password="pass",
            latitude=39.0,
            longitude=-8.0,
            date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            **kwargs,
        )


def test_fetch_meteomatics_data_converts_date_to_utc(monkeypatch):
    """A non-UTC aware datetime must be converted, not stamped with a bare Z.

    The request path carries the instant with a trailing "Z", so 12:00 at
    UTC+03:00 has to be sent as 09:00Z.
    """
    calls = []
    monkeypatch.setattr(fetchers.requests, "get", _make_fake_meteomatics_get(calls))

    fetchers.fetch_atmospheric_data_from_meteomatics(
        username="user",
        password="pass",
        latitude=39.0,
        longitude=-8.0,
        date=datetime(2024, 1, 1, 12, tzinfo=timezone(timedelta(hours=3))),
        wind_resolution=2,
        temperature_pressure_resolution=2,
    )

    data_calls = [c for c in calls if c[0] != fetchers.METEOMATICS_LOGIN_URL]
    assert data_calls, "expected at least one data request"
    assert all("2024-01-01T09:00:00Z" in url for url, _ in data_calls)
