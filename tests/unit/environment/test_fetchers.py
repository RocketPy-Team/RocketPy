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
