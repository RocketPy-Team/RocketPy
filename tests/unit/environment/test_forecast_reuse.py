"""Reuse, in memory, of a forecast already downloaded in the same Python
session. The forecast server is replaced by small netCDF files shaped like the
GFS forecast, read by the real processing, so nothing here needs a network."""

from datetime import datetime

import netCDF4
import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.environment import environment as environment_module
from rocketpy.environment.tools import get_model_run

UNITS = "Hour since 2026-01-01T00:00:00Z"
HOURS = (24, 30, 36, 42)
LAUNCH = (2026, 1, 2, 12)  # 36 hours after the start of the time axis
LAUNCH_DATE = datetime(*LAUNCH)
SURFACE_HEIGHT = 1400.0


FIELDS = ("time", "isobaric", "lat", "lon")


def _write_coordinates(data, runs, hours, masked_run):
    """Write the times, the model run behind each of them and the grid."""
    for name, size in zip(FIELDS, (len(hours), 5, 3, 3)):
        data.createDimension(name, size)
    time = data.createVariable("time", "f8", ("time",))
    time.units = UNITS
    time[:] = hours
    if runs is not None:
        reference_time = data.createVariable("reftime", "f8", ("time",))
        reference_time.units = UNITS
        reference_time[:] = runs
        if masked_run is not None:
            reference_time[masked_run] = np.ma.masked
    data.createVariable("lat", "f8", ("lat",))[:] = [34.0, 33.0, 32.0]
    data.createVariable("lon", "f8", ("lon",))[:] = [252.0, 253.0, 254.0]
    level = data.createVariable("isobaric", "f8", ("isobaric",))
    level.units = "Pa"
    level[:] = [30000.0, 50000.0, 70000.0, 85000.0, 100000.0]
    # Named by the AIGFS shortcut, whose forecast has no surface height
    data.createVariable("LatLon_Projection", "i4")


def _write_fields(data, runs, hours):
    """Write profiles that are the same over the grid. The temperature and
    the wind depend on the model run, so two runs give different profiles."""
    shape = (len(hours), 5, 3, 3)
    run = np.array([0] * len(hours) if runs is None else runs, dtype=float)
    run = run[:, None, None, None]

    def field(name, values, shift=0.0):
        values = np.array(values, dtype=float)[None, :, None, None]
        variable = data.createVariable(name, "f8", FIELDS)
        variable[:] = np.broadcast_to(values, shape) + shift

    field("Geopotential_height_isobaric", [9200, 5600, 3000, 1500, 100])
    field("Temperature_isobaric", [229, 253, 268, 278, 288], shift=run / 6)
    field("u-component_of_wind_isobaric", [3, 2, 1, -1, -2])
    field("v-component_of_wind_isobaric", [12, 10, 8, 6, 4], shift=run / 6)
    surface = data.createVariable(
        "Geopotential_height_surface", "f8", FIELDS[:1] + FIELDS[2:]
    )
    surface[:] = SURFACE_HEIGHT


def _write_forecast(path, runs, hours=HOURS, masked_run=None):
    """Write a forecast with a few pressure levels around the launch site.

    ``runs`` holds the hour of the model run behind each forecast time, or is
    None for a dataset that does not say.
    """
    with netCDF4.Dataset(path, "w") as data:
        _write_coordinates(data, runs, hours, masked_run)
        _write_fields(data, runs, hours)


class _Server:
    """Stands in for the forecast server: one address, whose dataset is
    replaced when the server publishes something new."""

    def __init__(self, folder, runs=(6, 6, 6, 6), **dataset):
        self.path = str(folder / "Best.nc")
        self.reachable = True
        self.datasets = []
        self.processed = 0
        self.publish(runs, **dataset)

    def publish(self, runs, **dataset):
        _write_forecast(self.path, runs, **dataset)

    def fetch(self):
        if not self.reachable:
            raise RuntimeError("Unable to load latest weather data")
        self.datasets.append(netCDF4.Dataset(self.path))
        return self.datasets[-1]

    def environment(self, **kwargs):
        site = {"date": LAUNCH, "latitude": 32.99, "longitude": -106.97}
        env = Environment(**{**site, **kwargs})
        env._Environment__atm_type_file_to_function_map = {
            "forecast": {"GFS": self.fetch, "AIGFS": self.fetch},
            "ensemble": {},
        }
        process = env.process_forecast_reanalysis

        def counted(*args, **kwargs):
            self.processed += 1
            return process(*args, **kwargs)

        env.process_forecast_reanalysis = counted
        return env

    def forecast(self, model="GFS", **kwargs):
        """An Environment with the forecast of the server already set."""
        env = self.environment(**kwargs)
        env.set_atmospheric_model(type="Forecast", file=model)
        return env


@pytest.fixture
def server(tmp_path):
    return _Server(tmp_path)


class _FakeVariable:
    units = UNITS

    def __init__(self, hours, dimensions=("time",)):
        self._hours = np.array(hours, dtype=float)
        self.dimensions = dimensions

    def __getitem__(self, index):
        return self._hours[index]


class _FakeDataset:
    """What identifying the model run needs from a dataset."""

    def __init__(self, runs=None):
        self.variables = {"time": _FakeVariable(HOURS)}
        if runs is not None:
            self.variables["reftime"] = _FakeVariable(runs)

    def filepath(self):
        return "https://example.test/forecast/Best"


def test_model_run_is_the_one_behind_the_wanted_time():
    """While a new run is being published, the first forecast times already
    come from it and the later ones still come from the previous run."""
    dataset = _FakeDataset(runs=(12, 12, 6, 6))
    address = "https://example.test/forecast/Best"

    assert get_model_run(dataset, "time", LAUNCH_DATE)[:2] == (
        address,
        "2026-01-01 06:00:00",
    )
    assert get_model_run(dataset, "time", datetime(2026, 1, 2, 0))[:2] == (
        address,
        "2026-01-01 12:00:00",
    )


def test_model_run_comes_with_the_times_the_dataset_covers():
    dataset = _FakeDataset(runs=(6, 6, 6, 6))

    assert get_model_run(dataset, "time", LAUNCH_DATE)[2:] == (24.0, 42.0)


def test_model_run_uses_the_reference_times_of_the_same_time_axis():
    dataset = _FakeDataset(runs=(6, 6, 6, 6))
    dataset.variables["reftime1"] = _FakeVariable((18, 18), dimensions=("time1",))

    assert get_model_run(dataset, "time", LAUNCH_DATE)[1] == "2026-01-01 06:00:00"


@pytest.mark.parametrize(
    ("dataset", "time_name", "date"),
    [
        (_FakeDataset(runs=None), "time", LAUNCH_DATE),
        (_FakeDataset(runs=(6, 6, 6, 6)), "another_time", LAUNCH_DATE),
        (_FakeDataset(runs=(6, 6, 6, 6)), "time", None),
        (_FakeDataset(runs=(6, 6, 6, 6)), "time", datetime(2026, 1, 1, 12)),
        (_FakeDataset(runs=(6, 6, 6, 6)), "time", datetime(2026, 1, 3, 0)),
        (_FakeDataset(runs=(6, 6, np.nan, 6)), "time", LAUNCH_DATE),
        ("a/local/file.nc", "time", LAUNCH_DATE),
    ],
    ids=[
        "no reference times",
        "no such time variable",
        "no date",
        "before the forecast",
        "after the forecast",
        "unreadable reference time",
        "not a dataset",
    ],
)
def test_model_run_is_unknown_when_the_dataset_does_not_say(dataset, time_name, date):
    assert get_model_run(dataset, time_name, date) is None


def test_model_run_is_unknown_when_its_reference_time_is_masked(tmp_path):
    path = str(tmp_path / "masked.nc")
    _write_forecast(path, runs=(6, 6, 6, 6), masked_run=2)

    with netCDF4.Dataset(path) as dataset:
        assert get_model_run(dataset, "time", LAUNCH_DATE) is None
        assert get_model_run(dataset, "time", datetime(2026, 1, 2, 0)) is not None


def test_same_request_and_same_model_run_is_not_processed_again(server):
    first = server.forecast()
    second = server.forecast()

    assert server.processed == 1
    assert len(server.datasets) == 2, "the server is asked for its run each time"
    assert not server.datasets[1].isopen()
    assert second.temperature(2000) == first.temperature(2000)


def test_a_reused_forecast_leaves_the_environment_as_a_processed_one(server):
    """Nothing the processing sets is missing or different after a reuse."""
    processed = server.forecast()
    reused = server.forecast()

    assert vars(reused).keys() == vars(processed).keys()
    assert reused.elevation == processed.elevation == pytest.approx(SURFACE_HEIGHT)
    for name in (
        "pressure",
        "temperature",
        "wind_velocity_x",
        "wind_velocity_y",
        "wind_speed",
        "wind_heading",
        "wind_direction",
        "density",
        "speed_of_sound",
    ):
        assert getattr(reused, name)(2000) == getattr(processed, name)(2000), name
    for name in (
        "atmospheric_model_init_date",
        "atmospheric_model_end_date",
        "atmospheric_model_interval",
        "lat_index",
        "lon_index",
        "time_array",
        "max_expected_height",
    ):
        assert getattr(reused, name) == getattr(processed, name), name
    reused.prints.all()


def test_reused_results_are_copies(server):
    first = server.forecast()
    second = server.forecast()
    second.temperature.set_discrete(0, 1000, 7)
    third = server.forecast()

    assert second.temperature is not first.temperature
    assert len(second.temperature.source) == 7
    assert len(first.temperature.source) == len(third.temperature.source) == 5


def test_reused_wind_angles_do_not_read_another_environment(server):
    """The wind heading and direction are read from their unwrapped versions,
    which have to be the ones of the Environment itself."""
    first = server.forecast()
    second = server.forecast()
    direction, heading = second.wind_direction(2000), second.wind_heading(2000)

    for unwrapped in (first.wind_direction_unwrapped, first.wind_heading_unwrapped):
        turned = np.array(unwrapped.get_source())
        turned[:, 1] += 90
        unwrapped.set_source(turned)
    third = server.forecast()

    assert first.wind_direction(2000) == pytest.approx((direction + 90) % 360)
    assert second.wind_direction(2000) == third.wind_direction(2000) == direction
    assert second.wind_heading(2000) == third.wind_heading(2000) == heading


def test_a_new_model_run_is_downloaded_again(server):
    first = server.forecast()
    server.publish(runs=(12, 12, 12, 12))
    second = server.forecast()
    third = server.forecast()

    assert server.processed == 2
    assert second.temperature(2000) == pytest.approx(first.temperature(2000) + 1)
    assert third.temperature(2000) == second.temperature(2000)


def test_a_new_model_run_counts_once_it_reaches_the_launch_time(server):
    """The launch is at the third forecast time. A new run that has only
    replaced the first two changes nothing for it; once it replaces the third,
    the forecast is downloaded again."""
    server.forecast()
    server.publish(runs=(12, 12, 6, 6))
    server.forecast()
    assert server.processed == 1

    server.publish(runs=(12, 12, 12, 6))
    server.forecast()
    assert server.processed == 2


@pytest.mark.parametrize(
    "other_request",
    [{"latitude": 32.5}, {"longitude": -106.5}, {"date": (2026, 1, 2, 6)}],
)
def test_a_different_request_is_downloaded(server, other_request):
    server.forecast()
    server.forecast(**other_request)

    assert server.processed == 2


def test_the_elevation_given_by_the_user_is_not_part_of_the_request(server):
    """The model has a surface height, which replaces the elevation."""
    first = server.forecast(elevation=0)
    second = server.forecast(elevation=500)

    assert server.processed == 1
    assert first.elevation == second.elevation == pytest.approx(SURFACE_HEIGHT)


def test_asking_twice_on_the_same_environment_processes_once(server):
    env = server.forecast()
    env.set_atmospheric_model(type="Forecast", file="GFS")
    other = server.forecast(elevation=env.elevation)

    assert server.processed == 1
    assert vars(other).keys() == vars(env).keys()
    other.prints.all()


def test_elevation_is_kept_when_the_model_has_no_surface_height(server):
    first = server.forecast("AIGFS", elevation=120)
    second = server.forecast("AIGFS", elevation=450)

    assert server.processed == 1
    assert (first.elevation, second.elevation) == (120, 450)


def test_a_launch_time_the_dataset_no_longer_covers_is_an_error(server):
    """Same model run behind every time, but the forecast has moved on."""
    server.forecast()
    server.publish(runs=(6, 6, 6, 6), hours=(42, 48, 54, 60))

    with pytest.raises(ValueError, match="not available in the provided file"):
        server.forecast()


def test_a_dataset_covering_other_times_is_downloaded_again(server):
    """The run behind the launch time is the same, but the forecast now goes
    further, which the Environment reports."""
    first = server.forecast()
    server.publish(runs=(6, 6, 6, 6, 12), hours=(24, 30, 36, 42, 48))
    second = server.forecast()

    assert server.processed == 2
    assert first.atmospheric_model_end_date == datetime(2026, 1, 2, 18)
    assert second.atmospheric_model_end_date == datetime(2026, 1, 3, 0)


@pytest.mark.parametrize("dataset", [{"runs": None}, {"masked_run": 2}])
def test_a_model_run_that_cannot_be_identified_is_never_reused(tmp_path, dataset):
    server = _Server(tmp_path, **{"runs": (6, 6, 6, 6), **dataset})
    server.forecast()
    server.forecast()

    assert server.processed == 2
    assert not environment_module._FORECASTS_IN_MEMORY


def test_an_unreachable_server_fails_even_with_a_forecast_in_memory(server):
    server.forecast()
    server.reachable = False
    env = server.environment()

    with pytest.raises(RuntimeError, match="Unable to load latest weather data"):
        env.set_atmospheric_model(type="Forecast", file="GFS")
    assert env.atmospheric_model_type == "Forecast"
    assert not hasattr(env, "atmospheric_model_end_date")


def test_a_dataset_given_by_the_user_is_not_kept(server):
    env = server.environment()

    env.set_atmospheric_model(type="Forecast", file=server.path, dictionary="GFS")

    assert server.processed == 1
    assert not environment_module._FORECASTS_IN_MEMORY


def test_only_the_latest_forecasts_are_kept(server, monkeypatch):
    monkeypatch.setattr(environment_module, "_MAX_FORECASTS_IN_MEMORY", 2)
    dates = [(2026, 1, 2, 0), (2026, 1, 2, 6), (2026, 1, 2, 12)]

    for date in dates:
        server.forecast(date=date)
    assert len(environment_module._FORECASTS_IN_MEMORY) == 2

    server.forecast(date=dates[2])
    assert server.processed == 3, "the latest one is still in memory"
    server.forecast(date=dates[0])
    assert server.processed == 4, "the oldest one was dropped"
