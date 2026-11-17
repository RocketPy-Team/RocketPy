"""Reuse, in memory, of a forecast already downloaded in the same Python
session. The forecast server is replaced by fake datasets, so nothing here
needs a network."""

from datetime import datetime

import numpy as np
import pytest

from rocketpy import Environment, Function
from rocketpy.environment import environment as environment_module
from rocketpy.environment.tools import get_model_run


LAUNCH = (2026, 1, 2, 12)  # 36 hours after the start of the fake time axis
LAUNCH_DATE = datetime(*LAUNCH)


class _FakeVariable:
    units = "Hour since 2026-01-01T00:00:00Z"

    def __init__(self, hours, dimensions=("time",)):
        self._hours = np.array(hours, dtype=float)
        self.dimensions = dimensions

    def __getitem__(self, index):
        return self._hours[index]


class _FakeDataset:
    """What the reuse needs from a dataset: its forecast times, the model run
    each of them comes from, its address and a way to close it.

    ``runs`` holds the hour of the model run behind each forecast time, or is
    None for a dataset that does not say.
    """

    forecast_hours = (24, 30, 36, 42)

    def __init__(self, runs=None):
        self.variables = {"time": _FakeVariable(self.forecast_hours)}
        if runs is not None:
            self.variables["reftime"] = _FakeVariable(runs)
        self.closed = False

    def filepath(self):
        return "https://example.test/forecast/Best"

    def close(self):
        self.closed = True


class _FakeServer:
    """Stands in for the forecast server and for the processing of its data."""

    def __init__(self, runs=(6, 6, 6, 6)):
        self.runs = runs
        self.reachable = True
        self.datasets = []
        self.processed = 0

    def fetch(self):
        if not self.reachable:
            raise RuntimeError("Unable to load latest weather data")
        self.datasets.append(_FakeDataset(self.runs))
        return self.datasets[-1]

    def environment(self, **kwargs):
        env = Environment(**{"date": LAUNCH, **kwargs})
        env._Environment__atm_type_file_to_function_map = {
            "forecast": {"GFS": self.fetch},
            "ensemble": {},
        }

        def process(dataset, dictionary, conversion_factor):  # pylint: disable=unused-argument
            self.processed += 1
            env.forecast_marker = f"runs {self.runs}"
            env.forecast_profile = Function([[0, 1.0], [1000, 2.0]])

        env.process_forecast_reanalysis = process
        return env


def test_model_run_is_the_one_behind_the_wanted_time():
    """While a new run is being published, the first forecast times already
    come from it and the later ones still come from the previous run."""
    dataset = _FakeDataset(runs=(12, 12, 6, 6))
    address = "https://example.test/forecast/Best"

    assert get_model_run(dataset, "time", LAUNCH_DATE) == (
        address,
        "2026-01-01 06:00:00",
    )
    assert get_model_run(dataset, "time", datetime(2026, 1, 2, 0)) == (
        address,
        "2026-01-01 12:00:00",
    )


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
        ("a/local/file.nc", "time", LAUNCH_DATE),
    ],
)
def test_model_run_is_unknown_when_the_dataset_does_not_say(dataset, time_name, date):
    assert get_model_run(dataset, time_name, date) is None


def test_same_request_and_same_model_run_is_not_processed_again():
    server = _FakeServer()
    first = server.environment()
    second = server.environment()

    first.set_atmospheric_model(type="Forecast", file="GFS")
    second.set_atmospheric_model(type="Forecast", file="GFS")

    assert server.processed == 1
    assert len(server.datasets) == 2, "the server is asked for its run each time"
    assert server.datasets[1].closed
    assert second.forecast_marker == "runs (6, 6, 6, 6)"
    assert second.forecast_profile(500) == first.forecast_profile(500)


def test_reused_results_are_copies():
    server = _FakeServer()
    first = server.environment()
    second = server.environment()
    third = server.environment()

    first.set_atmospheric_model(type="Forecast", file="GFS")
    second.set_atmospheric_model(type="Forecast", file="GFS")
    second.forecast_profile.set_discrete(0, 1000, 5)
    third.set_atmospheric_model(type="Forecast", file="GFS")

    assert second.forecast_profile is not first.forecast_profile
    assert len(first.forecast_profile.source) == 2
    assert len(third.forecast_profile.source) == 2


def test_a_new_model_run_is_downloaded_again():
    server = _FakeServer(runs=(6, 6, 6, 6))
    first = server.environment()
    second = server.environment()
    third = server.environment()

    first.set_atmospheric_model(type="Forecast", file="GFS")
    server.runs = (12, 12, 12, 12)
    second.set_atmospheric_model(type="Forecast", file="GFS")
    third.set_atmospheric_model(type="Forecast", file="GFS")

    assert server.processed == 2
    assert second.forecast_marker == "runs (12, 12, 12, 12)"
    assert third.forecast_marker == "runs (12, 12, 12, 12)"


def test_a_new_model_run_counts_once_it_reaches_the_launch_time():
    """The launch is at the third forecast time. A new run that has only
    replaced the first two changes nothing for it; once it replaces the third,
    the forecast is downloaded again."""
    server = _FakeServer(runs=(6, 6, 6, 6))
    environments = [server.environment() for _ in range(3)]

    environments[0].set_atmospheric_model(type="Forecast", file="GFS")
    server.runs = (12, 12, 6, 6)
    environments[1].set_atmospheric_model(type="Forecast", file="GFS")
    assert server.processed == 1

    server.runs = (12, 12, 12, 6)
    environments[2].set_atmospheric_model(type="Forecast", file="GFS")
    assert server.processed == 2


@pytest.mark.parametrize(
    "other_request",
    [
        {"latitude": 10.0},
        {"longitude": 10.0},
        {"elevation": 500.0},
        {"date": (2026, 1, 2, 6)},
    ],
)
def test_a_different_request_is_downloaded(other_request):
    server = _FakeServer()
    first = server.environment()
    second = server.environment(**other_request)

    first.set_atmospheric_model(type="Forecast", file="GFS")
    second.set_atmospheric_model(type="Forecast", file="GFS")

    assert server.processed == 2


def test_a_model_run_that_cannot_be_identified_is_never_reused():
    server = _FakeServer(runs=None)
    first = server.environment()
    second = server.environment()

    first.set_atmospheric_model(type="Forecast", file="GFS")
    second.set_atmospheric_model(type="Forecast", file="GFS")

    assert server.processed == 2
    assert not environment_module._FORECASTS_IN_MEMORY


def test_an_unreachable_server_fails_even_with_a_forecast_in_memory():
    server = _FakeServer()
    first = server.environment()
    second = server.environment()

    first.set_atmospheric_model(type="Forecast", file="GFS")
    server.reachable = False

    with pytest.raises(RuntimeError, match="Unable to load latest weather data"):
        second.set_atmospheric_model(type="Forecast", file="GFS")
    assert not hasattr(second, "forecast_marker")


def test_a_dataset_given_by_the_user_is_not_kept():
    server = _FakeServer()
    env = server.environment()

    env.set_atmospheric_model(
        type="Forecast", file=_FakeDataset(runs=(6, 6, 6, 6)), dictionary="GFS"
    )

    assert server.processed == 1
    assert not environment_module._FORECASTS_IN_MEMORY
