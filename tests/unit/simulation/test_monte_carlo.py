import builtins
import csv
import json
import os
import pathlib
import types
from collections import namedtuple
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy.simulation import MonteCarlo
from rocketpy.simulation.monte_carlo import (
    _refuse_logs_this_run_cannot_write,
)

plt.rcParams.update({"figure.max_open_warning": 0})


def test_stochastic_environment_create_object_with_wind_x(stochastic_environment):
    """Tests the stochastic environment object by checking if the wind velocity
    can be generated properly. The goal is to check if the create_object()
    method is being called without any problems.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        The stochastic environment object, this is a pytest fixture.
    """
    wind_x_at_1000m = []
    for _ in range(10):
        random_env = stochastic_environment.create_object()
        wind_x_at_1000m.append(random_env.wind_velocity_x(1000))

    assert np.isclose(np.mean(wind_x_at_1000m), 0, atol=0.1)
    assert np.isclose(np.std(wind_x_at_1000m), 0, atol=0.1)
    # TODO: add a new test for the special case of ensemble member


def test_stochastic_solid_motor_create_object_with_impulse(stochastic_solid_motor):
    """Tests the stochastic solid motor object by checking if the total impulse
    can be generated properly. The goal is to check if the create_object()
    method is being called without any problems.

    Parameters
    ----------
    stochastic_solid_motor : StochasticSolidMotor
        The stochastic solid motor object, this is a pytest fixture.
    """
    total_impulse = [
        stochastic_solid_motor.create_object().total_impulse for _ in range(200)
    ]

    assert np.isclose(np.mean(total_impulse), 6500, rtol=0.3)
    assert np.isclose(np.std(total_impulse), 1000, rtol=0.4)


def test_stochastic_calisto_create_object_with_static_margin(stochastic_calisto):
    """Tests the stochastic calisto object by checking if the static margin
    can be generated properly. The goal is to check if the create_object()
    method is being called without any problems.

    Parameters
    ----------
    stochastic_calisto : StochasticCalisto
        The stochastic calisto object, this is a pytest fixture.
    """

    all_margins = []
    for _ in range(10):
        random_rocket = stochastic_calisto.create_object()
        all_margins.append(random_rocket.static_margin(0))

    assert np.isclose(np.mean(all_margins), 2.2625350013000434, rtol=0.15)
    assert np.isclose(np.std(all_margins), 0.1, atol=0.2)


class MockMonteCarlo(MonteCarlo):
    """Create a mock class to test the method without running a real simulation"""

    def __init__(self):
        # pylint: disable=super-init-not-called

        # Simulate pre-calculated results
        # Example: a normal distribution centered on 100 for the apogee
        self.results = {
            "apogee": [98, 102, 100, 99, 101, 100, 97, 103],
            "max_velocity": [250, 255, 245, 252, 248],
            "single_point": [100],
            "empty_attribute": [],
        }


def test_append_simulation_record_rolls_back_inputs_on_output_failure(tmp_path):
    """If the outputs append fails, the inputs row must not remain on disk."""
    mc = MockMonteCarlo()
    input_file = tmp_path / "inputs.json"
    output_file = tmp_path / "outputs.json"
    input_file.write_text('{"index": 0}\n', encoding="utf-8")
    output_file.write_text('{"index": 0}\n', encoding="utf-8")
    mc._input_file = str(input_file)
    mc._output_file = str(output_file)

    mc._append_simulation_record('{"index": 1}\n', '{"index": 1}\n')

    original_open = builtins.open
    output_path = os.fspath(output_file)

    def failing_output_open(*args, **kwargs):
        # Match builtins.open call shapes without keyword-before-vararg (W1113).
        file = args[0] if args else kwargs["file"]
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        if os.fspath(file) == output_path and "a" in mode:
            raise OSError("no space left on device")
        return original_open(*args, **kwargs)

    with pytest.raises(OSError, match="no space left on device"):
        with patch("builtins.open", side_effect=failing_output_open):
            mc._append_simulation_record('{"index": 2}\n', '{"index": 2}\n')

    assert input_file.read_text(encoding="utf-8") == '{"index": 0}\n{"index": 1}\n'
    assert output_file.read_text(encoding="utf-8") == '{"index": 0}\n{"index": 1}\n'


def test_estimate_confidence_interval_contains_known_mean():
    """Checks that the confidence interval contains the known mean."""
    mc = MockMonteCarlo()

    ci = mc.estimate_confidence_interval("apogee", confidence_level=0.95)

    assert ci.low < 100 < ci.high
    assert ci.low < ci.high


def test_estimate_confidence_interval_supports_custom_statistic():
    """Checks that the statistic can be changed (e.g., standard deviation instead of mean)."""
    mc = MockMonteCarlo()

    ci_std = mc.estimate_confidence_interval("apogee", statistic=np.std)

    assert ci_std.low > 0
    assert ci_std.low < ci_std.high


def test_estimate_confidence_interval_raises_value_error_when_attribute_missing():
    """Checks that the code raises an error if the key does not exist."""
    mc = MockMonteCarlo()

    # Request a variable that does not exist ("altitude" is not in our mock)
    with pytest.raises(ValueError) as excinfo:
        mc.estimate_confidence_interval("altitude")

    assert "not found in results" in str(excinfo.value)


def test_estimate_confidence_interval_increases_width_with_higher_confidence_level():
    """Checks that a higher confidence level yields a wider interval."""
    mc = MockMonteCarlo()

    ci_90 = mc.estimate_confidence_interval("apogee", confidence_level=0.90)
    width_90 = ci_90.high - ci_90.low

    ci_99 = mc.estimate_confidence_interval("apogee", confidence_level=0.99)
    width_99 = ci_99.high - ci_99.low

    # The more confident we want to be (99%), the wider the interval must be
    assert width_99 >= width_90


def test_estimate_confidence_interval_raises_value_error_when_confidence_level_out_of_bounds():
    """Checks that validation fails if confidence_level is not strictly between 0 and 1."""
    mc = MockMonteCarlo()

    # Case 1: Value <= 0
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        mc.estimate_confidence_interval("apogee", confidence_level=0)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        mc.estimate_confidence_interval("apogee", confidence_level=-0.5)

    # Case 2: Value >= 1
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        mc.estimate_confidence_interval("apogee", confidence_level=1)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        mc.estimate_confidence_interval("apogee", confidence_level=1.5)


def test_estimate_confidence_interval_raises_value_error_when_n_resamples_invalid():
    """Checks that validation fails if n_resamples is not a positive integer."""
    mc = MockMonteCarlo()

    # Case 1: Not an integer (e.g. float)
    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        mc.estimate_confidence_interval("apogee", n_resamples=1000.5)

    # Case 2: Zero or Negative
    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        mc.estimate_confidence_interval("apogee", n_resamples=0)

    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        mc.estimate_confidence_interval("apogee", n_resamples=-100)


def test_estimate_confidence_interval_raises_value_error_on_empty_data_list():
    """Checks behavior when the attribute exists but contains no data (empty list)."""
    mc = MockMonteCarlo()

    with pytest.raises(ValueError):
        mc.estimate_confidence_interval("empty_attribute")


def test_estimate_confidence_interval_handles_single_data_point():
    """Checks behavior with only one data point. The CI should be [val, val]."""
    mc = MockMonteCarlo()

    with pytest.raises(ValueError):  # two or more value
        mc.estimate_confidence_interval("single_point", n_resamples=50)


def test_estimate_confidence_interval_raises_type_error_for_invalid_statistic():
    """Checks that passing a non-callable object (like a string/int) as statistic raises TypeError."""
    mc = MockMonteCarlo()
    with pytest.raises(TypeError):
        mc.estimate_confidence_interval("apogee", statistic=1)

    with pytest.raises(TypeError):
        mc.estimate_confidence_interval("apogee", statistic="not_a_function")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -5}, "batch_size"),
        ({"max_simulations": 0}, "max_simulations"),
        ({"tolerance": 0}, "tolerance"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"target_confidence": 1.5}, "target_confidence"),
        ({"target_confidence": 0}, "target_confidence"),
    ],
)
def test_simulate_convergence_validates_inputs(kwargs, match):
    """simulate_convergence must reject invalid inputs up front. In particular a
    non-positive batch_size would otherwise make the loop run zero simulations
    per iteration and spin forever."""
    mc = MockMonteCarlo()
    with pytest.raises(ValueError, match=match):
        mc.simulate_convergence(**kwargs)


# --- CSV and JSON export/import tests ---


class MockMonteCarloWithLogs(MonteCarlo):
    """Mock class with populated logs for testing export/import methods."""

    def __init__(self):
        # pylint: disable=super-init-not-called
        self.outputs_log = [
            {"apogee": 5742.42, "x_impact": 553.49, "index": 0},
            {"apogee": 3844.41, "x_impact": 402.31, "index": 1},
            {"apogee": 4500.00, "x_impact": 480.10, "index": 2},
        ]
        self.inputs_log = [
            {
                "elevation": 1413.6,
                "radius": 0.0635,
                "parachutes": [{"cd_s": 9.84}],
                "index": 0,
            },
            {
                "elevation": 1400.0,
                "radius": 0.0640,
                "parachutes": [{"cd_s": 10.0}],
                "index": 1,
            },
            {
                "elevation": 1420.0,
                "radius": 0.0630,
                "parachutes": [{"cd_s": 9.50}],
                "index": 2,
            },
        ]
        self.errors_log = []
        self.results = {}
        self.processed_results = {}
        self.num_of_loaded_sims = 3


def test_set_processed_results_summarizes_real_scalars():
    mc = MockMonteCarloWithLogs()
    mc.results = {"value": [1, np.int64(2), np.float32(3)]}

    mc.set_processed_results()

    mean, median, stdev, pi_low, pi_high = mc.processed_results["value"]
    assert mean == pytest.approx(2)
    assert median == pytest.approx(2)
    assert stdev == pytest.approx(np.std([1, 2, 3]))
    assert pi_low == pytest.approx(np.quantile([1, 2, 3], 0.025))
    assert pi_high == pytest.approx(np.quantile([1, 2, 3], 0.975))


@pytest.mark.parametrize(
    "values",
    [
        ["ascent", "descent"],
        [[1, 2], [3, 4]],
        [[1], [2, 3]],
        [{"x": 1}, {"x": 2}],
        [np.array([1, 2]), np.array([3, 4])],
        [1, "two"],
        [True, False],
        [],
    ],
)
def test_set_processed_results_preserves_structured_results(values):
    mc = MockMonteCarloWithLogs()
    mc.results = {"structured": values}

    mc.set_processed_results()

    assert mc.results["structured"] is values
    assert mc.processed_results["structured"] == (None, None, None, None, None)


def test_export_outputs_to_csv(tmp_path):
    """Tests that outputs are correctly exported to CSV."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.csv"

    mc.export_outputs_to_csv(str(filepath))

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert float(rows[0]["apogee"]) == pytest.approx(5742.42)
    assert float(rows[1]["x_impact"]) == pytest.approx(402.31)


def test_export_outputs_to_json(tmp_path):
    """Tests that outputs are correctly exported to JSON."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.json"

    mc.export_outputs_to_json(str(filepath))

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 3
    assert data[0]["apogee"] == pytest.approx(5742.42)
    assert data[2]["index"] == 2


def test_export_inputs_to_csv_no_flatten(tmp_path):
    """Tests that inputs with nested values are serialized as JSON in CSV cells."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "inputs.csv"

    mc.export_inputs_to_csv(str(filepath), flatten=False)

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    # The parachutes column should contain a JSON string
    parachutes_val = json.loads(rows[0]["parachutes"])
    assert parachutes_val == [{"cd_s": 9.84}]


def test_export_inputs_to_csv_flatten(tmp_path):
    """Tests that flatten=True omits non-scalar columns."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "inputs.csv"

    mc.export_inputs_to_csv(str(filepath), flatten=True)

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert "parachutes" not in rows[0]
    assert "elevation" in rows[0]
    assert "radius" in rows[0]


def test_export_inputs_to_json(tmp_path):
    """Tests that inputs are correctly exported to JSON."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "inputs.json"

    mc.export_inputs_to_json(str(filepath))

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 3
    assert data[0]["parachutes"] == [{"cd_s": 9.84}]


def test_export_empty_log_raises_error(tmp_path):
    """Tests that exporting an empty log raises ValueError."""
    mc = MockMonteCarloWithLogs()
    mc.outputs_log = []

    with pytest.raises(ValueError, match="No data to export"):
        mc.export_outputs_to_csv(str(tmp_path / "empty.csv"))

    with pytest.raises(ValueError, match="No data to export"):
        mc.export_outputs_to_json(str(tmp_path / "empty.json"))


def test_import_outputs_from_csv(tmp_path):
    """Tests that outputs can be imported from a CSV file."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.csv"

    # Export first
    mc.export_outputs_to_csv(str(filepath))

    # Create a fresh mock and import
    mc2 = MockMonteCarloWithLogs()
    mc2.output_file = str(filepath)

    assert len(mc2.outputs_log) == 3
    assert mc2.outputs_log[0]["apogee"] == pytest.approx(5742.42)
    assert mc2.outputs_log[1]["x_impact"] == pytest.approx(402.31)


def test_import_outputs_from_json(tmp_path):
    """Tests that outputs can be imported from a JSON file."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.json"

    # Export first
    mc.export_outputs_to_json(str(filepath))

    # Create a fresh mock and import
    mc2 = MockMonteCarloWithLogs()
    mc2.output_file = str(filepath)

    assert len(mc2.outputs_log) == 3
    assert mc2.outputs_log[0]["apogee"] == pytest.approx(5742.42)


def test_round_trip_outputs_csv(tmp_path):
    """Tests that outputs survive a CSV export/import round trip."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.csv"

    mc.export_outputs_to_csv(str(filepath))
    mc.output_file = str(filepath)

    for i, original in enumerate(MockMonteCarloWithLogs().outputs_log):
        for key, value in original.items():
            assert mc.outputs_log[i][key] == pytest.approx(value)


def test_round_trip_outputs_json(tmp_path):
    """Tests that outputs survive a JSON export/import round trip."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.json"

    mc.export_outputs_to_json(str(filepath))
    mc.output_file = str(filepath)

    for i, original in enumerate(MockMonteCarloWithLogs().outputs_log):
        for key, value in original.items():
            assert mc.outputs_log[i][key] == pytest.approx(value)


def test_round_trip_inputs_csv(tmp_path):
    """Tests that inputs with nested values survive a CSV round trip."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "inputs.csv"

    mc.export_inputs_to_csv(str(filepath), flatten=False)
    mc.input_file = str(filepath)

    assert mc.inputs_log[0]["parachutes"] == [{"cd_s": 9.84}]
    assert mc.inputs_log[0]["elevation"] == pytest.approx(1413.6)


def test_detect_file_format_unsupported():
    """Tests that unsupported file extensions raise ValueError."""
    mc = MockMonteCarloWithLogs()

    with pytest.raises(ValueError, match="Unsupported file extension"):
        mc._detect_file_format("data.xlsx")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        mc._detect_file_format("data.parquet")


def test_set_num_of_loaded_sims_csv(tmp_path):
    """Tests that set_num_of_loaded_sims works with CSV files."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.csv"

    mc.export_outputs_to_csv(str(filepath))
    mc._output_file = str(filepath)
    mc.set_num_of_loaded_sims()

    assert mc.num_of_loaded_sims == 3


def test_set_num_of_loaded_sims_json(tmp_path):
    """Tests that set_num_of_loaded_sims works with JSON files."""
    mc = MockMonteCarloWithLogs()
    filepath = tmp_path / "outputs.json"

    mc.export_outputs_to_json(str(filepath))
    mc._output_file = str(filepath)
    mc.set_num_of_loaded_sims()

    assert mc.num_of_loaded_sims == 3


# --- Adaptive Monte Carlo convergence (PR #922) ---

_CI = namedtuple("_CI", ["low", "high"])


class ConvergenceMockMonteCarlo(MonteCarlo):
    """Mock that fakes batch simulation and a scripted confidence-interval
    width, so ``simulate_convergence``'s stopping decision can be unit-tested
    without running any real flight simulation."""

    def __init__(self, width_model):
        # pylint: disable=super-init-not-called
        self.num_of_loaded_sims = 0
        self.filename = pathlib.Path("dummy_mc")
        self._width_model = width_model
        self.simulate_calls = 0

    def import_outputs(self, *args, **kwargs):  # no-op, avoids file I/O
        pass

    def simulate(self, number_of_simulations, append=True, **kwargs):
        # pylint: disable=arguments-differ
        self.simulate_calls += 1
        self.num_of_loaded_sims = number_of_simulations

    def estimate_confidence_interval(self, attribute, confidence_level=0.95, **kwargs):
        # pylint: disable=arguments-differ,unused-argument
        width = self._width_model(self.num_of_loaded_sims)
        return _CI(low=0.0, high=width)


def test_simulate_convergence_stops_early_when_tolerance_met():
    """The convergence loop must stop as soon as the CI width drops below the
    tolerance, well before reaching max_simulations."""
    # width = 40 / n  ->  50 sims: 0.8 (> 0.5),  100 sims: 0.4 (<= 0.5) -> stop
    mc = ConvergenceMockMonteCarlo(width_model=lambda n: 40.0 / n)

    history = mc.simulate_convergence(
        target_attribute="apogee_time",
        tolerance=0.5,
        max_simulations=1000,
        batch_size=50,
    )

    assert history[-1] <= 0.5
    assert len(history) == 2  # stopped after the second batch
    assert mc.num_of_loaded_sims == 100
    assert mc.num_of_loaded_sims < 1000  # did not exhaust the simulation budget


def test_simulate_convergence_runs_until_max_when_not_converging():
    """When the CI width never drops below the tolerance, the loop must run
    until max_simulations and never exceed it."""
    mc = ConvergenceMockMonteCarlo(width_model=lambda n: 10.0)  # constant, > tol

    history = mc.simulate_convergence(
        target_attribute="apogee_time",
        tolerance=0.5,
        max_simulations=200,
        batch_size=50,
    )

    assert mc.num_of_loaded_sims == 200
    assert all(width > 0.5 for width in history)
    assert len(history) == 4  # 200 / 50 batches


def test_a_monte_carlo_flight_keeps_the_configuration_it_was_given(monkeypatch):
    """A run must build the same ``Flight`` ``StochasticFlight`` would.

    Monte Carlo wrote out the constructor by hand and stopped at
    ``time_overshoot``, so ``max_time``, the tolerances, the solver, the
    equations of motion and the simulation mode were silently reset to their
    defaults. #1070 added StochasticFlight's handling of exactly those.
    """
    base = types.SimpleNamespace(
        max_time_step=0.5,
        min_time_step=0.01,
        rtol=1e-9,
        atol=1e-9,
        name="named",
        equations_of_motion="solid_propulsion",
        ode_solver="RK23",
        simulation_mode="native",
    )
    stochastic_flight = types.SimpleNamespace(
        obj=base,
        max_time=123.0,
        initial_solution=None,
        terminate_on_apogee=True,
        time_overshoot=False,
        _randomize_rail_length=lambda: 5.0,
        _randomize_inclination=lambda: 84.0,
        _randomize_heading=lambda: 133.0,
    )
    analysis = object.__new__(MonteCarlo)
    analysis.flight = stochastic_flight
    analysis.rocket = types.SimpleNamespace(create_object=lambda: "rocket")
    analysis.environment = types.SimpleNamespace(create_object=lambda: "environment")
    monkeypatch.setattr("rocketpy.simulation.monte_carlo.Flight", types.SimpleNamespace)

    flight = MonteCarlo._MonteCarlo__run_single_simulation(analysis)

    assert flight.max_time == 123.0
    assert (flight.rtol, flight.atol) == (1e-9, 1e-9)
    assert (flight.max_time_step, flight.min_time_step) == (0.5, 0.01)
    assert flight.ode_solver == "RK23"
    assert flight.equations_of_motion == "solid_propulsion"
    assert flight.simulation_mode == "native"
    assert flight.name == "named"


@pytest.mark.parametrize(
    "suffix, payload",
    [
        (".csv", "apogee,index\n1234.0,0\n1250.0,1\n"),
        (".json", '[{"apogee": 1234.0, "index": 0}]\n'),
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_simulate_refuses_a_results_file_it_cannot_write(
    monte_carlo_calisto, tmp_path, suffix, payload, append
):
    """Importing CSV or JSON results must not let simulate() write over them.

    ``import_outputs`` accepts both and points ``output_file`` at the file, and
    its docstring offers continuing a simulation. simulate() only writes JSONL,
    so ``append=False`` truncated the file before this check existed.
    """
    results = tmp_path / f"results{suffix}"
    results.write_text(payload, encoding="utf-8")
    monte_carlo_calisto.output_file = str(results)
    before = results.read_bytes()

    with pytest.raises(ValueError, match="one JSON object per line"):
        monte_carlo_calisto.simulate(number_of_simulations=1, append=append)

    assert results.read_bytes() == before


def _three_logs(tmp_path):
    """Three distinct, acceptable working logs."""
    return [
        str(tmp_path / f"run.{part}.txt") for part in ("inputs", "outputs", "errors")
    ]


def test_simulation_log_check_names_the_file_that_is_wrong(tmp_path):
    """The message says which of the three paths has to change."""
    good = _three_logs(tmp_path)

    _refuse_logs_this_run_cannot_write(*good)  # canonical, no raise

    for label, args in (
        ("input_file", (str(tmp_path / "a.csv"), good[1], good[2])),
        ("output_file", (good[0], str(tmp_path / "b.json"), good[2])),
        ("error_file", (good[0], good[1], str(tmp_path / "c.csv"))),
    ):
        with pytest.raises(ValueError, match=label):
            _refuse_logs_this_run_cannot_write(*args)


def test_simulation_log_check_accepts_an_uppercase_suffix(tmp_path):
    """A .TXT log is the same file to the filesystem, so it is accepted."""
    upper = [str(tmp_path / f"run.{part}.TXT") for part in ("in", "out", "err")]
    _refuse_logs_this_run_cannot_write(*upper)


def _three_logs(tmp_path):
    """Three distinct, acceptable working logs."""
    return [
        str(tmp_path / f"run.{part}.txt") for part in ("inputs", "outputs", "errors")
    ]


def test_working_logs_must_be_three_different_files(tmp_path):
    """``import_results`` points all three at one path, which cannot work.

    A run appends input rows and output rows separately, so one shared log ends
    up holding both and neither reader can make sense of it.
    """
    shared = str(tmp_path / "result.txt")

    with pytest.raises(ValueError, match="same file"):
        _refuse_logs_this_run_cannot_write(shared, shared, shared)


@pytest.mark.parametrize("alias", ["dotdot", "symlink", "hardlink"])
def test_a_log_named_two_ways_is_still_one_file(tmp_path, alias):
    """Text comparison misses every way one file answers to two names."""
    inputs, _, errors = _three_logs(tmp_path)
    pathlib.Path(inputs).write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    if alias == "dotdot":
        other = str(tmp_path / "sub" / ".." / "run.inputs.txt")
    else:
        other = str(tmp_path / f"run.{alias}.txt")
        try:
            if alias == "symlink":
                pathlib.Path(other).symlink_to(inputs)
            else:
                os.link(inputs, other)
        except (OSError, NotImplementedError):
            pytest.skip(f"{alias} not available on this filesystem")

    with pytest.raises(ValueError, match="same file"):
        _refuse_logs_this_run_cannot_write(inputs, other, errors)


def test_three_separate_logs_are_accepted(tmp_path):
    """The control: distinct .txt paths raise nothing."""
    _refuse_logs_this_run_cannot_write(*_three_logs(tmp_path))


@pytest.mark.parametrize("indent", [2, 0, ""])
def test_an_indented_record_is_refused_before_anything_is_written(tmp_path, indent):
    """``indent`` splits a record over lines the readers take one at a time.

    Without this the run finished, then the completeness check called the file
    it had just written damaged.
    """
    with pytest.raises(ValueError, match="indent"):
        _refuse_logs_this_run_cannot_write(*_three_logs(tmp_path), {"indent": indent})


def test_a_newline_in_the_separators_is_refused_too(tmp_path):
    """The same hazard by another name."""
    with pytest.raises(ValueError, match="separators"):
        _refuse_logs_this_run_cannot_write(
            *_three_logs(tmp_path), {"separators": (",\n", ": ")}
        )


@pytest.mark.parametrize(
    "harmless", [{"indent": None}, {"sort_keys": True}, {"ensure_ascii": False}]
)
def test_export_options_that_keep_one_line_are_left_alone(tmp_path, harmless):
    """Only what puts a newline inside a record is refused."""
    _refuse_logs_this_run_cannot_write(*_three_logs(tmp_path), harmless)


def test_two_names_for_a_file_that_does_not_exist_yet_are_still_one_file(tmp_path):
    """``samefile`` needs both to exist, and a first run has created neither.

    Every other case here writes the file first, so the resolved-path branch
    that a first run actually takes was never exercised.
    """
    (tmp_path / "sub").mkdir()
    missing = str(tmp_path / "run.inputs.txt")
    same_by_another_name = str(tmp_path / "sub" / ".." / "run.inputs.txt")
    errors = str(tmp_path / "run.errors.txt")

    assert not pathlib.Path(missing).exists()
    with pytest.raises(ValueError, match="same file"):
        _refuse_logs_this_run_cannot_write(missing, same_by_another_name, errors)
