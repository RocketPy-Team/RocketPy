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
from rocketpy.simulation import monte_carlo as mc_module
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
        # One draw for all three, as #1090 requires: three separate calls
        # meant the flight flew one sample and the exported row logged another.
        _sample_flight_inputs=lambda: {
            "rail_length": 5.0,
            "inclination": 84.0,
            "heading": 133.0,
        },
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
    # and the single draw is what the flight is actually built from
    assert (flight.rail_length, flight.inclination, flight.heading) == (
        5.0,
        84.0,
        133.0,
    )


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


class _InterruptingMonteCarlo(MonteCarlo):
    """A MonteCarlo that raises ``KeyboardInterrupt`` where Ctrl-C would land.

    Only the attributes ``simulate`` and ``__run_in_serial`` touch are set, so no
    stochastic object graph or real flight is needed. The name-mangled overrides
    stand in for the members ``MonteCarlo`` calls on itself.
    """

    # pylint: disable=super-init-not-called,invalid-name,unused-argument

    def __init__(self, filename, interrupt_after):
        self.filename = filename
        self._input_file = filename + ".inputs.txt"
        self._output_file = filename + ".outputs.txt"
        self._error_file = filename + ".errors.txt"
        self.num_of_loaded_sims = 0
        self.number_of_simulations = 0
        self._export_config = {}
        self._initial_sim_idx = 0
        self.interrupt_after = interrupt_after
        self.completed = 0

    def _MonteCarlo__run_single_simulation(self):
        if self.completed >= self.interrupt_after:
            raise KeyboardInterrupt("ctrl-c")
        self.completed += 1
        return object()

    def _MonteCarlo__evaluate_flight_inputs(self, index):
        return json.dumps({"index": index}) + "\n"

    def _MonteCarlo__evaluate_flight_outputs(self, flight, index):
        return json.dumps({"index": index, "apogee": 1000.0 + index}) + "\n"


def test_interrupted_serial_run_reaches_the_caller(tmp_path):
    """``simulate`` used to return normally after Ctrl-C.

    A caller could not tell a partial run from a complete one without opening
    the output file and counting rows.
    """
    mc = _InterruptingMonteCarlo(str(tmp_path / "run"), interrupt_after=2)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=10, parallel=False)


def test_interrupted_serial_run_keeps_the_rows_that_finished(tmp_path):
    """The two simulations that completed stay readable and paired."""
    mc = _InterruptingMonteCarlo(str(tmp_path / "run"), interrupt_after=2)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=10, parallel=False)

    inputs = (tmp_path / "run.inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs = (tmp_path / "run.outputs.txt").read_text(encoding="utf-8").splitlines()

    assert len(inputs) == 2
    assert len(outputs) == 2
    assert [json.loads(row)["index"] for row in inputs] == [
        json.loads(row)["index"] for row in outputs
    ]


def test_interrupted_serial_run_still_reloads_the_logs(tmp_path):
    """``__terminate_simulation`` runs before the interrupt leaves ``simulate``.

    It is what reloads the logs through the file setters. Asserting on the
    state those setters produce, rather than on the call, is what shows the
    reload actually happened.
    """
    mc = _InterruptingMonteCarlo(str(tmp_path / "run"), interrupt_after=2)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=10, parallel=False)

    assert mc.num_of_loaded_sims == 2
    assert len(mc.inputs_log) == 2
    assert len(mc.outputs_log) == 2
    assert mc.results["apogee"] == pytest.approx([1001.0, 1002.0])


def test_an_interrupted_run_can_be_continued_with_append(tmp_path):
    """The behavior the ``simulate`` docstring promises after an interrupt.

    ``set_num_of_loaded_sims`` is what ``append=True`` reads to decide where to
    resume, and it is only set by the reload above. This runs the whole path:
    interrupt, then continue, and check the indices on disk have no gap and no
    repeat.
    """
    stem = str(tmp_path / "run")
    mc = _InterruptingMonteCarlo(stem, interrupt_after=2)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=10, parallel=False)

    mc.interrupt_after = 10
    mc.simulate(number_of_simulations=10, append=True, parallel=False)

    rows = pathlib.Path(stem + ".outputs.txt").read_text(encoding="utf-8").splitlines()
    assert [json.loads(row)["index"] for row in rows] == list(range(1, 11))


def test_ctrl_c_before_the_first_simulation_is_still_the_interrupt(
    tmp_path, monkeypatch
):
    """The handler appends ``inputs_json``, which used to be unbound this early.

    Ctrl-C during the first ``keep_simulating()`` call reached the handler
    before the loop body had bound the name, so the run died with
    ``UnboundLocalError`` from inside the cleanup instead of with the interrupt.
    """

    def interrupt(self):
        raise KeyboardInterrupt("ctrl-c before the first simulation")

    monkeypatch.setattr(mc_module._SimMonitor, "keep_simulating", interrupt)
    mc = _InterruptingMonteCarlo(str(tmp_path / "run"), interrupt_after=0)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=5, parallel=False)


class _FakeWorker:
    """Stands in for a ``multiprocess.Process`` without starting anything.

    It leaves on the first join it is given, so every test built on it covers
    workers that exit cooperatively once the stop event is set — the guarantee
    the ``simulate`` docstring makes. A worker that outlives the grace period
    is terminated and then killed by ``_stop_the_workers_still_running``, and
    ``tests/unit/simulation/test_monte_carlo_worker_join.py`` is where that
    bounded shutdown is pinned; these tests are about the interrupt reaching
    the caller, not about the bound.
    """

    def __init__(self, interrupt_on_first_join=False, interrupt_on_start=False):
        self.starts = 0
        self.joins = 0
        self.timeouts = []
        self.terminated = False
        self.killed = False
        self.exitcode = None
        self._interrupt_on_first_join = interrupt_on_first_join
        self._interrupt_on_start = interrupt_on_start

    def is_alive(self):
        return self.exitcode is None

    def start(self):
        self.starts += 1
        if self._interrupt_on_start:
            raise KeyboardInterrupt("ctrl-c inside Process.start()")

    def join(self, timeout=None):
        self.joins += 1
        self.timeouts.append(timeout)
        if self._interrupt_on_first_join and self.joins == 1:
            raise KeyboardInterrupt("ctrl-c while waiting for the workers")
        self.exitcode = 0

    def terminate(self):
        self.terminated = True
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self.exitcode = -9


class _FakeManager:
    """The subset of the multiprocess manager that ``__run_in_parallel`` uses."""

    # pylint: disable=invalid-name

    def __init__(self):
        self.event = _FakeEvent()
        self.monitor = _FakeSimMonitor()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def Lock(self):
        return object()

    def Event(self):
        return self.event

    def _SimMonitor(self, **_kwargs):
        return self.monitor


class _FakeEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


class _FakeSimMonitor:
    def __init__(self, **_kwargs):
        self.final_status_calls = 0

    def print_final_status(self):
        self.final_status_calls += 1


def test_interrupted_parallel_run_signals_joins_and_reaches_the_caller(
    tmp_path, monkeypatch
):
    """Ctrl-C while waiting on the workers must not end as a successful run.

    The handler already signalled and joined the workers, but it then swallowed
    the interrupt, so ``simulate`` went on to report the study as finished.
    """
    manager = _FakeManager()
    workers = []

    class _FakeMultiprocess:
        # pylint: disable=invalid-name
        @staticmethod
        def Process(target=None, args=()):  # pylint: disable=unused-argument
            worker = _FakeWorker(interrupt_on_first_join=not workers)
            workers.append(worker)
            return worker

    monkeypatch.setattr(
        mc_module, "_import_multiprocess", lambda: (_FakeMultiprocess, None)
    )
    monkeypatch.setattr(
        mc_module, "_create_multiprocess_manager", lambda *_args: manager
    )
    mc = _InterruptingMonteCarlo(str(tmp_path / "run"), interrupt_after=0)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=4, parallel=True, n_workers=2)

    assert len(workers) == 2
    assert all(worker.starts == 1 for worker in workers)
    assert manager.event.is_set(), "the workers were never told to stop"
    # At least, not exactly: the bounded shutdown joins each worker once per
    # escalation stage, and how many stages it walks is its own business.
    assert workers[0].joins >= 2, (
        "the interrupted join was not retried after signalling"
    )
    assert workers[1].joins >= 1, "the second worker was never joined"
    assert manager.monitor.final_status_calls == 0, "a partial run reported completion"
    # __init__ above never sets inputs_log; only the reload in
    # __terminate_simulation does. Interrupted before any row was written, the
    # reload of the empty files must produce empty logs, not be skipped.
    assert mc.inputs_log == [], "the reload did not run on the interrupted path"
    assert mc.outputs_log == []


def test_ctrl_c_during_worker_startup_still_stops_the_started_workers(
    tmp_path, monkeypatch
):
    """Ctrl-C in the middle of the startup loop must clean up what came up.

    The cleanup handler used to begin only after every worker had started, so
    an interrupt during ``Process.start()`` left the earlier workers running
    with nobody signalling or joining them.
    """
    manager = _FakeManager()
    workers = []

    class _FakeMultiprocess:
        # pylint: disable=invalid-name
        @staticmethod
        def Process(target=None, args=()):  # pylint: disable=unused-argument
            # The second start() raises where a real Ctrl-C could land.
            worker = _FakeWorker(interrupt_on_start=len(workers) == 1)
            workers.append(worker)
            return worker

    monkeypatch.setattr(
        mc_module, "_import_multiprocess", lambda: (_FakeMultiprocess, None)
    )
    monkeypatch.setattr(
        mc_module, "_create_multiprocess_manager", lambda *_args: manager
    )
    mc = _InterruptingMonteCarlo(str(tmp_path / "run"), interrupt_after=0)

    with pytest.raises(KeyboardInterrupt):
        mc.simulate(number_of_simulations=4, parallel=True, n_workers=2)

    assert manager.event.is_set(), "the started worker was never told to stop"
    assert workers[0].joins >= 1, "the started worker was never joined"
    # The second worker's start() raised, so it never entered the started list
    # and must not be joined: joining a never-started process raises.
    assert workers[1].joins == 0


def test_ctrl_c_between_the_two_appends_rolls_the_inputs_row_back(tmp_path):
    """The rollback in ``_append_simulation_record`` must catch the interrupt.

    It caught ``Exception``, and ``KeyboardInterrupt`` derives from
    ``BaseException``, so Ctrl-C after the inputs append but before the outputs
    append left the inputs file one row longer — a one-sided record that the
    reload then disagrees with. This is the boundary between this change and
    the pairing that #1125 introduced.
    """
    stem = str(tmp_path / "run")
    mc = _InterruptingMonteCarlo(stem, interrupt_after=10)

    real_open = builtins.open
    outputs_path = stem + ".outputs.txt"
    appends = {"count": 0}

    def interrupt_third_outputs_append(*args, **kwargs):
        file = args[0] if args else kwargs["file"]
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        if os.fspath(file) == outputs_path and "a" in mode:
            appends["count"] += 1
            if appends["count"] == 3:
                raise KeyboardInterrupt("ctrl-c between the appends")
        return real_open(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        with patch("builtins.open", side_effect=interrupt_third_outputs_append):
            mc.simulate(number_of_simulations=10, parallel=False)

    inputs = pathlib.Path(stem + ".inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs = pathlib.Path(outputs_path).read_text(encoding="utf-8").splitlines()
    errors = pathlib.Path(stem + ".errors.txt").read_text(encoding="utf-8").splitlines()

    assert len(inputs) == 2, "the third inputs row was not rolled back"
    assert len(outputs) == 2
    assert [json.loads(row)["index"] for row in inputs] == [
        json.loads(row)["index"] for row in outputs
    ]
    # The simulation that was cut short is recorded where errors go, so the
    # rolled-back row is preserved rather than lost.
    assert [json.loads(row)["index"] for row in errors] == [3]
    assert mc.num_of_loaded_sims == 2


def test_ctrl_c_in_the_progress_print_leaves_the_error_file_empty(tmp_path):
    """A committed row must not be reported as one that never finished.

    After ``_append_simulation_record`` returns, the pair is on disk. The
    handler used to write ``inputs_json`` to the error file anyway when the
    interrupt landed in ``print_update_status()`` — or in the next
    ``keep_simulating()`` call — because the name still held the committed row.
    """
    stem = str(tmp_path / "run")
    mc = _InterruptingMonteCarlo(stem, interrupt_after=10)

    updates = {"count": 0}
    real_update = mc_module._SimMonitor.print_update_status

    def counting_update(self, *args, **kwargs):
        updates["count"] += 1
        if updates["count"] == 2:
            raise KeyboardInterrupt("ctrl-c during the progress print")
        return real_update(self, *args, **kwargs)

    with patch.object(mc_module._SimMonitor, "print_update_status", counting_update):
        with pytest.raises(KeyboardInterrupt):
            mc.simulate(number_of_simulations=10, parallel=False)

    inputs = pathlib.Path(stem + ".inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs = (
        pathlib.Path(stem + ".outputs.txt").read_text(encoding="utf-8").splitlines()
    )
    errors = pathlib.Path(stem + ".errors.txt").read_text(encoding="utf-8")

    assert len(inputs) == 2
    assert len(outputs) == 2
    assert errors == "", "a committed row was written to the error file"
    assert mc.num_of_loaded_sims == 2


class _TornWrite:
    """A context manager that writes half the row, flushes, and interrupts."""

    def __init__(self, real_file):
        self._real_file = real_file

    def __enter__(self):
        self._file = self._real_file.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._real_file.__exit__(*exc_info)

    def write(self, text):
        self._file.write(text[: len(text) // 2])
        self._file.flush()
        raise KeyboardInterrupt("ctrl-c inside the outputs write")


def test_a_torn_outputs_write_rolls_both_files_back(tmp_path):
    """An interrupt inside the write itself must not leave half a row.

    Rolling back only the inputs file handled the interrupt *between* the two
    appends; one landing *inside* the outputs write leaves a torn partial row
    that the rollback then has to remove too, or the reload dies with
    ``JSONDecodeError`` instead of the interrupt.
    """
    stem = str(tmp_path / "run")
    mc = _InterruptingMonteCarlo(stem, interrupt_after=10)

    outputs_path = stem + ".outputs.txt"
    appends = {"count": 0}
    real_open = builtins.open

    def torn_third_outputs_append(*args, **kwargs):
        file = args[0] if args else kwargs["file"]
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        if os.fspath(file) == outputs_path and "a" in mode:
            appends["count"] += 1
            if appends["count"] == 3:
                return _TornWrite(real_open(*args, **kwargs))
        return real_open(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        with patch("builtins.open", side_effect=torn_third_outputs_append):
            mc.simulate(number_of_simulations=10, parallel=False)

    inputs = pathlib.Path(stem + ".inputs.txt").read_text(encoding="utf-8").splitlines()
    outputs = pathlib.Path(outputs_path).read_text(encoding="utf-8").splitlines()
    errors = pathlib.Path(stem + ".errors.txt").read_text(encoding="utf-8").splitlines()

    assert len(inputs) == 2, "the inputs row of the torn record was not rolled back"
    assert len(outputs) == 2, "the torn outputs row was not rolled back"
    for row in inputs + outputs:
        json.loads(row)  # every surviving row must still parse
    assert [json.loads(row)["index"] for row in errors] == [3]
    assert mc.num_of_loaded_sims == 2
