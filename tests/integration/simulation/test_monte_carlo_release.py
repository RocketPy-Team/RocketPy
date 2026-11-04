"""Release regressions exercised through real Monte Carlo flights."""

import json

import matplotlib.pyplot as plt
import pytest

from rocketpy import MonteCarlo
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticRocket,
    StochasticSolidMotor,
)


@pytest.fixture
def release_study(
    tmp_path, example_spaceport_env, calisto_robust, flight_calisto_robust
):
    """Build an isolated study with a small set of scalar outputs."""
    stochastic_flight = StochasticFlight(flight_calisto_robust, max_time=1)
    stochastic_flight.obj.equations_of_motion = "generalized"
    stochastic_rocket = StochasticRocket(calisto_robust, mass=(15, 0.1))
    stochastic_rocket.add_motor(
        StochasticSolidMotor(calisto_robust.motor),
        position=[calisto_robust.motor_position],
    )
    return MonteCarlo(
        filename=str(tmp_path / "study"),
        environment=StochasticEnvironment(example_spaceport_env),
        rocket=stochastic_rocket,
        flight=stochastic_flight,
        export_list=["max_speed", "t_final"],
    )


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_new_run_plots_only_physical_results(release_study):
    """Metadata must not reach boxplots or summary statistics."""
    release_study.simulate(2, random_seed=42, include_function_data=False)

    try:
        release_study.plots.all()
        assert set(release_study.results) == {"max_speed", "t_final"}
    finally:
        plt.close("all")


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("target", [3, 4])
def test_resume_fills_holes_without_repeating_completed_flights(
    release_study, parallel, target
):
    """A completed later index must survive recovery of an interrupted run."""
    release_study.simulate(4, random_seed=42, include_function_data=False)
    expected_inputs = _rows(release_study.input_file)
    expected_outputs = _rows(release_study.output_file)
    for path in (release_study.input_file, release_study.output_file):
        _write_rows(path, [row for row in _rows(path) if row["index"] != 1])

    release_study.simulate(
        target, append=True, parallel=parallel, n_workers=2, include_function_data=False
    )

    actual_inputs = _rows(release_study.input_file)
    actual_outputs = _rows(release_study.output_file)
    assert [row["index"] for row in actual_inputs] == [0, 2, 3, 1]
    assert {row["index"]: row["mass"] for row in actual_inputs} == {
        row["index"]: row["mass"] for row in expected_inputs
    }
    assert sorted(actual_outputs, key=lambda row: row["index"]) == expected_outputs


def test_each_input_row_records_one_integer_seed(release_study):
    """The checkpoint needs the study seed, without NumPy state or digests."""
    release_study.simulate(2, random_seed=42, include_function_data=False)

    assert all(row["seed"] == 42 for row in _rows(release_study.input_file))
    assert all("run_root" not in row for row in _rows(release_study.input_file))
    assert all(
        "seed" not in row and "run_root" not in row
        for row in _rows(release_study.output_file)
    )


@pytest.mark.parametrize("component", ["environment", "motor"])
def test_object_seed_without_study_seed_warns_and_reports_replay_seed(
    release_study, capsys, component
):
    """An overridden object seed must not silently suggest reproducibility."""
    model = (
        release_study.environment
        if component == "environment"
        else release_study.rocket.motors[0][0]
    )
    model._set_stochastic(42)

    with pytest.warns(UserWarning, match="object seeds are ignored"):
        release_study.simulate(1, include_function_data=False)

    seed = _rows(release_study.input_file)[0]["seed"]
    assert isinstance(seed, int)
    assert f"Monte Carlo random_seed: {seed}" in capsys.readouterr().out


@pytest.mark.parametrize("parallel", [False, True])
def test_v113_study_can_be_extended_with_a_warning(release_study, parallel):
    """Keep one-based legacy rows and append without claiming reproducibility."""
    release_study.simulate(2, random_seed=42, include_function_data=False)
    for path in (release_study.input_file, release_study.output_file):
        rows = _rows(path)
        for row in rows:
            row.pop("run_root", None)
            row.pop("seed", None)
            row["index"] += 1
        _write_rows(path, rows)
    original = _rows(release_study.output_file)

    with pytest.warns(UserWarning, match="v1.13|legacy"):
        release_study.simulate(
            4,
            append=True,
            random_seed=42,
            parallel=parallel,
            n_workers=2,
            include_function_data=False,
        )

    assert _rows(release_study.output_file)[:2] == original
    assert sorted(row["index"] for row in _rows(release_study.output_file)) == [
        1,
        2,
        3,
        4,
    ]


def test_convergence_seed_reproduces_inputs_across_batch_sizes(release_study):
    """Batches must continue the same random stream."""
    release_study.simulate_convergence(
        target_attribute="max_speed",
        tolerance=1e-12,
        max_simulations=4,
        batch_size=2,
        random_seed=42,
    )
    expected = [row["mass"] for row in _rows(release_study.input_file)]
    release_study.simulate(0, append=False, random_seed=42)

    release_study.simulate_convergence(
        target_attribute="max_speed",
        tolerance=1e-12,
        max_simulations=4,
        batch_size=4,
        random_seed=42,
    )

    assert [row["mass"] for row in _rows(release_study.input_file)] == expected
