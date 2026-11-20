"""Release regressions exercised through real Monte Carlo flights."""

import json

import matplotlib.pyplot as plt
import pytest

from rocketpy import MonteCarlo
from rocketpy.simulation import monte_carlo as mc_module
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


def test_each_row_records_one_integer_seed(release_study):
    """Both halves of a record name the study, without NumPy state or digests."""
    release_study.simulate(2, random_seed=42, include_function_data=False)

    for path in (release_study.input_file, release_study.output_file):
        assert all(row["seed"] == 42 for row in _rows(path))
    assert "seed" not in release_study.results


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
def test_a_second_study_does_not_warn_about_seeds_the_first_one_set(
    release_study, recwarn, parallel
):
    """An object seed is ignored once, whichever path ran the first study."""
    release_study.environment._set_stochastic(42)
    run = {"include_function_data": False, "parallel": parallel, "n_workers": 2}
    release_study.simulate(1, random_seed=42, **run)
    release_study.simulate(1, append=False, **run)

    assert not [w for w in recwarn if "object seeds are ignored" in str(w.message)]


def test_seed_warnings_name_the_caller_of_simulate_convergence(release_study):
    """Both warnings point at the user's line, not inside rocketpy."""
    release_study.environment._set_stochastic(42)
    with pytest.warns(UserWarning, match="object seeds") as caught:
        release_study.simulate_convergence(
            target_attribute="max_speed", max_simulations=2, batch_size=2
        )
    assert caught[0].filename == __file__

    for path in (release_study.input_file, release_study.output_file):
        rows = _rows(path)
        for row in rows:
            row.pop("seed")
        _write_rows(path, rows)
    with pytest.warns(UserWarning, match="legacy") as caught:
        release_study.simulate_convergence(
            target_attribute="max_speed", max_simulations=3, batch_size=1
        )
    assert [w.filename for w in caught if "legacy" in str(w.message)] == [__file__]


@pytest.mark.parametrize("parallel", [False, True])
def test_v113_study_can_be_extended_with_a_warning(release_study, parallel):
    """Keep legacy rows, fill index 0 and warn about the random stream."""
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
    # v1.13 serial rows were one-based and parallel ones may hold a hole at
    # 0; the logs cannot tell which, so index 0 is run either way.
    assert sorted(row["index"] for row in _rows(release_study.output_file)) == [
        0,
        1,
        2,
        3,
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
    expected = _rows(release_study.input_file)
    release_study.simulate(0, append=False, random_seed=42)

    release_study.simulate_convergence(
        target_attribute="max_speed",
        tolerance=1e-12,
        max_simulations=4,
        batch_size=4,
        random_seed=42,
    )

    assert _rows(release_study.input_file) == expected


def _converge_by_threes(study, random_seed=None):
    # Stops after the second batch for seed 42: 0.269 at three samples, 0.252
    # at six. Unseeded, the bootstrap puts the second width in 0.236-0.262.
    history = study.simulate_convergence(
        target_attribute="max_speed",
        tolerance=0.26,
        max_simulations=12,
        batch_size=3,
        random_seed=random_seed,
    )
    return history, _rows(study.input_file)


def test_the_same_seed_gives_back_the_same_study(release_study):
    """The same seed draws the same inputs and stops after the same batch."""
    first = _converge_by_threes(release_study, random_seed=42)
    release_study.simulate(0, append=False, random_seed=42)

    assert _converge_by_threes(release_study, random_seed=42) == first
    assert len(first[0]) < 4  # stopped on the tolerance, not on max_simulations


def test_an_unseeded_study_repeats_from_the_seed_it_recorded(release_study):
    """The seed a study draws for itself reproduces it, stopping point included."""
    first = _converge_by_threes(release_study)
    recorded = first[1][0]["seed"]
    release_study.simulate(0, append=False, random_seed=recorded)

    assert _converge_by_threes(release_study, random_seed=recorded) == first


def test_convergence_reads_the_checkpoint_once(release_study, monkeypatch):
    """Batches reuse what the study just wrote instead of parsing the logs again."""
    reads = []
    real = mc_module._checkpoint_rows
    monkeypatch.setattr(
        mc_module, "_checkpoint_rows", lambda path: reads.append(path) or real(path)
    )
    release_study.simulate(1, random_seed=42, include_function_data=False)

    release_study.simulate_convergence(
        target_attribute="max_speed",
        tolerance=1e-12,
        max_simulations=4,
        batch_size=1,
    )

    assert not reads  # the logs are the ones the first simulate wrote
    assert sorted(row["index"] for row in _rows(release_study.input_file)) == [
        0,
        1,
        2,
        3,
    ]


def test_a_log_changed_after_the_run_is_read_again(release_study):
    """The shortcut only holds while the logs are what the study wrote."""
    release_study.simulate(2, random_seed=42, include_function_data=False)
    for path in (release_study.input_file, release_study.output_file):
        _write_rows(path, [row for row in _rows(path) if row["index"] != 1])

    release_study.simulate(2, append=True, include_function_data=False)

    assert sorted(row["index"] for row in _rows(release_study.input_file)) == [0, 1]
