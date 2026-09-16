import json

import pytest

from rocketpy.simulation.monte_carlo import MonteCarlo


def _study(tmp_path, stem, models):
    environment, rocket, flight = models
    return MonteCarlo(
        filename=str(tmp_path / stem),
        environment=environment,
        rocket=rocket,
        flight=flight,
    )


def _drawn(analysis):
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    return {row["index"]: row.get("mass") for row in rows}


@pytest.fixture(name="models")
def _models(stochastic_environment, stochastic_calisto, stochastic_flight):
    return stochastic_environment, stochastic_calisto, stochastic_flight


def _converge(analysis, batch_size, seed):
    return analysis.simulate_convergence(
        target_attribute="apogee_time",
        tolerance=1e-9,
        max_simulations=4,
        batch_size=batch_size,
        random_seed=seed,
    )


def test_a_convergence_study_repeats_from_one_seed(models, tmp_path):
    """The same seed gives the study back, which it could not before."""
    first = _study(tmp_path, "first", models)
    second = _study(tmp_path, "second", models)

    _converge(first, batch_size=2, seed=42)
    _converge(second, batch_size=2, seed=42)

    assert _drawn(first) == _drawn(second)


def test_the_batch_size_does_not_reach_the_samples(models, tmp_path):
    """Splitting the study differently must not change which samples it takes."""
    # Each batch used to derive a root of its own, so the same
    # max_simulations reached a different set of samples per batch size.
    in_twos = _study(tmp_path, "twos", models)
    in_fours = _study(tmp_path, "fours", models)

    _converge(in_twos, batch_size=2, seed=42)
    _converge(in_fours, batch_size=4, seed=42)

    assert _drawn(in_twos) == _drawn(in_fours)


def test_a_study_without_a_seed_still_runs(models, tmp_path):
    """The control. No seed is still allowed, and still reproduces nothing."""
    analysis = _study(tmp_path, "unseeded", models)

    history = _converge(analysis, batch_size=2, seed=None)

    assert history
    assert sorted(_drawn(analysis)) == [0, 1, 2, 3]
