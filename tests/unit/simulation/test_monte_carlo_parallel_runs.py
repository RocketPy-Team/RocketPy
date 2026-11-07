import pytest

from rocketpy.simulation.monte_carlo import MonteCarlo


@pytest.mark.parametrize("parallel", [False, True])
def test_a_monte_carlo_run_finishes(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path, parallel
):
    """A real run completes and records every simulation, both modes."""
    # The parallel path hands each worker a SeedSequence rather than an int, and
    # nothing else in the suite exercises that. A worker that dies on it is not
    # reported, so this reads as a hang rather than as a failure.
    #
    # Built here rather than taken from the monte_carlo_calisto fixture, whose
    # own filename is fixed, since `filename` is a plain attribute and the three
    # working paths are settled when the object is constructed.
    analysis = MonteCarlo(
        filename=str(tmp_path / "study"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )

    analysis.simulate(
        number_of_simulations=2,
        append=False,
        parallel=parallel,
        n_workers=2 if parallel else None,
    )

    assert analysis.num_of_loaded_sims == 2
    assert str(tmp_path) in str(analysis.output_file)
