"""Defines the fixtures for the Monte Carlo tests. The fixtures should be
instances of the MonteCarlo class, ideally."""

from pathlib import Path

import pytest

from rocketpy.simulation import MonteCarlo


@pytest.fixture
def monte_carlo_calisto(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """Creates a MonteCarlo object with the stochastic environment, stochastic
    calisto and stochastic flight.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        The stochastic environment object, this is a pytest fixture.
    stochastic_calisto : StochasticRocket
        The stochastic rocket object, this is a pytest fixture.
    stochastic_flight : StochasticFlight
        The stochastic flight object, this is a pytest fixture.
    tmp_path : pathlib.Path
        Directory for this test's logs, this is a pytest fixture. A bare name
        would resolve against whatever directory pytest was started from.

    Returns
    -------
    MonteCarlo
        The MonteCarlo object with the stochastic environment, stochastic
        calisto and stochastic flight.
    """
    return MonteCarlo(
        filename=str(tmp_path / "monte_carlo_test"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )


@pytest.fixture
def monte_carlo_calisto_pre_loaded(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """Creates a MonteCarlo object with some already imported simulations."""
    monte_carlo = MonteCarlo(
        filename=str(tmp_path / "monte_carlo_test"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    # Resolved against this file rather than the working directory, which is
    # what the caller happened to start pytest from. import_outputs opens it
    # "r+", and "w+" if it is missing, so a wrong guess writes into the tree.
    monte_carlo.import_results(
        filename=str(Path(__file__).parent / "example.outputs.txt")
    )
    return monte_carlo
