"""Defines the fixtures for the Monte Carlo tests. The fixtures should be
instances of the MonteCarlo class, ideally."""

import pytest

from rocketpy.simulation import MonteCarlo


@pytest.fixture
def monte_carlo_calisto(stochastic_environment, stochastic_calisto, stochastic_flight):
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

    Returns
    -------
    MonteCarlo
        The MonteCarlo object with the stochastic environment, stochastic
        calisto and stochastic flight.
    """
    return MonteCarlo(
        filename="monte_carlo_test",
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )


@pytest.fixture
def monte_carlo_calisto_pre_loaded(
    stochastic_environment, stochastic_calisto, stochastic_flight
):
    """Creates a MonteCarlo object with some already imported simulations."""
    monte_carlo = MonteCarlo(
        filename="monte_carlo_test",
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    monte_carlo.import_results(
        filename="tests/fixtures/monte_carlo/example.outputs.txt"
    )
    return monte_carlo
