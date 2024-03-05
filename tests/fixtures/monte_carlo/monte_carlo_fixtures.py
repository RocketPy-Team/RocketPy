import pytest

from rocketpy.simulation import MonteCarlo


@pytest.fixture
def calisto_dispersion(stochastic_environment, stochastic_calisto, stochastic_flight):
    return MonteCarlo(
        filename="monte_carlo_test",
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
