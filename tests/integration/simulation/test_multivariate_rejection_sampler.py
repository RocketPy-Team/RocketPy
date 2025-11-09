import os

import pytest
from scipy.stats import norm

from rocketpy import MultivariateRejectionSampler


@pytest.mark.parametrize(
    "monte_carlo_filepath, distribution_dict",
    [
        (
            "tests/fixtures/monte_carlo/example",
            {"mass": (norm(15.5, 1).pdf, norm(15.2, 1).pdf)},
        ),
        (
            "tests/fixtures/monte_carlo/example",
            {"motors_total_impulse": (norm(6000, 1000).pdf, norm(6500, 1000).pdf)},
        ),
        (
            "tests/fixtures/monte_carlo/example",
            {
                "parachutes_calisto_drogue_chute_lag": (
                    norm(1.5, 0.25).pdf,
                    norm(1.2, 0.25).pdf,
                )
            },
        ),
    ],
)
def test_mrs_sample_integration(monte_carlo_filepath, distribution_dict):
    """Tests the set_inputs_log method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    mrs_filepath = "mrs"
    mrs = MultivariateRejectionSampler(monte_carlo_filepath, mrs_filepath)
    mrs.sample(distribution_dict)

    os.remove("mrs.inputs.txt")
    os.remove("mrs.outputs.txt")
    os.remove("mrs.errors.txt")
