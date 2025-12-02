import matplotlib as plt
import numpy as np
import pytest

from rocketpy.simulation import MonteCarlo

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
        }


def test_ci_calculation_basic():
    """Checks that the confidence interval contains the known mean."""
    mc = MockMonteCarlo()

    ci = mc.estimate_confidence_interval("apogee", confidence_level=0.95)

    assert ci.low < 100 < ci.high
    assert ci.low < ci.high


def test_ci_custom_statistic():
    """Checks that the statistic can be changed (e.g., standard deviation instead of mean)."""
    mc = MockMonteCarlo()

    ci_std = mc.estimate_confidence_interval("apogee", statistic=np.std)

    assert ci_std.low > 0
    assert ci_std.low < ci_std.high


def test_ci_error_handling():
    """Checks that the code raises an error if the key does not exist."""
    mc = MockMonteCarlo()

    # Request a variable that does not exist ("altitude" is not in our mock)
    with pytest.raises(ValueError) as excinfo:
        mc.estimate_confidence_interval("altitude")

    assert "not found in results" in str(excinfo.value)


def test_ci_consistency():
    """Checks that a higher confidence level yields a wider interval."""
    mc = MockMonteCarlo()

    ci_90 = mc.estimate_confidence_interval("apogee", confidence_level=0.90)
    width_90 = ci_90.high - ci_90.low

    ci_99 = mc.estimate_confidence_interval("apogee", confidence_level=0.99)
    width_99 = ci_99.high - ci_99.low

    # The more confident we want to be (99%), the wider the interval must be
    assert width_99 >= width_90
