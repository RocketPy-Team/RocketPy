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
            "single_point": [100],
            "empty_attribute": [],
        }


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
