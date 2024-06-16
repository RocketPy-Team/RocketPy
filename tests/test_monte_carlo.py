import os
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

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
    total_impulse = []
    for _ in range(20):
        random_motor = stochastic_solid_motor.create_object()
        total_impulse.append(random_motor.total_impulse)

    assert np.isclose(np.mean(total_impulse), 6500, rtol=0.3)
    assert np.isclose(np.std(total_impulse), 1000, rtol=0.3)


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


@pytest.mark.slow
def test_monte_carlo_simulate(monte_carlo_calisto):
    """Tests the simulate method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    # NOTE: this is really slow, it runs 10 flight simulations
    monte_carlo_calisto.simulate(number_of_simulations=10, append=False)

    assert monte_carlo_calisto.num_of_loaded_sims == 10
    assert monte_carlo_calisto.number_of_simulations == 10
    assert monte_carlo_calisto.filename == "monte_carlo_test"
    assert monte_carlo_calisto.error_file == "monte_carlo_test.errors.txt"
    assert monte_carlo_calisto.output_file == "monte_carlo_test.outputs.txt"
    assert np.isclose(
        monte_carlo_calisto.processed_results["apogee"][0], 4711, rtol=0.15
    )
    assert np.isclose(
        monte_carlo_calisto.processed_results["impact_velocity"][0],
        -5.234,
        rtol=0.15,
    )
    os.remove("monte_carlo_test.errors.txt")
    os.remove("monte_carlo_test.outputs.txt")
    os.remove("monte_carlo_test.inputs.txt")


def test_monte_carlo_set_inputs_log(monte_carlo_calisto):
    """Tests the set_inputs_log method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    monte_carlo_calisto.input_file = "tests/fixtures/monte_carlo/example.inputs.txt"
    monte_carlo_calisto.set_inputs_log()
    assert len(monte_carlo_calisto.inputs_log) == 100
    assert all(isinstance(item, dict) for item in monte_carlo_calisto.inputs_log)
    assert all(
        "gravity" in item and "elevation" in item
        for item in monte_carlo_calisto.inputs_log
    )


def test_monte_carlo_set_outputs_log(monte_carlo_calisto):
    """Tests the set_outputs_log method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    monte_carlo_calisto.output_file = "tests/fixtures/monte_carlo/example.outputs.txt"
    monte_carlo_calisto.set_outputs_log()
    assert len(monte_carlo_calisto.outputs_log) == 100
    assert all(isinstance(item, dict) for item in monte_carlo_calisto.outputs_log)
    assert all(
        "apogee" in item and "impact_velocity" in item
        for item in monte_carlo_calisto.outputs_log
    )


# def test_monte_carlo_set_errors_log(monte_carlo_calisto):
#     monte_carlo_calisto.error_file = "tests/fixtures/monte_carlo/example.errors.txt"
#     monte_carlo_calisto.set_errors_log()
#     assert


def test_monte_carlo_prints(monte_carlo_calisto):
    """Tests the prints methods of the MonteCarlo class."""
    monte_carlo_calisto.info()


@patch("matplotlib.pyplot.show")
def test_monte_carlo_plots(mock_show, monte_carlo_calisto_pre_loaded):
    """Tests the plots methods of the MonteCarlo class."""
    assert monte_carlo_calisto_pre_loaded.all_info() is None


def test_monte_carlo_export_ellipses_to_kml(monte_carlo_calisto_pre_loaded):
    """Tests the export_ellipses_to_kml method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    assert (
        monte_carlo_calisto_pre_loaded.export_ellipses_to_kml(
            filename="monte_carlo_class_example.kml",
            origin_lat=32.990254,
            origin_lon=-106.974998,
            type="impact",
        )
        is None
    )

    os.remove("monte_carlo_class_example.kml")
