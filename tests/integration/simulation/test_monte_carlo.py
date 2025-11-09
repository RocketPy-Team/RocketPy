# pylint: disable=unused-argument
import os
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

plt.rcParams.update({"figure.max_open_warning": 0})


def _post_test_file_cleanup():
    """Clean monte carlo files after test session if they exist."""
    if os.path.exists("monte_carlo_class_example.kml"):
        os.remove("monte_carlo_class_example.kml")
    if os.path.exists("monte_carlo_test.errors.txt"):
        os.remove("monte_carlo_test.errors.txt")
    if os.path.exists("monte_carlo_test.inputs.txt"):
        os.remove("monte_carlo_test.inputs.txt")
    if os.path.exists("monte_carlo_test.outputs.txt"):
        os.remove("monte_carlo_test.outputs.txt")


@pytest.mark.slow
@pytest.mark.parametrize("parallel", [False, True])
def test_monte_carlo_simulate(monte_carlo_calisto, parallel):
    """Tests the simulate method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # NOTE: this is really slow, it runs 10 flight simulations
        monte_carlo_calisto.simulate(
            number_of_simulations=10, append=False, parallel=parallel
        )

        assert monte_carlo_calisto.num_of_loaded_sims == 10
        assert monte_carlo_calisto.number_of_simulations == 10
        assert str(monte_carlo_calisto.filename.name) == "monte_carlo_test"
        assert str(monte_carlo_calisto.error_file.name) == "monte_carlo_test.errors.txt"
        assert (
            str(monte_carlo_calisto.output_file.name) == "monte_carlo_test.outputs.txt"
        )
        assert np.isclose(
            monte_carlo_calisto.processed_results["apogee"][0], 4711, rtol=0.2
        )
        assert np.isclose(
            monte_carlo_calisto.processed_results["impact_velocity"][0],
            -5.234,
            rtol=0.2,
        )
    finally:
        _post_test_file_cleanup()


def test_monte_carlo_set_inputs_log(monte_carlo_calisto):
    """Tests the set_inputs_log method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        monte_carlo_calisto.input_file = "tests/fixtures/monte_carlo/example.inputs.txt"
        monte_carlo_calisto.set_inputs_log()
        assert len(monte_carlo_calisto.inputs_log) == 100
        assert all(isinstance(item, dict) for item in monte_carlo_calisto.inputs_log)
        assert all(
            "gravity" in item and "elevation" in item
            for item in monte_carlo_calisto.inputs_log
        )
    finally:
        _post_test_file_cleanup()


def test_monte_carlo_set_outputs_log(monte_carlo_calisto):
    """Tests the set_outputs_log method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        monte_carlo_calisto.output_file = (
            "tests/fixtures/monte_carlo/example.outputs.txt"
        )
        monte_carlo_calisto.set_outputs_log()
        assert len(monte_carlo_calisto.outputs_log) == 100
        assert all(isinstance(item, dict) for item in monte_carlo_calisto.outputs_log)
        assert all(
            "apogee" in item and "impact_velocity" in item
            for item in monte_carlo_calisto.outputs_log
        )
    finally:
        _post_test_file_cleanup()


# def test_monte_carlo_set_errors_log(monte_carlo_calisto):
#     monte_carlo_calisto.error_file = "tests/fixtures/monte_carlo/example.errors.txt"
#     monte_carlo_calisto.set_errors_log()
#     assert


def test_monte_carlo_prints(monte_carlo_calisto):
    """Tests the prints methods of the MonteCarlo class."""
    try:
        monte_carlo_calisto.info()
        monte_carlo_calisto.compare_info(monte_carlo_calisto)
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")  # pylint: disable=unused-argument
def test_monte_carlo_plots(mock_show, monte_carlo_calisto_pre_loaded):
    """Tests the plots methods of the MonteCarlo class."""
    try:
        assert monte_carlo_calisto_pre_loaded.all_info() is None
        assert (
            monte_carlo_calisto_pre_loaded.compare_plots(monte_carlo_calisto_pre_loaded)
            is None
        )
        assert (
            monte_carlo_calisto_pre_loaded.compare_ellipses(
                monte_carlo_calisto_pre_loaded
            )
            is None
        )
    finally:
        _post_test_file_cleanup()


def test_monte_carlo_export_ellipses_to_kml(monte_carlo_calisto_pre_loaded):
    """Tests the export_ellipses_to_kml method of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        assert (
            monte_carlo_calisto_pre_loaded.export_ellipses_to_kml(
                filename="monte_carlo_class_example.kml",
                origin_lat=32.990254,
                origin_lon=-106.974998,
                type="all",
            )
            is None
        )
    finally:
        _post_test_file_cleanup()


@pytest.mark.slow
def test_monte_carlo_callback(monte_carlo_calisto):
    """Tests the data_collector argument of the MonteCarlo class.

    Parameters
    ----------
    monte_carlo_calisto : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # define valid data collector
        valid_data_collector = {
            "name": lambda flight: flight.name,
            "density_t0": lambda flight: flight.env.density(0),
        }

        monte_carlo_calisto.data_collector = valid_data_collector
        # NOTE: this is really slow, it runs 10 flight simulations
        monte_carlo_calisto.simulate(number_of_simulations=10, append=False)

        # tests if print works when we have None in summary
        monte_carlo_calisto.info()

        ## tests if an error is raised for invalid data_collector definitions
        # invalid type
        def invalid_data_collector(flight):
            return flight.name

        with pytest.raises(ValueError):
            monte_carlo_calisto._check_data_collector(invalid_data_collector)

        # invalid key overwrite
        invalid_data_collector = {"apogee": lambda flight: flight.apogee}
        with pytest.raises(ValueError):
            monte_carlo_calisto._check_data_collector(invalid_data_collector)

        # invalid callback definition
        invalid_data_collector = {"name": "Calisto"}  # callbacks must be callables!
        with pytest.raises(ValueError):
            monte_carlo_calisto._check_data_collector(invalid_data_collector)

        # invalid logic (division by zero)
        invalid_data_collector = {
            "density_t0": lambda flight: flight.env.density(0) / "0",
        }
        monte_carlo_calisto.data_collector = invalid_data_collector
        # NOTE: this is really slow, it runs 10 flight simulations
        with pytest.raises(ValueError):
            monte_carlo_calisto.simulate(number_of_simulations=10, append=False)
    finally:
        _post_test_file_cleanup()
