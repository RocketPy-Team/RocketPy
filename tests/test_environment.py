import datetime
from unittest.mock import patch

import pytest


@patch("matplotlib.pyplot.show")
def test_standard_atmosphere(mock_show, example_plain_env):
    """Tests the standard atmosphere model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_plain_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_plain_env.set_atmospheric_model(type="standard_atmosphere")
    assert example_plain_env.info() == None
    assert example_plain_env.all_info() == None
    assert abs(example_plain_env.pressure(0) - 101325.0) < 1e-8
    assert abs(example_plain_env.barometric_height(101325.0)) < 1e-2
    assert example_plain_env.prints.print_earth_details() == None


@patch("matplotlib.pyplot.show")
def test_custom_atmosphere(mock_show, example_plain_env):
    """Tests the custom atmosphere model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_plain_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_plain_env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=None,
        temperature=300,
        wind_u=[(0, 5), (1000, 10)],
        wind_v=[(0, -2), (500, 3), (1600, 2)],
    )
    assert example_plain_env.all_info() == None
    assert abs(example_plain_env.pressure(0) - 101325.0) < 1e-8
    assert abs(example_plain_env.barometric_height(101325.0)) < 1e-2
    assert abs(example_plain_env.wind_velocity_x(0) - 5) < 1e-8
    assert abs(example_plain_env.temperature(100) - 300) < 1e-8


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gfs_atmosphere(mock_show, example_spaceport_env):
    """Tests the Forecast model with the GFS file. It does not test the values,
    instead the test checks if the method runs without errors.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Forecast", file="GFS")
    assert example_spaceport_env.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_nam_atmosphere(mock_show, example_spaceport_env):
    """Tests the Forecast model with the NAM file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Forecast", file="NAM")
    assert example_spaceport_env.all_info() == None


# Deactivated since it is hard to figure out and appropriate date to use RAP forecast
@pytest.mark.skip(reason="legacy tests")
@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_rap_atmosphere(mock_show, example_spaceport_env):
    today = datetime.date.today()
    example_spaceport_env.set_date((today.year, today.month, today.day, 8))
    example_spaceport_env.set_atmospheric_model(type="Forecast", file="RAP")
    assert example_spaceport_env.all_info() == None


@patch("matplotlib.pyplot.show")
def test_era5_atmosphere(mock_show, example_spaceport_env):
    """Tests the Reanalysis model with the ERA5 file. It uses an example file
    available in the data/weather folder of the RocketPy repository.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_date((2018, 10, 15, 12))
    example_spaceport_env.set_atmospheric_model(
        type="Reanalysis",
        file="data/weather/SpaceportAmerica_2018_ERA-5.nc",
        dictionary="ECMWF",
    )
    assert example_spaceport_env.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gefs_atmosphere(mock_show, example_spaceport_env):
    """Tests the Ensemble model with the GEFS file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Ensemble", file="GEFS")
    assert example_spaceport_env.all_info() == None


@patch("matplotlib.pyplot.show")
def test_info_returns(mock_show, example_plain_env):
    """Tests the all_info_returned() all_plot_info_returned() and methods of the
    Environment class.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_plain_env : rocketpy.Environment
        Example environment object to be tested.
    """
    returned_plots = example_plain_env.all_plot_info_returned()
    returned_infos = example_plain_env.all_info_returned()
    expected_info = {
        "grav": example_plain_env.gravity,
        "elevation": 0,
        "model_type": "standard_atmosphere",
        "model_type_max_expected_height": 80000,
        "wind_speed": 0,
        "wind_direction": 0,
        "wind_heading": 0,
        "surface_pressure": 1013.25,
        "surface_temperature": 288.15,
        "surface_air_density": 1.225000018124288,
        "surface_speed_of_sound": 340.293988026089,
        "lat": 0,
        "lon": 0,
    }
    expected_plots_keys = [
        "grid",
        "wind_speed",
        "wind_direction",
        "speed_of_sound",
        "density",
        "wind_vel_x",
        "wind_vel_y",
        "pressure",
        "temperature",
    ]
    assert list(returned_infos.keys()) == list(expected_info.keys())
    assert list(returned_infos.values()) == list(expected_info.values())
    assert list(returned_plots.keys()) == expected_plots_keys


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_cmc_atmosphere(mock_show, example_spaceport_env):
    """Tests the Ensemble model with the CMC file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Ensemble", file="CMC")
    assert example_spaceport_env.all_info() == None
