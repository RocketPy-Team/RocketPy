import time
from datetime import datetime
from unittest.mock import patch

import numpy.ma as ma
import pytest

from rocketpy import Environment


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


@pytest.mark.skip(reason="legacy tests")  # deprecated method
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
def test_wyoming_sounding_atmosphere(mock_show, example_plain_env):
    """Asserts whether the Wyoming sounding model in the environment
    object behaves as expected with respect to some attributes such
    as pressure, barometric_height, wind_velocity and temperature.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_plain_env : rocketpy.Environment
        Example environment object to be tested.
    """

    # TODO:: this should be added to the set_atmospheric_model() method as a
    #        "file" option, instead of receiving the URL as a string.
    URL = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"
    # give it at least 5 times to try to download the file
    for i in range(5):
        try:
            example_plain_env.set_atmospheric_model(type="wyoming_sounding", file=URL)
            break
        except:
            time.sleep(1)  # wait 1 second before trying again
            pass
    assert example_plain_env.all_info() == None
    assert abs(example_plain_env.pressure(0) - 93600.0) < 1e-8
    assert (
        abs(example_plain_env.barometric_height(example_plain_env.pressure(0)) - 722.0)
        < 1e-8
    )
    assert abs(example_plain_env.wind_velocity_x(0) - -2.9005178894925043) < 1e-8
    assert abs(example_plain_env.temperature(100) - 291.75) < 1e-8


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_hiresw_ensemble_atmosphere(mock_show, example_spaceport_env):
    """Tests the Forecast model with the HIRESW file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    # TODO: why isn't the HIRESW a built-in option in the set_atmospheric_model() method?
    HIRESW_dictionary = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    today = datetime.date.today()
    date_info = (today.year, today.month, today.day, 12)  # Hour given in UTC time
    date_string = f"{date_info[0]}{date_info[1]:02}{date_info[2]:02}"

    example_spaceport_env.set_date(date_info)

    example_spaceport_env.set_atmospheric_model(
        type="Forecast",
        file=f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/hiresw_conusarw_12z",
        dictionary=HIRESW_dictionary,
    )

    assert example_spaceport_env.all_info() == None


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
