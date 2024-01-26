import datetime
import time
from unittest.mock import patch

import pytest


@patch("matplotlib.pyplot.show")
def test_standard_atmosphere(mock_show, example_env):
    """Tests the standard atmosphere model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env.set_atmospheric_model(type="standard_atmosphere")
    assert example_env.info() == None
    assert example_env.all_info() == None
    assert abs(example_env.pressure(0) - 101325.0) < 1e-8
    assert abs(example_env.barometric_height(101325.0)) < 1e-2
    assert example_env.prints.print_earth_details() == None


@patch("matplotlib.pyplot.show")
def test_custom_atmosphere(mock_show, example_env):
    """Tests the custom atmosphere model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=None,
        temperature=300,
        wind_u=[(0, 5), (1000, 10)],
        wind_v=[(0, -2), (500, 3), (1600, 2)],
    )
    assert example_env.all_info() == None
    assert abs(example_env.pressure(0) - 101325.0) < 1e-8
    assert abs(example_env.barometric_height(101325.0)) < 1e-2
    assert abs(example_env.wind_velocity_x(0) - 5) < 1e-8
    assert abs(example_env.temperature(100) - 300) < 1e-8


@patch("matplotlib.pyplot.show")
def test_wyoming_sounding_atmosphere(mock_show, example_env):
    """Tests the Wyoming sounding model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    # TODO:: this should be added to the set_atmospheric_model() method as a
    #        "file" option, instead of receiving the URL as a string.
    URL = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"
    # give it at least 5 times to try to download the file
    for i in range(5):
        try:
            example_env.set_atmospheric_model(type="wyoming_sounding", file=URL)
            break
        except:
            time.sleep(1)  # wait 1 second before trying again
            pass
    assert example_env.all_info() == None
    assert abs(example_env.pressure(0) - 93600.0) < 1e-8
    assert abs(example_env.barometric_height(example_env.pressure(0)) - 722.0) < 1e-8
    assert abs(example_env.wind_velocity_x(0) - -2.9005178894925043) < 1e-8
    assert abs(example_env.temperature(100) - 291.75) < 1e-8


@pytest.mark.skip(reason="legacy tests")
@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_noaa_ruc_sounding_atmosphere(mock_show, example_env):
    URL = r"https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest"
    example_env.set_atmospheric_model(type="NOAARucSounding", file=URL)
    assert example_env.all_info() == None
    assert example_env.pressure(0) == 100000.0


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gfs_atmosphere(mock_show, example_env_robust):
    """Tests the Forecast model with the GFS file. It does not test the values,
    instead the test checks if the method runs without errors.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env_robust : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env_robust.set_atmospheric_model(type="Forecast", file="GFS")
    assert example_env_robust.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_nam_atmosphere(mock_show, example_env_robust):
    """Tests the Forecast model with the NAM file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env_robust : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env_robust.set_atmospheric_model(type="Forecast", file="NAM")
    assert example_env_robust.all_info() == None


# Deactivated since it is hard to figure out and appropriate date to use RAP forecast
@pytest.mark.skip(reason="legacy tests")
@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_rap_atmosphere(mock_show, example_env_robust):
    today = datetime.date.today()
    example_env_robust.set_date((today.year, today.month, today.day, 8))
    example_env_robust.set_atmospheric_model(type="Forecast", file="RAP")
    assert example_env_robust.all_info() == None


@patch("matplotlib.pyplot.show")
def test_era5_atmosphere(mock_show, example_env_robust):
    """Tests the Reanalysis model with the ERA5 file. It uses an example file
    available in the data/weather folder of the RocketPy repository.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env_robust : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env_robust.set_date((2018, 10, 15, 12))
    example_env_robust.set_atmospheric_model(
        type="Reanalysis",
        file="data/weather/SpaceportAmerica_2018_ERA-5.nc",
        dictionary="ECMWF",
    )
    assert example_env_robust.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gefs_atmosphere(mock_show, example_env_robust):
    """Tests the Ensemble model with the GEFS file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env_robust : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env_robust.set_atmospheric_model(type="Ensemble", file="GEFS")
    assert example_env_robust.all_info() == None


@patch("matplotlib.pyplot.show")
def test_info_returns(mock_show, example_env):
    """Tests the all_info_returned() all_plot_info_returned() and methods of the
    Environment class.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env : rocketpy.Environment
        Example environment object to be tested.
    """
    returned_plots = example_env.all_plot_info_returned()
    returned_infos = example_env.all_info_returned()
    expected_info = {
        "grav": example_env.gravity,
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
def test_cmc_atmosphere(mock_show, example_env_robust):
    """Tests the Ensemble model with the CMC file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env_robust : rocketpy.Environment
        Example environment object to be tested.
    """
    example_env_robust.set_atmospheric_model(type="Ensemble", file="CMC")
    assert example_env_robust.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_hiresw_ensemble_atmosphere(mock_show, example_env_robust):
    """Tests the Forecast model with the HIRESW file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_env_robust : rocketpy.Environment
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

    example_env_robust.set_date(date_info)
    example_env_robust.set_atmospheric_model(
        type="Forecast",
        file=f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/hiresw_conusarw_12z",
        dictionary=HIRESW_dictionary,
    )
    assert example_env_robust.all_info() == None
