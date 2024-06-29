import datetime
from unittest.mock import patch

import pytest


@patch("matplotlib.pyplot.show")
def test_standard_atmosphere(
    mock_show, example_plain_env
):  # pylint: disable=unused-argument
    """Tests the standard atmosphere model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_plain_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_plain_env.set_atmospheric_model(type="standard_atmosphere")
    assert example_plain_env.info() is None
    assert example_plain_env.all_info() is None
    assert abs(example_plain_env.pressure(0) - 101325.0) < 1e-8
    assert abs(example_plain_env.barometric_height(101325.0)) < 1e-2
    assert example_plain_env.prints.print_earth_details() is None


@patch("matplotlib.pyplot.show")
def test_custom_atmosphere(
    mock_show, example_plain_env
):  # pylint: disable=unused-argument
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
    assert example_plain_env.all_info() is None
    assert abs(example_plain_env.pressure(0) - 101325.0) < 1e-8
    assert abs(example_plain_env.barometric_height(101325.0)) < 1e-2
    assert abs(example_plain_env.wind_velocity_x(0) - 5) < 1e-8
    assert abs(example_plain_env.temperature(100) - 300) < 1e-8


@patch("matplotlib.pyplot.show")
def test_wyoming_sounding_atmosphere(
    mock_show, example_plain_env
):  # pylint: disable=unused-argument
    """Tests the Wyoming sounding model in the environment object.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_plain_env : rocketpy.Environment
        Example environment object to be tested.
    """
    # TODO:: this should be added to the set_atmospheric_model() method as a
    #        "file" option, instead of receiving the URL as a string.
    URL = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"  # pylint: disable=invalid-name
    # give it at least 5 times to try to download the file
    example_plain_env.set_atmospheric_model(type="wyoming_sounding", file=URL)

    assert example_plain_env.all_info() is None
    assert abs(example_plain_env.pressure(0) - 93600.0) < 1e-8
    assert (
        abs(example_plain_env.barometric_height(example_plain_env.pressure(0)) - 722.0)
        < 1e-8
    )
    assert abs(example_plain_env.wind_velocity_x(0) - -2.9005178894925043) < 1e-8
    assert abs(example_plain_env.temperature(100) - 291.75) < 1e-8


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_noaa_ruc_sounding_atmosphere(
    mock_show, example_plain_env
):  # pylint: disable=unused-argument
    URL = r"https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest"  # pylint: disable=invalid-name
    example_plain_env.set_atmospheric_model(type="NOAARucSounding", file=URL)
    assert example_plain_env.all_info() is None
    assert example_plain_env.pressure(0) == 100000.0


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gfs_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
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
    assert example_spaceport_env.all_info() is None


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_nam_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
    """Tests the Forecast model with the NAM file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Forecast", file="NAM")
    assert example_spaceport_env.all_info() is None


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_rap_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
    today = datetime.date.today()
    example_spaceport_env.set_date((today.year, today.month, today.day, 8))
    example_spaceport_env.set_atmospheric_model(type="Forecast", file="RAP")
    assert example_spaceport_env.all_info() is None


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_era5_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
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
    assert example_spaceport_env.all_info() is None


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gefs_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
    """Tests the Ensemble model with the GEFS file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Ensemble", file="GEFS")
    assert example_spaceport_env.all_info() is None


@pytest.mark.skip(reason="legacy tests")  # deprecated method
@patch("matplotlib.pyplot.show")
def test_info_returns(mock_show, example_plain_env):  # pylint: disable=unused-argument
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


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_cmc_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
    """Tests the Ensemble model with the CMC file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    example_spaceport_env.set_atmospheric_model(type="Ensemble", file="CMC")
    assert example_spaceport_env.all_info() is None


# @pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_hiresw_ensemble_atmosphere(
    mock_show, example_spaceport_env
):  # pylint: disable=unused-argument
    """Tests the Forecast model with the HIRESW file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    today = datetime.date.today()
    date_info = (today.year, today.month, today.day, 12)  # Hour given in UTC time

    example_spaceport_env.set_date(date_info)
    example_spaceport_env.set_atmospheric_model(
        type="Forecast",
        file="HIRESW",
        dictionary="HIRESW",
    )
    assert example_spaceport_env.all_info() is None


@pytest.mark.parametrize(
    "model_name",
    [
        "ECMWF",
        "GFS",
        "ICON",
        "ICONEU",
    ],
)
def test_windy_atmosphere(example_euroc_env, model_name):
    """Tests the Windy model in the environment object. The test ensures the
    pressure, temperature, and wind profiles are working and giving reasonable
    values. The tolerances may be higher than usual due to the nature of the
    atmospheric uncertainties, but it is ok since we are just testing if the
    method is working.

    Parameters
    ----------
    example_euroc_env : Environment
        Example environment object to be tested. The EuRoC launch site is used
        to test the ICONEU model, which only works in Europe.
    model_name : str
        The name of the model to be passed to the set_atmospheric_model() method
        as the "file" parameter.
    """
    example_euroc_env.set_atmospheric_model(type="Windy", file=model_name)
    assert pytest.approx(100000.0, rel=0.1) == example_euroc_env.pressure(100)
    assert 0 + 273 < example_euroc_env.temperature(100) < 40 + 273
    assert abs(example_euroc_env.wind_velocity_x(100)) < 20.0
    assert abs(example_euroc_env.wind_velocity_y(100)) < 20.0
