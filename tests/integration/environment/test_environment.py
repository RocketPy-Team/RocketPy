import time
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "lat, lon, theoretical_elevation",
    [
        (48.858844, 2.294351, 34),  # The Eiffel Tower
        (32.990254, -106.974998, 1401),  # Spaceport America
    ],
)
def test_set_elevation_open_elevation(
    lat, lon, theoretical_elevation, example_plain_env
):
    example_plain_env.set_location(lat, lon)

    # either successfully gets the elevation or raises RuntimeError
    try:
        example_plain_env.set_elevation(elevation="Open-Elevation")
        assert example_plain_env.elevation == pytest.approx(
            theoretical_elevation, abs=1
        ), "The Open-Elevation API returned an unexpected value for the elevation"
    except RuntimeError:
        pass  # Ignore the error and pass the test


@patch("matplotlib.pyplot.show")
def test_era5_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
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


@patch("matplotlib.pyplot.show")
def test_custom_atmosphere(mock_show, example_plain_env):  # pylint: disable=unused-argument
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
def test_standard_atmosphere(mock_show, example_plain_env):  # pylint: disable=unused-argument
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


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gfs_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
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


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_nam_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
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


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_rap_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
    today = date.today()
    now = datetime.now(timezone.utc)
    example_spaceport_env.set_date((today.year, today.month, today.day, now.hour))
    example_spaceport_env.set_atmospheric_model(type="Forecast", file="RAP")
    assert example_spaceport_env.all_info() is None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gefs_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
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


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_wyoming_sounding_atmosphere(mock_show, example_plain_env):  # pylint: disable=unused-argument
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
    url = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"
    # give it at least 5 times to try to download the file
    for i in range(5):
        try:
            example_plain_env.set_atmospheric_model(type="wyoming_sounding", file=url)
            break
        except Exception:  # pylint: disable=broad-except
            time.sleep(2**i)
    assert example_plain_env.all_info() is None
    assert abs(example_plain_env.pressure(0) - 93600.0) < 1e-8
    assert (
        abs(example_plain_env.barometric_height(example_plain_env.pressure(0)) - 722.0)
        < 1e-8
    )
    assert abs(example_plain_env.wind_velocity_x(0) - -2.9005178894925043) < 1e-8
    assert abs(example_plain_env.temperature(100) - 291.75) < 1e-8


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_hiresw_ensemble_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
    """Tests the Forecast model with the HIRESW file.

    Parameters
    ----------
    mock_show : mock
        Mock object to replace matplotlib.pyplot.show() method.
    example_spaceport_env : rocketpy.Environment
        Example environment object to be tested.
    """
    today = date.today()
    date_info = (today.year, today.month, today.day, 12)  # Hour given in UTC time

    example_spaceport_env.set_date(date_info)

    example_spaceport_env.set_atmospheric_model(
        type="Forecast",
        file="HIRESW",
        dictionary="HIRESW",
    )

    assert example_spaceport_env.all_info() is None


@pytest.mark.skip(reason="CMC model is currently not working")
@patch("matplotlib.pyplot.show")
def test_cmc_atmosphere(mock_show, example_spaceport_env):  # pylint: disable=unused-argument
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
