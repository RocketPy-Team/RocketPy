import datetime
import os
from unittest.mock import patch

import numpy as np
import pytest
import pytz

from rocketpy import Environment


def test_env_set_date(example_env):
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    example_env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
    assert example_env.datetime_date == datetime.datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, 12, tzinfo=pytz.utc
    )


def test_env_set_date_time_zone(example_env):
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    example_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12), timezone="America/New_York"
    )
    date_naive = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 12)
    timezone = pytz.timezone("America/New_York")
    date_aware_local_date = timezone.localize(date_naive)
    date_aware_utc = date_aware_local_date.astimezone(pytz.UTC)
    assert example_env.datetime_date == date_aware_utc


def test_env_set_location(example_env):
    example_env.set_location(-21.960641, -47.482122)
    assert example_env.latitude == -21.960641 and example_env.longitude == -47.482122


def test_set_elevation(example_env):
    example_env.set_elevation(elevation=200)
    assert example_env.elevation == 200


def test_set_topographic_profile(example_env):
    example_env.set_location(46.90479, 8.07575)
    example_env.set_topographic_profile(
        type="NASADEM_HGT",
        file="data/sites/switzerland/NASADEM_NC_n46e008.nc",
        dictionary="netCDF4",
    )
    assert (
        example_env.get_elevation_from_topographic_profile(
            example_env.latitude, example_env.longitude
        )
        == 1565
    )


@patch("matplotlib.pyplot.show")
def test_standard_atmosphere(mock_show, example_env):
    example_env.set_atmospheric_model(type="standard_atmosphere")
    assert example_env.info() == None
    assert example_env.all_info() == None
    assert example_env.pressure(0) == 101325.0
    assert example_env.prints.print_earth_details() == None


@patch("matplotlib.pyplot.show")
def test_custom_atmosphere(mock_show, example_env):
    example_env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=None,
        temperature=300,
        wind_u=[(0, 5), (1000, 10)],
        wind_v=[(0, -2), (500, 3), (1600, 2)],
    )
    assert example_env.all_info() == None
    assert example_env.pressure(0) == 101325.0
    assert example_env.wind_velocity_x(0) == 5
    assert example_env.temperature(100) == 300


@patch("matplotlib.pyplot.show")
def test_wyoming_sounding_atmosphere(mock_show, example_env):
    URL = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"
    example_env.set_atmospheric_model(type="wyoming_sounding", file=URL)
    assert example_env.all_info() == None
    assert example_env.pressure(0) == 93600.0
    assert example_env.wind_velocity_x(0) == -2.9005178894925043
    assert example_env.temperature(100) == 291.75


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
    example_env_robust.set_atmospheric_model(type="Forecast", file="GFS")
    assert example_env_robust.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_nam_atmosphere(mock_show, example_env_robust):
    example_env_robust.set_atmospheric_model(type="Forecast", file="NAM")
    assert example_env_robust.all_info() == None


# Deactivated since it is hard to figure out and appropriate date to use RAP forecast
# @pytest.mark.slow
# @patch("matplotlib.pyplot.show")
# def test_rap_atmosphere(mock_show, example_env_robust):
#     today = datetime.date.today()
#     example_env_robust.set_date((today.year, today.month, today.day, 8)) # Hour given in UTC time
#     example_env_robust.set_atmospheric_model(type='Forecast', file='RAP')
#     assert example_env_robust.all_info() == None


@patch("matplotlib.pyplot.show")
def test_era5_atmosphere(mock_show):
    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        date=(2018, 10, 15, 12),
        datum="WGS84",
    )
    env.set_atmospheric_model(
        type="Reanalysis",
        file="data/weather/SpaceportAmerica_2018_ERA-5.nc",
        dictionary="ECMWF",
    )
    assert env.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gefs_atmosphere(mock_show, example_env_robust):
    example_env_robust.set_atmospheric_model(type="Ensemble", file="GEFS")
    assert example_env_robust.all_info() == None


@patch("matplotlib.pyplot.show")
def test_info_returns(mock_show, example_env):
    returned_plots = example_env.all_plot_info_returned()
    returned_infos = example_env.all_info_returned()
    expected_info = {
        "grav": example_env.gravity,
        "elevation": 0,
        "modelType": "standard_atmosphere",
        "modelTypeMaxExpectedHeight": 80000,
        "windSpeed": 0,
        "windDirection": 0,
        "windHeading": 0,
        "surfacePressure": 1013.25,
        "surfaceTemperature": 288.15,
        "surfaceAirDensity": 1.225000018124288,
        "surfaceSpeedOfSound": 340.293988026089,
        "lat": 0,
        "lon": 0,
    }
    expected_plots_keys = [
        "grid",
        "windSpeed",
        "windDirection",
        "speed_of_sound",
        "density",
        "windVelX",
        "windVelY",
        "pressure",
        "temperature",
    ]
    assert list(returned_infos.keys()) == list(expected_info.keys())
    assert list(returned_infos.values()) == list(expected_info.values())
    assert list(returned_plots.keys()) == expected_plots_keys


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_cmc_atmosphere(mock_show, example_env_robust):
    example_env_robust.set_atmospheric_model(type="Ensemble", file="CMC")
    assert example_env_robust.all_info() == None


# Deactivated since example file CuritibaRioSaoPauloEnsemble_2018_ERA-5.nc does not exist anymore
# @patch("matplotlib.pyplot.show")
# def test_era5_ensemble_atmosphere(mock_show, example_env_robust):
#     example_env_robust.set_atmospheric_model(
#         type='Reanalysis',
#         file='data/weather/CuritibaRioSaoPauloEnsemble_2018_ERA-5.nc',
#         dictionary='ECMWF'
#     )
#     assert example_env_robust.all_info() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_hiresw_ensemble_atmosphere(mock_show, example_env_robust):
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


def test_export_environment(example_env_robust):
    assert example_env_robust.export_environment(filename="environment") == None
    os.remove("environment.json")


def test_utm_to_geodesic(example_env_robust):
    lat, lon = example_env_robust.utm_to_geodesic(
        x=315468.64, y=3651938.65, utm_zone=13, hemis="N"
    )
    assert np.isclose(lat, 32.99025, atol=1e-5) == True
    assert np.isclose(lon, -106.9750, atol=1e-5) == True
