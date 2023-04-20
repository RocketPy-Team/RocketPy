import datetime
from unittest.mock import patch

import pytest
import pytz
from rocketpy import Environment, Flight, Rocket, SolidMotor


@pytest.fixture
def example_env():
    Env = Environment(railLength=5, datum="WGS84")
    return Env


@pytest.fixture
def example_env_robust():
    Env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    Env.setDate(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time
    return Env


def test_env_set_date(example_env):
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    example_env.setDate((tomorrow.year, tomorrow.month, tomorrow.day, 12))
    assert example_env.date == datetime.datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, 12, tzinfo=pytz.utc
    )


def test_env_set_date_time_zone(example_env):
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    example_env.setDate(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12), timeZone="America/New_York"
    )
    dateNaive = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 12)
    timezone = pytz.timezone("America/New_York")
    dateAwareLocalDate = timezone.localize(dateNaive)
    dateAwareUTC = dateAwareLocalDate.astimezone(pytz.UTC)
    assert example_env.date == dateAwareUTC


def test_env_set_location(example_env):
    example_env.setLocation(-21.960641, -47.482122)
    assert example_env.lat == -21.960641 and example_env.lon == -47.482122


def test_set_elevation(example_env):
    example_env.setElevation(elevation=200)
    assert example_env.elevation == 200


def test_set_topographic_profile(example_env):
    example_env.setLocation(46.90479, 8.07575)
    example_env.setTopographicProfile(
        type="NASADEM_HGT",
        file="data/sites/switzerland/NASADEM_NC_n46e008.nc",
        dictionary="netCDF4",
    )
    assert (
        example_env.getElevationFromTopographicProfile(example_env.lat, example_env.lon)
        == 1565
    )


@patch("matplotlib.pyplot.show")
def test_standard_atmosphere(mock_show, example_env):
    example_env.setAtmosphericModel(type="StandardAtmosphere")
    assert example_env.allInfo() == None
    assert example_env.pressure(0) == 101325.0


@patch("matplotlib.pyplot.show")
def test_custom_atmosphere(mock_show, example_env):
    example_env.setAtmosphericModel(
        type="CustomAtmosphere",
        pressure=None,
        temperature=300,
        wind_u=[(0, 5), (1000, 10)],
        wind_v=[(0, -2), (500, 3), (1600, 2)],
    )
    assert example_env.allInfo() == None
    assert example_env.pressure(0) == 101325.0
    assert example_env.windVelocityX(0) == 5
    assert example_env.temperature(100) == 300


@patch("matplotlib.pyplot.show")
def test_wyoming_sounding_atmosphere(mock_show, example_env):
    URL = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"
    example_env.setAtmosphericModel(type="WyomingSounding", file=URL)
    assert example_env.allInfo() == None
    assert example_env.pressure(0) == 93600.0
    assert example_env.windVelocityX(0) == -2.9005178894925043
    assert example_env.temperature(100) == 291.75


@pytest.mark.skip(reason="legacy tests")
@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_noaa_ruc_sounding_atmosphere(mock_show, example_env):
    URL = r"https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest"
    example_env.setAtmosphericModel(type="NOAARucSounding", file=URL)
    assert example_env.allInfo() == None
    assert example_env.pressure(0) == 100000.0


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gfs_atmosphere(mock_show, example_env_robust):
    example_env_robust.setAtmosphericModel(type="Forecast", file="GFS")
    assert example_env_robust.allInfo() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_nam_atmosphere(mock_show, example_env_robust):
    example_env_robust.setAtmosphericModel(type="Forecast", file="NAM")
    assert example_env_robust.allInfo() == None


# Deactivated since it is hard to figure out and appropriate date to use RAP forecast
# @pytest.mark.slow
# @patch("matplotlib.pyplot.show")
# def test_rap_atmosphere(mock_show, example_env_robust):
#     today = datetime.date.today()
#     example_env_robust.setDate((today.year, today.month, today.day, 8)) # Hour given in UTC time
#     example_env_robust.setAtmosphericModel(type='Forecast', file='RAP')
#     assert example_env_robust.allInfo() == None


@patch("matplotlib.pyplot.show")
def test_era5_atmosphere(mock_show):
    Env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        date=(2018, 10, 15, 12),
        datum="WGS84",
    )
    Env.setAtmosphericModel(
        type="Reanalysis",
        file="data/weather/SpaceportAmerica_2018_ERA-5.nc",
        dictionary="ECMWF",
    )
    assert Env.allInfo() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_gefs_atmosphere(mock_show, example_env_robust):
    example_env_robust.setAtmosphericModel(type="Ensemble", file="GEFS")
    assert example_env_robust.allInfo() == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_cmc_atmosphere(mock_show, example_env_robust):
    example_env_robust.setAtmosphericModel(type="Ensemble", file="CMC")
    assert example_env_robust.allInfo() == None


# Deactivated since example file CuritibaRioSaoPauloEnsemble_2018_ERA-5.nc does not exist anymore
# @patch("matplotlib.pyplot.show")
# def test_era5_ensemble_atmosphere(mock_show, example_env_robust):
#     example_env_robust.setAtmosphericModel(
#         type='Reanalysis',
#         file='data/weather/CuritibaRioSaoPauloEnsemble_2018_ERA-5.nc',
#         dictionary='ECMWF'
#     )
#     assert example_env_robust.allInfo() == None


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

    example_env_robust.setDate(date_info)
    example_env_robust.setAtmosphericModel(
        type="Forecast",
        file=f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/hiresw_conusarw_12z",
        dictionary=HIRESW_dictionary,
    )
    assert example_env_robust.allInfo() == None
