from datetime import datetime, timedelta

import pytest

from rocketpy import Environment, EnvironmentAnalysis


@pytest.fixture
def example_plain_env():
    """Simple object of the Environment class to be used in the tests.

    Returns
    -------
    rocketpy.Environment
    """
    return Environment()


@pytest.fixture
def example_date_naive():
    """Naive tomorrow date

    Returns
    -------
    datetime.datetime
    """
    return datetime.now() + timedelta(days=1)


@pytest.fixture
def example_spaceport_env(example_date_naive):
    """Environment class with location set to Spaceport America Cup launch site

    Returns
    -------
    rocketpy.Environment
    """
    spaceport_env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    spaceport_env.set_date(example_date_naive)
    return spaceport_env


@pytest.fixture
def example_euroc_env(example_date_naive):
    """Environment class with location set to EuRoC launch site

    Returns
    -------
    rocketpy.Environment
    """
    euroc_env = Environment(
        latitude=39.3897,
        longitude=-8.28896388889,
        elevation=100,
        datum="WGS84",
    )
    euroc_env.set_date(example_date_naive)
    return euroc_env


@pytest.fixture
def env_analysis():
    """Environment Analysis class with hardcoded parameters

    Returns
    -------
    EnvironmentAnalysis
    """
    env_analysis = EnvironmentAnalysis(
        start_date=datetime(2019, 10, 23),
        end_date=datetime(2021, 10, 23),
        latitude=39.3897,
        longitude=-8.28896388889,
        start_hour=6,
        end_hour=18,
        surface_data_file="./data/weather/EuroC_single_level_reanalysis_2002_2021.nc",
        pressure_level_data_file="./data/weather/EuroC_pressure_levels_reanalysis_2001-2021.nc",
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        max_expected_altitude=None,
    )

    return env_analysis


@pytest.fixture
def environment_spaceport_america_2023():
    """Creates an Environment object for Spaceport America with a 2023 launch
    conditions.

    Returns
    -------
    rocketpy.Environment
        Environment object configured for Spaceport America in 2023.
    """
    env = Environment(
        latitude=32.939377,
        longitude=-106.911986,
        elevation=1401,
    )
    env.set_date(date=(2023, 6, 24, 9), timezone="America/Denver")

    env.set_atmospheric_model(
        type="Reanalysis",
        file="data/weather/spaceport_america_pressure_levels_2023_hourly.nc",
        dictionary="ECMWF",
    )

    env.max_expected_height = 6000
    return env


@pytest.fixture
def example_kennedy_env(example_date_naive):
    """Environment class with location set to Kennedy Space Center launch site.

    Kennedy Space Center coordinates:
    - Latitude: 28.5721° N
    - Longitude: -80.6480° W
    - Elevation: ~3 meters

    Returns
    -------
    rocketpy.Environment
        Environment object configured for Kennedy Space Center.
    """
    kennedy_env = Environment(
        latitude=28.5721,
        longitude=-80.6480,
        elevation=3.0,
        datum="WGS84",
    )
    # Set date to tomorrow at noon UTC
    tomorrow = example_date_naive
    kennedy_env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12), timezone="UTC"
    )
    return kennedy_env
