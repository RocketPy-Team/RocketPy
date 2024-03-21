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
    spaceport_env.height = 1425
    return spaceport_env


@pytest.fixture
def env_analysis():
    """Environment Analysis class with hardcoded parameters

    Returns
    -------
    EnvironmentAnalysis
    """
    env_analysis = EnvironmentAnalysis(
        start_date=datetime.datetime(2019, 10, 23),
        end_date=datetime.datetime(2021, 10, 23),
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
