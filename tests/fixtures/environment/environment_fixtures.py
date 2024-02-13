import datetime

import pytest

from rocketpy import Environment, EnvironmentAnalysis


@pytest.fixture
def example_env():
    """Create a simple object of the Environment class to be used in the tests.
    This allows to avoid repeating the same code in all tests. The environment
    set here is the simplest possible, with no parameters set.

    Returns
    -------
    rocketpy.Environment
        The simplest object of the Environment class
    """
    return Environment()


@pytest.fixture
def example_env_robust():
    """Create an object of the Environment class to be used in the tests. This
    allows to avoid repeating the same code in all tests. The environment set
    here is a bit more complex than the one in the example_env fixture. This
    time the latitude, longitude and elevation are set, as well as the datum and
    the date. The location refers to the Spaceport America Cup launch site,
    while the date is set to tomorrow at noon.

    Returns
    -------
    rocketpy.Environment
        An object of the Environment class
    """
    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
    return env


@pytest.fixture
def env_analysis():
    """Create a simple object of the Environment Analysis class to be used in
    the tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    EnvironmentAnalysis
        A simple object of the Environment Analysis class
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