import copy
import datetime
import os
from unittest.mock import patch

import ipywidgets as widgets
import matplotlib as plt
import pytest
from IPython.display import HTML

from rocketpy import EnvironmentAnalysis

plt.rcParams.update({"figure.max_open_warning": 0})

# Create a simple object of the Environment Analysis class
@pytest.fixture
def env():
    """Create a simple object of the Environment Analysis class to be used in
    the tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    EnvironmentAnalysis
        A simple object of the Environment Analysis class
    """
    env = EnvironmentAnalysis(
        start_date=datetime.datetime(2019, 10, 23),
        end_date=datetime.datetime(2021, 10, 23),
        latitude=39.3897,
        longitude=-8.28896388889,
        start_hour=6,
        end_hour=18,
        surfaceDataFile="./data/weather/EuroC_single_level_reanalysis_2002_2021.nc",
        pressureLevelDataFile="./data/weather/EuroC_pressure_levels_reanalysis_2001-2021.nc",
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        maxExpectedAltitude=None,
    )

    return env


def test_allInfo(env):
    """Test the EnvironmentAnalysis.allInfo() method, which already invokes
    several other methods. It is a good way to test the whole class in a first view.
    However, if it fails, it is hard to know which method is failing.

    Parameters
    ----------
    env : EnvironmentAnalysis
        A simple object of the Environment Analysis class

    Returns
    -------
    None
    """
    assert env.allInfo() == None


@patch("matplotlib.pyplot.show")
def test_distribution_plots(mock_show, env):
    """Tests the distribution plots method of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    plots are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    env : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.

    Returns
    -------
    None
    """

    # Check distribution plots
    assert env.plot_wind_gust_distribution() == None
    assert env.plot_surface10m_wind_speed_distribution() == None
    assert env.plot_wind_gust_distribution_over_average_day() == None
    assert env.plot_sustained_surface_wind_speed_distribution_over_average_day() == None


@patch("matplotlib.pyplot.show")
def test_average_plots(mock_show, env):
    """Tests the average plots method of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    plots are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    env : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.

    Returns
    -------
    None
    """
    # Check "average" plots
    assert env.plot_average_temperature_along_day() == None
    assert env.plot_average_surface10m_wind_speed_along_day(False) == None
    assert env.plot_average_surface10m_wind_speed_along_day(True) == None
    assert env.plot_average_sustained_surface100m_wind_speed_along_day() == None
    assert env.plot_average_day_wind_rose_all_hours() == None
    assert env.plot_average_day_wind_rose_specific_hour(12) == None
    assert env.plot_average_day_wind_rose_specific_hour(12) == None


@patch("matplotlib.pyplot.show")
def test_profile_plots(mock_show, env):
    # Check profile plots
    assert env.plot_wind_heading_profile_over_average_day() == None
    assert env.plot_average_wind_heading_profile(clear_range_limits=False) == None
    assert env.plot_average_wind_heading_profile(clear_range_limits=True) == None
    assert env.plot_average_wind_speed_profile(clear_range_limits=False) == None
    assert env.plot_average_wind_speed_profile(clear_range_limits=True) == None
    assert env.plot_average_pressure_profile(clear_range_limits=False) == None
    assert env.plot_average_pressure_profile(clear_range_limits=True) == None
    assert env.plot_wind_profile_over_average_day() == None


@patch("matplotlib.pyplot.show")
def test_animation_plots(mock_show, env):

    # Check animation plots
    assert isinstance(env.animate_average_wind_rose(), widgets.Image)
    assert isinstance(env.animate_wind_gust_distribution_over_average_day(), HTML)
    assert isinstance(env.animate_wind_heading_profile_over_average_day(), HTML)
    assert isinstance(env.animate_wind_profile_over_average_day(), HTML)
    assert isinstance(
        env.animate_sustained_surface_wind_speed_distribution_over_average_day(),
        HTML,
    )


def test_exports(env):

    assert env.exportMeanProfiles() == None
    assert env.save("EnvAnalysisDict") == None

    env2 = copy.deepcopy(env)
    env2.load("EnvAnalysisDict")
    assert env2.allInfo() == None

    # Delete file created by save method
    os.remove("EnvAnalysisDict")
