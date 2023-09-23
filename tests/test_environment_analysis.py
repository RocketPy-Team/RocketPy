import copy
import os
from unittest.mock import patch

import matplotlib as plt
import pytest

from rocketpy.tools import import_optional_dependency

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_all_info(mock_show, env_analysis):
    """Test the EnvironmentAnalysis.all_info() method, which already invokes
    several other methods. It is a good way to test the whole class in a first view.
    However, if it fails, it is hard to know which method is failing.

    Parameters
    ----------
    env_analysis : rocketpy.EnvironmentAnalysis
        A simple object of the Environment Analysis class

    Returns
    -------
    None
    """
    assert env_analysis.info() == None
    assert env_analysis.all_info() == None
    assert env_analysis.plots.info() == None
    os.remove("wind_rose.gif")  # remove the files created by the method


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_distribution_plots(mock_show, env_analysis):
    """Tests the distribution plots method of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    plots are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    env_analysis : rocketpy.EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.

    Returns
    -------
    None
    """

    # Check distribution plots
    assert env_analysis.plots.wind_gust_distribution() == None
    assert (
        env_analysis.plots.surface10m_wind_speed_distribution(wind_speed_limit=True)
        == None
    )
    assert env_analysis.plots.wind_gust_distribution_grid() == None
    assert (
        env_analysis.plots.surface_wind_speed_distribution_grid(wind_speed_limit=True)
        == None
    )


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_average_plots(mock_show, env_analysis):
    """Tests the average plots method of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    plots are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    env_analysis : rocketpy.EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.

    Returns
    -------
    None
    """
    # Check "average" plots
    assert env_analysis.plots.average_surface_temperature_evolution() == None
    assert env_analysis.plots.average_surface10m_wind_speed_evolution(False) == None
    assert env_analysis.plots.average_surface10m_wind_speed_evolution(True) == None
    assert env_analysis.plots.average_surface100m_wind_speed_evolution() == None
    assert env_analysis.plots.average_wind_rose_grid() == None
    assert env_analysis.plots.average_wind_rose_specific_hour(12) == None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_profile_plots(mock_show, env_analysis):
    """Check the profile plots method of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    plots are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    mock_show : Mock
        Mock of the matplotlib.pyplot.show() method
    env_analysis : rocketpy.EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.
    """
    # Check profile plots
    assert env_analysis.plots.wind_heading_profile_grid(clear_range_limits=True) == None
    assert (
        env_analysis.plots.average_wind_heading_profile(clear_range_limits=False)
        == None
    )
    assert (
        env_analysis.plots.average_wind_heading_profile(clear_range_limits=True) == None
    )
    assert (
        env_analysis.plots.average_wind_speed_profile(clear_range_limits=False) == None
    )
    assert (
        env_analysis.plots.average_wind_speed_profile(clear_range_limits=True) == None
    )
    assert env_analysis.plots.average_pressure_profile(clear_range_limits=False) == None
    assert env_analysis.plots.average_pressure_profile(clear_range_limits=True) == None
    assert env_analysis.plots.wind_speed_profile_grid(clear_range_limits=True) == None
    assert (
        env_analysis.plots.average_wind_velocity_xy_profile(clear_range_limits=True)
        == None
    )
    assert (
        env_analysis.plots.average_temperature_profile(clear_range_limits=True) == None
    )


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_animation_plots(mock_show, env_analysis):
    """Check the animation plots method of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    plots are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    mock_show : Mock
        Mock of the matplotlib.pyplot.show() method
    env_analysis : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.
    """
    # import dependencies
    widgets = import_optional_dependency("ipywidgets")
    HTML = import_optional_dependency("IPython.display").HTML

    # Check animation plots
    assert isinstance(env_analysis.plots.animate_average_wind_rose(), widgets.Image)
    assert isinstance(env_analysis.plots.animate_wind_gust_distribution(), HTML)
    assert isinstance(env_analysis.plots.animate_wind_heading_profile(), HTML)
    assert isinstance(env_analysis.plots.animate_wind_speed_profile(), HTML)
    assert isinstance(
        env_analysis.plots.animate_surface_wind_speed_distribution(),
        HTML,
    )
    os.remove("wind_rose.gif")  # remove the files created by the method


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_exports(mock_show, env_analysis):
    """Check the export methods of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    files are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    env_analysis : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.
    """

    assert env_analysis.export_mean_profiles() == None
    assert env_analysis.save("env_analysis_dict") == None

    env2 = copy.deepcopy(env_analysis)
    env2.load("env_analysis_dict")
    assert env2.all_info() == None

    # Delete file created by save method
    os.remove("env_analysis_dict")
    os.remove("wind_rose.gif")
    os.remove("export_env_analysis.json")


@pytest.mark.slow
def test_values(env_analysis):
    """Check the numeric properties of the EnvironmentAnalysis class. It computes
    a few values and compares them to the expected values. Not all the values are
    tested since the most of them were already invoke in the previous tests.

    Parameters
    ----------
    env_analysis : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.

    Returns
    -------
    None
    """
    assert pytest.approx(env_analysis.record_min_surface_wind_speed, 1e-6) == 5.190407
    assert (
        pytest.approx(env_analysis.max_average_temperature_at_altitude, 1e-6)
        == 24.52549
    )
    assert (
        pytest.approx(env_analysis.min_average_temperature_at_altitude, 1e-6)
        == -63.18178
    )
    assert pytest.approx(env_analysis.std_pressure_at_10000ft, 1e-6) == 13.58699
    assert pytest.approx(env_analysis.std_pressure_at_30000ft, 1e-6) == 38.48947
