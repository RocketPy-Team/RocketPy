import os
from unittest.mock import patch

import matplotlib as plt
import pytest

from rocketpy.tools import import_optional_dependency

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_distribution_plots(mock_show, env_analysis):  # pylint: disable=unused-argument
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
    assert env_analysis.plots.wind_gust_distribution() is None
    assert (
        env_analysis.plots.surface10m_wind_speed_distribution(wind_speed_limit=True)
        is None
    )
    assert env_analysis.plots.wind_gust_distribution_grid() is None
    assert (
        env_analysis.plots.surface_wind_speed_distribution_grid(wind_speed_limit=True)
        is None
    )


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_average_plots(mock_show, env_analysis):  # pylint: disable=unused-argument
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
    assert env_analysis.plots.average_surface_temperature_evolution() is None
    assert env_analysis.plots.average_surface10m_wind_speed_evolution(False) is None
    assert env_analysis.plots.average_surface10m_wind_speed_evolution(True) is None
    assert env_analysis.plots.average_surface100m_wind_speed_evolution() is None
    assert env_analysis.plots.average_wind_rose_grid() is None
    assert env_analysis.plots.average_wind_rose_specific_hour(12) is None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_profile_plots(mock_show, env_analysis):  # pylint: disable=unused-argument
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
    assert env_analysis.plots.wind_heading_profile_grid(clear_range_limits=True) is None
    assert (
        env_analysis.plots.average_wind_heading_profile(clear_range_limits=False)
        is None
    )
    assert (
        env_analysis.plots.average_wind_heading_profile(clear_range_limits=True) is None
    )
    assert (
        env_analysis.plots.average_wind_speed_profile(clear_range_limits=False) is None
    )
    assert (
        env_analysis.plots.average_wind_speed_profile(clear_range_limits=True) is None
    )
    assert env_analysis.plots.average_pressure_profile(clear_range_limits=False) is None
    assert env_analysis.plots.average_pressure_profile(clear_range_limits=True) is None
    assert env_analysis.plots.wind_speed_profile_grid(clear_range_limits=True) is None
    assert (
        env_analysis.plots.average_wind_velocity_xy_profile(clear_range_limits=True)
        is None
    )
    assert (
        env_analysis.plots.average_temperature_profile(clear_range_limits=True) is None
    )


@pytest.mark.slow
def test_values(env_analysis):
    """Check the numeric properties of the EnvironmentAnalysis class. It computes
    a few values and compares them to the expected values. Not all the values are
    tested since the most of them were already invoke in the previous tests.

    Parameters
    ----------
    env_analysis : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.
    """
    assert pytest.approx(0.07569172, 1e-2) == env_analysis.record_min_surface_wind_speed
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


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_animation_plots(mock_show, env_analysis):  # pylint: disable=unused-argument
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
def test_pressure_level_wind_profile_uses_velocity_components(env_analysis):
    """Regression for PR #1041: the redundant per-level ``wind_heading`` and
    ``wind_direction`` functions were removed from the pressure-level data.

    The surviving wind profile is expressed through ``wind_velocity_x`` /
    ``wind_velocity_y`` (and ``wind_speed``), from which heading/direction are
    derivable. This guards against re-introducing the removed entries and
    checks that the surviving API is internally consistent.
    """
    data = env_analysis.original_pressure_level_data
    first_date = next(iter(data))
    first_hour = next(iter(data[first_date]))
    hour_data = data[first_date][first_hour]

    # Removed, redundant entries must not come back.
    assert "wind_heading" not in hour_data
    assert "wind_direction" not in hour_data

    # Surviving, non-redundant wind-profile entries.
    assert "wind_velocity_x" in hour_data
    assert "wind_velocity_y" in hour_data
    assert "wind_speed" in hour_data

    # wind_speed must remain derivable from the velocity components. Sample at
    # an actual grid node so interpolation and the sqrt() commute exactly.
    grid_heights = hour_data["wind_speed"].x_array
    height = float(grid_heights[len(grid_heights) // 2])
    vx = hour_data["wind_velocity_x"](height)
    vy = hour_data["wind_velocity_y"](height)
    speed = hour_data["wind_speed"](height)
    assert speed == pytest.approx((vx**2 + vy**2) ** 0.5, rel=1e-6)
