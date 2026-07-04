import builtins
import os
import sys
import types
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation

from rocketpy.plots.compare import Compare
from rocketpy.plots.plot_helpers import (
    show_or_save_animation,
    show_or_save_fig,
    show_or_save_plot,
)


@patch("matplotlib.pyplot.show")
def test_compare(mock_show, flight_calisto):  # pylint: disable=unused-argument
    """Here we want to test the 'x_attributes' argument, which is the only one
    that is not tested in the other tests.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    flight_calisto : rocketpy.Flight
        Flight object to be used in the tests. See conftest.py for more details.
    """
    flight = flight_calisto

    objects = [flight, flight, flight]

    comparison = Compare(object_list=objects)

    fig, _ = comparison.create_comparison_figure(
        y_attributes=["z"],
        n_rows=1,
        n_cols=1,
        figsize=(10, 10),
        legend=False,
        title="Test",
        x_labels=["Time (s)"],
        y_labels=["Altitude (m)"],
        x_lim=(0, 3),
        y_lim=(0, 1000),
        x_attributes=["time"],
    )

    assert isinstance(fig, plt.Figure)


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("filename", [None, "test.png"])
def test_show_or_save_plot(mock_show, filename):
    """This test is to check if the show_or_save_plot function is
    working properly.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing
        the plots.
    filename : str
        Name of the file to save the plot. If None, the plot will be
        shown instead.
    """
    plt.subplots()
    show_or_save_plot(filename)

    if filename is None:
        mock_show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


@pytest.mark.parametrize("filename", [None, "test.png"])
def test_show_or_save_fig(filename):
    """This test is to check if the show_or_save_fig function is
    working properly.

    Parameters
    ----------
    filename : str
        Name of the file to save the plot. If None, the plot will be
        shown instead.
    """
    fig, _ = plt.subplots()

    fig.show = MagicMock()
    show_or_save_fig(fig, filename)

    if filename is None:
        fig.show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


@pytest.mark.parametrize("filename", [None, "test.gif"])
@patch("matplotlib.pyplot.show")
def test_show_or_save_animation(mock_show, filename):
    """This test is to check if the show_or_save_animation function is
    working properly.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the animation.
    filename : str
        Name of the file to save the animation. If None, the animation will be
        shown instead.
    """

    # Create a simple animation object
    fig, ax = plt.subplots()

    def update(frame):
        ax.plot([0, frame], [0, frame])
        return ax

    animation = FuncAnimation(fig, update, frames=5)

    show_or_save_animation(animation, filename)

    if filename is None:
        mock_show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


def test_show_or_save_animation_unsupported_format():
    # Test that show_or_save_animation raises ValueError for unsupported formats.
    fig, ax = plt.subplots()

    def update(frame):
        ax.plot([0, frame], [0, frame])
        return ax

    animation = FuncAnimation(fig, update, frames=5)

    with pytest.raises(ValueError, match="Unsupported file ending"):
        show_or_save_animation(animation, "test.mp4")


def test_animate_propellant_mass(cesaroni_m1670):
    """Test that animate_propellant_mass saves a .gif file correctly."""

    motor = cesaroni_m1670
    animation = motor.plots.animate_propellant_mass(filename="cesaroni_m1670.gif")

    # Check animation type
    assert isinstance(animation, FuncAnimation)

    # check if file exists
    assert os.path.exists("cesaroni_m1670.gif")

    os.remove("cesaroni_m1670.gif")


def test_animate_fluid_volume(example_mass_flow_rate_based_tank_seblm):
    """Test that animate_fluid_volume saves a .gif file correctly."""

    tank = example_mass_flow_rate_based_tank_seblm
    animation = tank.plots.animate_fluid_volume(filename="test_fluid_volume.gif")

    # Check animation type
    assert isinstance(animation, FuncAnimation)

    # Check if file exists
    assert os.path.exists("test_fluid_volume.gif")

    os.remove("test_fluid_volume.gif")


class _DummyVedoActor:
    """Minimal actor mock that supports the methods used by animation plots."""

    def __init__(self):
        self.rotations = []

    def c(self, *_args, **_kwargs):
        return self

    def pos(self, *_args, **_kwargs):
        return self

    def wireframe(self):
        return self

    def rotate(self, angle, axis=None):
        self.rotations.append((angle, axis))
        return self

    def clone(self):
        return _DummyVedoActor()


class _DummyPlotter:
    """Minimal plotter mock for non-interactive animation tests."""

    def __init__(self, *_args, **_kwargs):
        self.escaped = False

    def show(self, *_args, **_kwargs):
        return self

    def render(self):
        return None

    def interactive(self):
        return self

    def close(self):
        return None


def _mock_vedo_module(monkeypatch):
    """Install a minimal vedo module in sys.modules for tests."""

    vedo_module = types.ModuleType("vedo")
    vedo_module.Mesh = lambda *_args, **_kwargs: _DummyVedoActor()
    vedo_module.Box = lambda *_args, **_kwargs: _DummyVedoActor()
    vedo_module.Line = lambda *_args, **_kwargs: _DummyVedoActor()
    vedo_module.Plotter = _DummyPlotter
    vedo_module.settings = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vedo", vedo_module)


def test_animate_trajectory_runs_with_mocked_vedo(flight_calisto, monkeypatch):
    """Test flight trajectory animation entry point through the plots layer."""

    # Arrange
    _mock_vedo_module(monkeypatch)

    # Act
    result = flight_calisto.plots.animate_trajectory(
        start=0.0,
        stop=0.001,
        time_step=0.001,
    )

    # Assert
    assert result is None


def test_animate_rotate_runs_with_mocked_vedo(flight_calisto, monkeypatch):
    """Test flight rotation animation entry point through the plots layer."""

    # Arrange
    _mock_vedo_module(monkeypatch)

    # Act
    result = flight_calisto.plots.animate_rotate(
        start=0.0,
        stop=0.001,
        time_step=0.001,
    )

    # Assert
    assert result is None


def test_animate_trajectory_raises_when_vedo_is_missing(flight_calisto, monkeypatch):
    """Test that an informative ImportError is raised when vedo is unavailable."""

    # Arrange
    real_import = builtins.__import__

    def import_without_vedo(name, *args, **kwargs):
        if name == "vedo" or name.startswith("vedo."):
            raise ImportError("No module named 'vedo'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_vedo)

    # Act / Assert
    with pytest.raises(ImportError, match="optional dependency"):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.001,
            time_step=0.001,
        )


def test_animate_rotate_raises_when_time_range_is_invalid(flight_calisto, monkeypatch):
    """Test validation error for invalid animation time range."""

    # Arrange
    _mock_vedo_module(monkeypatch)
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid animation time range"):
        flight_calisto.plots.animate_rotate(
            start=1.0,
            stop=0.5,
            time_step=0.1,
        )


def test_animate_trajectory_raises_when_stl_file_is_missing(
    flight_calisto, monkeypatch
):
    """Test file validation when STL path does not exist."""

    # Arrange
    _mock_vedo_module(monkeypatch)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Could not find the 3D model file"):
        flight_calisto.plots.animate_trajectory(
            "missing_model.stl",
            start=0.0,
            stop=0.1,
            time_step=0.1,
        )


@pytest.mark.parametrize("invalid_time_step", [0, -0.1])
def test_animate_trajectory_raises_when_time_step_is_non_positive(
    flight_calisto, monkeypatch, invalid_time_step
):
    """Test validation error when animation time_step is not strictly positive."""

    # Arrange
    _mock_vedo_module(monkeypatch)
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid time_step"):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.1,
            time_step=invalid_time_step,
        )


def test_animate_rotate_raises_when_stop_exceeds_flight_end(
    flight_calisto, monkeypatch
):
    """Test validation error when stop time exceeds available simulation range."""

    # Arrange
    _mock_vedo_module(monkeypatch)
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid animation time range"):
        flight_calisto.plots.animate_rotate(
            start=0.0,
            stop=flight_calisto.t_final + 0.1,
            time_step=0.1,
        )


def test_animate_trajectory_raises_when_default_model_is_missing(
    flight_calisto, monkeypatch
):
    """Test failure path when default packaged STL model is unavailable."""

    # Arrange
    _mock_vedo_module(monkeypatch)
    monkeypatch.setattr(
        flight_calisto.plots,
        "_resolve_animation_model_path",
        lambda _file_name: "missing_default_model.stl",
    )

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Could not find the 3D model file"):
        flight_calisto.plots.animate_trajectory(
            start=0.0,
            stop=0.1,
            time_step=0.1,
        )
