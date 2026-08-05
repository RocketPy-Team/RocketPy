import os
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb

from rocketpy.plots.compare import Compare
from rocketpy.plots.flight_plots import _FlightPlots
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


def test_animate_propellant_mass(cesaroni_m1670, monkeypatch):
    """Test that animate_propellant_mass saves a .gif file correctly."""

    def mock_show_or_save(animation, filename=None, fps=30):  # pylint: disable=unused-argument
        if filename:
            with open(filename, "a"):
                pass

    monkeypatch.setattr(
        "rocketpy.plots.motor_plots.show_or_save_animation", mock_show_or_save
    )

    motor = cesaroni_m1670
    animation = motor.plots.animate_propellant_mass(filename="cesaroni_m1670.gif")

    # Check animation type
    assert isinstance(animation, FuncAnimation)

    # check if file exists
    assert os.path.exists("cesaroni_m1670.gif")

    os.remove("cesaroni_m1670.gif")


def test_animate_fluid_volume(example_mass_flow_rate_based_tank_seblm, monkeypatch):
    """Test that animate_fluid_volume saves a .gif file correctly."""

    def mock_show_or_save(animation, filename=None, fps=30):  # pylint: disable=unused-argument
        if filename:
            with open(filename, "a"):
                pass

    monkeypatch.setattr(
        "rocketpy.plots.tank_plots.show_or_save_animation", mock_show_or_save
    )

    tank = example_mass_flow_rate_based_tank_seblm
    animation = tank.plots.animate_fluid_volume(filename="test_fluid_volume.gif")

    # Check animation type
    assert isinstance(animation, FuncAnimation)

    # Check if file exists
    assert os.path.exists("test_fluid_volume.gif")

    os.remove("test_fluid_volume.gif")


# ---------------------------------------------------------------------------
# PyVista flight animation helpers
#
# The interactive animation drivers (``animate_trajectory``/``animate_rotate``)
# are exercised off screen by the integration tests. The unit tests below cover
# the pure, render-independent helpers that back them so their logic is checked
# without a live VTK render loop.
# ---------------------------------------------------------------------------


def test_rotation_matrix_from_quaternion():
    """Identity quaternion yields identity; a zero quaternion is handled."""
    identity = _FlightPlots._rotation_matrix_from_quaternion(1, 0, 0, 0)
    assert np.allclose(identity, np.eye(4))

    # 180 deg about z: x -> -x, y -> -y
    half_turn = _FlightPlots._rotation_matrix_from_quaternion(0, 0, 0, 1)
    assert np.allclose(half_turn[:3, :3] @ np.array([1, 0, 0]), [-1, 0, 0])

    # Zero-norm quaternion falls back to the identity transform.
    assert np.allclose(
        _FlightPlots._rotation_matrix_from_quaternion(0, 0, 0, 0), np.eye(4)
    )


def test_safe_unit_vector():
    """Vectors are normalized; degenerate input returns the fallback."""
    assert np.allclose(_FlightPlots._safe_unit_vector([0, 0, 5]), [0, 0, 1])
    assert np.allclose(
        _FlightPlots._safe_unit_vector([0, 0, 0], fallback=(1, 0, 0)), [1, 0, 0]
    )


def test_animation_color_scheme_and_resolved_colors():
    """The color scheme is complete and user overrides are validated."""
    colors = _FlightPlots._animation_color_scheme()
    assert "day_top" in colors and "rocket" in colors

    resolved = _FlightPlots._resolved_animation_colors({"rocket": "#000000"})
    assert resolved["rocket"] == "#000000"
    # Passing None returns the untouched default scheme.
    assert _FlightPlots._resolved_animation_colors(None)["rocket"] == colors["rocket"]

    with pytest.raises(ValueError):
        _FlightPlots._resolved_animation_colors({"not_a_key": "#fff"})


def test_animation_scalar_metadata():
    """Every supported scalar has a display label and unit."""
    for key in ["speed", "mach", "dynamic_pressure", "acceleration", "altitude"]:
        label, unit = _FlightPlots._animation_scalar_metadata(key)
        assert isinstance(label, str) and isinstance(unit, str)


def test_animation_background_at_altitude():
    """The palette blends from launch colors to the near-space colors."""
    palette = {
        "launch_bottom": np.array([1.0, 0.0, 0.0]),
        "launch_top": np.array([0.0, 1.0, 0.0]),
        "space_bottom": np.array([0.0, 0.0, 1.0]),
        "space_top": np.array([1.0, 1.0, 0.0]),
    }
    bottom, top = _FlightPlots._animation_background_at_altitude(palette, 0)
    assert np.allclose(bottom, [1.0, 0.0, 0.0])
    assert np.allclose(top, [0.0, 1.0, 0.0])
    # Well above 50 km AGL the palette is fully near-space.
    bottom_high, _ = _FlightPlots._animation_background_at_altitude(palette, 100_000)
    assert np.allclose(bottom_high, [0.0, 0.0, 1.0])


def test_interpolated_camera_path():
    """Camera positions are linearly interpolated along the path fraction."""
    path = [
        [(0, 0, 0), (0, 0, 0), (0, 0, 1)],
        [(2, 0, 0), (0, 0, 0), (0, 0, 1)],
    ]
    midpoint = _FlightPlots._interpolated_camera_path(path, 0.5)
    assert np.allclose(midpoint[0], (1, 0, 0))

    with pytest.raises(ValueError):
        _FlightPlots._interpolated_camera_path([[(0, 0, 0)]], 0.5)


def test_update_animation_camera_modes():
    """Each camera preset and custom path updates the plotter camera."""

    class _FakePlotter:
        camera_position = None

    rotation = np.eye(4)
    position = np.array([1.0, 2.0, 3.0])

    # "static" leaves the camera untouched.
    static_plotter = _FakePlotter()
    _FlightPlots._update_animation_camera(
        static_plotter, "static", position, rotation, 10.0, 5.0, 0.0, 10.0, None
    )
    assert static_plotter.camera_position is None

    for mode in ["follow", "ground", "body"]:
        plotter = _FakePlotter()
        _FlightPlots._update_animation_camera(
            plotter, mode, position, rotation, 10.0, 5.0, 0.0, 10.0, None
        )
        assert plotter.camera_position is not None

    # Callable camera path.
    callable_plotter = _FakePlotter()
    _FlightPlots._update_animation_camera(
        callable_plotter,
        "static",
        position,
        rotation,
        10.0,
        5.0,
        0.0,
        10.0,
        lambda t: [(t, t, t), (0, 0, 0), (0, 0, 1)],
    )
    assert callable_plotter.camera_position is not None

    # Sequence camera path.
    sequence_plotter = _FakePlotter()
    _FlightPlots._update_animation_camera(
        sequence_plotter,
        "static",
        position,
        rotation,
        10.0,
        5.0,
        0.0,
        10.0,
        [[(0, 0, 0), (0, 0, 0), (0, 0, 1)], [(1, 0, 0), (0, 0, 0), (0, 0, 1)]],
    )
    assert sequence_plotter.camera_position is not None


def test_update_animation_chart_cursors():
    """Chart cursors are moved to the requested time value."""

    class _FakeCursor:
        def __init__(self):
            self.x = None
            self.y = None

        def update(self, x, y):
            self.x = x
            self.y = y

    cursor = _FakeCursor()
    _FlightPlots._update_animation_chart_cursors([(cursor, 0.0, 1.0)], 3.0)
    assert cursor.x == [3.0, 3.0]
    assert cursor.y == [0.0, 1.0]


def test_set_animation_background():
    """The scene background is set from the altitude-blended palette."""

    class _FakePlotter:
        def __init__(self):
            self.background = None

        def set_background(self, bottom, top=None):
            self.background = (bottom, top)

    palette = {
        "launch_bottom": np.array([1.0, 0.0, 0.0]),
        "launch_top": np.array([0.0, 1.0, 0.0]),
        "space_bottom": np.array([0.0, 0.0, 1.0]),
        "space_top": np.array([1.0, 1.0, 0.0]),
    }
    plotter = _FakePlotter()
    _FlightPlots._set_animation_background(plotter, palette, 0)
    assert plotter.background is not None


def test_polyline_helpers():
    """Polyline builders return meshes and carry optional scalar arrays."""
    pv = pytest.importorskip("pyvista")

    single = _FlightPlots._polyline(pv, [[0, 0, 0]])
    assert single.n_points == 1

    line = _FlightPlots._polyline(pv, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert line.n_points == 3

    closed = _FlightPlots._polyline(pv, [[0, 0, 0], [1, 0, 0]], closed=True)
    assert closed.n_points == 2

    with_scalars = _FlightPlots._polyline_with_scalars(
        pv, [[0, 0, 0], [1, 0, 0]], [0.0, 1.0], "speed"
    )
    assert "speed" in with_scalars.point_data


def test_dashed_polyline():
    """The dashed line builder handles degenerate and normal inputs."""
    pv = pytest.importorskip("pyvista")

    # Fewer than two points returns the raw point cloud.
    degenerate = _FlightPlots._dashed_polyline(pv, [[0, 0, 0]])
    assert degenerate.n_points == 1

    points = [[float(i), 0.0, 0.0] for i in range(10)]
    scalars = list(range(10))
    dashed = _FlightPlots._dashed_polyline(
        pv, points, scalars=scalars, scalar_name="speed", dash_count=4
    )
    assert "speed" in dashed.point_data


def test_direction_arrow():
    """A direction arrow mesh is built for a non-degenerate vector."""
    pv = pytest.importorskip("pyvista")
    arrow = _FlightPlots._direction_arrow(pv, [0, 0, 1], scale=2.0)
    assert arrow.n_points > 0


def test_animation_options_defaults_and_normalization():
    """Option parsing fills defaults and normalizes booleans/aliases."""
    options = _FlightPlots._animation_options({})
    assert options["backend"] == "auto"
    assert options["color_by"] == "speed"
    assert options["camera_mode"] == "static"

    normalized = _FlightPlots._animation_options(
        {"color_by": False, "camera_mode": True}
    )
    assert normalized["color_by"] is None
    assert normalized["camera_mode"] == "follow"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"backend": "invalid"}, ValueError),
        ({"trajectory_line_width": True}, TypeError),
        ({"trajectory_line_width": -1}, ValueError),
        ({"color_by": "not_a_scalar"}, ValueError),
        ({"camera_mode": "sideways"}, ValueError),
        ({"camera_path": "not_callable"}, TypeError),
        ({"export_fps": 0}, ValueError),
        ({"export_resolution": (1, 2, 3)}, ValueError),
        ({"export_file": "movie.avi"}, ValueError),
        ({"export_file": "movie.gif", "force_external": True}, ValueError),
        ({"color_scheme": "not_a_mapping"}, TypeError),
    ],
)
def test_animation_options_validation_errors(kwargs, error):
    """Invalid animation options raise the documented errors."""
    with pytest.raises(error):
        _FlightPlots._animation_options(kwargs)


def test_ground_bounds_from_spec(flight_calisto):
    """Ground image bounds convert from ENU and lat/lon, and validate input."""
    plots = flight_calisto.plots
    fallback = (-1.0, 1.0, -1.0, 1.0)

    # No explicit bounds returns the fallback.
    assert plots._ground_bounds_from_spec({}, fallback) == fallback

    # Explicit ENU bounds pass through unchanged.
    enu = plots._ground_bounds_from_spec(
        {"bounds": (-2.0, 2.0, -3.0, 3.0), "coordinates": "enu"}, fallback
    )
    assert enu == (-2.0, 2.0, -3.0, 3.0)

    # Lat/lon bounds are projected to a finite ENU box.
    latitude = float(flight_calisto.env.latitude)
    longitude = float(flight_calisto.env.longitude)
    latlon = plots._ground_bounds_from_spec(
        {
            "bounds": (
                longitude - 0.01,
                longitude + 0.01,
                latitude - 0.01,
                latitude + 0.01,
            ),
            "coordinates": "latlon",
        },
        fallback,
    )
    assert len(latlon) == 4 and all(np.isfinite(latlon))

    with pytest.raises(ValueError):
        plots._ground_bounds_from_spec({"bounds": (1.0, 2.0, 3.0)}, fallback)
    with pytest.raises(ValueError):
        plots._ground_bounds_from_spec({"bounds": (2.0, 1.0, -1.0, 1.0)}, fallback)
    with pytest.raises(ValueError):
        plots._ground_bounds_from_spec(
            {"bounds": (-1.0, 1.0, -1.0, 1.0), "coordinates": "polar"}, fallback
        )


def test_resolve_and_validate_animation_inputs(flight_calisto):
    """Model-path resolution and time-range validation behave as documented."""
    plots = flight_calisto.plots

    # Explicit paths pass through; None resolves to the bundled default STL.
    assert plots._resolve_animation_model_path("rocket.stl") == "rocket.stl"
    default_path = plots._resolve_animation_model_path(None)
    assert default_path.endswith(".stl") and os.path.isfile(default_path)

    # A valid range returns the resolved stop time.
    stop = plots._validate_animation_inputs(default_path, 0, 1, 0.1)
    assert stop == 1

    with pytest.raises(ValueError):  # non-positive time step
        plots._validate_animation_inputs(default_path, 0, 1, 0)
    with pytest.raises(ValueError):  # start after stop
        plots._validate_animation_inputs(default_path, 2, 1, 0.1)
    with pytest.raises(FileNotFoundError):  # missing model file
        plots._validate_animation_inputs("does_not_exist.stl", 0, 1, 0.1)


def test_animation_state_helpers(flight_calisto):
    """Position, velocity, wind and scalar samplers return finite SI values."""
    plots = flight_calisto.plots
    time_value = 1.0

    assert plots._animation_position(time_value).shape == (3,)
    assert plots._animation_velocity(time_value).shape == (3,)
    assert plots._animation_wind(time_value).shape == (3,)
    assert plots._animation_transformation(time_value).shape == (4, 4)

    for scalar in ["speed", "mach", "dynamic_pressure", "acceleration", "altitude"]:
        assert np.isfinite(plots._animation_scalar(time_value, scalar))

    times = np.linspace(0, 1, 5)
    assert len(plots._animation_kinematic_series(times)) == 3
    assert len(plots._animation_attitude_series(times)) == 3


def test_animation_event_markers_and_phase(flight_calisto):
    """Event markers span the flight and phase labels match the timeline."""
    plots = flight_calisto.plots
    colors = _FlightPlots._animation_color_scheme()
    stop = flight_calisto.t_final

    markers = plots._animation_event_markers(0, stop, colors)
    labels = [label for _time, label, _color in markers]
    assert labels[0] == "Start" and labels[-1] == "End"
    # Markers are returned in chronological order.
    times = [marker_time for marker_time, _label, _color in markers]
    assert times == sorted(times)

    burn_out_time = flight_calisto.rocket.motor.burn_out_time
    assert plots._animation_phase(0) == "POWERED ASCENT"
    coast_time = 0.5 * (burn_out_time + flight_calisto.apogee_time)
    assert plots._animation_phase(coast_time) == "COAST"
    # Late in the flight the rocket is descending (with or without a parachute).
    assert plots._animation_phase(stop) in ("DESCENT", "PARACHUTE DESCENT")


def test_animation_telemetry_panels(flight_calisto):
    """Telemetry panels render as multi-line strings, optionally with margins."""
    plots = flight_calisto.plots
    trajectory_panel = plots._trajectory_telemetry(1.0)
    assert "ALTITUDE" in trajectory_panel

    rotation = plots._animation_transformation(1.0)
    rotation_panel = plots._rotation_telemetry(1.0, rotation)
    assert "ATTITUDE" in rotation_panel
    stability_panel = plots._rotation_telemetry(1.0, rotation, include_stability=True)
    assert "STATIC MARGIN" in stability_panel


def test_rocket_axial_display_coordinate(flight_calisto):
    """Axial coordinates map onto the centered display model symmetrically."""
    plots = flight_calisto.plots
    display_length = 2.0
    value = plots._rocket_axial_display_coordinate(0.0, display_length)
    assert np.isfinite(value)
    assert abs(value) <= display_length


def test_animation_background_palette(flight_calisto):
    """The palette responds to an explicit color and the launch time of day."""
    plots = flight_calisto.plots
    default_palette = plots._animation_background_palette()
    assert "launch_bottom" in default_palette

    override_palette = plots._animation_background_palette(background_color="#123456")
    assert np.allclose(
        override_palette["launch_bottom"], override_palette["launch_top"]
    )


def test_animation_background_palette_night(flight_calisto, monkeypatch):
    """A night-time launch selects the darker background palette."""
    plots = flight_calisto.plots
    colors = _FlightPlots._animation_color_scheme()

    class _Night:
        hour = 23

    monkeypatch.setattr(flight_calisto.env, "local_date", _Night(), raising=False)
    palette = plots._animation_background_palette(colors=colors)
    assert np.allclose(palette["launch_bottom"], to_rgb(colors["night_bottom"]))
