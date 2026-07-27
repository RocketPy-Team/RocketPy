from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from rocketpy import (
    Earth,
    Epoch,
    Flight,
    FlightState,
    PointMassRocket,
    ReferenceFrame,
    Space,
    ZeroAtmosphereLayer,
)


@pytest.fixture
def orbital_flight():
    """Build a short deterministic LEO propagation with no external data."""
    epoch = Epoch.from_datetime(datetime(2026, 1, 2, tzinfo=UTC))
    earth = Earth(
        geopotential="point_mass",
        atmosphere=ZeroAtmosphereLayer(),
        relativistic_correction=True,
    )
    space = Space()
    rocket = PointMassRocket(0.1, 100.0, 0.0, 0.0, 0.0)
    radius = earth.datum.semi_major_axis + 500e3
    speed = np.sqrt(earth.datum.gravitational_parameter / radius)
    state = FlightState.cartesian(
        epoch=epoch,
        position=[radius, 0.0, 0.0],
        velocity=[0.0, speed, 0.0],
        frame=ReferenceFrame.GCRF,
    )
    return Flight.from_orbit(
        rocket,
        earth,
        state,
        space=space,
        simulation_mode="3 DOF",
        max_time=600.0,
        max_time_step=30.0,
        rtol=1e-10,
        atol=1e-10,
        ode_solver="DOP853",
    )


def test_orbital_flight_preserves_near_circular_two_body_orbit(orbital_flight):
    """A short LEO propagation exposes stable state and orbital output views."""
    # Arrange
    initial = orbital_flight.orbit.initial

    # Act
    final = orbital_flight.orbit.final
    state = orbital_flight.state_at_time(300.0)

    # Assert
    assert orbital_flight.reference_frame is ReferenceFrame.GCRF
    assert state.epoch - orbital_flight.start_epoch == pytest.approx(300.0)
    assert final.semi_major_axis == pytest.approx(initial.semi_major_axis, rel=2e-7)
    assert orbital_flight.orbit.period(0.0) > orbital_flight.t_final
    assert orbital_flight.position("itrf").shape == (
        len(orbital_flight.time),
        3,
    )
    assert orbital_flight.position("teme").shape == (
        len(orbital_flight.time),
        3,
    )
    assert np.all(np.isfinite(orbital_flight.latitude[:, 1]))
    assert set(orbital_flight.orbital_accelerations) == {
        "gravity",
        "drag",
        "thrust",
        "third_body_gravity",
        "space_radiation_pressure",
        "relativistic_correction",
    }
    assert set(orbital_flight.orbital_accelerations_rtn) == set(
        orbital_flight.orbital_accelerations
    )
    assert np.all(np.isfinite(orbital_flight.specific_orbital_energy[:, 1]))
    assert "orbit" in orbital_flight.to_dict(include_outputs=True)


def test_ground_station_observation_returns_geometric_pass_outputs(orbital_flight):
    """Ground-station access is derived from the frame-aware Flight history."""
    # Arrange / Act
    observation = orbital_flight.ground_station_observation(
        latitude=0.0,
        longitude=0.0,
        minimum_elevation=10.0,
    )

    # Assert
    assert set(observation) == {"range", "elevation", "azimuth", "visible"}
    assert np.all(observation["range"][:, 1] >= 0.0)
    assert set(np.unique(observation["visible"][:, 1])).issubset({0.0, 1.0})


def test_orbital_plots_and_animation_use_flight_output_contract(
    orbital_flight, tmp_path
):
    """Orbital static plots save and 3D animations support multiple backends."""
    # Arrange
    orbit_path = tmp_path / "orbit.png"
    track_path = tmp_path / "ground-track.png"
    state_path = tmp_path / "state.png"
    geodetic_path = tmp_path / "geodetic.png"

    # Act
    orbital_flight.plots.orbit_3d(filename=orbit_path)
    orbital_flight.plots.ground_track(filename=track_path)
    orbital_flight.plots.earth_centered_state(filename=state_path)
    orbital_flight.plots.geodetic_coordinates(filename=geodetic_path)
    animation_mpl = orbital_flight.plots.animate_orbit_3d(
        interval=1, backend="matplotlib"
    )
    plotly_fig = orbital_flight.plots.plot_3d_trajectory()
    plotly_anim = orbital_flight.plots.animate_orbit_3d(interval=1, backend="plotly")
    pyvista_plotter = orbital_flight.plots.animate_orbit_3d(
        interval=1, backend="pyvista"
    )

    # Assert
    assert orbit_path.is_file()
    assert track_path.is_file()
    assert state_path.is_file()
    assert geodetic_path.is_file()
    assert orbital_flight.plots.is_high_altitude_flight is True
    assert orbital_flight.plots.has_low_altitude_segment is False
    assert isinstance(animation_mpl, FuncAnimation)
    assert plotly_fig.__class__.__name__ == "Figure"
    assert plotly_anim.__class__.__name__ == "Figure"
    assert pyvista_plotter.__class__.__name__ == "Plotter"

    animation_mpl._draw_was_started = True
    plt.close(animation_mpl._fig)
