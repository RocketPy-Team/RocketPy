"""Unit tests for the ``udot_rail2`` 3-DOF "tip-off" rail phase (issue #28).

The phase runs whenever the rocket has two rail buttons and the flight uses the
standard 6 DOF equations of motion. Coverage:

- Phase ordering: ``udot_rail1`` -> ``udot_rail2`` -> ``u_dot_generalized``.
- The window starts and ends where the two buttons reach the end of the rail.
- Constraint: the lower button stays on the rail line and roll stays zero.
- Physics: the motion matches an independent model of a rigid body pivoting
  about a point that slides along the rail.
- Outputs: the accelerations reported inside the window are the integrated
  ones, and the window shows up in ``.info()`` and in the attitude plots.
- The phase is skipped where it cannot run.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rocketpy import Environment, Rocket, SolidMotor
from rocketpy._encoders import rebuild_minimal_flight
from rocketpy.mathutils import Matrix, Vector
from rocketpy.simulation.flight import Flight


def _make_flight(rocket, environment, **kwargs):
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        **kwargs,
    )


def _bare_flight(grain_density=1815, **kwargs):
    """Flight of a rocket with no aerodynamics (no surfaces, zero drag) under a
    constant thrust, in still air and without Earth rotation. Only thrust and
    gravity act, so the motion on the rail can be written down exactly. A very
    low ``grain_density`` leaves almost no propellant, which makes the rocket a
    rigid body of constant mass.
    """
    environment = Environment()
    environment.earth_rotation_vector = [0.0, 0.0, 0.0]
    motor = SolidMotor(
        thrust_source=2000.0,
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=grain_density,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.0,
        power_on_drag=0.0,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.255)
    rocket.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.6182,
        angular_position=45,
    )
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5.2,
        heading=0,
        max_time=1.0,
        terminate_on_apogee=False,
        **kwargs,
    )


def _phase_names(flight):
    return [
        phase.derivative.__name__ if phase.derivative is not None else None
        for phase in flight.flight_phases.list
    ]


def _lower_button_body_position(rocket):
    """Lower rail button position from the center of dry mass, body frame."""
    z = (rocket.rail_buttons[0].position.z - rocket.center_of_dry_mass_position) * (
        rocket._csys
    )
    return Vector([0.0, 0.0, z])


def _tip_off_rows(flight):
    """Solution rows [t, *state] within the tip-off window (inclusive)."""
    t0, t1 = flight.out_of_rail_time, flight.tip_off_end_time
    return [row for row in flight.solution if t0 - 1e-12 <= row[0] <= t1 + 1e-12]


def _distance_travelled(flight, state):
    return math.hypot(state[0], state[1], state[2] - flight.env.elevation)


def _attitude_inclination_deg(state):
    """Inclination (deg from horizontal) of the body axis in the inertial frame."""
    body_axis = Matrix.transformation(state[6:10]) @ Vector([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(body_axis.z, math.hypot(body_axis.x, body_axis.y)))


def test_udot_rail2_runs_between_the_rail_phase_and_free_flight(
    calisto_robust, example_spaceport_env
):
    """A rocket with two rail buttons goes through the tip-off phase, between
    the 1-DOF rail phase and free flight, without being asked to."""
    flight = _make_flight(calisto_robust, example_spaceport_env)
    names = _phase_names(flight)

    assert "udot_rail2" in names
    assert names.index("udot_rail1") < names.index("udot_rail2")
    assert names.index("udot_rail2") < names.index("u_dot_generalized")


def test_udot_rail2_window_starts_and_ends_at_the_button_exits(
    calisto_robust, example_spaceport_env
):
    """The window opens when the upper button reaches the end of the rail and
    closes when the lower one does."""
    flight = _make_flight(calisto_robust, example_spaceport_env)

    assert flight.effective_2rl > flight.effective_1rl
    assert 0 < flight.out_of_rail_time < flight.tip_off_end_time
    assert flight.tip_off_duration == pytest.approx(
        flight.tip_off_end_time - flight.out_of_rail_time
    )
    assert _distance_travelled(flight, flight.out_of_rail_state) == pytest.approx(
        flight.effective_1rl, rel=1e-9
    )
    assert _distance_travelled(flight, flight.tip_off_end_state) == pytest.approx(
        flight.effective_2rl, rel=1e-9
    )


def test_udot_rail2_button_stays_on_rail(calisto_robust, example_spaceport_env):
    """The lower rail button stays on the rail line throughout the window."""
    flight = _make_flight(calisto_robust, example_spaceport_env)
    r_b = _lower_button_body_position(flight.rocket)
    rail_dir = flight.attitude_unit
    rows = _tip_off_rows(flight)
    first_state = rows[0][1:]
    rail_origin = (
        Vector(first_state[0:3]) + Matrix.transformation(first_state[6:10]) @ r_b
    )

    max_perp = 0.0
    for row in rows:
        state = row[1:]
        button = Vector(state[0:3]) + Matrix.transformation(state[6:10]) @ r_b
        offset = button - rail_origin
        perpendicular = offset - rail_dir * (offset @ rail_dir)
        max_perp = max(max_perp, abs(perpendicular))

    assert len(rows) >= 2, "tip-off window not exercised"
    assert max_perp < 1e-6, f"button drifted off the rail (max offset {max_perp} m)"


def test_udot_rail2_no_roll(calisto_robust, example_spaceport_env):
    """Neither the roll angular acceleration nor the roll rate leaves zero."""
    flight = _make_flight(calisto_robust, example_spaceport_env)

    u_dot = flight.udot_rail2(flight.tip_off_end_time, flight.tip_off_end_state)
    max_roll_rate = max(abs(row[1:][12]) for row in _tip_off_rows(flight))

    assert u_dot[12] == 0
    assert max_roll_rate < 1e-9, f"expected zero roll rate, got {max_roll_rate}"


def test_udot_rail2_gravity_tip_off_direction(calisto_robust, example_spaceport_env):
    """With no wind, gravity on the center of mass, which sits ahead of the
    lower button, pitches the nose down by a small amount."""
    flight = _make_flight(calisto_robust, example_spaceport_env)
    rows = _tip_off_rows(flight)

    delta = _attitude_inclination_deg(rows[-1][1:]) - _attitude_inclination_deg(
        rows[0][1:]
    )

    assert delta < 0, f"nose should pitch down (gravity tip-off), got {delta:+.4f} deg"
    assert abs(delta) < 1.0, f"tip-off rotation implausibly large: {delta:+.4f} deg"


def _pivot_model_parameters(flight):
    """Mass, distance from the lower button to the center of mass, inertia
    about the center of mass, thrust and gravity when the window opens."""
    rocket = flight.rocket
    t1 = flight.out_of_rail_time
    mass = rocket.total_mass(t1)
    center_of_mass = rocket.center_of_mass(t1)
    cm_from_cdm = (center_of_mass - rocket.center_of_dry_mass_position) * rocket._csys
    arm = (center_of_mass - rocket.rail_buttons[0].position.z) * rocket._csys
    inertia = rocket.I_11(t1) - mass * cm_from_cdm**2
    thrust = rocket.motor.thrust(t1)
    gravity = flight.env.gravity(flight.out_of_rail_state[2])
    return mass, arm, inertia, thrust, gravity


def _pivot_model_pitch_acceleration(flight, theta, phi, phi_dot):
    """Second derivative of ``phi`` with respect to time in the model described
    in ``_pivot_model_end_state``."""
    mass, arm, inertia, thrust, gravity = _pivot_model_parameters(flight)
    return (
        arm
        * math.cos(phi)
        * (
            mass * arm * phi_dot**2 * math.sin(phi)
            + thrust * math.sin(phi)
            - mass * gravity * math.cos(theta)
        )
        / (inertia + mass * arm**2 * math.cos(phi) ** 2)
    )


def _pivot_model_end_state(flight, inclination):
    """Pitch angle and pitch rate, at the end of the tip-off window, of a rigid
    body of constant mass whose lower button slides along the rail under thrust
    and gravity only.

    The model has two coordinates: ``s``, the position of the button along the
    rail, and ``phi``, the angle of the rocket axis from the rail. With ``l``
    the distance from the button to the center of mass, ``I`` the inertia about
    the center of mass and ``theta`` the rail inclination:

        (I + m l^2 cos^2 phi) phi'' = l cos phi (m l phi'^2 sin phi
                                                 + T sin phi - m g cos theta)
        s'' = T cos phi / m - g sin theta + l phi'' sin phi + l phi'^2 cos phi
    """
    t1, t2 = flight.out_of_rail_time, flight.tip_off_end_time
    start = np.array(flight.out_of_rail_state)
    theta = math.radians(inclination)
    mass, arm, _, thrust, gravity = _pivot_model_parameters(flight)

    def derivative(_, y):
        _, phi, s_dot, phi_dot = y
        phi_ddot = _pivot_model_pitch_acceleration(flight, theta, phi, phi_dot)
        s_ddot = (
            thrust * math.cos(phi) / mass
            - gravity * math.sin(theta)
            + arm * phi_ddot * math.sin(phi)
            + arm * phi_dot**2 * math.cos(phi)
        )
        return [s_dot, phi_dot, s_ddot, phi_ddot]

    speed = start[3:6] @ np.array(list(flight.attitude_unit))
    end = solve_ivp(
        derivative,
        (t1, t2),
        [0.0, 0.0, speed, 0.0],
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    ).y[:, -1]
    return end[1], end[3]


def _pitch_from_the_rail(flight, state):
    """Pitch angle and pitch rate of the rocket in the vertical plane of the
    rail, measured from the rail and positive when the nose rises."""
    state = np.array(state)
    rail = np.array(list(flight.attitude_unit))
    up = np.array([0.0, 0.0, 1.0])
    across = up - (up @ rail) * rail
    across /= np.linalg.norm(across)
    K = np.array(Matrix.transformation(state[6:10]))
    body_axis = K @ np.array([0.0, 0.0, 1.0])
    pitch = math.atan2(body_axis @ across, body_axis @ rail)
    pitch_rate = (K @ state[10:13]) @ np.cross(rail, across)
    return pitch, pitch_rate


@pytest.mark.parametrize("inclination", [85, 60])
def test_udot_rail2_matches_a_rigid_body_pivoting_on_the_rail(inclination):
    """With only thrust and gravity acting on a rocket of constant mass, the
    phase must reproduce a rigid body whose lower button slides along the rail.
    """
    flight = _bare_flight(
        grain_density=1e-3, inclination=inclination, rtol=1e-11, atol=1e-13
    )

    expected_pitch, expected_pitch_rate = _pivot_model_end_state(flight, inclination)
    pitch, pitch_rate = _pitch_from_the_rail(flight, flight.tip_off_end_state)

    assert pitch_rate < 0
    assert pitch == pytest.approx(expected_pitch, rel=1e-5)
    assert pitch_rate == pytest.approx(expected_pitch_rate, rel=1e-5)


def test_udot_rail2_reports_the_accelerations_it_integrates():
    """The accelerations stored for the window are the ones of the rocket on
    the rail, not the ones of a free rocket. Without aerodynamics a free rocket
    has no angular acceleration, so the two are easy to tell apart."""
    flight = _bare_flight(inclination=85)
    rows = _tip_off_rows(flight)
    time, *state = rows[len(rows) // 2]

    u_dot = flight.udot_rail2(time, state)

    assert abs(u_dot[10]) > 0.1
    assert flight.alpha1(time) == pytest.approx(u_dot[10], rel=1e-9)
    assert flight.alpha2(time) == pytest.approx(u_dot[11], rel=1e-9)
    assert u_dot[12] == 0
    # Interpolation may leave roundoff even at a stored zero sample.
    assert flight.alpha3(time) == pytest.approx(u_dot[12], abs=1e-12)
    assert flight.ax(time) == pytest.approx(u_dot[3], abs=1e-9)
    assert flight.ay(time) == pytest.approx(u_dot[4], rel=1e-9)
    assert flight.az(time) == pytest.approx(u_dot[5], rel=1e-9)


def test_udot_rail2_reports_the_force_on_the_lower_button():
    """During the window the lower button carries the rocket alone. In the
    pivot model the rail pushes the rocket, across the rail, with

        N = m g cos theta - T sin phi + m l (phi'' cos phi - phi'^2 sin phi)

    and the rocket pushes the rail with the opposite force. The upper button,
    already off the rail, carries nothing.
    """
    inclination = 60
    flight = _bare_flight(
        grain_density=1e-3, inclination=inclination, rtol=1e-11, atol=1e-13
    )
    time, state = flight.tip_off_end_time, np.array(flight.tip_off_end_state)
    theta = math.radians(inclination)
    mass, arm, _, thrust, gravity = _pivot_model_parameters(flight)
    phi, phi_dot = _pitch_from_the_rail(flight, state)
    phi_ddot = _pivot_model_pitch_acceleration(flight, theta, phi, phi_dot)
    expected = (
        mass * gravity * math.cos(theta)
        - thrust * math.sin(phi)
        + mass * arm * (phi_ddot * math.cos(phi) - phi_dot**2 * math.sin(phi))
    )

    # Rebuild the force of the button on the rail from its two components
    angle = flight.rocket.rail_buttons[0].component.angular_position_rad
    normal = flight.rail_button2_normal_force(time)
    shear = flight.rail_button2_shear_force(time)
    force_body = np.array(
        [
            normal * math.cos(angle) - shear * math.sin(angle),
            normal * math.sin(angle) + shear * math.cos(angle),
            0.0,
        ]
    )
    force = np.array(Matrix.transformation(state[6:10])) @ force_body
    rail = np.array(list(flight.attitude_unit))
    up_across_the_rail = np.array([0.0, 0.0, 1.0]) - rail[2] * rail
    up_across_the_rail /= np.linalg.norm(up_across_the_rail)

    assert expected > 1
    assert force @ up_across_the_rail == pytest.approx(-expected, rel=1e-5)
    assert np.linalg.norm(force) == pytest.approx(expected, rel=1e-5)
    assert flight.rail_button1_normal_force(time) == 0
    assert flight.rail_button1_shear_force(time) == 0
    assert flight.max_rail_button2_normal_force >= abs(normal) * (1 - 1e-12)


def test_rail_exits_are_found_inside_a_long_solver_step():
    """A constant thrust on a rocket of constant mass is smooth enough for the
    solver to cross the whole rail in one step. The button exits must still be
    placed where the buttons reach the end of the rail."""
    flight = _bare_flight(grain_density=1e-3, inclination=85)

    assert _distance_travelled(flight, flight.out_of_rail_state) == pytest.approx(
        flight.effective_1rl, rel=1e-6
    )
    assert _distance_travelled(flight, flight.tip_off_end_state) == pytest.approx(
        flight.effective_2rl, rel=1e-6
    )


@pytest.mark.parametrize(
    "options",
    [{"simulation_mode": "3 DOF"}, {"equations_of_motion": "solid_propulsion"}],
)
def test_udot_rail2_is_skipped_where_it_cannot_run(
    calisto_robust, example_spaceport_env, options, capsys
):
    """The phase corrects the generalized 6 DOF equations, so flights that use
    the 3 DOF or the solid propulsion equations go straight to free flight and
    report no window."""
    flight = _make_flight(calisto_robust, example_spaceport_env, **options)
    figure, axes = plt.subplots()

    flight.prints.out_of_rail_conditions()
    flight.plots._mark_tip_off_window(axes)

    assert "udot_rail2" not in _phase_names(flight)
    assert flight.tip_off_end_time == flight.out_of_rail_time
    assert flight.tip_off_duration == 0
    assert "Tip-Off State" not in capsys.readouterr().out
    assert not axes.patches
    plt.close(figure)


def test_udot_rail2_is_skipped_for_a_given_initial_solution(
    calisto_robust, example_spaceport_env
):
    """A flight that starts from a given state has no rail phases. The rail
    direction is still defined, and no window is reported."""
    launch = _make_flight(calisto_robust, example_spaceport_env)
    mid_flight_state = list(launch.solution[len(launch.solution) // 2])

    continuation = _make_flight(
        calisto_robust, example_spaceport_env, initial_solution=mid_flight_state
    )

    assert "udot_rail2" not in _phase_names(continuation)
    assert continuation.tip_off_duration == 0
    assert abs(continuation.attitude_unit) == pytest.approx(1)
    for axis in range(3):
        assert continuation.attitude_unit[axis] == launch.attitude_unit[axis]


def test_udot_rail2_reports_the_window_in_prints(
    calisto_robust, example_spaceport_env, capsys
):
    """The window is printed with the rail departure conditions."""
    flight = _make_flight(calisto_robust, example_spaceport_env)

    flight.prints.out_of_rail_conditions()
    out = capsys.readouterr().out

    assert flight.tip_off_duration > 0
    assert "Tip-Off State" in out
    assert f"{flight.tip_off_duration:.3f} s" in out
    assert f"{flight.tip_off_end_time:.3f} s" in out


def test_udot_rail2_shades_the_window_in_attitude_plots(
    calisto_robust, example_spaceport_env, tmp_path
):
    """The attitude plots shade the window and are written to the file."""
    flight = _make_flight(calisto_robust, example_spaceport_env)
    target = tmp_path / "attitude.png"
    figure, axes = plt.subplots()

    flight.plots._mark_tip_off_window(axes)
    flight.plots.attitude_data(filename=str(target))

    assert len(axes.patches) == 1
    assert target.exists()
    plt.close(figure)


def test_udot_rail2_window_survives_saving_the_flight(
    calisto_robust, example_spaceport_env
):
    """A flight rebuilt from its saved data reports the same window and the
    same lower button force. Data saved before the phase existed has no window
    to report and still gives the rail button forces."""
    flight = _make_flight(calisto_robust, example_spaceport_env)
    data = flight.to_dict()

    restored = rebuild_minimal_flight(Flight, data, None)
    del data["tip_off_end_time"]
    del data["tip_off_button_forces"]
    restored_from_old_data = rebuild_minimal_flight(Flight, data, None)

    assert restored.tip_off_duration == flight.tip_off_duration
    assert restored.max_rail_button2_normal_force == pytest.approx(
        flight.max_rail_button2_normal_force
    )
    assert restored_from_old_data.tip_off_duration == 0
    assert (
        restored_from_old_data.max_rail_button2_normal_force
        < flight.max_rail_button2_normal_force
    )


def test_udot_rail2_gives_way_to_a_parachute_opened_inside_the_window(
    calisto_robust, example_spaceport_env
):
    """A parachute that opens while the lower button is still on the rail takes
    over the flight. The lower button exit must not hand it back to free
    flight afterwards, and no complete window is reported."""
    window = _make_flight(calisto_robust, example_spaceport_env)
    inside = 0.5 * (window.out_of_rail_time + window.tip_off_end_time)
    calisto_robust.add_parachute(
        name="early", cd_s=1.0, trigger=("time", inside), sampling_rate=500
    )

    flight = _make_flight(calisto_robust, example_spaceport_env)
    names = _phase_names(flight)

    assert "udot_rail2" in names
    assert "u_dot_generalized" not in names[names.index("u_dot_parachute") :]
    assert flight.tip_off_duration == 0


def test_udot_rail2_reports_no_window_when_the_flight_ends_inside_it(
    calisto_robust, example_spaceport_env
):
    """A flight stopped while the lower button is still on the rail has no
    complete window to report."""
    window = _make_flight(calisto_robust, example_spaceport_env)
    inside = 0.5 * (window.out_of_rail_time + window.tip_off_end_time)

    flight = _make_flight(calisto_robust, example_spaceport_env, max_time=inside)

    names = _phase_names(flight)
    assert "udot_rail2" in names
    assert "u_dot_generalized" not in names
    assert flight.tip_off_duration == 0


def test_udot_rail2_is_skipped_when_the_upper_button_starts_off_the_rail(
    calisto_robust, example_spaceport_env
):
    """On a rail shorter than the distance from the nozzle to the upper button,
    the upper button is never on the rail and the phase is not used."""
    flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=1.0,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    assert flight.effective_1rl < 0 < flight.effective_2rl
    assert "udot_rail2" not in _phase_names(flight)
    assert flight.tip_off_duration == 0
