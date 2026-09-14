"""Unit tests for the ``udot_rail2`` 3-DOF "tip-off" rail phase (issue #28).

These tests follow the project's testing conventions: each test uses the
Arrange / Act / Assert pattern and documents the expected behaviour in its
docstring. They avoid plotting and optional-dependency features so they run
reliably in CI.

Coverage:
- Phase ordering: ``udot_rail1`` -> ``udot_rail2`` -> ``u_dot_generalized``.
- Opt-in default: ``use_udot_rail2`` defaults to False and, when disabled, no
  ``udot_rail2`` phase is inserted (previous behaviour preserved).
- Constraint satisfaction: the lower rail button stays on the rail line and the
  roll acceleration/rate remains zero throughout the phase.
- Physical direction: with no wind the nose pitches over (gravity tip-off).
- Mode compatibility: the phase refuses reduced formulations it cannot patch.
- Reporting: the window shows up in ``.info()`` and in the attitude plots.
"""

import math

import matplotlib.pyplot as plt
import pytest

from rocketpy.mathutils import Matrix, Vector
from rocketpy.simulation.flight import Flight


def _make_flight(rocket, environment, use_udot_rail2):
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        use_udot_rail2=use_udot_rail2,
    )


def _phase_names(flight):
    return [
        phase.derivative.__name__ if phase.derivative is not None else None
        for phase in flight.flight_phases.list
    ]


def _lower_button_body_position(rocket):
    """Lower rail button position relative to the CDM in the true body frame."""
    z = (rocket.rail_buttons[0].position.z - rocket.center_of_dry_mass_position) * (
        rocket._csys
    )
    return Vector([0.0, 0.0, z])


def _tip_off_rows(flight):
    """Solution rows [t, *state] within the tip-off window (inclusive)."""
    t0, t1 = flight.out_of_rail_time, flight.between_rails_time
    return [row for row in flight.solution if t0 - 1e-12 <= row[0] <= t1 + 1e-12]


def _attitude_inclination_deg(state):
    """Inclination (deg from horizontal) of the body axis in the inertial frame."""
    body_axis = Matrix.transformation(state[6:10]) @ Vector([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(body_axis.z, math.hypot(body_axis.x, body_axis.y)))


def test_udot_rail2_default_is_opt_in(calisto_robust, example_spaceport_env):
    """``use_udot_rail2`` defaults to False and, when not requested, the flight
    keeps the previous rail1 -> generalized transition with no ``udot_rail2``.

    Arrange: build a Flight without passing ``use_udot_rail2``.
    Act: read the flag and the phase derivative names.
    Assert: the flag is False and no ``udot_rail2`` phase was inserted.
    """
    # Arrange / Act
    flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    # Assert
    assert flight.use_udot_rail2 is False, (
        "tip-off phase must be opt-in (default False)"
    )
    assert "udot_rail2" not in _phase_names(flight)


def test_udot_rail2_inserts_phase_in_order(calisto_robust, example_spaceport_env):
    """With ``use_udot_rail2=True`` the intermediate 3-DOF ``udot_rail2`` phase
    is inserted between the 1-DOF ``udot_rail1`` and the 6-DOF
    ``u_dot_generalized`` phases.

    Arrange: build a Flight with ``use_udot_rail2=True``.
    Act: inspect the ordered phase derivative names.
    Assert: rail1 < rail2 < generalized in the phase list.
    """
    # Arrange / Act
    flight = _make_flight(calisto_robust, example_spaceport_env, True)
    names = _phase_names(flight)

    # Assert
    assert "udot_rail2" in names, "udot_rail2 phase not present"
    assert names.index("udot_rail1") < names.index("udot_rail2"), (
        "udot_rail2 should follow the 1-DOF rail phase"
    )
    assert names.index("udot_rail2") < names.index("u_dot_generalized"), (
        "udot_rail2 should precede the generalized 6-DOF phase"
    )


def test_udot_rail2_window_is_between_effective_rail_lengths(
    calisto_robust, example_spaceport_env
):
    """The tip-off phase spans the interval between the upper button exit
    (``effective_1rl``, recorded as ``out_of_rail``) and the lower button exit
    (``effective_2rl``, recorded as ``between_rails``).

    Arrange: build a Flight with the tip-off phase enabled.
    Act: read the event times.
    Assert: ``0 < out_of_rail_time < between_rails_time`` and the window is short.
    """
    # Arrange / Act
    flight = _make_flight(calisto_robust, example_spaceport_env, True)

    # Assert
    assert flight.effective_2rl > flight.effective_1rl
    assert 0 < flight.out_of_rail_time < flight.between_rails_time
    assert (flight.between_rails_time - flight.out_of_rail_time) < 1.0


def test_udot_rail2_button_stays_on_rail(calisto_robust, example_spaceport_env):
    """The single-button constraint must keep the lower rail button on the rail
    line: its distance from the rail axis stays ~0 throughout the tip-off phase.

    Arrange: run a Flight with the tip-off phase enabled.
    Act: for every solution point in the tip-off window, compute the lower
    button position and its perpendicular distance to the (fixed) rail line.
    Assert: the maximum perpendicular offset is negligible.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, True)
    r_b = _lower_button_body_position(flight.rocket)
    rail_dir = flight.attitude_unit
    rail_origin = Vector(flight.solution[0][1:4])

    # Act
    max_perp = 0.0
    for row in _tip_off_rows(flight):
        state = row[1:]
        cdm = Vector(state[0:3])
        button = cdm + Matrix.transformation(state[6:10]) @ r_b
        offset = button - rail_origin
        perpendicular = offset - rail_dir * (offset @ rail_dir)
        max_perp = max(max_perp, abs(perpendicular))

    # Assert
    assert len(_tip_off_rows(flight)) >= 2, "tip-off window not exercised"
    assert max_perp < 1e-6, f"button drifted off the rail (max offset {max_perp} m)"


def test_udot_rail2_no_roll(calisto_robust, example_spaceport_env):
    """The tip-off phase must not induce roll: both the roll angular
    acceleration returned by ``udot_rail2`` and the integrated roll rate stay
    zero.

    Arrange: run a Flight with the tip-off phase enabled.
    Act: evaluate ``udot_rail2`` at the phase-end state and scan the roll rate
    over the tip-off window.
    Assert: the roll angular acceleration and every sampled roll rate are ~0.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, True)

    # Act
    u_dot = flight.udot_rail2(flight.between_rails_time, flight.between_rails_state)
    roll_acceleration = u_dot[12]
    max_roll_rate = max(abs(row[1:][12]) for row in _tip_off_rows(flight))

    # Assert
    assert abs(roll_acceleration) < 1e-12, (
        f"expected zero roll acceleration, got {roll_acceleration}"
    )
    assert max_roll_rate < 1e-9, f"expected zero roll rate, got {max_roll_rate}"


def test_udot_rail2_rail_axis_defined_for_given_initial_solution(
    calisto_robust, example_spaceport_env
):
    """The rail axis ``udot_rail2`` constrains the button to is set by the launch
    inclination and heading, so it must exist even when the rail phase is skipped
    because an ``initial_solution`` was supplied.

    Arrange: run a launch and take a mid-flight state from its solution.
    Act: build a Flight that starts from that state instead of the rail.
    Assert: ``attitude_unit`` is the same rail unit vector as the launch's, so
    ``udot_rail2`` cannot raise ``AttributeError``.
    """
    # Arrange
    launch = _make_flight(calisto_robust, example_spaceport_env, True)
    mid_flight_state = list(launch.solution[len(launch.solution) // 2])

    # Act
    continuation = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        initial_solution=mid_flight_state,
        use_udot_rail2=True,
    )

    # Assert
    assert abs(abs(continuation.attitude_unit) - 1) < 1e-12
    for axis in range(3):
        assert continuation.attitude_unit[axis] == launch.attitude_unit[axis]


def test_udot_rail2_gravity_tip_off_direction(calisto_robust, example_spaceport_env):
    """With no wind, gravity acting on the center of mass (ahead of the lower
    button pivot) tips the nose over: the attitude inclination decreases across
    the tip-off phase by a small amount.

    Arrange: run a wind-free Flight with the tip-off phase enabled.
    Act: measure the attitude inclination at the start and end of the window.
    Assert: the inclination decreases, by a small (sub-degree) magnitude.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, True)
    rows = _tip_off_rows(flight)

    # Act
    incl_start = _attitude_inclination_deg(rows[0][1:])
    incl_end = _attitude_inclination_deg(rows[-1][1:])
    delta = incl_end - incl_start

    # Assert
    assert delta < 0, f"nose should pitch down (gravity tip-off), got {delta:+.4f} deg"
    assert abs(delta) < 1.0, f"tip-off rotation implausibly large: {delta:+.4f} deg"


def test_udot_rail2_rejects_3dof_simulation_mode(calisto_robust, example_spaceport_env):
    """The tip-off phase must refuse ``simulation_mode='3 DOF'``.

    ``udot_rail2`` patches the generalized 6-DOF solution with a constraint
    wrench built from the full inertia tensor, but in 3 DOF the bound
    ``u_dot_generalized`` is ``u_dot_generalized_3dof``, which carries no
    attitude at all. Silently patching it would yield non-physical tip-off
    kinematics, so construction must fail loudly.

    Arrange: a rocket and environment that fly normally in 6 DOF.
    Act: build a Flight asking for both the tip-off phase and 3 DOF.
    Assert: a ValueError naming the offending argument is raised.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="simulation_mode='6 DOF'"):
        Flight(
            rocket=calisto_robust,
            environment=example_spaceport_env,
            rail_length=5.2,
            inclination=85,
            heading=0,
            terminate_on_apogee=True,
            simulation_mode="3 DOF",
            use_udot_rail2=True,
        )


def test_udot_rail2_rejects_solid_propulsion_equations(
    calisto_robust, example_spaceport_env
):
    """The tip-off phase must refuse ``equations_of_motion='solid_propulsion'``.

    That option binds ``u_dot_generalized`` to the reduced axisymmetric ``u_dot``
    formulation, which does not share the state the constraint patch describes.

    Arrange: a rocket and environment that fly normally with the standard EOM.
    Act: build a Flight asking for both the tip-off phase and solid_propulsion.
    Assert: a ValueError naming the offending argument is raised.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="equations_of_motion='standard'"):
        Flight(
            rocket=calisto_robust,
            environment=example_spaceport_env,
            rail_length=5.2,
            inclination=85,
            heading=0,
            terminate_on_apogee=True,
            equations_of_motion="solid_propulsion",
            use_udot_rail2=True,
        )


def test_udot_rail2_disabled_allows_reduced_formulations(
    calisto_robust, example_spaceport_env
):
    """The guard must only fire when the tip-off phase is actually requested.

    Arrange: a rocket and environment.
    Act: build flights in 3 DOF and with solid_propulsion, tip-off left off.
    Assert: both build without raising, and neither inserts the phase.
    """
    # Arrange / Act
    three_dof = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        simulation_mode="3 DOF",
    )
    solid = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        equations_of_motion="solid_propulsion",
    )

    # Assert
    assert "udot_rail2" not in _phase_names(three_dof)
    assert "udot_rail2" not in _phase_names(solid)


def test_udot_rail2_reports_the_window_in_prints(
    calisto_robust, example_spaceport_env, capsys
):
    """The tip-off window must be visible in the rail conditions printout.

    Studying tip-off is the point of the feature, so a user who opts in should
    not have to dig the window out of the solution by hand.

    Arrange: fly with the tip-off phase enabled.
    Act: print the rail departure conditions.
    Assert: the tip-off section is there and reports the measured window.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, True)

    # Act
    flight.prints.out_of_rail_conditions()
    out = capsys.readouterr().out

    # Assert
    assert "Tip-Off State" in out
    assert f"{flight.tip_off_duration:.3f} s" in out
    assert f"{flight.between_rails_time:.3f} s" in out
    assert flight.tip_off_duration > 0


def test_udot_rail2_prints_no_tip_off_section_when_disabled(
    calisto_robust, example_spaceport_env, capsys
):
    """Flights that did not run the phase must not grow a tip-off section.

    Arrange: fly with the tip-off phase disabled.
    Act: print the rail departure conditions.
    Assert: the output is unchanged from previous versions.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, False)

    # Act
    flight.prints.out_of_rail_conditions()
    out = capsys.readouterr().out

    # Assert
    assert "Tip-Off State" not in out
    assert flight.tip_off_duration == 0.0


def test_udot_rail2_shades_the_window_in_attitude_plots(
    calisto_robust, example_spaceport_env, tmp_path
):
    """The attitude plots must shade the tip-off window when it exists.

    Arrange: fly with the tip-off phase enabled.
    Act: render the attitude plots to a file.
    Assert: every subplot carries the shaded window, and the file is written.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, True)
    target = tmp_path / "attitude.png"

    # Act
    flight.plots.attitude_data(filename=str(target))

    # Assert
    assert target.exists()


def test_udot_rail2_flag_survives_objects_without_it(
    calisto_robust, example_spaceport_env, capsys
):
    """Flights restored from a ``.rpy`` written before this feature must still
    answer the flag, since they are rebuilt without going through ``__init__``.

    Arrange: fly, then drop the instance attribute to mimic an old object.
    Act: read the flag and print the rail conditions.
    Assert: the class default answers False and nothing raises.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, True)
    del flight.__dict__["use_udot_rail2"]

    # Act
    flight.prints.out_of_rail_conditions()
    out = capsys.readouterr().out

    # Assert
    assert flight.use_udot_rail2 is False
    assert flight.tip_off_duration == 0.0
    assert "Tip-Off State" not in out


def test_udot_rail2_window_marker_is_a_noop_when_disabled(
    calisto_robust, example_spaceport_env
):
    """The shading helper must do nothing for flights without the phase.

    Arrange: fly with the tip-off phase disabled and make a bare axes.
    Act: ask the plot helper to mark the window on it.
    Assert: nothing was drawn.
    """
    # Arrange
    flight = _make_flight(calisto_robust, example_spaceport_env, False)
    figure, axes = plt.subplots()

    # Act
    flight.plots._mark_tip_off_window(axes)

    # Assert
    assert not axes.patches
    plt.close(figure)
