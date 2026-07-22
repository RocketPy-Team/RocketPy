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
"""

import math

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
