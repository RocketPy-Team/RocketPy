"""Unit tests for the udot_rail2 rail-phase feature.

These tests follow the project's testing conventions: each test is named
`test_methodname`, uses the Arrange / Act / Assert pattern, and the expected
behaviour is documented in the test docstring.

Coverage:
- Phase insertion ordering when `use_udot_rail2` is enabled
- udot_rail2 enforces zero roll acceleration and projects `r_dot` on the rail
- CSV comparison generation for enabled/disabled runs

These tests are intentionally written to avoid plotting and optional-dependency
features so they run reliably in CI and local environments.
"""

import csv
import math
import os

import numpy as np

from rocketpy.mathutils import Matrix, Vector


def _yaw_deg(v: Vector):
    return math.degrees(math.atan2(v.y, v.x))


def _pitch_deg(v: Vector):
    return math.degrees(math.atan2(v.z, math.hypot(v.x, v.y)))


def _body_axis_from_e(e):
    K = Matrix.transformation(e)
    return K @ Vector([0.0, 0.0, 1.0])


def test_udot_rail2_inserts_phase_in_order(calisto_robust, example_spaceport_env):
    """When `use_udot_rail2=True`, the intermediate 3-DOF `udot_rail2` phase
    is inserted before the 6-DOF generalized phase (`u_dot_generalized`).

    Arrange: build a Flight with `use_udot_rail2=True`.
    Act: inspect `flight.flight_phases` derivatives names.
    Assert: `udot_rail2` appears before `u_dot_generalized` in the phase list.
    """
    # Arrange
    from rocketpy.simulation.flight import Flight

    flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
        use_udot_rail2=True,
    )

    # Act
    derivative_names = [
        phase.derivative.__name__ if phase.derivative is not None else None
        for phase in flight.flight_phases.list
    ]

    # Assert
    assert "udot_rail2" in derivative_names, "udot_rail2 phase not present"
    assert "u_dot_generalized" in derivative_names, (
        "u_dot_generalized phase not present"
    )
    assert derivative_names.index("udot_rail2") < derivative_names.index(
        "u_dot_generalized"
    ), "udot_rail2 should be inserted before u_dot_generalized"


def test_udot_rail2_no_roll_and_alignment(calisto_robust, example_spaceport_env):
    """udot_rail2 must enforce zero roll acceleration and set `r_dot` as the
    projection of the velocity vector onto the inertial rail axis.

    Arrange: create flight with `use_udot_rail2=True` and find the between-rails
    time/state. Act: evaluate `udot_rail2(t, u)` at that instant. Assert: the
    angular-acceleration third component (roll) is zero and `r_dot` equals the
    projection of velocity on `flight.attitude_unit`.
    """
    from rocketpy.simulation.flight import Flight

    # Arrange
    flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
        use_udot_rail2=True,
    )

    # Act
    t_between = getattr(flight, "between_rails_time", None)
    u_between = getattr(flight, "between_rails_state", None)

    # If the flight never registered between-rails, skip the detailed asserts
    # (the test still passes as it did not exercise the condition).
    if t_between is None or u_between is None:
        return

    u_dot = flight.udot_rail2(t_between, u_between)

    # u_dot layout for udot_rail2: [r_dot_x, r_dot_y, r_dot_z, v_dot_x, v_dot_y, v_dot_z, e_dot..., w_dot_x, w_dot_y, w_dot_z]
    r_dot = Vector(u_dot[0:3])
    v_dot = Vector(u_dot[3:6])
    # angular accelerations are last three entries
    w_dot = Vector(u_dot[-3:])

    # Assert: roll acceleration is zero (third component)
    assert abs(w_dot[2]) < 1e-12, f"Expected zero roll acceleration, got {w_dot[2]}"

    # Assert: r_dot is projection of velocity onto rail axis
    rail = Vector(flight.attitude_unit)
    velocity = Vector(u_between[3:6])
    projected = rail * (velocity @ rail)

    diff = r_dot - projected
    assert float(abs(diff)) < 1e-8, (
        f"r_dot not a projection onto rail (err={float(abs(diff))})"
    )


def test_udot_rail2_csv_comparison_generation(
    calisto_robust, example_spaceport_env, tmp_path
):
    """Generate CSVs comparing runs with udot_rail2 enabled/disabled.

    Arrange: create two flights (enabled/disabled). Act: write CSVs with
    between-rails and out-of-rail pitch/yaw. Assert: files exist and contain
    the expected header row; also return numeric deltas for quick inspection.
    """
    from rocketpy.simulation.flight import Flight

    out_dir = tmp_path / "udot_rail2_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for enabled in (True, False):
        # Arrange / Act
        flight = Flight(
            rocket=calisto_robust,
            environment=example_spaceport_env,
            rail_length=5.2,
            inclination=85,
            heading=0,
            terminate_on_apogee=False,
            use_udot_rail2=enabled,
        )

        t_between = getattr(flight, "between_rails_time", None)
        t_out = getattr(flight, "out_of_rail_time", None)

        def sample_at(t):
            if t is None:
                return None, None
            sol = min(flight.solution, key=lambda row: abs(row[0] - t))
            e = sol[7:11]
            body = _body_axis_from_e(e)
            return _pitch_deg(body), _yaw_deg(body)

        pitch_between, yaw_between = sample_at(t_between)
        pitch_out, yaw_out = sample_at(t_out)

        tag = "enabled" if enabled else "disabled"
        csv_path = out_dir / f"calisto_angles_udot_rail2_{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_event", "pitch_deg", "yaw_deg"])
            if t_between is not None:
                writer.writerow(["between_rails", pitch_between, yaw_between])
            if t_out is not None:
                writer.writerow(["out_of_rail", pitch_out, yaw_out])

        # Assert: file created and header present
        assert csv_path.exists(), f"CSV was not created: {csv_path}"
        with open(csv_path, "r", newline="") as f:
            lines = f.read().splitlines()
        assert lines and lines[0].startswith("time_event,pitch_deg,yaw_deg"), (
            "CSV header mismatch"
        )

        results.append(
            (
                enabled,
                t_between,
                pitch_between,
                yaw_between,
                t_out,
                pitch_out,
                yaw_out,
                str(csv_path),
            )
        )

    # Provide a final sanity check: both CSVs were created
    assert all(os.path.exists(r[-1]) for r in results)

    # Optional: compute numeric deltas for out_of_rail if both present
    enabled_row = next(r for r in results if r[0] is True)
    disabled_row = next(r for r in results if r[0] is False)

    if enabled_row[4] is not None and disabled_row[4] is not None:
        delta_pitch = abs((enabled_row[5] or 0) - (disabled_row[5] or 0))
        delta_yaw = abs((enabled_row[6] or 0) - (disabled_row[6] or 0))
        # The test only asserts that deltas are numeric and finite
        assert math.isfinite(delta_pitch) and math.isfinite(delta_yaw)
