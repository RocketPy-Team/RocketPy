"""Integration tests for multi-dimensional drag coefficient support."""

import numpy as np

from rocketpy import Flight, Function, Rocket


def test_flight_with_1d_drag(flight_calisto):
    """Test that flights with 1D drag curves still work (backward compatibility)."""

    # `flight_calisto` is a fixture that already runs the simulation
    flight = flight_calisto

    # Check that flight completed successfully
    assert flight.t_final > 0
    assert flight.apogee > 0
    assert flight.apogee_time > 0


def test_flight_with_3d_drag_basic(example_plain_env, cesaroni_m1670):
    """Test that a simple 3D drag function works."""
    # Use fixtures for environment and motor
    env = example_plain_env
    env.set_atmospheric_model(type="standard_atmosphere")
    motor = cesaroni_m1670

    # Create 3D drag
    mach = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    reynolds = np.array([1e5, 5e5, 1e6])
    alpha = np.array([0.0, 2.0, 4.0, 6.0])

    M, Re, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    cd_data = 0.3 + 0.1 * M - 1e-7 * Re + 0.01 * A
    cd_data = np.clip(cd_data, 0.2, 1.0)

    power_off_drag = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )
    power_on_drag = Function.from_grid(
        cd_data * 1.1,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    # Create rocket
    rocket = Rocket(
        radius=0.0635,
        mass=16.24,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=power_off_drag,
        power_on_drag=power_on_drag,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.set_rail_buttons(0.2, -0.5, 30)
    rocket.add_motor(motor, position=-1.255)

    # Run flight
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    # Check results - should launch and have non-zero apogee
    assert flight.apogee > 100, f"Apogee too low: {flight.apogee}m"
    assert flight.apogee < 5000, f"Apogee too high: {flight.apogee}m"
    assert hasattr(flight, "angle_of_attack")


def test_3d_drag_with_varying_alpha():
    """Test that 3D drag responds to angle of attack changes.

    This test only verifies the Function mapping from alpha -> Cd. The
    integration-level comparison is placed in a separate test to keep each
    test function small and easier to lint/maintain.
    """
    # Create drag function with strong alpha dependency
    mach = np.array([0.0, 0.5, 1.0, 1.5])
    reynolds = np.array([1e5, 1e6])
    alpha = np.array([0.0, 5.0, 10.0, 15.0])

    M, _, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    # Strong alpha dependency: Cd increases significantly with alpha
    cd_data = 0.3 + 0.05 * M + 0.03 * A
    cd_data = np.clip(cd_data, 0.2, 2.0)

    drag_func = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    # Test at different angles of attack (direct function call)
    # At zero alpha, Cd should be lower
    cd_0 = drag_func(0.8, 5e5, 0.0)
    cd_10 = drag_func(0.8, 5e5, 10.0)

    # Cd should increase with alpha
    assert cd_10 > cd_0
    assert cd_10 - cd_0 > 0.2  # Should show significant difference


def test_flight_apogee_diff(flight_alpha, flight_flat):
    """Run paired flights (fixtures) and assert their apogees differ."""

    # Flights should both launch
    assert flight_alpha.apogee > 100
    assert flight_flat.apogee > 100

    # Apogees should differ
    assert flight_alpha.apogee != flight_flat.apogee


def test_flight_cd_sample_consistency(flight_alpha, flight_flat):
    """Sample Cd during a flight and ensure Cd difference matches apogee ordering.

    Uses the `flight_alpha` and `flight_flat` fixtures which provide paired
    flights constructed with alpha-dependent and alpha-averaged Cd functions.
    """

    # Sample a mid-ascent time and compare Cd evaluations
    speeds = flight_alpha.free_stream_speed[:, 1]
    idx_candidates = np.where(speeds > 5)[0]
    assert idx_candidates.size > 0
    idx = idx_candidates[len(idx_candidates) // 2]
    t_sample = flight_alpha.time[idx]

    mach_sample = flight_alpha.mach_number.get_value_opt(t_sample)
    v_sample = flight_alpha.free_stream_speed.get_value_opt(t_sample)
    reynolds_sample = (
        flight_alpha.density.get_value_opt(t_sample)
        * v_sample
        * (2 * flight_alpha.rocket.radius)
        / flight_alpha.dynamic_viscosity.get_value_opt(t_sample)
    )
    alpha_sample = flight_alpha.angle_of_attack.get_value_opt(t_sample)

    cd_alpha_sample = flight_alpha.rocket.power_on_drag.get_value_opt(
        mach_sample, reynolds_sample, alpha_sample
    )
    cd_flat_sample = flight_flat.rocket.power_on_drag.get_value_opt(
        mach_sample, reynolds_sample
    )

    assert cd_alpha_sample != cd_flat_sample
    if cd_alpha_sample > cd_flat_sample:
        assert flight_alpha.apogee < flight_flat.apogee
    else:
        assert flight_alpha.apogee > flight_flat.apogee
