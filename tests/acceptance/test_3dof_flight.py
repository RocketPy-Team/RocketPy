"""Acceptance tests for 3 DOF flight simulation.

This module contains acceptance tests for validating the 3 DOF (Degrees of Freedom)
flight simulation mode in RocketPy. These tests ensure that the 3 DOF implementation
produces realistic and physically consistent results, including:

- Basic 3 DOF trajectory simulation
- Weathercocking behavior with different coefficients
- Validation of key flight metrics

The tests use realistic rocket configurations based on the Bella Lui rocket
to ensure the robustness of the 3 DOF implementation.

All fixtures are defined in tests/fixtures/flight/flight_fixtures.py.
"""

import numpy as np

from rocketpy import Flight
from tests.fixtures.flight.flight_fixtures import LAUNCH_HEADING, LAUNCH_INCLINATION

# Test tolerance constants
# Based on Bella Lui rocket performance (~459m apogee, K828FJ motor)
# Apogee range allows for variation in atmospheric conditions and drag models
MIN_APOGEE_ALTITUDE = 300  # meters - lower bound for point mass approximation
MAX_APOGEE_ALTITUDE = 600  # meters - upper bound considering Bella Lui achieves ~459m
MIN_APOGEE_TIME = 5  # seconds - minimum time to apogee
MAX_APOGEE_TIME = 30  # seconds - maximum time to apogee for this class of rocket
MIN_VELOCITY = 30  # m/s - minimum peak velocity
MAX_VELOCITY = 150  # m/s - maximum peak velocity (Bella Lui is subsonic)
APOGEE_SPEED_RATIO = 0.3  # Max ratio of apogee speed to max speed
MAX_LATERAL_TO_ALTITUDE_RATIO = 0.5  # Max lateral displacement vs altitude ratio
QUATERNION_CHANGE_TOLERANCE = 0.2  # Max quaternion change without weathercocking
# Note: Accounts for passive aerodynamic effects, numerical integration, and wind
WEATHERCOCK_COEFFICIENTS = [0.0, 0.5, 1.0, 2.0]  # Test weathercock coefficients
# Note: Weathercocking effects are verified by checking for changes in trajectory
# rather than specific tolerance values, as the magnitude is hard to quantify
# LAUNCH_INCLINATION and LAUNCH_HEADING imported from flight_fixtures
MASS_TOLERANCE = 0.001  # kg
THRUST_TOLERANCE = 1e-6  # N


def test_3dof_flight_basic_trajectory(flight_3dof_no_weathercock):
    """Test that 3 DOF flight produces reasonable trajectory.

    This test validates that the basic 3 DOF flight simulation produces
    physically reasonable results for key flight metrics using Bella Lui
    based rocket parameters.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Validate apogee is reasonable
    apogee_altitude = flight.apogee - flight.env.elevation
    assert MIN_APOGEE_ALTITUDE < apogee_altitude < MAX_APOGEE_ALTITUDE, (
        f"Apogee altitude {apogee_altitude:.1f} m is outside expected range "
        f"[{MIN_APOGEE_ALTITUDE}, {MAX_APOGEE_ALTITUDE}]"
    )

    # Validate apogee time is reasonable
    assert MIN_APOGEE_TIME < flight.apogee_time < MAX_APOGEE_TIME, (
        f"Apogee time {flight.apogee_time:.1f} s is outside expected range "
        f"[{MIN_APOGEE_TIME}, {MAX_APOGEE_TIME}]"
    )

    # Validate maximum velocity is reasonable (subsonic)
    max_velocity = flight.max_speed
    assert MIN_VELOCITY < max_velocity < MAX_VELOCITY, (
        f"Maximum velocity {max_velocity:.1f} m/s is outside expected range "
        f"[{MIN_VELOCITY}, {MAX_VELOCITY}]"
    )

    # Validate impact velocity is reasonable (with no parachute, terminal velocity)
    impact_velocity = abs(flight.speed(flight.t_final))
    assert impact_velocity > 0, "Impact velocity should be positive"

    # Validate flight time is reasonable
    assert flight.t_final > flight.apogee_time, (
        "Total flight time should be greater than apogee time"
    )


def test_3dof_flight_energy_conservation(flight_3dof_no_weathercock):
    """Test energy conservation principles in 3 DOF flight.

    This test validates that the 3 DOF simulation respects basic energy
    conservation principles (accounting for drag losses).

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # At apogee, kinetic energy should be minimal (mainly horizontal)
    apogee_speed = flight.speed(flight.apogee_time)
    max_speed = flight.max_speed

    # Apogee speed should be significantly less than max speed
    assert apogee_speed < APOGEE_SPEED_RATIO * max_speed, (
        f"Apogee speed {apogee_speed:.1f} m/s should be much less than "
        f"max speed {max_speed:.1f} m/s (ratio < {APOGEE_SPEED_RATIO})"
    )

    # Potential energy at apogee should be positive
    apogee_altitude = flight.apogee - flight.env.elevation
    assert apogee_altitude > 0, "Apogee altitude should be positive"


def test_3dof_flight_lateral_motion_no_weathercock(flight_3dof_no_weathercock):
    """Test lateral motion in 3 DOF flight without weathercocking.

    Without weathercocking, the rocket should maintain a relatively fixed
    attitude, but still have lateral motion due to the inclined launch.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Calculate lateral displacement at apogee
    x_apogee = flight.x(flight.apogee_time)
    y_apogee = flight.y(flight.apogee_time)
    lateral_displacement = np.sqrt(x_apogee**2 + y_apogee**2)

    # With 85 degree inclination (5 degrees from vertical), we expect some
    # lateral displacement due to the inclined launch
    assert lateral_displacement > 0, "Lateral displacement should be positive"

    # Lateral displacement should be reasonable (not too extreme)
    apogee_altitude = flight.apogee - flight.env.elevation
    assert lateral_displacement < MAX_LATERAL_TO_ALTITUDE_RATIO * apogee_altitude, (
        f"Lateral displacement {lateral_displacement:.1f} m seems too large "
        f"compared to apogee altitude {apogee_altitude:.1f} m "
        f"(ratio > {MAX_LATERAL_TO_ALTITUDE_RATIO})"
    )


def test_3dof_weathercocking_affects_trajectory(
    flight_3dof_no_weathercock, flight_3dof_with_weathercock
):
    """Test that weathercocking affects the flight trajectory.

    This test validates that enabling weathercocking (quasi-static attitude
    adjustment) produces different trajectory results compared to fixed attitude.
    Rather than checking for specific tolerance values, we verify that the
    physics implementation is working by checking if properties change.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    flight_3dof_with_weathercock : rocketpy.Flight
        A 3 DOF flight simulation with weathercocking enabled.
    """
    flight_no_wc = flight_3dof_no_weathercock
    flight_with_wc = flight_3dof_with_weathercock

    # Apogees should be different (weathercocking affects drag and lift)
    apogee_no_wc = flight_no_wc.apogee - flight_no_wc.env.elevation
    apogee_with_wc = flight_with_wc.apogee - flight_with_wc.env.elevation

    # Verify that weathercocking causes a change in trajectory
    # We don't specify how much change, just that there IS a change
    assert apogee_no_wc != apogee_with_wc, (
        "Weathercocking should affect apogee altitude. "
        f"Got same value: {apogee_no_wc:.2f} m for both simulations."
    )

    # Both should still be in reasonable range
    assert MIN_APOGEE_ALTITUDE < apogee_no_wc < MAX_APOGEE_ALTITUDE
    assert MIN_APOGEE_ALTITUDE < apogee_with_wc < MAX_APOGEE_ALTITUDE

    # Verify lateral displacement is also affected
    x_no_wc = flight_no_wc.x(flight_no_wc.apogee_time)
    y_no_wc = flight_no_wc.y(flight_no_wc.apogee_time)
    lateral_no_wc = (x_no_wc**2 + y_no_wc**2) ** 0.5

    x_with_wc = flight_with_wc.x(flight_with_wc.apogee_time)
    y_with_wc = flight_with_wc.y(flight_with_wc.apogee_time)
    lateral_with_wc = (x_with_wc**2 + y_with_wc**2) ** 0.5

    # Weathercocking should cause different lateral displacement
    assert lateral_no_wc != lateral_with_wc, (
        "Weathercocking should affect lateral displacement. "
        f"Got same value: {lateral_no_wc:.2f} m for both simulations."
    )


def test_3dof_weathercocking_coefficient_stored(flight_3dof_with_weathercock):
    """Test that weathercock coefficient is correctly stored.

    Parameters
    ----------
    flight_3dof_with_weathercock : rocketpy.Flight
        A 3 DOF flight simulation with weathercocking enabled.
    """
    assert flight_3dof_with_weathercock.weathercock_coeff == 1.0


def test_3dof_flight_post_processing_attributes(flight_3dof_no_weathercock):
    """Test that 3 DOF flight has necessary post-processing attributes.

    This test ensures that all necessary flight data and post-processing
    attributes are available after a 3 DOF simulation.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Check for essential position attributes
    assert hasattr(flight, "x"), "Flight should have x position attribute"
    assert hasattr(flight, "y"), "Flight should have y position attribute"
    assert hasattr(flight, "z"), "Flight should have z position attribute"

    # Check for essential velocity attributes
    assert hasattr(flight, "vx"), "Flight should have vx velocity attribute"
    assert hasattr(flight, "vy"), "Flight should have vy velocity attribute"
    assert hasattr(flight, "vz"), "Flight should have vz velocity attribute"

    # Check for essential acceleration attributes
    assert hasattr(flight, "ax"), "Flight should have ax acceleration attribute"
    assert hasattr(flight, "ay"), "Flight should have ay acceleration attribute"
    assert hasattr(flight, "az"), "Flight should have az acceleration attribute"

    # Check for derived attributes
    assert hasattr(flight, "speed"), "Flight should have speed attribute"
    assert hasattr(flight, "max_speed"), "Flight should have max_speed attribute"
    assert hasattr(flight, "apogee"), "Flight should have apogee attribute"
    assert hasattr(flight, "apogee_time"), "Flight should have apogee_time attribute"

    # Check that these attributes can be called
    assert flight.x(0) is not None
    assert flight.speed(0) is not None


def test_3dof_flight_rail_exit_velocity(flight_3dof_no_weathercock):
    """Test that rail exit velocity is reasonable.

    The rocket should exit the rail with sufficient velocity to ensure
    stable flight.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Get velocity at rail exit (stored in out_of_rail_velocity)
    rail_exit_velocity = flight.out_of_rail_velocity

    # Rail exit velocity should be reasonable (typically 15-50 m/s)
    assert 10 < rail_exit_velocity < 100, (
        f"Rail exit velocity {rail_exit_velocity:.1f} m/s is outside expected range"
    )


def test_3dof_flight_quaternion_evolution_no_weathercock(flight_3dof_no_weathercock):
    """Test that quaternions remain relatively fixed without active weathercocking.

    Without active weathercocking, the quaternions should not evolve significantly
    during flight. Note that some quaternion evolution may still occur due to:
    - Passive aerodynamic effects
    - Numerical integration effects
    - Wind conditions in the environment

    This test verifies that without the active weathercocking model, the
    attitude changes remain within reasonable bounds.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Get quaternions at different times
    e0_initial = flight.e0(0)
    e1_initial = flight.e1(0)
    e2_initial = flight.e2(0)
    e3_initial = flight.e3(0)

    # Get quaternions at mid-flight
    mid_time = flight.apogee_time / 2
    e0_mid = flight.e0(mid_time)
    e1_mid = flight.e1(mid_time)
    e2_mid = flight.e2(mid_time)
    e3_mid = flight.e3(mid_time)

    # Calculate quaternion change magnitude
    quat_change = np.sqrt(
        (e0_mid - e0_initial) ** 2
        + (e1_mid - e1_initial) ** 2
        + (e2_mid - e2_initial) ** 2
        + (e3_mid - e3_initial) ** 2
    )

    # Without active weathercocking, quaternion change should be limited
    # Tolerance accounts for passive aerodynamic effects and numerical integration
    assert quat_change < QUATERNION_CHANGE_TOLERANCE, (
        f"Quaternion change {quat_change:.6f} exceeds expected bounds without "
        f"active weathercocking (tolerance: {QUATERNION_CHANGE_TOLERANCE}). "
        f"This may indicate unexpected attitude dynamics."
    )


def test_3dof_flight_mass_variation(flight_3dof_no_weathercock):
    """Test that rocket mass varies correctly during flight.

    The rocket mass should decrease during motor burn and remain constant
    after burnout. Based on Bella Lui motor with ~2.43s burn time.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Get initial mass (at t=0)
    initial_mass = flight.rocket.total_mass(0)

    # Get mass during burn (at t=1s)
    mass_during_burn = flight.rocket.total_mass(1.0)

    # Get mass after burnout (at t=4s, after ~2.43s burn time)
    post_burn_mass = flight.rocket.total_mass(4.0)

    # Initial mass should be greater than mass during burn
    assert initial_mass > mass_during_burn, "Mass should decrease during burn"

    # Mass during burn should be greater than post-burn mass
    # (propellant is still being consumed)
    assert mass_during_burn > post_burn_mass, (
        "Mass should continue decreasing until burnout"
    )

    # Mass should remain constant after burnout
    # Check at t=5s and t=6s to verify constant mass
    mass_at_5s = flight.rocket.total_mass(5.0)
    mass_at_6s = flight.rocket.total_mass(6.0)
    assert abs(mass_at_5s - mass_at_6s) < MASS_TOLERANCE, (
        f"Mass should remain constant after burnout (difference: {abs(mass_at_5s - mass_at_6s):.4f} kg, "
        f"tolerance: {MASS_TOLERANCE} kg)"
    )


def test_3dof_flight_thrust_profile(flight_3dof_no_weathercock):
    """Test that thrust profile is correct for point mass motor.

    For the Bella Lui motor (K828FJ), thrust should be positive during burn
    (~2.43s) and zero after burnout.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Get thrust during burn (at t=1s)
    thrust_during_burn = flight.rocket.motor.thrust(1.0)

    # Get thrust after burnout (at t=4s, after ~2.43s burn time)
    thrust_after_burnout = flight.rocket.motor.thrust(4.0)

    # Thrust during burn should be positive
    assert thrust_during_burn > 0, "Thrust should be positive during burn"

    # Thrust after burnout should be zero
    assert abs(thrust_after_burnout) < THRUST_TOLERANCE, (
        f"Thrust should be zero after burnout (got {thrust_after_burnout:.9f} N, "
        f"tolerance: {THRUST_TOLERANCE} N)"
    )


def test_3dof_flight_reproducibility(
    example_spaceport_env, acceptance_point_mass_rocket
):
    """Test that 3 DOF flights are reproducible.

    Running the same simulation multiple times should produce identical results.

    Parameters
    ----------
    example_spaceport_env : rocketpy.Environment
        Environment fixture for Spaceport America.
    acceptance_point_mass_rocket : rocketpy.PointMassRocket
        Rocket fixture for testing.
    """
    # Run simulation twice with same parameters
    flight1 = Flight(
        rocket=acceptance_point_mass_rocket,
        environment=example_spaceport_env,
        rail_length=5.0,
        inclination=LAUNCH_INCLINATION,
        heading=LAUNCH_HEADING,
        simulation_mode="3 DOF",
        weathercock_coeff=0.5,
    )

    flight2 = Flight(
        rocket=acceptance_point_mass_rocket,
        environment=example_spaceport_env,
        rail_length=5.0,
        inclination=LAUNCH_INCLINATION,
        heading=LAUNCH_HEADING,
        simulation_mode="3 DOF",
        weathercock_coeff=0.5,
    )

    # Results should be identical
    assert abs(flight1.apogee - flight2.apogee) < 1e-6, (
        "Apogee should be identical for same simulation"
    )
    assert abs(flight1.apogee_time - flight2.apogee_time) < 1e-6, (
        "Apogee time should be identical for same simulation"
    )
    assert abs(flight1.max_speed - flight2.max_speed) < 1e-6, (
        "Max speed should be identical for same simulation"
    )


def test_3dof_flight_different_weathercock_coefficients(
    example_spaceport_env, acceptance_point_mass_rocket
):
    """Test 3 DOF flight with various weathercock coefficients.

    This test validates that different weathercock coefficients produce
    different results, verifying the physics implementation rather than
    checking specific tolerance values.

    Parameters
    ----------
    example_spaceport_env : rocketpy.Environment
        Environment fixture for Spaceport America.
    acceptance_point_mass_rocket : rocketpy.PointMassRocket
        Rocket fixture for testing.
    """
    coefficients = WEATHERCOCK_COEFFICIENTS
    flights = []

    for coeff in coefficients:
        flight = Flight(
            rocket=acceptance_point_mass_rocket,
            environment=example_spaceport_env,
            rail_length=5.0,
            inclination=LAUNCH_INCLINATION,
            heading=LAUNCH_HEADING,
            simulation_mode="3 DOF",
            weathercock_coeff=coeff,
        )
        flights.append(flight)

    # All flights should have reasonable apogees
    for flight, coeff in zip(flights, coefficients):
        apogee = flight.apogee - flight.env.elevation
        assert MIN_APOGEE_ALTITUDE < apogee < MAX_APOGEE_ALTITUDE, (
            f"Apogee {apogee:.1f} m with weathercock_coeff={coeff} is outside expected range "
            f"[{MIN_APOGEE_ALTITUDE}, {MAX_APOGEE_ALTITUDE}]"
        )

    # Verify that different coefficients produce different results
    # This confirms the weathercocking physics is being applied
    apogees = [f.apogee for f in flights]

    # Check if there's meaningful variation in apogees (use range instead of set
    # to avoid floating-point precision issues)
    apogee_range = max(apogees) - min(apogees)
    apogee_tolerance = 0.01  # meters - meaningful physical difference
    assert apogee_range > apogee_tolerance, (
        f"Different weathercock coefficients should produce different apogees. "
        f"Range of apogees: {apogee_range:.4f} m (threshold: {apogee_tolerance} m)"
    )

    # Verify lateral displacements also vary with coefficients
    lateral_displacements = []
    for flight in flights:
        x = flight.x(flight.apogee_time)
        y = flight.y(flight.apogee_time)
        lateral = (x**2 + y**2) ** 0.5
        lateral_displacements.append(lateral)

    # Check if there's meaningful variation in lateral displacements
    lateral_tolerance = 0.001  # meters - meaningful physical difference
    lateral_range = max(lateral_displacements) - min(lateral_displacements)
    assert lateral_range > lateral_tolerance, (
        "Different weathercock coefficients should produce different lateral displacements. "
        f"Range of lateral displacements: {lateral_range:.4f} m (threshold: {lateral_tolerance} m)"
    )
