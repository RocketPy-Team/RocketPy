"""Acceptance tests for 3 DOF flight simulation.

This module contains acceptance tests for validating the 3 DOF (Degrees of Freedom)
flight simulation mode in RocketPy. These tests ensure that the 3 DOF implementation
produces realistic and physically consistent results, including:

- Basic 3 DOF trajectory simulation
- Weathercocking behavior with different coefficients
- Comparison between 3 DOF and 6 DOF simulations
- Validation of key flight metrics

The tests use realistic rocket configurations and scenarios to ensure the
robustness of the 3 DOF implementation.

Note: These tests are designed for the 3 DOF feature implemented in issue #882.
They will be skipped until PointMassMotor and PointMassRocket are available.
All fixtures are defined in tests/fixtures/flight/flight_fixtures.py.
"""

import numpy as np
import pytest

from rocketpy import Flight

# Try to import 3DOF-specific classes, skip tests if not available
try:
    from rocketpy.motors.point_mass_motor import PointMassMotor
    from rocketpy.rocket.point_mass_rocket import PointMassRocket

    THREEDOF_AVAILABLE = True
except ImportError:
    THREEDOF_AVAILABLE = False

# Skip all tests in this module if 3DOF is not available
pytestmark = pytest.mark.skipif(
    not THREEDOF_AVAILABLE,
    reason="3 DOF feature (PointMassMotor, PointMassRocket) not yet available. "
    "These tests will be enabled when issue #882 is merged.",
)


def test_3dof_flight_basic_trajectory(flight_3dof_no_weathercock):
    """Test that 3 DOF flight produces reasonable trajectory.

    This test validates that the basic 3 DOF flight simulation produces
    physically reasonable results for key flight metrics.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Validate apogee is reasonable (between 500m and 3000m)
    apogee_altitude = flight.apogee - flight.env.elevation
    assert 500 < apogee_altitude < 3000, (
        f"Apogee altitude {apogee_altitude:.1f} m is outside expected range"
    )

    # Validate apogee time is reasonable (between 5s and 60s)
    assert 5 < flight.apogee_time < 60, (
        f"Apogee time {flight.apogee_time:.1f} s is outside expected range"
    )

    # Validate maximum velocity is reasonable (subsonic to low supersonic)
    max_velocity = flight.max_speed
    assert 50 < max_velocity < 400, (
        f"Maximum velocity {max_velocity:.1f} m/s is outside expected range"
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
    assert apogee_speed < 0.3 * max_speed, (
        f"Apogee speed {apogee_speed:.1f} m/s should be much less than "
        f"max speed {max_speed:.1f} m/s"
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
    assert lateral_displacement < 0.5 * apogee_altitude, (
        f"Lateral displacement {lateral_displacement:.1f} m seems too large "
        f"compared to apogee altitude {apogee_altitude:.1f} m"
    )


def test_3dof_weathercocking_affects_trajectory(
    flight_3dof_no_weathercock, flight_3dof_with_weathercock
):
    """Test that weathercocking affects the flight trajectory.

    This test validates that enabling weathercocking (quasi-static attitude
    adjustment) produces different trajectory results compared to fixed attitude.

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

    # They should be reasonably close but not identical
    apogee_difference = abs(apogee_no_wc - apogee_with_wc)
    assert apogee_difference > 0.1, "Weathercocking should affect apogee altitude"

    # Both should still be in reasonable range
    assert 500 < apogee_no_wc < 3000
    assert 500 < apogee_with_wc < 3000


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
    """Test that quaternions remain relatively fixed without weathercocking.

    Without weathercocking, the quaternions should not evolve significantly
    during flight.

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

    # Without weathercocking, quaternion change should be minimal
    # (allowing for some numerical drift)
    assert quat_change < 0.1, (
        f"Quaternion change {quat_change:.6f} is too large without weathercocking"
    )


def test_3dof_flight_mass_variation(flight_3dof_no_weathercock):
    """Test that rocket mass varies correctly during flight.

    The rocket mass should decrease during motor burn and remain constant
    after burnout.

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

    # Get mass after burnout (at t=5s, after 3.5s burn time)
    post_burn_mass = flight.rocket.total_mass(5.0)

    # Initial mass should be greater than mass during burn
    assert initial_mass > mass_during_burn, "Mass should decrease during burn"

    # Mass during burn should be greater than post-burn mass
    # (propellant is still being consumed)
    assert mass_during_burn > post_burn_mass, (
        "Mass should continue decreasing until burnout"
    )

    # Mass should remain constant after burnout
    # Check at t=6s and t=7s to verify constant mass
    mass_at_6s = flight.rocket.total_mass(6.0)
    mass_at_7s = flight.rocket.total_mass(7.0)
    assert abs(mass_at_6s - mass_at_7s) < 0.001, (
        "Mass should remain constant after burnout"
    )


def test_3dof_flight_thrust_profile(flight_3dof_no_weathercock):
    """Test that thrust profile is correct for point mass motor.

    For a constant thrust motor, thrust should be constant during burn
    and zero after burnout.

    Parameters
    ----------
    flight_3dof_no_weathercock : rocketpy.Flight
        A 3 DOF flight simulation without weathercocking.
    """
    flight = flight_3dof_no_weathercock

    # Get thrust during burn (at t=1s)
    thrust_during_burn = flight.rocket.motor.thrust(1.0)

    # Get thrust after burnout (at t=5s)
    thrust_after_burnout = flight.rocket.motor.thrust(5.0)

    # Thrust during burn should be positive
    assert thrust_during_burn > 0, "Thrust should be positive during burn"

    # Thrust after burnout should be zero
    assert abs(thrust_after_burnout) < 1e-6, "Thrust should be zero after burnout"


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
        inclination=85,
        heading=0,
        simulation_mode="3 DOF",
        weathercock_coeff=0.5,
    )

    flight2 = Flight(
        rocket=acceptance_point_mass_rocket,
        environment=example_spaceport_env,
        rail_length=5.0,
        inclination=85,
        heading=0,
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
    different but reasonable results.

    Parameters
    ----------
    example_spaceport_env : rocketpy.Environment
        Environment fixture for Spaceport America.
    acceptance_point_mass_rocket : rocketpy.PointMassRocket
        Rocket fixture for testing.
    """
    coefficients = [0.0, 0.5, 1.0, 2.0]
    flights = []

    for coeff in coefficients:
        flight = Flight(
            rocket=acceptance_point_mass_rocket,
            environment=example_spaceport_env,
            rail_length=5.0,
            inclination=85,
            heading=0,
            simulation_mode="3 DOF",
            weathercock_coeff=coeff,
        )
        flights.append(flight)

    # All flights should have reasonable apogees
    for flight, coeff in zip(flights, coefficients):
        apogee = flight.apogee - flight.env.elevation
        assert 500 < apogee < 3000, (
            f"Apogee {apogee:.1f} m with weathercock_coeff={coeff} is outside expected range"
        )

    # Apogees should vary with weathercock coefficient
    # Calculate the range of apogees to ensure they're different
    apogees = [f.apogee for f in flights]
    apogee_range = max(apogees) - min(apogees)
    assert apogee_range > 1.0, (
        f"Different weathercock coefficients should produce different apogees. "
        f"Range was only {apogee_range:.2f} m"
    )
