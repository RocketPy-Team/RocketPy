import numpy as np
import pytest

from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.point_mass_rocket import PointMassRocket
from rocketpy.simulation.flight import Flight


@pytest.fixture
def point_mass_motor():
    """Creates a simple PointMassMotor for 3-DOF flight tests.

    Returns
    -------
    rocketpy.PointMassMotor
        A point mass motor with constant thrust of 10 N, 1.0 kg dry mass,
        0.5 kg propellant mass, and 2.2 s burn time.
    """
    return PointMassMotor(
        thrust_source=10,
        dry_mass=1.0,
        propellant_initial_mass=0.5,
        burn_time=2.2,
    )


@pytest.fixture
def point_mass_rocket(point_mass_motor):
    """Creates a simple PointMassRocket for 3-DOF flight tests.

    Parameters
    ----------
    point_mass_motor : rocketpy.PointMassMotor
        The motor to be added to the rocket.

    Returns
    -------
    rocketpy.PointMassRocket
        A point mass rocket with 0.05 m radius, 2.0 kg mass, and the
        provided motor attached at position 0.
    """
    rocket = PointMassRocket(
        radius=0.05,
        mass=2.0,
        center_of_mass_without_motor=0.1,
        power_off_drag=0.5,
        power_on_drag=0.6,
    )
    rocket.add_motor(point_mass_motor, position=0)
    return rocket


@pytest.fixture
def flight_weathercock_zero(example_plain_env, point_mass_rocket):
    """Creates a Flight fixture with weathercock_coeff set to 0.0.

    Returns
    -------
    rocketpy.simulation.flight.Flight
        A Flight object configured for 3-DOF with zero weathercock coefficient.
    """
    return Flight(
        rocket=point_mass_rocket,
        environment=example_plain_env,
        rail_length=1,
        simulation_mode="3 DOF",
        weathercock_coeff=0.0,
    )


@pytest.fixture
def flight_3dof(example_plain_env, point_mass_rocket):
    """Creates a standard 3-DOF Flight fixture with default weathercock_coeff=0.0.

    Returns
    -------
    rocketpy.simulation.flight.Flight
        A Flight object configured for 3-DOF with default weathercock coefficient.
    """
    return Flight(
        rocket=point_mass_rocket,
        environment=example_plain_env,
        rail_length=1,
        simulation_mode="3 DOF",
    )


@pytest.fixture
def flight_weathercock_pos(example_plain_env, point_mass_rocket):
    """Creates a Flight fixture with weathercock_coeff set to 1.0.

    Returns
    -------
    rocketpy.simulation.flight.Flight
        A Flight object configured for 3-DOF with weathercocking enabled.
    """
    return Flight(
        rocket=point_mass_rocket,
        environment=example_plain_env,
        rail_length=1,
        simulation_mode="3 DOF",
        weathercock_coeff=1.0,
    )


def test_simulation_mode_sets_3dof_with_point_mass_rocket(flight_3dof):
    """Tests that simulation mode is correctly set to 3 DOF for PointMassRocket.

    Parameters
    ----------
    flight_3dof : rocketpy.simulation.flight.Flight
        A Flight fixture configured for 3-DOF simulation with a PointMassRocket.
    """
    assert flight_3dof.simulation_mode == "3 DOF"


def test_3dof_simulation_mode_warning(example_plain_env, point_mass_rocket):
    """Tests that a warning is issued when 6 DOF mode is requested with PointMassRocket.

    When a PointMassRocket is used with simulation_mode="6 DOF", the Flight
    class should emit a UserWarning and automatically switch to 3 DOF mode.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
        A basic environment fixture for flight simulation.
    point_mass_rocket : rocketpy.PointMassRocket
        A point mass rocket fixture for 3-DOF simulation.
    """
    with pytest.warns(UserWarning):
        flight = Flight(
            rocket=point_mass_rocket,
            environment=example_plain_env,
            rail_length=1,
            simulation_mode="6 DOF",
        )
        assert flight.simulation_mode == "3 DOF"


def test_u_dot_generalized_3dof_returns_valid_result(flight_3dof):
    """Tests that 3-DOF equations of motion return valid derivative results.

    Verifies that the u_dot_generalized_3dof method returns a list or numpy
    array representing the state derivative vector.

    """
    flight = flight_3dof
    u = [0] * 13  # Generalized state vector size
    result = flight.u_dot_generalized_3dof(0, u)
    assert isinstance(result, (list, np.ndarray))


def test_invalid_simulation_mode(example_plain_env, calisto):
    """Tests that invalid simulation mode raises ValueError.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
        A basic environment fixture for flight simulation.
    calisto : rocketpy.Rocket
        The Calisto rocket fixture from the test suite.
    """
    with pytest.raises(ValueError):
        Flight(
            rocket=calisto,
            environment=example_plain_env,
            rail_length=1,
            simulation_mode="2 DOF",
        )


def test_weathercock_coeff_stored(example_plain_env, point_mass_rocket):
    """Tests that the weathercock_coeff parameter is correctly stored.

    Parameters
    ----------
    example_plain_env : rocketpy.Environment
        A basic environment fixture for flight simulation.
    point_mass_rocket : rocketpy.PointMassRocket
        A point mass rocket fixture for 3-DOF simulation.
    """
    flight = Flight(
        rocket=point_mass_rocket,
        environment=example_plain_env,
        rail_length=1,
        simulation_mode="3 DOF",
        weathercock_coeff=2.5,
    )
    assert flight.weathercock_coeff == 2.5


def test_weathercock_coeff_default(flight_3dof):
    """Tests that the default weathercock_coeff is 0.0 (no weathercocking).

    Parameters
    ----------
    flight_3dof : rocketpy.Flight
        A Flight object for a 3-DOF simulation, provided by the flight_3dof fixture.
    """
    assert flight_3dof.weathercock_coeff == 0.0


def test_weathercock_zero_gives_fixed_attitude(flight_weathercock_zero):
    """Tests that weathercock_coeff=0 results in fixed attitude (no quaternion change).
    When weathercock_coeff is 0, the quaternion derivatives should be zero,
    meaning the attitude does not evolve.

    Parameters
    ----------
    flight_weathercock_zero : rocketpy.simulation.Flight
        A Flight fixture with weathercock_coeff set to 0. Used to verify that
        the attitude (quaternion) does not evolve when weathercocking is disabled.
    """
    flight = flight_weathercock_zero
    # Create a state vector with non-zero velocity (to have freestream)
    # [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    u = [0, 0, 100, 10, 5, 50, 1, 0, 0, 0, 0, 0, 0]
    result = flight.u_dot_generalized_3dof(0, u)

    # Quaternion derivatives (indices 6-9) should be zero
    e_dot = result[6:10]
    assert all(abs(ed) < 1e-10 for ed in e_dot), "Quaternion derivatives should be zero"


def test_weathercock_nonzero_evolves_attitude(flight_weathercock_pos):
    """Tests that non-zero weathercock_coeff causes attitude evolution.
    When the body axis is misaligned with the relative wind and weathercock_coeff
    is positive, the quaternion derivatives should be non-zero.

    Parameters
    ----------
    flight_weathercock_pos : rocketpy.simulation.Flight
        A Flight fixture with a positive weathercock coefficient for 3-DOF simulation.
    """
    flight = flight_weathercock_pos
    # Create a state with misaligned body axis
    # Body pointing straight up (e0=1, e1=e2=e3=0) but velocity is horizontal
    # [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    u = [0, 0, 100, 50, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    result = flight.u_dot_generalized_3dof(0, u)

    # With misalignment, quaternion derivatives should be non-zero
    e_dot = result[6:10]
    e_dot_magnitude = sum(ed**2 for ed in e_dot) ** 0.5
    assert e_dot_magnitude > 1e-6, "Quaternion derivatives should be non-zero"


def test_weathercock_aligned_no_evolution(flight_weathercock_pos):
    """Tests that when body axis is aligned with relative wind, no rotation occurs.
    When the rocket's body z-axis is already aligned with the negative of the
    freestream velocity, the quaternion derivatives should be approximately zero.

    Parameters
    ----------
    flight_weathercock_pos : rocketpy.Flight
        A Flight fixture configured for weathercocking tests with a nonzero initial angle.
    """
    flight = flight_weathercock_pos
    # Body pointing in +x direction (into the wind for vx=50)
    # Quaternion for 90 degree rotation about y-axis uses half-angle:
    # e0=cos(90°/2)=cos(45°), e2=sin(90°/2)=sin(45°)
    sqrt2_2 = np.sqrt(2) / 2
    # [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    u = [0, 0, 100, 50, 0, 0, sqrt2_2, 0, sqrt2_2, 0, 0, 0, 0]
    result = flight.u_dot_generalized_3dof(0, u)

    # With alignment, quaternion derivatives should be very small
    e_dot = result[6:10]
    e_dot_magnitude = sum(ed**2 for ed in e_dot) ** 0.5
    assert e_dot_magnitude < 1e-8, (
        "Quaternion derivatives should be very small when aligned"
    )


def test_weathercock_anti_aligned_uses_perp_axis_and_evolves(flight_weathercock_pos):
    """Tests the anti-aligned case where body z-axis is opposite freestream.

    This should exercise the branch that selects a perpendicular axis (y-axis)
    when the cross with x-axis is nearly zero, producing a non-zero quaternion
    derivative.
    """
    flight = flight_weathercock_pos

    sqrt2_2 = np.sqrt(2) / 2
    # Build quaternion that makes body z-axis = [-1, 0, 0]
    # This corresponds to a -90 deg rotation about the y-axis: e0=cos(45°), e2=-sin(45°)
    e0 = sqrt2_2
    e1 = 0
    e2 = -sqrt2_2
    e3 = 0

    # State: [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    # Set velocity so desired_direction becomes [1,0,0]
    u = [0, 0, 100, 50, 0, 0, e0, e1, e2, e3, 0, 0, 0]

    result = flight.u_dot_generalized_3dof(0, u)

    # Quaternion derivatives (indices 6-9) should be non-zero in anti-aligned case
    e_dot = result[6:10]
    e_dot_magnitude = sum(ed**2 for ed in e_dot) ** 0.5
    assert e_dot_magnitude > 1e-6, (
        "Quaternion derivatives should be non-zero for anti-aligned"
    )
