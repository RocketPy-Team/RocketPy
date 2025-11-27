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


def test_simulation_mode_sets_3dof_with_point_mass_rocket(
    example_plain_env, point_mass_rocket
):
    """Tests that simulation mode is correctly set to 3 DOF for PointMassRocket.

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
    )
    assert flight.simulation_mode == "3 DOF"


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


def test_u_dot_generalized_3dof_returns_valid_result(
    example_plain_env, point_mass_rocket
):
    """Tests that 3-DOF equations of motion return valid derivative results.

    Verifies that the u_dot_generalized_3dof method returns a list or numpy
    array representing the state derivative vector.

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
    )
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
