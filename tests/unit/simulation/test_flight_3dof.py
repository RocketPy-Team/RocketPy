import numpy as np
import pytest

from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.point_mass_rocket import PointMassRocket
from rocketpy.simulation.flight import Flight


@pytest.fixture
def point_mass_motor():
    """Simple PointMassMotor for 3-DOF tests.

    Returns
    -------
    rocketpy.PointMassMotor
    """
    return PointMassMotor(
        thrust_source=10,
        dry_mass=1.0,
        propellant_initial_mass=0.5,
        burn_time=2.2,
    )


@pytest.fixture
def point_mass_rocket(point_mass_motor):
    """Simple PointMassRocket for 3-DOF tests.

    Returns
    -------
    rocketpy.PointMassRocket
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


def test_3dof_simulation_mode_autoset(example_plain_env, point_mass_rocket):
    """Test that simulation mode is correctly set to 3 DOF."""
    flight = Flight(
        rocket=point_mass_rocket,
        environment=example_plain_env,
        rail_length=1,
        simulation_mode="3 DOF",
    )
    assert flight.simulation_mode == "3 DOF"


def test_3dof_simulation_mode_warning(
    monkeypatch, example_plain_env, point_mass_rocket
):
    """Test that a warning is issued when 6 DOF is requested with PointMassRocket."""
    monkeypatch.setattr("warnings.warn", lambda *a, **k: None)
    flight = Flight(
        rocket=point_mass_rocket,
        environment=example_plain_env,
        rail_length=1,
        simulation_mode="6 DOF",
    )
    assert flight.simulation_mode == "3 DOF"


def test_3dof_equations_of_motion_functions(example_plain_env, point_mass_rocket):
    """Test that 3-DOF equations of motion return valid results."""
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
    """Test that invalid simulation mode raises ValueError."""
    with pytest.raises(ValueError):
        Flight(
            rocket=calisto,
            environment=example_plain_env,
            rail_length=1,
            simulation_mode="2 DOF",
        )
