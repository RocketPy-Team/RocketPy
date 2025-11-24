import pytest
import numpy as np
from rocketpy.simulation.flight import Flight
from rocketpy.rocket.rocket import PointMassRocket
from rocketpy.motors.point_mass_motor import PointMassMotor

class DummyEnv:
    # Minimal stub, adapt with real Environment in your RocketPy setup
    windvelocityx = windvelocityy = speedofsound = pressure = density = dynamicviscosity = lambda self: 0
    gravity = lambda self: 9.81
    elevation = 0
    def __getattr__(self, name):
        return lambda *a, **k: 0

def make_simple_3dof_components():
    env = DummyEnv()
    motor = PointMassMotor(10, dry_mass=1.0, propellant_initial_mass=0.5, burn_time=2.2)
    rocket = PointMassRocket(0.05, 2.0, 0.1, 0.5, 0.6)
    rocket.addmotor(motor, 0)
    return env, rocket

def test_3dof_simulation_mode_autoset():
    env, rocket = make_simple_3dof_components()
    flight = Flight(rocket=rocket, environment=env, rail_length=1, simulation_mode="3 DOF")
    assert flight.simulation_mode == "3 DOF"

def test_3dof_simulation_mode_warning(monkeypatch):
    env, rocket = make_simple_3dof_components()
    monkeypatch.setattr("warnings.warn", lambda *a, **k: None)
    f = Flight(rocket=rocket, environment=env, rail_length=1, simulation_mode="6 DOF")
    assert f.simulation_mode == "3 DOF"

def test_3dof_equations_of_motion_functions():
    env, rocket = make_simple_3dof_components()
    flight = Flight(rocket=rocket, environment=env, rail_length=1, simulation_mode="3 DOF")
    u = [0]*13  # Proper size for generalized state for 3/6 DOF probably
    result = flight.udotgeneralized3dof(0, u)
    assert isinstance(result, list) or isinstance(result, np.ndarray)

def test_invalid_simulation_mode():
    env, rocket = make_simple_3dof_components()
    with pytest.raises(ValueError):
        Flight(rocket=rocket, environment=env, rail_length=1, simulation_mode="2 DOF")
