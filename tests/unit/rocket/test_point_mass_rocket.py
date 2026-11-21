import json

import pytest

from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.point_mass_rocket import PointMassRocket


def test_init_sets_basic_properties_correctly():
    """Tests that PointMassRocket initializes with correct basic properties.

    Verifies that radius, mass, motor assignment, and zero inertias are
    correctly set when a PointMassRocket is created and a motor is added.
    """
    motor = PointMassMotor(10, 1.0, 0.5, 1.0)
    rocket = PointMassRocket(
        radius=0.05,
        mass=2.0,
        center_of_mass_without_motor=0.1,
        power_off_drag=0.7,
        power_on_drag=0.8,
    )
    rocket.add_motor(motor, 0)
    assert rocket.radius == 0.05
    assert rocket.mass == 2.0
    assert rocket.motor is motor
    assert rocket.dry_I_11 == 0.0  # 3-DOF: inertias are forced zero


def test_structural_and_total_mass():
    """Tests structural and total mass properties of PointMassRocket.

    Verifies that the rocket's structural mass and total mass (including motor
    dry mass and propellant) are calculated correctly at time t=0.
    """
    motor = PointMassMotor(10, 1.0, 1.1, 2.0)
    rocket = PointMassRocket(
        radius=0.03,
        mass=2.5,
        center_of_mass_without_motor=0,
        power_off_drag=0.3,
        power_on_drag=0.4,
    )
    rocket.add_motor(motor, 0)

    # Test that structural mass and total mass are calculated correctly
    assert rocket.mass == 2.5
    expected_total_mass = rocket.mass + motor.dry_mass + motor.propellant_initial_mass
    assert abs(rocket.total_mass(0) - expected_total_mass) < 1e-6


def test_add_motor_overwrites():
    """Tests that adding a new motor to PointMassRocket overwrites the previous motor.

    Verifies that when add_motor is called multiple times, only the most
    recently added motor is retained.
    """
    motor1 = PointMassMotor(10, 1, 1.1, 2.0)
    motor2 = PointMassMotor(20, 2, 1.5, 3.0)
    rocket = PointMassRocket(0.02, 1.0, 0.0, 0.2, 0.5)
    rocket.add_motor(motor1, position=0)
    rocket.add_motor(motor2, position=0)
    assert rocket.motor == motor2


@pytest.mark.parametrize("with_motor", [True, False])
def test_save_and_load_give_the_same_rocket(with_motor):
    """A point mass rocket is rebuilt with its own arguments, and keeps its
    weathercock coefficient and its motor."""
    rocket = PointMassRocket(
        radius=0.05,
        mass=2.0,
        center_of_mass_without_motor=0.1,
        power_off_drag=0.3,
        power_on_drag=0.4,
        weathercock_coeff=2.5,
    )
    if with_motor:
        motor = PointMassMotor(
            thrust_source=10, dry_mass=1.0, propellant_initial_mass=0.5, burn_time=1.2
        )
        rocket.add_motor(motor, position=0)

    loaded = json.loads(json.dumps(rocket, cls=RocketPyEncoder), cls=RocketPyDecoder)

    assert isinstance(loaded, PointMassRocket)
    assert loaded.radius == rocket.radius
    assert loaded.mass == rocket.mass
    assert loaded.weathercock_coeff == 2.5
    assert loaded.center_of_mass_without_motor == rocket.center_of_mass_without_motor
    assert loaded.power_off_drag(0.5) == pytest.approx(0.3)
    assert loaded.power_on_drag(0.5) == pytest.approx(0.4)
    assert (loaded.dry_I_11, loaded.dry_I_22, loaded.dry_I_33) == (0.0, 0.0, 0.0)
    if with_motor:
        assert isinstance(loaded.motor, PointMassMotor)
        for time in (0.0, 0.6, 1.2):
            assert loaded.total_mass(time) == pytest.approx(rocket.total_mass(time))


def test_a_file_saved_without_the_weathercock_coefficient_still_loads():
    rocket = PointMassRocket(0.05, 2.0, 0.0, 0.3, 0.4, weathercock_coeff=2.5)
    saved = json.loads(json.dumps(rocket, cls=RocketPyEncoder))
    del saved["weathercock_coeff"]

    loaded = json.loads(json.dumps(saved), cls=RocketPyDecoder)

    assert loaded.weathercock_coeff == 0.0
