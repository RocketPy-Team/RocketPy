from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.point_mass_rocket import PointMassRocket


def test_point_mass_rocket_basic_properties():
    motor = PointMassMotor(10, 1.0, 0.5, 1.0)
    rocket = PointMassRocket(
        radius=0.05,
        mass=2.0,
        center_of_mass_without_motor=0.1,
        power_off_drag=0.7,
        power_on_drag=0.8,
    )
    rocket.addmotor(motor, 0)
    assert rocket.radius == 0.05
    assert rocket.mass == 2.0
    assert rocket.motor is motor
    assert rocket.dryI11 == 0.0  # 3-DOF: inertias are forced zero


def test_structural_and_total_mass():
    """Test structural and total mass properties of point mass rocket."""
    motor = PointMassMotor(10, 1.0, 1.1, 2.0)
    rocket = PointMassRocket(
        radius=0.03,
        mass=2.5,
        center_of_mass_without_motor=0,
        power_off_drag=0.3,
        power_on_drag=0.4,
    )
    rocket.addmotor(motor, 0)

    # Test that structural mass and total mass are calculated correctly
    assert rocket.mass == 2.5
    expected_total_mass = rocket.mass + motor.propellant_initial_mass
    assert abs(rocket.total_mass(0) - expected_total_mass) < 1e-6


def test_add_motor_overwrites():
    """Test that adding a motor overwrites the previous motor."""
    motor1 = PointMassMotor(10, 1, 1.1, 2.0)
    motor2 = PointMassMotor(20, 2, 1.5, 3.0)
    rocket = PointMassRocket(0.02, 1.0, 0.0, 0.2, 0.5)
    rocket.addmotor(motor1, position=0)
    rocket.addmotor(motor2, position=0)
    assert rocket.motor == motor2
