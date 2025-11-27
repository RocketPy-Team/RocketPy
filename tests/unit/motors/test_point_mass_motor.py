import pytest

from rocketpy.motors.point_mass_motor import PointMassMotor


def test_init_required_args():
    """Tests that PointMassMotor initializes correctly with required arguments.

    Verifies that the motor is properly instantiated and that basic properties
    like dry_mass, propellant_initial_mass, and burn_time are correctly set.
    """
    m = PointMassMotor(
        thrust_source=10, dry_mass=1.0, propellant_initial_mass=0.5, burn_time=1.2
    )
    assert isinstance(m, PointMassMotor)
    assert m.dry_mass == 1.0
    assert m.propellant_initial_mass == 0.5
    assert m.burn_time == (0, 1.2)  # burn_time is always a tuple (start, end)
    assert m.burn_duration == 1.2


def test_missing_required_args_raises():
    """Tests that PointMassMotor raises errors for missing required arguments.

    Verifies that ValueError is raised when propellant_initial_mass is None
    or when burn_time is not provided with constant thrust. Also verifies
    TypeError is raised for invalid thrust_source types.
    """
    # TODO: in the future we would like to capture specific RocketPy Exceptions
    with pytest.raises(ValueError):
        PointMassMotor(10, 1.0, None, 1)
    with pytest.raises(ValueError):
        PointMassMotor(10, 1.0, 1.2)
    with pytest.raises(TypeError):
        PointMassMotor([], 1.0, 1.2, 1)


def test_exhaustvelocity_and_totalmassflowrate():
    """Tests that exhaust_velocity and total_mass_flow_rate return Function objects.

    Verifies that both properties are callable functions with get_value method,
    which is required for numerical evaluation during simulation.
    """
    m = PointMassMotor(
        thrust_source=10, dry_mass=1.0, propellant_initial_mass=1.0, burn_time=2.0
    )
    ev_func = m.exhaust_velocity
    assert hasattr(ev_func, "get_value")
    tmf_func = m.total_mass_flow_rate
    assert hasattr(tmf_func, "get_value")


def test_zero_inertias():
    """Tests that all propellant inertia components are zero for point mass motor.

    Verifies that propellant_I_11, propellant_I_22, and propellant_I_33 all
    return zero, as expected for a point mass model without rotational dynamics.
    """
    m = PointMassMotor(
        thrust_source=10, dry_mass=1.0, propellant_initial_mass=1.0, burn_time=2.0
    )
    assert m.propellant_I_11.get_value(0) == 0
    assert m.propellant_I_22.get_value(0) == 0
    assert m.propellant_I_33.get_value(0) == 0


def test_callable_thrust():
    """Tests that PointMassMotor accepts a callable function as thrust_source.

    Verifies that when a lambda function is used for thrust_source, the motor
    correctly evaluates thrust values at different times.
    """
    m = PointMassMotor(
        thrust_source=lambda t: 100 * t,
        dry_mass=2,
        propellant_initial_mass=2,
        burn_time=4,
    )
    assert m.thrust_source(0.5) == 50
