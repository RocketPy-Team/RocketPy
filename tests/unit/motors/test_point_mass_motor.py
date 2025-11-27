import pytest

from rocketpy.motors.point_mass_motor import PointMassMotor


def test_init_required_args():
    m = PointMassMotor(
        thrust_source=10, dry_mass=1.0, propellant_initial_mass=0.5, burn_time=1.2
    )
    assert isinstance(m, PointMassMotor)
    assert m.dry_mass == 1.0
    assert m.propellant_initial_mass == 0.5
    assert m.burn_time == (0, 1.2)  # burn_time is always a tuple (start, end)
    assert m.burn_duration == 1.2


def test_missing_required_args_raises():
    # TODO: in the future we would like to capture specific RocketPy Exceptions
    with pytest.raises(ValueError):
        PointMassMotor(10, 1.0, None, 1)
    with pytest.raises(ValueError):
        PointMassMotor(10, 1.0, 1.2)
    with pytest.raises(TypeError):
        PointMassMotor([], 1.0, 1.2, 1)


def test_exhaustvelocity_and_totalmassflowrate():
    m = PointMassMotor(
        thrust_source=10, dry_mass=1.0, propellant_initial_mass=1.0, burn_time=2.0
    )
    ev_func = m.exhaust_velocity
    assert hasattr(ev_func, "get_value")
    tmf_func = m.total_mass_flow_rate
    assert hasattr(tmf_func, "get_value")


def test_zero_inertias():
    m = PointMassMotor(
        thrust_source=10, dry_mass=1.0, propellant_initial_mass=1.0, burn_time=2.0
    )
    assert m.propellant_I_11.get_value(0) == 0
    assert m.propellant_I_22.get_value(0) == 0
    assert m.propellant_I_33.get_value(0) == 0


def test_callable_thrust():
    m = PointMassMotor(
        thrust_source=lambda t: 100 * t,
        dry_mass=2,
        propellant_initial_mass=2,
        burn_time=4,
    )
    assert m.thrust_source(0.5) == 50
