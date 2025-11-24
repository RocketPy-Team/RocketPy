import pytest
from rocketpy.motors.point_mass_motor import PointMassMotor

def test_init_required_args():
    m = PointMassMotor(thrust_source=10, dry_mass=1.0, propellant_initial_mass=0.5, burn_time=1.2)
    assert isinstance(m, PointMassMotor)
    assert m.dry_mass == 1.0
    assert m.propellant_initial_mass == 0.5

def test_missing_required_args_raises():
    with pytest.raises(ValueError):
        PointMassMotor(10, 1.0, None, 1)
    with pytest.raises(ValueError):
        PointMassMotor(10, 1.0, 1.2)
    with pytest.raises(TypeError):
        PointMassMotor([], 1.0, 1.2, 1)

def test_exhaustvelocity_and_totalmassflowrate():
    m = PointMassMotor(thrust_source=10, dry_mass=1.0, propellant_initial_mass=1.0, burn_time=2.0)
    ev_func = m.exhaustvelocity()
    assert hasattr(ev_func, 'getValue')
    tmf_func = m.totalmassflowrate
    assert hasattr(tmf_func, 'getValue')

def test_zero_inertias():
    m = PointMassMotor(thrust_source=10, dry_mass=1.0, propellant_initial_mass=1.0, burn_time=2.0)
    assert m.propellantI11().getValue(0) == 0
    assert m.propellantI22().getValue(0) == 0
    assert m.propellantI33().getValue(0) == 0

def test_callable_thrust():
    m = PointMassMotor(thrust_source=lambda t: 100*t, dry_mass=2, propellant_initial_mass=2, burn_time=4)
    assert m.thrust_source(0.5) == 50
