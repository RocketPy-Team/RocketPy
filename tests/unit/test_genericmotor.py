import numpy as np
import pytest
import scipy.integrate

BURN_TIME = (2, 7)


def thrust_source(t):
    return 2000 - 100 * (t - 2)


CHAMBER_HEIGHT = 0.5
CHAMBER_RADIUS = 0.075
CHAMBER_POSITION = -0.25
PROPELLANT_INITIAL_MASS = 5.0
NOZZLE_POSITION = -0.5
NOZZLE_RADIUS = 0.075
DRY_MASS = 8.0
DRY_INERTIA = (0.2, 0.2, 0.08)


def test_generic_motor_basic_parameters(generic_motor):
    """Tests the GenericMotor class construction parameters.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    assert generic_motor.burn_time == BURN_TIME
    assert generic_motor.dry_mass == DRY_MASS
    assert (
        generic_motor.dry_I_11,
        generic_motor.dry_I_22,
        generic_motor.dry_I_33,
    ) == DRY_INERTIA
    assert generic_motor.nozzle_position == NOZZLE_POSITION
    assert generic_motor.nozzle_radius == NOZZLE_RADIUS
    assert generic_motor.chamber_position == CHAMBER_POSITION
    assert generic_motor.chamber_radius == CHAMBER_RADIUS
    assert generic_motor.chamber_height == CHAMBER_HEIGHT
    assert generic_motor.propellant_initial_mass == PROPELLANT_INITIAL_MASS


def test_generic_motor_thrust_parameters(generic_motor):
    """Tests the GenericMotor thrust parameters.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    expected_thrust = np.array(
        [(t, thrust_source(t)) for t in np.linspace(*BURN_TIME, 50)]
    )
    expected_total_impulse = scipy.integrate.trapezoid(
        expected_thrust[:, 1], expected_thrust[:, 0]
    )
    expected_exhaust_velocity = expected_total_impulse / PROPELLANT_INITIAL_MASS
    expected_mass_flow_rate = -1 * expected_thrust[:, 1] / expected_exhaust_velocity

    # Discretize mass flow rate for testing purposes
    mass_flow_rate = generic_motor.total_mass_flow_rate.set_discrete(*BURN_TIME, 50)

    assert generic_motor.thrust.y_array == pytest.approx(expected_thrust[:, 1])
    assert generic_motor.total_impulse == pytest.approx(expected_total_impulse)
    assert generic_motor.exhaust_velocity.average(*BURN_TIME) == pytest.approx(
        expected_exhaust_velocity
    )
    assert mass_flow_rate.y_array == pytest.approx(expected_mass_flow_rate)


def test_generic_motor_center_of_mass(generic_motor):
    """Tests the GenericMotor center of mass.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    center_of_propellant_mass = -0.25
    center_of_dry_mass = -0.25
    center_of_mass = -0.25

    # Discretize center of mass for testing purposes
    generic_motor.center_of_propellant_mass.set_discrete(*BURN_TIME, 50)
    generic_motor.center_of_mass.set_discrete(*BURN_TIME, 50)

    assert generic_motor.center_of_propellant_mass.y_array == pytest.approx(
        center_of_propellant_mass
    )
    assert generic_motor.center_of_dry_mass_position == pytest.approx(
        center_of_dry_mass
    )
    assert generic_motor.center_of_mass.y_array == pytest.approx(center_of_mass)


def test_generic_motor_inertia(generic_motor):
    """Tests the GenericMotor inertia.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    # Tests the inertia formulation from the propellant mass
    propellant_mass = generic_motor.propellant_mass.set_discrete(*BURN_TIME, 50).y_array

    propellant_I_11 = propellant_mass * (CHAMBER_RADIUS**2 / 4 + CHAMBER_HEIGHT**2 / 12)
    propellant_I_22 = propellant_I_11
    propellant_I_33 = propellant_mass * (CHAMBER_RADIUS**2 / 2)

    # Centers of mass coincide, so no translation is needed
    I_11 = propellant_I_11 + DRY_INERTIA[0]
    I_22 = propellant_I_22 + DRY_INERTIA[1]
    I_33 = propellant_I_33 + DRY_INERTIA[2]

    # Discretize inertia for testing purposes
    generic_motor.propellant_I_11.set_discrete(*BURN_TIME, 50)
    generic_motor.propellant_I_22.set_discrete(*BURN_TIME, 50)
    generic_motor.propellant_I_33.set_discrete(*BURN_TIME, 50)
    generic_motor.I_11.set_discrete(*BURN_TIME, 50)
    generic_motor.I_22.set_discrete(*BURN_TIME, 50)
    generic_motor.I_33.set_discrete(*BURN_TIME, 50)

    assert generic_motor.propellant_I_11.y_array == pytest.approx(propellant_I_11)
    assert generic_motor.propellant_I_22.y_array == pytest.approx(propellant_I_22)
    assert generic_motor.propellant_I_33.y_array == pytest.approx(propellant_I_33)
    assert generic_motor.I_11.y_array == pytest.approx(I_11)
    assert generic_motor.I_22.y_array == pytest.approx(I_22)
    assert generic_motor.I_33.y_array == pytest.approx(I_33)
