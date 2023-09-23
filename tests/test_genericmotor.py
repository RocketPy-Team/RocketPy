from unittest.mock import patch

import numpy as np
import pytest
import scipy.integrate

burn_time = (2, 7)
thrust_source = lambda t: 2000 - 100 * (t - 2)
chamber_height = 0.5
chamber_radius = 0.075
chamber_position = -0.25
propellant_initial_mass = 5.0
nozzle_position = -0.5
nozzle_radius = 0.075
dry_mass = 8.0
dry_inertia = (0.2, 0.2, 0.08)


@patch("matplotlib.pyplot.show")
def test_generic_motor_info(mock_show, generic_motor):
    """Tests the GenericMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    assert generic_motor.info() == None
    assert generic_motor.all_info() == None


def test_generic_motor_basic_parameters(generic_motor):
    """Tests the GenericMotor class construction parameters.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    assert generic_motor.burn_time == burn_time
    assert generic_motor.dry_mass == dry_mass
    assert (
        generic_motor.dry_I_11,
        generic_motor.dry_I_22,
        generic_motor.dry_I_33,
    ) == dry_inertia
    assert generic_motor.nozzle_position == nozzle_position
    assert generic_motor.nozzle_radius == nozzle_radius
    assert generic_motor.chamber_position == chamber_position
    assert generic_motor.chamber_radius == chamber_radius
    assert generic_motor.chamber_height == chamber_height
    assert generic_motor.propellant_initial_mass == propellant_initial_mass


def test_generic_motor_thrust_parameters(generic_motor):
    """Tests the GenericMotor thrust parameters.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    expected_thrust = np.array(
        [(t, thrust_source(t)) for t in np.linspace(*burn_time, 50)]
    )
    expected_total_impulse = scipy.integrate.trapezoid(
        expected_thrust[:, 1], expected_thrust[:, 0]
    )
    expected_exhaust_velocity = expected_total_impulse / propellant_initial_mass
    expected_mass_flow_rate = -1 * expected_thrust[:, 1] / expected_exhaust_velocity

    # Discretize mass flow rate for testing purposes
    mass_flow_rate = generic_motor.total_mass_flow_rate.set_discrete(*burn_time, 50)

    assert generic_motor.thrust.y_array == pytest.approx(expected_thrust[:, 1])
    assert generic_motor.total_impulse == pytest.approx(expected_total_impulse)
    assert generic_motor.exhaust_velocity.average(*burn_time) == pytest.approx(
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
    generic_motor.center_of_propellant_mass.set_discrete(*burn_time, 50)
    generic_motor.center_of_mass.set_discrete(*burn_time, 50)

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
    propellant_mass = generic_motor.propellant_mass.set_discrete(*burn_time, 50).y_array

    propellant_I_11 = propellant_mass * (
        chamber_radius**2 / 4 + chamber_height**2 / 12
    )
    propellant_I_22 = propellant_I_11
    propellant_I_33 = propellant_mass * (chamber_radius**2 / 2)

    # Centers of mass coincide, so no translation is needed
    I_11 = propellant_I_11 + dry_inertia[0]
    I_22 = propellant_I_22 + dry_inertia[1]
    I_33 = propellant_I_33 + dry_inertia[2]

    # Discretize inertia for testing purposes
    generic_motor.propellant_I_11.set_discrete(*burn_time, 50)
    generic_motor.propellant_I_22.set_discrete(*burn_time, 50)
    generic_motor.propellant_I_33.set_discrete(*burn_time, 50)
    generic_motor.I_11.set_discrete(*burn_time, 50)
    generic_motor.I_22.set_discrete(*burn_time, 50)
    generic_motor.I_33.set_discrete(*burn_time, 50)

    assert generic_motor.propellant_I_11.y_array == pytest.approx(propellant_I_11)
    assert generic_motor.propellant_I_22.y_array == pytest.approx(propellant_I_22)
    assert generic_motor.propellant_I_33.y_array == pytest.approx(propellant_I_33)
    assert generic_motor.I_11.y_array == pytest.approx(I_11)
    assert generic_motor.I_22.y_array == pytest.approx(I_22)
    assert generic_motor.I_33.y_array == pytest.approx(I_33)
