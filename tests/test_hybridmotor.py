from unittest.mock import patch

import numpy as np
import pytest
import scipy.integrate

from rocketpy import Function

thrust_function = lambda t: 2000 - 100 * t
burn_time = 10
center_of_dry_mass = 0
dry_inertia = (4, 4, 0.1)
dry_mass = 8
grain_density = 1700
grain_number = 4
grain_initial_height = 0.1
grain_separation = 0
grain_initial_inner_radius = 0.04
grain_outer_radius = 0.1
nozzle_position = -0.4
nozzle_radius = 0.07
grains_center_of_mass_position = -0.1
oxidizer_tank_position = 0.3


@patch("matplotlib.pyplot.show")
def test_hybrid_motor_info(mock_show, hybrid_motor):
    """Tests the HybridMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    """
    assert hybrid_motor.info() == None
    assert hybrid_motor.all_info() == None


def test_hybrid_motor_basic_parameters(hybrid_motor):
    """Tests the HybridMotor class construction parameters.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    """
    assert hybrid_motor.burn_time == (0, burn_time)
    assert hybrid_motor.dry_mass == dry_mass
    assert (
        hybrid_motor.dry_I_11,
        hybrid_motor.dry_I_22,
        hybrid_motor.dry_I_33,
    ) == dry_inertia
    assert hybrid_motor.center_of_dry_mass_position == center_of_dry_mass
    assert hybrid_motor.nozzle_position == nozzle_position
    assert hybrid_motor.nozzle_radius == nozzle_radius
    assert hybrid_motor.solid.grain_number == grain_number
    assert hybrid_motor.solid.grain_density == grain_density
    assert hybrid_motor.solid.grain_initial_height == grain_initial_height
    assert hybrid_motor.solid.grain_separation == grain_separation
    assert hybrid_motor.solid.grain_initial_inner_radius == grain_initial_inner_radius
    assert hybrid_motor.solid.grain_outer_radius == grain_outer_radius
    assert (
        hybrid_motor.solid.grains_center_of_mass_position
        == grains_center_of_mass_position
    )
    assert hybrid_motor.liquid.positioned_tanks[0]["position"] == 0.3


def test_hybrid_motor_thrust_parameters(hybrid_motor, spherical_oxidizer_tank):
    """Tests the HybridMotor class thrust parameters.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    spherical_oxidizer_tank : rocketpy.SphericalTank
        The SphericalTank object to be used in the tests.
    """
    # Function dependency for discretization validation
    expected_thrust = Function(thrust_function).set_discrete(0, 10, 50)
    expected_total_impulse = scipy.integrate.quad(expected_thrust, 0, 10)[0]

    initial_grain_mass = (
        grain_density
        * np.pi
        * (grain_outer_radius**2 - grain_initial_inner_radius**2)
        * grain_initial_height
        * grain_number
    )
    initial_oxidizer_mass = spherical_oxidizer_tank.fluid_mass(0)
    initial_mass = initial_grain_mass + initial_oxidizer_mass

    expected_exhaust_velocity = expected_total_impulse / initial_mass
    expected_mass_flow_rate = -expected_thrust / expected_exhaust_velocity
    expected_grain_mass_flow_rate = (
        expected_mass_flow_rate - spherical_oxidizer_tank.net_mass_flow_rate
    )

    assert pytest.approx(hybrid_motor.thrust.y_array) == expected_thrust.y_array
    assert pytest.approx(hybrid_motor.total_impulse) == expected_total_impulse
    assert pytest.approx(hybrid_motor.exhaust_velocity(0)) == expected_exhaust_velocity

    # Assert mass flow rate grain/oxidizer balance
    for t in np.linspace(0, 10, 100)[1:-1]:
        assert pytest.approx(
            hybrid_motor.total_mass_flow_rate(t)
        ) == expected_mass_flow_rate(t)
        assert pytest.approx(
            hybrid_motor.solid.mass_flow_rate(t)
        ) == expected_grain_mass_flow_rate(t)


def test_hybrid_motor_center_of_mass(hybrid_motor, spherical_oxidizer_tank):
    """Tests the HybridMotor class center of mass.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    spherical_oxidizer_tank : rocketpy.SphericalTank
        The SphericalTank object to be used in the tests.
    """
    oxidizer_mass = spherical_oxidizer_tank.fluid_mass
    grain_mass = hybrid_motor.solid.propellant_mass

    propellant_balance = grain_mass * grains_center_of_mass_position + oxidizer_mass * (
        oxidizer_tank_position + spherical_oxidizer_tank.center_of_mass
    )
    balance = propellant_balance + dry_mass * center_of_dry_mass

    propellant_center_of_mass = propellant_balance / (grain_mass + oxidizer_mass)
    center_of_mass = balance / (grain_mass + oxidizer_mass + dry_mass)

    for t in np.linspace(0, 100, 100):
        assert pytest.approx(
            hybrid_motor.center_of_propellant_mass(t)
        ) == propellant_center_of_mass(t)
        assert pytest.approx(hybrid_motor.center_of_mass(t)) == center_of_mass(t)


def test_hybrid_motor_inertia(hybrid_motor, spherical_oxidizer_tank):
    """Tests the HybridMotor class inertia.

    Parameters
    ----------
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    spherical_oxidizer_tank : rocketpy.SphericalTank
        The SphericalTank object to be used in the tests.
    """
    oxidizer_mass = spherical_oxidizer_tank.fluid_mass
    oxidizer_inertia = spherical_oxidizer_tank.inertia
    grain_mass = hybrid_motor.solid.propellant_mass
    grain_inertia = hybrid_motor.solid.propellant_I_11
    propellant_mass = oxidizer_mass + grain_mass

    # Validate parallel axis theorem translation
    grain_inertia += (
        grain_mass
        * (grains_center_of_mass_position - hybrid_motor.center_of_propellant_mass) ** 2
    )
    oxidizer_inertia += (
        oxidizer_mass
        * (
            oxidizer_tank_position
            + spherical_oxidizer_tank.center_of_mass
            - hybrid_motor.center_of_propellant_mass
        )
        ** 2
    )

    propellant_inertia = grain_inertia + oxidizer_inertia

    # Adding dry mass contributions
    inertia = (
        propellant_inertia
        + propellant_mass
        * (hybrid_motor.center_of_propellant_mass - hybrid_motor.center_of_mass) ** 2
        + dry_inertia[0]
        + dry_mass * (-hybrid_motor.center_of_mass + center_of_dry_mass) ** 2
    )

    for t in np.linspace(0, 100, 100):
        assert pytest.approx(hybrid_motor.propellant_I_11(t)) == propellant_inertia(t)
        assert pytest.approx(hybrid_motor.I_11(t)) == inertia(t)

        # Assert cylindrical symmetry
        assert pytest.approx(hybrid_motor.propellant_I_22(t)) == propellant_inertia(t)
