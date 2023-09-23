from unittest.mock import patch

import numpy as np
import pytest
import scipy.integrate

from rocketpy import Function

burn_time = (8, 20)
dry_mass = 10
dry_inertia = (5, 5, 0.2)
center_of_dry_mass = 0
nozzle_position = -1.364
nozzle_radius = 0.069 / 2
pressurant_tank_position = 2.007
fuel_tank_position = -1.048
oxidizer_tank_position = 0.711


@patch("matplotlib.pyplot.show")
def test_liquid_motor_info(mock_show, liquid_motor):
    """Tests the LiquidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.info() == None
    assert liquid_motor.all_info() == None


def test_liquid_motor_basic_parameters(liquid_motor):
    """Tests the LiquidMotor class construction parameters.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.burn_time == burn_time
    assert liquid_motor.dry_mass == dry_mass
    assert (
        liquid_motor.dry_I_11,
        liquid_motor.dry_I_22,
        liquid_motor.dry_I_33,
    ) == dry_inertia
    assert liquid_motor.center_of_dry_mass_position == center_of_dry_mass
    assert liquid_motor.nozzle_position == nozzle_position
    assert liquid_motor.nozzle_radius == nozzle_radius
    assert liquid_motor.positioned_tanks[0]["position"] == pressurant_tank_position
    assert liquid_motor.positioned_tanks[1]["position"] == fuel_tank_position
    assert liquid_motor.positioned_tanks[2]["position"] == oxidizer_tank_position


def test_liquid_motor_thrust_parameters(
    liquid_motor, pressurant_tank, fuel_tank, oxidizer_tank
):
    """Tests the LiquidMotor class thrust parameters.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_tank : rocketpy.Tank
        The expected pressurant tank.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    """
    expected_thrust = np.loadtxt("data/SEBLM/test124_Thrust_Curve.csv", delimiter=",")
    expected_mass_flow = (
        pressurant_tank.net_mass_flow_rate
        + fuel_tank.net_mass_flow_rate
        + oxidizer_tank.net_mass_flow_rate
    )
    expected_total_impulse = scipy.integrate.trapezoid(
        expected_thrust[:, 1], expected_thrust[:, 0]
    )

    assert pytest.approx(liquid_motor.thrust.y_array) == expected_thrust[:, 1]
    assert (
        pytest.approx(liquid_motor.mass_flow_rate.y_array) == expected_mass_flow.y_array
    )
    assert pytest.approx(liquid_motor.total_impulse) == expected_total_impulse


def test_liquid_motor_mass_volume(
    liquid_motor,
    pressurant_fluid,
    fuel_tank,
    fuel_fluid,
    fuel_pressurant,
    oxidizer_tank,
    oxidizer_fluid,
    oxidizer_pressurant,
):
    """Tests the LiquidMotor class tanks flow and method values.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_fluid : rocketpy.Fluid
        The expected pressurant fluid.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    fuel_fluid : rocketpy.Fluid
        The expected fuel fluid.
    fuel_pressurant : rocketpy.Fluid
        The expected fuel pressurant.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    oxidizer_fluid : rocketpy.Fluid
        The expected oxidizer fluid.
    oxidizer_pressurant : rocketpy.Fluid
        The expected oxidizer pressurant.
    """
    test_pressurant_tank = liquid_motor.positioned_tanks[0]["tank"]
    test_fuel_tank = liquid_motor.positioned_tanks[1]["tank"]
    test_oxidizer_tank = liquid_motor.positioned_tanks[2]["tank"]

    # Test is Function dependent for discretization validation
    expected_pressurant_mass = Function("data/SEBLM/pressurantMassFiltered.csv")
    expected_pressurant_volume = expected_pressurant_mass / pressurant_fluid.density
    expected_fuel_volume = Function("data/SEBLM/test124_Propane_Volume.csv") * 1e-3
    expected_fuel_mass = (
        expected_fuel_volume * fuel_fluid.density
        + (-expected_fuel_volume + fuel_tank.geometry.total_volume)
        * fuel_pressurant.density
    )
    expected_oxidizer_volume = Function("data/SEBLM/test124_Lox_Volume.csv") * 1e-3
    expected_oxidizer_mass = (
        expected_oxidizer_volume * oxidizer_fluid.density
        + (-expected_oxidizer_volume + oxidizer_tank.geometry.total_volume)
        * oxidizer_pressurant.density
    )

    # Perform default discretization
    expected_pressurant_mass.set_discrete(*burn_time, 100)
    expected_fuel_mass.set_discrete(*burn_time, 100)
    expected_oxidizer_mass.set_discrete(*burn_time, 100)
    expected_pressurant_volume.set_discrete(*burn_time, 100)
    expected_fuel_volume.set_discrete(*burn_time, 100)
    expected_oxidizer_volume.set_discrete(*burn_time, 100)

    assert (
        pytest.approx(expected_pressurant_mass.y_array, 0.01)
        == test_pressurant_tank.fluid_mass.y_array
    )
    assert (
        pytest.approx(expected_fuel_mass.y_array, 0.01)
        == test_fuel_tank.fluid_mass.y_array
    )
    assert (
        pytest.approx(expected_oxidizer_mass.y_array, 0.01)
        == test_oxidizer_tank.fluid_mass.y_array
    )
    assert (
        pytest.approx(expected_pressurant_volume.y_array, 0.01)
        == test_pressurant_tank.gas_volume.y_array
    )
    assert (
        pytest.approx(expected_fuel_volume.y_array, 0.01)
        == test_fuel_tank.liquid_volume.y_array
    )
    assert (
        pytest.approx(expected_oxidizer_volume.y_array, 0.01)
        == test_oxidizer_tank.liquid_volume.y_array
    )


def test_liquid_motor_center_of_mass(
    liquid_motor, pressurant_tank, fuel_tank, oxidizer_tank
):
    """Tests the LiquidMotor class center of mass.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_tank : rocketpy.Tank
        The expected pressurant tank.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    """
    pressurant_mass = pressurant_tank.fluid_mass
    fuel_mass = fuel_tank.fluid_mass
    oxidizer_mass = oxidizer_tank.fluid_mass
    propellant_mass = pressurant_mass + fuel_mass + oxidizer_mass

    propellant_balance = (
        pressurant_mass * (pressurant_tank.center_of_mass + pressurant_tank_position)
        + fuel_mass * (fuel_tank.center_of_mass + fuel_tank_position)
        + oxidizer_mass * (oxidizer_tank.center_of_mass + oxidizer_tank_position)
    )
    balance = propellant_balance + dry_mass * center_of_dry_mass

    propellant_center_of_mass = propellant_balance / propellant_mass
    center_of_mass = balance / (propellant_mass + dry_mass)

    assert (
        pytest.approx(liquid_motor.center_of_propellant_mass.y_array)
        == propellant_center_of_mass.y_array
    )
    assert pytest.approx(liquid_motor.center_of_mass.y_array) == center_of_mass.y_array


def test_liquid_motor_inertia(liquid_motor, pressurant_tank, fuel_tank, oxidizer_tank):
    """Tests the LiquidMotor class inertia.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    pressurant_tank : rocketpy.Tank
        The expected pressurant tank.
    fuel_tank : rocketpy.Tank
        The expected fuel tank.
    oxidizer_tank : rocketpy.Tank
        The expected oxidizer tank.
    """
    pressurant_inertia = pressurant_tank.inertia
    fuel_inertia = fuel_tank.inertia
    oxidizer_inertia = oxidizer_tank.inertia
    propellant_mass = (
        pressurant_tank.fluid_mass + fuel_tank.fluid_mass + oxidizer_tank.fluid_mass
    )

    # Validate parallel axis theorem translation
    pressurant_inertia += (
        pressurant_tank.fluid_mass
        * (
            pressurant_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + pressurant_tank_position
        )
        ** 2
    )
    fuel_inertia += (
        fuel_tank.fluid_mass
        * (
            fuel_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + fuel_tank_position
        )
        ** 2
    )
    oxidizer_inertia += (
        oxidizer_tank.fluid_mass
        * (
            oxidizer_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + oxidizer_tank_position
        )
        ** 2
    )

    propellant_inertia = pressurant_inertia + fuel_inertia + oxidizer_inertia

    # Adding dry mass contributions
    inertia = (
        propellant_inertia
        + propellant_mass
        * (liquid_motor.center_of_propellant_mass - liquid_motor.center_of_mass) ** 2
        + dry_inertia[0]
        + dry_mass * (-liquid_motor.center_of_mass + center_of_dry_mass) ** 2
    )

    assert (
        pytest.approx(liquid_motor.propellant_I_11.y_array)
        == propellant_inertia.y_array
    )
    assert pytest.approx(liquid_motor.I_11.y_array) == inertia.y_array

    # Assert cylindrical symmetry
    assert (
        pytest.approx(liquid_motor.propellant_I_22.y_array)
        == propellant_inertia.y_array
    )
