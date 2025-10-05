import numpy as np
import pytest
import scipy.integrate

from rocketpy import Function

BURN_TIME = (8, 20)
DRY_MASS = 10
DRY_INERTIA = (5, 5, 0.2)
CENTER_OF_DRY_MASS = 0
NOZZLE_POSITION = -1.364
NOZZLE_RADIUS = 0.069 / 2
PRESSURANT_TANK_POSITION = 2.007
FUEL_TANK_POSITION = -1.048
OXIDIZER_TANK_POSITION = 0.711


def test_liquid_motor_basic_parameters(liquid_motor):
    """Tests the LiquidMotor class construction parameters.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.burn_time == BURN_TIME
    assert liquid_motor.dry_mass == DRY_MASS
    assert (
        liquid_motor.dry_I_11,
        liquid_motor.dry_I_22,
        liquid_motor.dry_I_33,
    ) == DRY_INERTIA
    assert liquid_motor.center_of_dry_mass_position == CENTER_OF_DRY_MASS
    assert liquid_motor.nozzle_position == NOZZLE_POSITION
    assert liquid_motor.nozzle_radius == NOZZLE_RADIUS
    assert liquid_motor.positioned_tanks[0]["position"] == PRESSURANT_TANK_POSITION
    assert liquid_motor.positioned_tanks[1]["position"] == FUEL_TANK_POSITION
    assert liquid_motor.positioned_tanks[2]["position"] == OXIDIZER_TANK_POSITION


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
    expected_thrust = np.loadtxt(
        "data/rockets/berkeley/test124_Thrust_Curve.csv", delimiter=","
    )
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
    expected_pressurant_mass = Function(
        "data/rockets/berkeley/pressurantMassFiltered.csv"
    )
    expected_pressurant_volume = expected_pressurant_mass / pressurant_fluid.density
    expected_fuel_volume = (
        Function("data/rockets/berkeley/test124_Propane_Volume.csv") * 1e-3
    )
    expected_fuel_mass = (
        expected_fuel_volume * fuel_fluid.density
        + (-expected_fuel_volume + fuel_tank.geometry.total_volume)
        * fuel_pressurant.density
    )
    expected_oxidizer_volume = (
        Function("data/rockets/berkeley/test124_Lox_Volume.csv") * 1e-3
    )
    expected_oxidizer_mass = (
        expected_oxidizer_volume * oxidizer_fluid.density
        + (-expected_oxidizer_volume + oxidizer_tank.geometry.total_volume)
        * oxidizer_pressurant.density
    )

    # Perform default discretization
    expected_pressurant_mass.set_discrete(*BURN_TIME, 100)
    expected_fuel_mass.set_discrete(*BURN_TIME, 100)
    expected_oxidizer_mass.set_discrete(*BURN_TIME, 100)
    expected_pressurant_volume.set_discrete(*BURN_TIME, 100)
    expected_fuel_volume.set_discrete(*BURN_TIME, 100)
    expected_oxidizer_volume.set_discrete(*BURN_TIME, 100)

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
        pressurant_mass * (pressurant_tank.center_of_mass + PRESSURANT_TANK_POSITION)
        + fuel_mass * (fuel_tank.center_of_mass + FUEL_TANK_POSITION)
        + oxidizer_mass * (oxidizer_tank.center_of_mass + OXIDIZER_TANK_POSITION)
    )
    balance = propellant_balance + DRY_MASS * CENTER_OF_DRY_MASS

    propellant_center_of_mass = propellant_balance / propellant_mass
    center_of_mass = balance / (propellant_mass + DRY_MASS)

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
            + PRESSURANT_TANK_POSITION
        )
        ** 2
    )
    fuel_inertia += (
        fuel_tank.fluid_mass
        * (
            fuel_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + FUEL_TANK_POSITION
        )
        ** 2
    )
    oxidizer_inertia += (
        oxidizer_tank.fluid_mass
        * (
            oxidizer_tank.center_of_mass
            - liquid_motor.center_of_propellant_mass
            + OXIDIZER_TANK_POSITION
        )
        ** 2
    )

    propellant_inertia = pressurant_inertia + fuel_inertia + oxidizer_inertia

    # Adding dry mass contributions
    inertia = (
        propellant_inertia
        + propellant_mass
        * (liquid_motor.center_of_propellant_mass - liquid_motor.center_of_mass) ** 2
        + DRY_INERTIA[0]
        + DRY_MASS * (-liquid_motor.center_of_mass + CENTER_OF_DRY_MASS) ** 2
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
