from math import isclose
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import scipy.integrate as spi

BASE_PATH = Path("./data/rockets/berkeley/")


@pytest.mark.parametrize(
    "params",
    [
        (
            "real_mass_based_tank_seblm",
            BASE_PATH / "Test135LoxMass.csv",
            BASE_PATH / "Test135GasMass.csv",
        ),
        (
            "example_mass_based_tank_seblm",
            BASE_PATH / "ExampleTankLiquidMassData.csv",
            BASE_PATH / "ExampleTankGasMassData.csv",
        ),
    ],
)
def test_mass_based_tank_fluid_mass(params, request):
    """Test the fluid_mass property of the MassBasedTank subclass of Tank
    class.

    Parameters
    ----------
    params : tuple
        A tuple containing test parameters.
    request : _pytest.fixtures.FixtureRequest
        A pytest fixture request object.
    """
    tank, liq_path, gas_path = params
    tank = request.getfixturevalue(tank)

    expected_liquid_mass = np.loadtxt(
        liq_path,
        skiprows=1,
        delimiter=",",
    )

    expected_gas_mass = np.loadtxt(
        gas_path,
        skiprows=1,
        delimiter=",",
    )

    liquid_time = expected_liquid_mass[:, 0]
    gas_time = expected_gas_mass[:, 0]

    npt.assert_allclose(
        tank.liquid_mass(liquid_time),
        expected_liquid_mass[:, 1],
        rtol=1e-2,
        atol=1e-4,
    )

    npt.assert_allclose(
        tank.gas_mass(gas_time),
        expected_gas_mass[:, 1],
        rtol=1e-1,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "params",
    [
        (
            "real_mass_based_tank_seblm",
            BASE_PATH / "Test135LoxMass.csv",
            BASE_PATH / "Test135GasMass.csv",
        ),
        (
            "example_mass_based_tank_seblm",
            BASE_PATH / "ExampleTankLiquidMassData.csv",
            BASE_PATH / "ExampleTankGasMassData.csv",
        ),
    ],
)
def test_mass_based_tank_net_mass_flow_rate(params, request):
    """Test the net_mass_flow_rate property of the MassBasedTank
    subclass of Tank.

    Parameters
    ----------
    params : tuple
        A tuple containing test parameters.
    request : _pytest.fixtures.FixtureRequest
        A pytest fixture request object.
    """
    tank, liq_path, gas_path = params
    tank = request.getfixturevalue(tank)

    expected_liquid_mass = np.loadtxt(
        liq_path,
        skiprows=1,
        delimiter=",",
    )

    expected_gas_mass = np.loadtxt(
        gas_path,
        skiprows=1,
        delimiter=",",
    )

    # Noisy derivatives, assert integrals
    initial_mass = expected_liquid_mass[0, 1] + expected_gas_mass[0, 1]

    expected_mass_variation = (
        expected_liquid_mass[-1, 1] + expected_gas_mass[-1, 1] - initial_mass
    )

    time = tank.net_mass_flow_rate.x_array

    computed_final_mass = spi.simpson(
        tank.net_mass_flow_rate(time),
        x=time,
    )

    assert isclose(
        expected_mass_variation,
        computed_final_mass,
        rel_tol=1e-2,
    )


def test_level_based_tank_liquid_level(real_level_based_tank_seblm):
    """Test the liquid_level property of LevelBasedTank
    subclass of Tank.

    Parameters
    ----------
    real_level_based_tank_seblm : LevelBasedTank
        The LevelBasedTank to be tested.
    """
    tank = real_level_based_tank_seblm

    level_data = np.loadtxt(
        BASE_PATH / "loxUllage.csv",
        delimiter=",",
    )

    time = level_data[:, 0]

    npt.assert_allclose(tank.liquid_height(time), level_data[:, 1], atol=1e-8)


def test_level_based_tank_mass(real_level_based_tank_seblm):
    """Test the mass property of LevelBasedTank subclass of Tank.

    Parameters
    ----------
    real_level_based_tank_seblm : LevelBasedTank
        The LevelBasedTank to be tested.
    """
    tank = real_level_based_tank_seblm

    mass_data = np.loadtxt(
        BASE_PATH / "loxMass.csv",
        delimiter=",",
    )

    time = mass_data[:, 0]

    computed_mass = tank.fluid_mass(time)

    # Soft tolerances for the whole curve
    npt.assert_allclose(
        computed_mass,
        mass_data[:, 1],
        rtol=1e-1,
        atol=6e-1,
    )

    # Tighter tolerances for middle of the curve
    npt.assert_allclose(
        computed_mass[100:401],
        mass_data[100:401, 1],
        rtol=5e-2,
        atol=1e-1,
    )


def test_mass_flow_rate_tank_mass_flow_rate(example_mass_flow_rate_based_tank_seblm):
    """Test the mass_flow_rate property of the MassFlowRateBasedTank
    subclass of Tank.

    Parameters
    ----------
    example_mass_flow_rate_based_tank_seblm : MassFlowRateBasedTank
        The MassFlowRateBasedTank to be tested.
    """
    tank = example_mass_flow_rate_based_tank_seblm

    expected_mass_flow_rate = 0.1 - 0.2 + 0.01 - 0.02

    time = tank.net_mass_flow_rate.x_array

    npt.assert_allclose(
        tank.net_mass_flow_rate(time),
        expected_mass_flow_rate,
        atol=1e-6,
    )


def test_mass_flow_rate_tank_fluid_mass(example_mass_flow_rate_based_tank_seblm):
    """Test the fluid_mass property of the MassFlowRateBasedTank
    subclass of Tank.

    Parameters
    ----------
    example_mass_flow_rate_based_tank_seblm : MassFlowRateBasedTank
        The MassFlowRateBasedTank to be tested.
    """
    tank = example_mass_flow_rate_based_tank_seblm

    expected_initial_liquid_mass = 5
    expected_initial_gas_mass = 0.1

    expected_initial_mass = expected_initial_liquid_mass + expected_initial_gas_mass

    expected_liquid_mass_flow = 0.1 - 0.2
    expected_gas_mass_flow = 0.01 - 0.02

    expected_total_mass_flow = expected_liquid_mass_flow + expected_gas_mass_flow

    time = np.linspace(0, 10, 11)

    npt.assert_allclose(
        tank.liquid_mass(time),
        expected_initial_liquid_mass + expected_liquid_mass_flow * time,
        atol=1e-6,
    )

    npt.assert_allclose(
        tank.gas_mass(time),
        expected_initial_gas_mass + expected_gas_mass_flow * time,
        atol=1e-6,
    )

    npt.assert_allclose(
        tank.fluid_mass(time),
        expected_initial_mass + expected_total_mass_flow * time,
        atol=1e-6,
    )


def test_mass_flow_rate_tank_liquid_height(
    example_mass_flow_rate_based_tank_seblm, lox_fluid_seblm, nitrogen_fluid_seblm
):
    """Test the liquid height properties of the MassFlowRateBasedTank
    subclass of Tank.

    Parameters
    ----------
    example_mass_flow_rate_based_tank_seblm : MassFlowRateBasedTank
        The MassFlowRateBasedTank to be tested.
    lox_fluid_seblm : Fluid
        The Fluid object representing liquid oxygen.
    nitrogen_fluid_seblm : Fluid
        The Fluid object representing nitrogen gas.
    """
    tank = example_mass_flow_rate_based_tank_seblm

    def expected_liquid_volume(t):
        return (5 + (0.1 - 0.2) * t) / lox_fluid_seblm.density

    def expected_gas_volume(t):
        return (0.1 + (0.01 - 0.02) * t) / nitrogen_fluid_seblm.density

    time = np.linspace(0, 10, 11)

    liquid_volume = expected_liquid_volume(time)
    gas_volume = expected_gas_volume(time)

    npt.assert_allclose(
        tank.liquid_volume(time),
        liquid_volume,
        atol=1e-6,
    )

    npt.assert_allclose(
        tank.liquid_height(time),
        liquid_volume / tank.geometry.area(0),
        atol=1e-8,
    )

    npt.assert_allclose(tank.gas_volume(time), gas_volume, atol=1e-6)

    npt.assert_allclose(
        tank.gas_height(time),
        (gas_volume + liquid_volume) / tank.geometry.area(0),
        atol=1e-8,
    )


def test_mass_flow_rate_tank_center_of_mass(
    example_mass_flow_rate_based_tank_seblm, lox_fluid_seblm, nitrogen_fluid_seblm
):
    """Test the center of mass properties of the MassFlowRateBasedTank
    subclass of Tank.

    Parameters
    ----------
    example_mass_flow_rate_based_tank_seblm : MassFlowRateBasedTank
        The MassFlowRateBasedTank to be tested.
    lox_fluid_seblm : Fluid
        The Fluid object representing liquid oxygen.
    nitrogen_fluid_seblm : Fluid
        The Fluid object representing nitrogen gas.
    """
    # TODO: improve code context and repetition
    tank = example_mass_flow_rate_based_tank_seblm

    def expected_liquid_center_of_mass(t):
        liquid_height = (5 + (0.1 - 0.2) * t) / lox_fluid_seblm.density / np.pi

        return liquid_height / 2

    def expected_gas_center_of_mass(t):
        liquid_height = (5 + (0.1 - 0.2) * t) / lox_fluid_seblm.density / np.pi

        gas_height = (0.1 + (0.01 - 0.02) * t) / nitrogen_fluid_seblm.density / np.pi

        return gas_height / 2 + liquid_height

    def expected_center_of_mass(t):
        liquid_mass = 5 + (0.1 - 0.2) * t
        gas_mass = 0.1 + (0.01 - 0.02) * t

        return (
            liquid_mass * expected_liquid_center_of_mass(t)
            + gas_mass * expected_gas_center_of_mass(t)
        ) / (liquid_mass + gas_mass)

    time = np.linspace(0, 10, 11)

    npt.assert_allclose(
        tank.liquid_center_of_mass(time),
        expected_liquid_center_of_mass(time),
        atol=1e-4,
        rtol=1e-3,
    )

    npt.assert_allclose(
        tank.gas_center_of_mass(time),
        expected_gas_center_of_mass(time),
        atol=1e-4,
        rtol=1e-3,
    )

    npt.assert_allclose(
        tank.center_of_mass(time),
        expected_center_of_mass(time),
        atol=1e-4,
        rtol=1e-3,
    )


def test_mass_flow_rate_tank_inertia(
    example_mass_flow_rate_based_tank_seblm, lox_fluid_seblm, nitrogen_fluid_seblm
):
    """Test the inertia properties of the MassFlowRateBasedTank
    subclass of Tank.

    Parameters
    ----------
    example_mass_flow_rate_based_tank_seblm : MassFlowRateBasedTank
        The MassFlowRateBasedTank to be tested.
    lox_fluid_seblm : Fluid
        The Fluid object representing liquid oxygen.
    nitrogen_fluid_seblm : Fluid
        The Fluid object representing nitrogen gas.
    """
    # TODO: improve code context and repetition
    tank = example_mass_flow_rate_based_tank_seblm

    def expected_center_of_mass(t):
        liquid_mass = 5 + (0.1 - 0.2) * t
        gas_mass = 0.1 + (0.01 - 0.02) * t

        liquid_height = liquid_mass / lox_fluid_seblm.density / np.pi

        gas_height = gas_mass / nitrogen_fluid_seblm.density / np.pi

        return (
            liquid_mass * liquid_height / 2
            + gas_mass * (gas_height / 2 + liquid_height)
        ) / (liquid_mass + gas_mass)

    def expected_liquid_inertia(t):
        r = 1

        liquid_mass = 5 + (0.1 - 0.2) * t

        liquid_height = liquid_mass / lox_fluid_seblm.density / np.pi

        liquid_com = liquid_height / 2

        return (
            1 / 4 * liquid_mass * r**2
            + 1 / 12 * liquid_mass * liquid_height**2
            + liquid_mass * (liquid_com - expected_center_of_mass(t)) ** 2
        )

    def expected_gas_inertia(t):
        r = 1

        liquid_mass = 5 + (0.1 - 0.2) * t
        gas_mass = 0.1 + (0.01 - 0.02) * t

        liquid_height = liquid_mass / lox_fluid_seblm.density / np.pi

        gas_height = gas_mass / nitrogen_fluid_seblm.density / np.pi

        gas_com = gas_height / 2 + liquid_height

        return (
            1 / 4 * gas_mass * r**2
            + 1 / 12 * gas_mass * (gas_height - liquid_height) ** 2
            + gas_mass * (gas_com - expected_center_of_mass(t)) ** 2
        )

    time = np.linspace(0, 10, 11)

    liquid_inertia = expected_liquid_inertia(time)
    gas_inertia = expected_gas_inertia(time)

    npt.assert_allclose(
        tank.liquid_inertia(time),
        liquid_inertia,
        atol=1e-3,
        rtol=1e-2,
    )

    npt.assert_allclose(
        tank.gas_inertia(time),
        gas_inertia,
        atol=1e-3,
        rtol=1e-2,
    )

    npt.assert_allclose(
        tank.inertia(time),
        liquid_inertia + gas_inertia,
        atol=1e-3,
        rtol=1e-2,
    )
