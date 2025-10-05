from math import isclose
from pathlib import Path

import numpy as np
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
    expected_liquid_mass = np.loadtxt(liq_path, skiprows=1, delimiter=",")
    expected_gas_mass = np.loadtxt(gas_path, skiprows=1, delimiter=",")

    assert np.allclose(
        expected_liquid_mass[:, 1],
        tank.liquid_mass(expected_liquid_mass[:, 0]),
        rtol=1e-2,
        atol=1e-4,
    )
    assert np.allclose(
        expected_gas_mass[:, 1],
        tank.gas_mass(expected_gas_mass[:, 0]),
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
    expected_liquid_mass = np.loadtxt(liq_path, skiprows=1, delimiter=",")
    expected_gas_mass = np.loadtxt(gas_path, skiprows=1, delimiter=",")

    # Noisy derivatives, assert integrals
    initial_mass = expected_liquid_mass[0, 1] + expected_gas_mass[0, 1]
    expected_mass_variation = (
        expected_liquid_mass[-1, 1] + expected_gas_mass[-1, 1] - initial_mass
    )
    computed_final_mass = spi.simpson(
        tank.net_mass_flow_rate.y_array,
        x=tank.net_mass_flow_rate.x_array,
    )

    assert isclose(expected_mass_variation, computed_final_mass, rel_tol=1e-2)


def test_level_based_tank_liquid_level(real_level_based_tank_seblm):
    """Test the liquid_level property of LevelBasedTank
    subclass of Tank.

    Parameters
    ----------
    real_level_based_tank_seblm : LevelBasedTank
        The LevelBasedTank to be tested.
    """
    tank = real_level_based_tank_seblm
    level_data = np.loadtxt(BASE_PATH / "loxUllage.csv", delimiter=",")

    assert np.allclose(level_data, tank.liquid_height.get_source())


def test_level_based_tank_mass(real_level_based_tank_seblm):
    """Test the mass property of LevelBasedTank subclass of Tank.

    Parameters
    ----------
    real_level_based_tank_seblm : LevelBasedTank
        The LevelBasedTank to be tested.
    """
    tank = real_level_based_tank_seblm
    mass_data = np.loadtxt(BASE_PATH / "loxMass.csv", delimiter=",")

    # Soft tolerances for the whole curve
    assert np.allclose(mass_data, tank.fluid_mass.get_source(), rtol=1e-1, atol=6e-1)

    # Tighter tolerances for middle of the curve
    assert np.allclose(
        mass_data[100:401], tank.fluid_mass.get_source()[100:401], rtol=5e-2, atol=1e-1
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

    assert np.allclose(expected_mass_flow_rate, tank.net_mass_flow_rate.y_array)


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

    times = np.linspace(0, 10, 11)

    assert np.allclose(
        expected_initial_liquid_mass + expected_liquid_mass_flow * times,
        tank.liquid_mass.y_array,
    )
    assert np.allclose(
        expected_initial_gas_mass + expected_gas_mass_flow * times,
        tank.gas_mass.y_array,
    )
    assert np.allclose(
        expected_initial_mass + expected_total_mass_flow * times,
        tank.fluid_mass.y_array,
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

    times = np.linspace(0, 10, 11)

    assert np.allclose(expected_liquid_volume(times), tank.liquid_volume.y_array)
    assert np.allclose(
        expected_liquid_volume(times) / tank.geometry.area(0),
        tank.liquid_height.y_array,
    )
    assert np.allclose(
        expected_gas_volume(times),
        tank.gas_volume.y_array,
    )
    assert np.allclose(
        (expected_gas_volume(times) + expected_liquid_volume(times))
        / tank.geometry.area(0),
        tank.gas_height.y_array,
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

    times = np.linspace(0, 10, 11)

    assert np.allclose(
        expected_liquid_center_of_mass(times),
        tank.liquid_center_of_mass.y_array,
        atol=1e-4,
        rtol=1e-3,
    )
    assert np.allclose(
        expected_gas_center_of_mass(times),
        tank.gas_center_of_mass.y_array,
        atol=1e-4,
        rtol=1e-3,
    )
    assert np.allclose(
        expected_center_of_mass(times),
        tank.center_of_mass.y_array,
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

    times = np.linspace(0, 10, 11)

    assert np.allclose(
        expected_liquid_inertia(times),
        tank.liquid_inertia.y_array,
        atol=1e-3,
        rtol=1e-2,
    )
    assert np.allclose(
        expected_gas_inertia(times), tank.gas_inertia.y_array, atol=1e-3, rtol=1e-2
    )
    assert np.allclose(
        expected_liquid_inertia(times) + expected_gas_inertia(times),
        tank.inertia.y_array,
        atol=1e-3,
        rtol=1e-2,
    )


def test_variable_density_mass_tank(cylindrical_variable_density_oxidizer_tank):
    """Tests a cylindrical tank with variable density fluids
    from its temperature and pressure values.

    Parameters
    ----------
    cylindrical_variable_density_oxidizer_tank: MassBasedTank
        The tank to be tested.
    """
    tank = cylindrical_variable_density_oxidizer_tank
    time_steps = np.linspace(*tank.flux_time, 75)

    assert (tank._liquid_density(time_steps) > 0).all()
    assert (tank._gas_density(time_steps) > 0).all()
    assert (tank._liquid_density(time_steps) < 1e5).all()
    assert (tank._gas_density(time_steps) < 1e5).all()
    np.testing.assert_allclose(
        tank.liquid_mass(time_steps),
        tank.liquid_volume(time_steps) * tank._liquid_density(time_steps),
        atol=1e-2,
    )
    np.testing.assert_allclose(
        tank.gas_mass(time_steps),
        tank.gas_volume(time_steps) * tank._gas_density(time_steps),
        atol=1e-2,
    )
    np.testing.assert_allclose(
        tank.gas_mass(time_steps),
        0,
        atol=1e-4,
    )
