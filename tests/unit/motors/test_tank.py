from math import isclose
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import scipy.integrate as spi

from rocketpy import CylindricalTank, Fluid, Function, MassFlowRateBasedTank
from rocketpy.motors.tank import _compose_clipped

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


def test_variable_density_mass_tank(cylindrical_variable_density_oxidizer_tank):
    """Test variable-density mass, volume, and density consistency.

    Parameters
    ----------
    cylindrical_variable_density_oxidizer_tank : MassBasedTank
        The variable-density oxidizer tank to be tested.
    """
    tank = cylindrical_variable_density_oxidizer_tank
    time = np.linspace(*tank.flux_time, 75)

    liquid_density = tank._liquid_density(time)
    gas_density = tank._gas_density(time)

    assert np.all(liquid_density > 0)
    assert np.all(gas_density > 0)
    assert np.all(liquid_density < 1e5)
    assert np.all(gas_density < 1e5)

    npt.assert_allclose(
        tank.liquid_mass(time),
        tank.liquid_volume(time) * liquid_density,
        atol=1e-2,
    )
    npt.assert_allclose(
        tank.gas_mass(time),
        tank.gas_volume(time) * gas_density,
        atol=1e-2,
    )
    npt.assert_allclose(
        tank.gas_mass(time),
        0,
        atol=1e-4,
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


def test_mass_flow_rate_tank_exact_depletion():
    """Regression test for a tank drained to exact zero mass via a constant
    (linear) mass flow rate.

    Before the fix, floating-point roundoff caused the computed liquid mass
    to land marginally below zero (e.g. -1e-15 kg) at the instant of exact
    depletion, which incorrectly tripped the tank's underfill check and/or
    the downstream height/volume `Function.compose` domain check, raising a
    spurious ValueError even though the tank is simply empty.
    """
    liquid = Fluid(name="water", density=1000)
    gas = Fluid(name="air", density=1.225)

    geometry = CylindricalTank(radius_function=0.1, height=1.2, spherical_caps=False)

    flux_time = 5.0
    initial_liquid_mass = 32.0  # chosen to reliably reproduce the roundoff

    tank = MassFlowRateBasedTank(
        name="linear drain tank",
        geometry=geometry,
        flux_time=flux_time,
        initial_liquid_mass=initial_liquid_mass,
        initial_gas_mass=0,
        liquid_mass_flow_rate_in=0,
        # Constant drain rate: mass hits exactly 0 at t = flux_time
        liquid_mass_flow_rate_out=initial_liquid_mass / flux_time,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=liquid,
        gas=gas,
    )

    time_points = np.array([0.0, 2.5, flux_time, flux_time + 1.0, flux_time + 5.0])

    # Should not raise, and should read as ~0 (not negative) at/after depletion
    liquid_mass = tank.liquid_mass(time_points)
    npt.assert_allclose(liquid_mass[-2:], 0, atol=1e-6)
    assert np.all(liquid_mass > -1e-6)

    # Height/volume properties must also remain well-defined past depletion
    liquid_height = tank.liquid_height(time_points)
    assert np.all(np.isfinite(liquid_height))

    gas_height = tank.gas_height(time_points)
    assert np.all(np.isfinite(gas_height))


@pytest.mark.parametrize(
    "outer_is_array, inner_is_array",
    [(True, False), (False, True), (False, False)],
    ids=["array-callable", "callable-array", "callable-callable"],
)
def test_compose_clipped_defers_when_a_source_is_not_an_array(
    outer_is_array, inner_is_array
):
    """Clipping needs ``x_array``, which only array-sourced Functions have.

    ``Function.compose`` handles a callable source on its own and performs no
    bounds check there, so there is no spurious error to absorb and nothing to
    clip. Without this the helper would raise ``AttributeError`` instead of
    composing, so it would not be a drop-in replacement for ``compose``.
    """
    doubling = np.column_stack([np.linspace(0, 10, 11), np.linspace(0, 20, 11)])
    outer = Function(doubling) if outer_is_array else Function(lambda v: v * 2.0)

    shift = np.column_stack([np.linspace(0, 5, 6), np.linspace(1, 6, 6)])
    inner = Function(shift) if inner_is_array else Function(lambda t: t + 1.0)

    composed = _compose_clipped(outer, inner)

    # outer(inner(3)) == (3 + 1) * 2
    assert float(composed(3.0)) == pytest.approx(8.0)


def test_compose_clipped_absorbs_only_boundary_noise():
    """Values a roundoff below the domain are pulled in, not rejected."""
    doubling = np.column_stack([np.linspace(0, 10, 11), np.linspace(0, 20, 11)])
    outer = Function(doubling)

    times = np.linspace(0.0, 5.0, 6)
    just_under_zero = np.column_stack([times, np.full_like(times, -1e-16)])
    inner = Function(just_under_zero)

    composed = _compose_clipped(outer, inner)

    assert float(composed(2.0)) == pytest.approx(0.0)
