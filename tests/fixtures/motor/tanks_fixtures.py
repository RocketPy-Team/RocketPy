from math import exp

import numpy as np
import pytest
from scipy.constants import atm, zero_Celsius

from rocketpy import (
    CylindricalTank,
    Fluid,
    Function,
    LevelBasedTank,
    MassBasedTank,
    MassFlowRateBasedTank,
    SphericalTank,
    TankGeometry,
    UllageBasedTank,
)


@pytest.fixture
def sample_full_mass_flow_rate_tank():
    """An example of a full MassFlowRateBasedTank.

    Returns
    -------
    rocketpy.MassFlowRateBasedTank
        An object of the MassFlowRateBasedTank class.
    """
    full_tank = MassFlowRateBasedTank(
        name="Full Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        initial_liquid_mass=9,
        initial_gas_mass=0.001,
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        liquid_mass_flow_rate_in=0,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid_mass_flow_rate_out=0,
        flux_time=(0, 10),
    )

    return full_tank


@pytest.fixture
def sample_empty_mass_flow_rate_tank():
    """An example of an empty MassFlowRateBasedTank.

    Returns
    -------
    rocketpy.MassFlowRateBasedTank
        An object of the MassFlowRateBasedTank class.
    """
    empty_tank = MassFlowRateBasedTank(
        name="Empty Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        initial_liquid_mass=0,
        initial_gas_mass=0,
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        liquid_mass_flow_rate_in=0,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid_mass_flow_rate_out=0,
        flux_time=(0, 10),
    )

    return empty_tank


@pytest.fixture
def sample_full_ullage_tank():
    """An example of a UllageBasedTank full of liquid.

    Returns
    -------
    rocketpy.UllageBasedTank
        An object of the UllageBasedTank class.
    """
    full_tank = UllageBasedTank(
        name="Full Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        ullage=0,
        flux_time=(0, 10),
    )

    return full_tank


@pytest.fixture
def sample_empty_ullage_tank():
    """An example of a UllageBasedTank with no liquid.

    Returns
    -------
    rocketpy.UllageBasedTank
        An object of the UllageBasedTank class.
    """
    empty_tank = UllageBasedTank(
        name="Empty Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        ullage=0.01,
        flux_time=(0, 10),
    )

    return empty_tank


@pytest.fixture
def sample_full_level_tank():
    """An example of a LevelBasedTank full of liquid.

    Returns
    -------
    rocketpy.LevelBasedTank
        An object of the LevelBasedTank class.
    """
    full_tank = LevelBasedTank(
        name="Full Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        liquid_height=1 / (2 * np.pi),
        flux_time=(0, 10),
    )

    return full_tank


@pytest.fixture
def sample_empty_level_tank():
    """An example of a LevelBasedTank with no liquid.

    Returns
    -------
    rocketpy.LevelBasedTank
        An object of the LevelBasedTank class.
    """
    empty_tank = LevelBasedTank(
        name="Empty Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        liquid_height=0,
        flux_time=(0, 10),
    )

    return empty_tank


@pytest.fixture
def sample_full_mass_tank():
    """An example of a full MassBasedTank.

    Returns
    -------
    rocketpy.MassBasedTank
        An object of the MassBasedTank class.
    """
    full_tank = MassBasedTank(
        name="Full Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        liquid_mass=9,
        gas_mass=0.001,
        flux_time=(0, 10),
    )

    return full_tank


@pytest.fixture
def sample_empty_mass_tank():
    """An example of an empty MassBasedTank.

    Returns
    -------
    rocketpy.MassBasedTank
        An object of the MassBasedTank class.
    """
    empty_tank = MassBasedTank(
        name="Empty Tank",
        geometry=CylindricalTank(0.1, 1 / np.pi),
        liquid=Fluid("Water", 1000),
        gas=Fluid("Air", 1),
        liquid_mass=0,
        gas_mass=0,
        flux_time=(0, 10),
    )

    return empty_tank


@pytest.fixture
def real_mass_based_tank_seblm(lox_fluid_seblm, nitrogen_fluid_seblm):
    """An instance of a real cylindrical tank with spherical caps.

    Parameters
    ----------
    lox_fluid_seblm : rocketpy.Fluid
        Liquid oxygen fluid. This is a pytest fixture.
    nitrogen_fluid_seblm : rocketpy.Fluid
        Nitrogen gas fluid. This is a pytest fixture.

    Returns
    -------
    rocketpy.MassBasedTank
        An object of the MassBasedTank class.
    """
    geometry = CylindricalTank(0.0744, 0.8698, True)

    lox_tank = MassBasedTank(
        name="Real Tank",
        geometry=geometry,
        flux_time=(0, 15.583),
        liquid_mass="./data/rockets/berkeley/Test135LoxMass.csv",
        gas_mass="./data/rockets/berkeley/Test135GasMass.csv",
        liquid=lox_fluid_seblm,
        gas=nitrogen_fluid_seblm,
        discretize=200,
    )

    return lox_tank


@pytest.fixture
def example_mass_based_tank_seblm(lox_fluid_seblm, nitrogen_fluid_seblm):
    """Example data of a cylindrical tank with spherical caps.

    Parameters
    ----------
    lox_fluid_seblm : rocketpy.Fluid
        Liquid oxygen fluid. This is a pytest fixture.
    nitrogen_fluid_seblm : rocketpy.Fluid
        Nitrogen gas fluid. This is a pytest fixture.

    Returns
    -------
    rocketpy.MassBasedTank
        An object of the MassBasedTank class.
    """
    geometry = TankGeometry({(0, 5): 1})

    example_tank = MassBasedTank(
        name="Example Tank",
        geometry=geometry,
        flux_time=(0, 10),
        liquid_mass="./data/rockets/berkeley/ExampleTankLiquidMassData.csv",
        gas_mass="./data/rockets/berkeley/ExampleTankGasMassData.csv",
        liquid=lox_fluid_seblm,
        gas=nitrogen_fluid_seblm,
        discretize=None,
    )

    return example_tank


@pytest.fixture
def real_level_based_tank_seblm(lox_fluid_seblm, nitrogen_fluid_seblm):
    """An instance of a real cylindrical tank with spherical caps.

    Parameters
    ----------
    lox_fluid_seblm : rocketpy.Fluid
        Liquid oxygen fluid. This is a pytest fixture.
    nitrogen_fluid_seblm : rocketpy.Fluid
        Nitrogen gas fluid. This is a pytest fixture.

    Returns
    -------
    rocketpy.LevelBasedTank
        An object of the LevelBasedTank class.
    """
    geometry = TankGeometry(
        {
            (0, 0.0559): lambda h: np.sqrt(0.0775**2 - (0.0775 - h) ** 2),
            (0.0559, 0.7139): 0.0744,
            (0.7139, 0.7698): lambda h: np.sqrt(0.0775**2 - (h - 0.6924) ** 2),
        }
    )

    level_tank = LevelBasedTank(
        name="Level Tank",
        geometry=geometry,
        flux_time=(0, 15.583),
        gas=nitrogen_fluid_seblm,
        liquid=lox_fluid_seblm,
        liquid_height="./data/rockets/berkeley/loxUllage.csv",
        discretize=None,
    )

    return level_tank


@pytest.fixture
def example_mass_flow_rate_based_tank_seblm(lox_fluid_seblm, nitrogen_fluid_seblm):
    """An instance of a example cylindrical tank whose flux
    is given by mass flow rates.

    Parameters
    ----------
    lox_fluid_seblm : rocketpy.Fluid
        Liquid oxygen fluid. This is a pytest fixture.
    nitrogen_fluid_seblm : rocketpy.Fluid
        Nitrogen gas fluid. This is a pytest fixture.

    Returns
    -------
    rocketpy.MassFlowRateBasedTank
        An object of the MassFlowRateBasedTank class.
    """
    mass_flow_rate_tank = MassFlowRateBasedTank(
        name="Test Tank",
        geometry=TankGeometry({(0, 1): 1}),
        flux_time=(0, 10),
        initial_liquid_mass=5,
        initial_gas_mass=0.1,
        liquid_mass_flow_rate_in=0.1,
        gas_mass_flow_rate_in=0.01,
        liquid_mass_flow_rate_out=0.2,
        gas_mass_flow_rate_out=0.02,
        liquid=lox_fluid_seblm,
        gas=nitrogen_fluid_seblm,
        discretize=11,
    )

    return mass_flow_rate_tank


@pytest.fixture
def pressurant_tank(pressurant_fluid, pressurant_tank_geometry):
    """An example of a pressurant cylindrical tank with spherical
    caps.

    Parameters
    ----------
    pressurant_fluid : rocketpy.Fluid
        Pressurizing fluid. This is a pytest fixture.
    pressurant_tank_geometry : rocketpy.CylindricalTank
        Geometry of the pressurant tank. This is a pytest fixture.

    Returns
    -------
    rocketpy.MassBasedTank
        An object of the CylindricalTank class.
    """
    pressurant_tank = MassBasedTank(
        name="Pressure Tank",
        geometry=pressurant_tank_geometry,
        liquid_mass=0,
        flux_time=(8, 20),
        gas_mass="data/rockets/berkeley/pressurantMassFiltered.csv",
        gas=pressurant_fluid,
        liquid=pressurant_fluid,
    )

    return pressurant_tank


@pytest.fixture
def fuel_tank(fuel_fluid, fuel_pressurant, propellant_tank_geometry):
    """An example of a fuel cylindrical tank with spherical
    caps.

    Parameters
    ----------
    fuel_fluid : rocketpy.Fluid
        Fuel fluid of the tank. This is a pytest fixture.
    fuel_pressurant : rocketpy.Fluid
        Pressurizing fluid of the fuel tank. This is a pytest
        fixture.
    propellant_tank_geometry : rocketpy.CylindricalTank
        Geometry of the fuel tank. This is a pytest fixture.

    Returns
    -------
    rocketpy.UllageBasedTank
    """
    ullage = (
        -Function("data/rockets/berkeley/test124_Propane_Volume.csv") * 1e-3
        + propellant_tank_geometry.total_volume
    )
    fuel_tank = UllageBasedTank(
        name="Propane Tank",
        flux_time=(8, 20),
        geometry=propellant_tank_geometry,
        liquid=fuel_fluid,
        gas=fuel_pressurant,
        ullage=ullage,
    )

    return fuel_tank


@pytest.fixture
def oxidizer_tank(oxidizer_fluid, oxidizer_pressurant, propellant_tank_geometry):
    """An example of a oxidizer cylindrical tank with spherical
    caps.

    Parameters
    ----------
    oxidizer_fluid : rocketpy.Fluid
        Oxidizer fluid of the tank. This is a pytest fixture.
    oxidizer_pressurant : rocketpy.Fluid
        Pressurizing fluid of the oxidizer tank. This is a pytest
        fixture.
    propellant_tank_geometry : rocketpy.CylindricalTank
        Geometry of the oxidizer tank. This is a pytest fixture.

    Returns
    -------
    rocketpy.UllageBasedTank
    """
    ullage = (
        -Function("data/rockets/berkeley/test124_Lox_Volume.csv") * 1e-3
        + propellant_tank_geometry.total_volume
    )
    oxidizer_tank = UllageBasedTank(
        name="Lox Tank",
        flux_time=(8, 20),
        geometry=propellant_tank_geometry,
        liquid=oxidizer_fluid,
        gas=oxidizer_pressurant,
        ullage=ullage,
    )

    return oxidizer_tank


@pytest.fixture
def spherical_oxidizer_tank(oxidizer_fluid, oxidizer_pressurant):
    """An example of a oxidizer spherical tank.

    Parameters
    ----------
    oxidizer_fluid : rocketpy.Fluid
        Oxidizer fluid of the tank. This is a pytest fixture.
    oxidizer_pressurant : rocketpy.Fluid
        Pressurizing fluid of the oxidizer tank. This is a pytest
        fixture.

    Returns
    -------
    rocketpy.LevelBasedTank
    """
    liquid_level = Function(lambda t: 0.1 * np.exp(-t / 2) - 0.05)
    oxidizer_tank = LevelBasedTank(
        name="Lox Tank",
        flux_time=10,
        geometry=SphericalTank(0.0501),
        liquid=oxidizer_fluid,
        gas=oxidizer_pressurant,
        liquid_height=liquid_level,
    )

    return oxidizer_tank


@pytest.fixture
def cylindrical_variable_density_oxidizer_tank(nitrous_oxide_non_constant_fluid):
    """Fixture for creating a cylindrical variable density oxidizer
    tank. This fixture returns a `MassBasedTank` object representing
    a cylindrical oxidizer tank with variable density properties.

    Parameters
    ----------
    nitrous_oxide_non_constant_fluid : rocketpy.Fluid
        The fluid object representing nitrous oxide.

    Returns
    -------
    MassBasedTank
        The configured MassBasedTank fixture.
    """

    def liquid_mass(t):
        if t < 4:
            return 4.2 - 3.7 / 4 * t
        else:
            return 0.5 * exp(4 - t)

    def pressure(t):
        if t < 4:
            return (60 - 25 / 4 * t) * atm
        else:
            return (34 * exp(4 - t)) * atm + atm

    def temperature(t):
        if t < 4:
            return (22 - 5 * t) + zero_Celsius
        else:
            return -40 * (t - 4) + zero_Celsius + 2

    return MassBasedTank(
        name="Variable Density N2O Tank",
        geometry=CylindricalTank(height=0.8, radius=0.06, spherical_caps=True),
        flux_time=7,
        liquid=nitrous_oxide_non_constant_fluid,
        gas=nitrous_oxide_non_constant_fluid,
        discretize=50,
        liquid_mass=liquid_mass,
        gas_mass=0,
        pressure=pressure,
        temperature=temperature,
    )
