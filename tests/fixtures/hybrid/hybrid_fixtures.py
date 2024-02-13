import numpy as np
import pytest

from rocketpy import (CylindricalTank, Fluid, Function, HybridMotor,
                      LevelBasedTank, LiquidMotor, MassBasedTank,
                      SphericalTank, UllageBasedTank)


@pytest.fixture
def pressurant_fluid():
    """An example of a pressurant fluid as N2 gas at
    273.15K and 30MPa.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="N2", density=300)


@pytest.fixture
def fuel_pressurant():
    """An example of a pressurant fluid as N2 gas at
    273.15K and 2MPa.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="N2", density=25)


@pytest.fixture
def oxidizer_pressurant():
    """An example of a pressurant fluid as N2 gas at
    273.15K and 3MPa.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="N2", density=35)


@pytest.fixture
def fuel_fluid():
    """An example of propane as fuel fluid at
    273.15K and 2MPa.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="Propane", density=500)


@pytest.fixture
def oxidizer_fluid():
    """An example of liquid oxygen as oxidizer fluid at
    100K and 3MPa.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="O2", density=1000)


@pytest.fixture
def pressurant_tank(pressurant_fluid):
    """An example of a pressurant cylindrical tank with spherical
    caps.

    Parameters
    ----------
    pressurant_fluid : rocketpy.Fluid
        Pressurizing fluid. This is a pytest fixture.

    Returns
    -------
    rocketpy.MassBasedTank
        An object of the CylindricalTank class.
    """
    geometry = CylindricalTank(0.135 / 2, 0.981, spherical_caps=True)
    pressurant_tank = MassBasedTank(
        name="Pressure Tank",
        geometry=geometry,
        liquid_mass=0,
        flux_time=(8, 20),
        gas_mass="data/SEBLM/pressurantMassFiltered.csv",
        gas=pressurant_fluid,
        liquid=pressurant_fluid,
    )

    return pressurant_tank


@pytest.fixture
def fuel_tank(fuel_fluid, fuel_pressurant):
    """An example of a fuel cylindrical tank with spherical
    caps.

    Parameters
    ----------
    fuel_fluid : rocketpy.Fluid
        Fuel fluid of the tank. This is a pytest fixture.
    fuel_pressurant : rocketpy.Fluid
        Pressurizing fluid of the fuel tank. This is a pytest
        fixture.

    Returns
    -------
    rocketpy.UllageBasedTank
    """
    geometry = CylindricalTank(0.0744, 0.8068, spherical_caps=True)
    ullage = (
        -Function("data/SEBLM/test124_Propane_Volume.csv") * 1e-3
        + geometry.total_volume
    )
    fuel_tank = UllageBasedTank(
        name="Propane Tank",
        flux_time=(8, 20),
        geometry=geometry,
        liquid=fuel_fluid,
        gas=fuel_pressurant,
        ullage=ullage,
    )

    return fuel_tank


@pytest.fixture
def oxidizer_tank(oxidizer_fluid, oxidizer_pressurant):
    """An example of a oxidizer cylindrical tank with spherical
    caps.

    Parameters
    ----------
    oxidizer_fluid : rocketpy.Fluid
        Oxidizer fluid of the tank. This is a pytest fixture.
    oxidizer_pressurant : rocketpy.Fluid
        Pressurizing fluid of the oxidizer tank. This is a pytest
        fixture.

    Returns
    -------
    rocketpy.UllageBasedTank
    """
    geometry = CylindricalTank(0.0744, 0.8068, spherical_caps=True)
    ullage = (
        -Function("data/SEBLM/test124_Lox_Volume.csv") * 1e-3 + geometry.total_volume
    )
    oxidizer_tank = UllageBasedTank(
        name="Lox Tank",
        flux_time=(8, 20),
        geometry=geometry,
        liquid=oxidizer_fluid,
        gas=oxidizer_pressurant,
        ullage=ullage,
    )

    return oxidizer_tank


@pytest.fixture
def liquid_motor(pressurant_tank, fuel_tank, oxidizer_tank):
    """An example of a liquid motor with pressurant, fuel and oxidizer tanks.

    Parameters
    ----------
    pressurant_tank : rocketpy.MassBasedTank
        Tank that contains pressurizing fluid. This is a pytest fixture.
    fuel_tank : rocketpy.UllageBasedTank
        Tank that contains the motor fuel. This is a pytest fixture.
    oxidizer_tank : rocketpy.UllageBasedTank
        Tank that contains the motor oxidizer. This is a pytest fixture.

    Returns
    -------
    rocketpy.LiquidMotor
    """
    liquid_motor = LiquidMotor(
        thrust_source="data/SEBLM/test124_Thrust_Curve.csv",
        burn_time=(8, 20),
        dry_mass=10,
        dry_inertia=(5, 5, 0.2),
        center_of_dry_mass_position=0,
        nozzle_position=-1.364,
        nozzle_radius=0.069 / 2,
    )
    liquid_motor.add_tank(pressurant_tank, position=2.007)
    liquid_motor.add_tank(fuel_tank, position=-1.048)
    liquid_motor.add_tank(oxidizer_tank, position=0.711)

    return liquid_motor


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
    rocketpy.UllageBasedTank
    """
    geometry = SphericalTank(0.05)
    liquid_level = Function(lambda t: 0.1 * np.exp(-t / 2) - 0.05)
    oxidizer_tank = LevelBasedTank(
        name="Lox Tank",
        flux_time=10,
        geometry=geometry,
        liquid=oxidizer_fluid,
        gas=oxidizer_pressurant,
        liquid_height=liquid_level,
    )

    return oxidizer_tank


@pytest.fixture
def hybrid_motor(spherical_oxidizer_tank):
    """An example of a hybrid motor with spherical oxidizer
    tank and fuel grains.

    Parameters
    ----------
    spherical_oxidizer_tank : rocketpy.LevelBasedTank
        Example Tank that contains the motor oxidizer. This is a
        pytest fixture.

    Returns
    -------
    rocketpy.HybridMotor
    """
    motor = HybridMotor(
        thrust_source=lambda t: 2000 - 100 * t,
        burn_time=10,
        center_of_dry_mass_position=0,
        dry_inertia=(4, 4, 0.1),
        dry_mass=8,
        grain_density=1700,
        grain_number=4,
        grain_initial_height=0.1,
        grain_separation=0,
        grain_initial_inner_radius=0.04,
        grain_outer_radius=0.1,
        nozzle_position=-0.4,
        nozzle_radius=0.07,
        grains_center_of_mass_position=-0.1,
    )

    motor.add_tank(spherical_oxidizer_tank, position=0.3)

    return motor
