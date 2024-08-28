import numpy as np
import pytest

from rocketpy import Function, LevelBasedTank, MassBasedTank, UllageBasedTank


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
        gas_mass="data/SEBLM/pressurantMassFiltered.csv",
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
        -Function("data/SEBLM/test124_Propane_Volume.csv") * 1e-3
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
        -Function("data/SEBLM/test124_Lox_Volume.csv") * 1e-3
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
def spherical_oxidizer_tank(
    oxidizer_fluid, oxidizer_pressurant, spherical_oxidizer_geometry
):
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
        geometry=spherical_oxidizer_geometry,
        liquid=oxidizer_fluid,
        gas=oxidizer_pressurant,
        liquid_height=liquid_level,
    )

    return oxidizer_tank
