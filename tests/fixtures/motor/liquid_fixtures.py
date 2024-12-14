import pytest

from rocketpy import Fluid, LiquidMotor


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
def lox_fluid_seblm():
    """A liquid oxygen fixture whose density comes
    from testing data.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="O2", density=1141.7)


@pytest.fixture
def nitrogen_fluid_seblm():
    """A nitrogen gas fixture whose density comes
    from testing data.

    Returns
    -------
    rocketpy.Fluid
        An object of the Fluid class.
    """
    return Fluid(name="N2", density=51.75)


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
        thrust_source="data/rockets/berkeley/test124_Thrust_Curve.csv",
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
