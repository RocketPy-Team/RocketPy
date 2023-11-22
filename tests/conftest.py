import datetime

import numericalunits
import numpy as np
import pytest

from rocketpy import (
    CylindricalTank,
    Environment,
    EnvironmentAnalysis,
    Flight,
    Fluid,
    Function,
    GenericMotor,
    HybridMotor,
    LevelBasedTank,
    LiquidMotor,
    MassBasedTank,
    NoseCone,
    Parachute,
    RailButtons,
    Rocket,
    SolidMotor,
    SphericalTank,
    Tail,
    TrapezoidalFins,
    UllageBasedTank,
)

# Pytest configuration


def pytest_addoption(parser):
    """Add option to run slow tests. This is used to skip slow tests by default.

    Parameters
    ----------
    parser : _pytest.config.argparsing.Parser
        Parser object to which the option is added.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add marker to run slow tests. This is used to skip slow tests by default.

    Parameters
    ----------
    config : _pytest.config.Config
        Config object to which the marker is added.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default. This is used to skip slow tests by default.

    Parameters
    ----------
    config : _pytest.config.Config
        Config object to which the marker is added.
    items : list
        List of tests to be modified.
    """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Fixtures
## Motors and rockets


@pytest.fixture
def cesaroni_m1670():  # old name: solid_motor
    """Create a simple object of the SolidMotor class to be used in the tests.
    This is the same motor that has been used in the getting started guide for
    years.

    Returns
    -------
    rocketpy.SolidMotor
        A simple object of the SolidMotor class
    """
    example_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    return example_motor


@pytest.fixture
def calisto_motorless():
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same rocket that has been used in the getting started guide for years
    but without a motor.

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    return calisto


@pytest.fixture
def calisto(calisto_motorless, cesaroni_m1670):  # old name: rocket
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same rocket that has been used in the getting started guide for
    years. The Calisto rocket is the Projeto Jupiter's project launched at the
    2019 Spaceport America Cup.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    cesaroni_m1670 : rocketpy.SolidMotor
        An object of the SolidMotor class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(cesaroni_m1670, position=-1.373)
    return calisto


@pytest.fixture
def calisto_nose_to_tail(cesaroni_m1670):
    """Create a simple object of the Rocket class to be used in the tests. This
    is the same as the calisto fixture, but with the coordinate system
    orientation set to "nose_to_tail" instead of "tail_to_nose". This allows to
    check if the coordinate system orientation is being handled correctly in
    the code.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        An object of the SolidMotor class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Rocket
        The Calisto rocket with the coordinate system orientation set to
        "nose_to_tail". Rail buttons are already set, as well as the motor.
    """
    calisto = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="nose_to_tail",
    )
    calisto.add_motor(cesaroni_m1670, position=1.373)
    calisto.set_rail_buttons(
        upper_button_position=-0.082,
        lower_button_position=0.618,
        angular_position=45,
    )
    return calisto


@pytest.fixture
def calisto_liquid_modded(calisto_motorless, liquid_motor):
    """Create a simple object of the Rocket class to be used in the tests. This
    is an example of the Calisto rocket with a liquid motor.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    liquid_motor : rocketpy.LiquidMotor

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(liquid_motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_hybrid_modded(calisto_motorless, hybrid_motor):
    """Create a simple object of the Rocket class to be used in the tests. This
    is an example of the Calisto rocket with a hybrid motor.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    hybrid_motor : rocketpy.HybridMotor

    Returns
    -------
    rocketpy.Rocket
        A simple object of the Rocket class
    """
    calisto = calisto_motorless
    calisto.add_motor(hybrid_motor, position=-1.373)
    return calisto


@pytest.fixture
def calisto_robust(
    calisto,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_rail_buttons,
    calisto_main_chute,
    calisto_drogue_chute,
):
    """Create an object class of the Rocket class to be used in the tests. This
    is the same Calisto rocket that was defined in the calisto fixture, but with
    all the aerodynamic surfaces and parachutes added. This avoids repeating the
    same code in all tests.

    Parameters
    ----------
    calisto : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    calisto_nose_cone : rocketpy.NoseCone
        The nose cone of the Calisto rocket. This is a pytest fixture too.
    calisto_tail : rocketpy.Tail
        The boat tail of the Calisto rocket. This is a pytest fixture too.
    calisto_trapezoidal_fins : rocketpy.TrapezoidalFins
        The trapezoidal fins of the Calisto rocket. This is a pytest fixture
    calisto_rail_buttons : rocketpy.RailButtons
        The rail buttons of the Calisto rocket. This is a pytest fixture too.
    calisto_main_chute : rocketpy.Parachute
        The main parachute of the Calisto rocket. This is a pytest fixture too.
    calisto_drogue_chute : rocketpy.Parachute
        The drogue parachute of the Calisto rocket. This is a pytest fixture

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    # we follow this format: calisto.add_surfaces(surface, position)
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.add_surfaces(calisto_trapezoidal_fins, -1.168)
    # calisto.add_surfaces(calisto_rail_buttons, -1.168)
    # TODO: if I use the line above, the calisto won't have rail buttons attribute
    #       we need to apply a check in the add_surfaces method to set the rail buttons
    calisto.set_rail_buttons(
        upper_button_position=0.082,
        lower_button_position=-0.618,
        angular_position=45,
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)
    return calisto


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


@pytest.fixture
def generic_motor():
    """An example of a generic motor for low accuracy simulations.

    Returns
    -------
    rocketpy.GenericMotor
    """
    motor = GenericMotor(
        burn_time=(2, 7),
        thrust_source=lambda t: 2000 - 100 * (t - 2),
        chamber_height=0.5,
        chamber_radius=0.075,
        chamber_position=-0.25,
        propellant_initial_mass=5.0,
        nozzle_position=-0.5,
        nozzle_radius=0.075,
        dry_mass=8.0,
        dry_inertia=(0.2, 0.2, 0.08),
    )

    return motor


## AeroSurfaces


@pytest.fixture
def calisto_nose_cone():
    """The nose cone of the Calisto rocket.

    Returns
    -------
    rocketpy.NoseCone
        The nose cone of the Calisto rocket.
    """
    return NoseCone(
        length=0.55829,
        kind="vonkarman",
        base_radius=0.0635,
        rocket_radius=0.0635,
        name="calisto_nose_cone",
    )


@pytest.fixture
def calisto_tail():
    """The boat tail of the Calisto rocket.

    Returns
    -------
    rocketpy.Tail
        The boat tail of the Calisto rocket.
    """
    return Tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        rocket_radius=0.0635,
        name="calisto_tail",
    )


@pytest.fixture
def calisto_trapezoidal_fins():
    """The trapezoidal fins of the Calisto rocket.

    Returns
    -------
    rocketpy.TrapezoidalFins
        The trapezoidal fins of the Calisto rocket.
    """
    return TrapezoidalFins(
        n=4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        rocket_radius=0.0635,
        name="calisto_trapezoidal_fins",
        cant_angle=0,
        sweep_length=None,
        sweep_angle=None,
        airfoil=None,
    )


@pytest.fixture
def calisto_rail_buttons():
    """The rail buttons of the Calisto rocket.

    Returns
    -------
    rocketpy.RailButtons
        The rail buttons of the Calisto rocket.
    """
    return RailButtons(
        buttons_distance=0.7,
        angular_position=45,
        name="Rail Buttons",
    )


## Parachutes


@pytest.fixture
def calisto_drogue_parachute_trigger():
    """The trigger for the drogue parachute of the Calisto rocket.

    Returns
    -------
    function
        The trigger for the drogue parachute of the Calisto rocket.
    """

    def drogue_trigger(p, h, y):
        # activate drogue when vertical velocity is negative
        return True if y[5] < 0 else False

    return drogue_trigger


@pytest.fixture
def calisto_main_parachute_trigger():
    """The trigger for the main parachute of the Calisto rocket.

    Returns
    -------
    function
        The trigger for the main parachute of the Calisto rocket.
    """

    def main_trigger(p, h, y):
        # activate main when vertical velocity is <0 and altitude is below 800m
        return True if y[5] < 0 and h < 800 else False

    return main_trigger


@pytest.fixture
def calisto_main_chute(calisto_main_parachute_trigger):
    """The main parachute of the Calisto rocket.

    Parameters
    ----------
    calisto_main_parachute_trigger : function
        The trigger for the main parachute of the Calisto rocket. This is a
        pytest fixture too.

    Returns
    -------
    rocketpy.Parachute
        The main parachute of the Calisto rocket.
    """
    return Parachute(
        name="calisto_main_chute",
        cd_s=10.0,
        trigger=calisto_main_parachute_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )


@pytest.fixture
def calisto_drogue_chute(calisto_drogue_parachute_trigger):
    """The drogue parachute of the Calisto rocket.

    Parameters
    ----------
    calisto_drogue_parachute_trigger : function
        The trigger for the drogue parachute of the Calisto rocket. This is a
        pytest fixture too.

    Returns
    -------
    rocketpy.Parachute
        The drogue parachute of the Calisto rocket.
    """
    return Parachute(
        name="calisto_drogue_chute",
        cd_s=1.0,
        trigger=calisto_drogue_parachute_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )


## Flights


@pytest.fixture
def flight_calisto(calisto, example_env):  # old name: flight
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    without the aerodynamic surfaces and parachutes. The environment is the
    simplest possible, with no parameters set.

    Parameters
    ----------
    calisto : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object of the Calisto rocket in the simplest possible
        conditions.
    """
    return Flight(
        environment=example_env,
        rocket=calisto,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_nose_to_tail(calisto_nose_to_tail, example_env):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with "nose_to_tail" coordinate system orientation, just as described in the
    calisto_nose_to_tail fixture.

    Parameters
    ----------
    calisto_nose_to_tail : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        The Calisto rocket with the coordinate system orientation set to
        "nose_to_tail".
    """
    return Flight(
        environment=example_env,
        rocket=calisto_nose_to_tail,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_robust(calisto_robust, example_env_robust):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with the aerodynamic surfaces and parachutes. The environment is a bit more
    complex than the one in the flight_calisto fixture. This time the latitude,
    longitude and elevation are set, as well as the datum and the date. The
    location refers to the Spaceport America Cup launch site, while the date is
    set to tomorrow at noon.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_env_robust : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object of the Calisto rocket in a more complex
        condition.
    """
    return Flight(
        environment=example_env_robust,
        rocket=calisto_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_custom_wind(calisto_robust, example_env_robust):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with the aerodynamic surfaces and parachutes. The environment is a bit more
    complex than the one in the flight_calisto_robust fixture. Now the wind is
    set to 5m/s (x direction) and 2m/s (y direction), constant with altitude.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_env_robust : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight

    """
    env = example_env_robust
    env.set_atmospheric_model(
        type="custom_atmosphere",
        temperature=300,
        wind_u=[(0, 5), (4000, 5)],
        wind_v=[(0, 2), (4000, 2)],
    )
    return Flight(
        environment=env,
        rocket=calisto_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


## Dimensionless motors and rockets


@pytest.fixture
def m():
    """Create a simple object of the numericalunits.m class to be used in the
    tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    numericalunits.m
        A simple object of the numericalunits.m class
    """
    return numericalunits.m


@pytest.fixture
def kg():
    """Create a simple object of the numericalunits.kg class to be used in the
    tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    numericalunits.kg
        A simple object of the numericalunits.kg class
    """
    return numericalunits.kg


@pytest.fixture
def dimensionless_cesaroni_m1670(kg, m):  # old name: dimensionless_motor
    """The dimensionless version of the Cesaroni M1670 motor. This is the same
    motor as defined in the cesaroni_m1670 fixture, but with all the parameters
    converted to dimensionless values. This allows to check if the dimensions
    are being handled correctly in the code.

    Parameters
    ----------
    kg : numericalunits.kg
        An object of the numericalunits.kg class. This is a pytest
    m : numericalunits.m
        An object of the numericalunits.m class. This is a pytest

    Returns
    -------
    rocketpy.SolidMotor
        An object of the SolidMotor class
    """
    example_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815 * kg,
        dry_inertia=(
            0.125 * (kg * m**2),
            0.125 * (kg * m**2),
            0.002 * (kg * m**2),
        ),
        center_of_dry_mass_position=0.317 * m,
        grain_number=5,
        grain_separation=5 / 1000 * m,
        grain_density=1815 * (kg / m**3),
        grain_outer_radius=33 / 1000 * m,
        grain_initial_inner_radius=15 / 1000 * m,
        grain_initial_height=120 / 1000 * m,
        nozzle_radius=33 / 1000 * m,
        throat_radius=11 / 1000 * m,
        interpolation_method="linear",
        grains_center_of_mass_position=0.397 * m,
        nozzle_position=0 * m,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    return example_motor


@pytest.fixture  # old name: dimensionless_rocket
def dimensionless_calisto(kg, m, dimensionless_cesaroni_m1670):
    """The dimensionless version of the Calisto rocket. This is the same rocket
    as defined in the calisto fixture, but with all the parameters converted to
    dimensionless values. This allows to check if the dimensions are being
    handled correctly in the code.

    Parameters
    ----------
    kg : numericalunits.kg
        An object of the numericalunits.kg class. This is a pytest fixture too.
    m : numericalunits.m
        An object of the numericalunits.m class. This is a pytest fixture too.
    dimensionless_cesaroni_m1670 : rocketpy.SolidMotor
        The dimensionless version of the Cesaroni M1670 motor. This is a pytest
        fixture too.

    Returns
    -------
    rocketpy.Rocket
        An object of the Rocket class
    """
    example_rocket = Rocket(
        radius=0.0635 * m,
        mass=14.426 * kg,
        inertia=(6.321 * (kg * m**2), 6.321 * (kg * m**2), 0.034 * (kg * m**2)),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0 * m,
        coordinate_system_orientation="tail_to_nose",
    )
    example_rocket.add_motor(dimensionless_cesaroni_m1670, position=(-1.373) * m)
    return example_rocket


## Environment


@pytest.fixture
def example_env():
    """Create a simple object of the Environment class to be used in the tests.
    This allows to avoid repeating the same code in all tests. The environment
    set here is the simplest possible, with no parameters set.

    Returns
    -------
    rocketpy.Environment
        The simplest object of the Environment class
    """
    return Environment()


@pytest.fixture
def example_env_robust():
    """Create an object of the Environment class to be used in the tests. This
    allows to avoid repeating the same code in all tests. The environment set
    here is a bit more complex than the one in the example_env fixture. This
    time the latitude, longitude and elevation are set, as well as the datum and
    the date. The location refers to the Spaceport America Cup launch site,
    while the date is set to tomorrow at noon.

    Returns
    -------
    rocketpy.Environment
        An object of the Environment class
    """
    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
    return env


@pytest.fixture
def env_analysis():
    """Create a simple object of the Environment Analysis class to be used in
    the tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    EnvironmentAnalysis
        A simple object of the Environment Analysis class
    """
    env_analysis = EnvironmentAnalysis(
        start_date=datetime.datetime(2019, 10, 23),
        end_date=datetime.datetime(2021, 10, 23),
        latitude=39.3897,
        longitude=-8.28896388889,
        start_hour=6,
        end_hour=18,
        surface_data_file="./data/weather/EuroC_single_level_reanalysis_2002_2021.nc",
        pressure_level_data_file="./data/weather/EuroC_pressure_levels_reanalysis_2001-2021.nc",
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        max_expected_altitude=None,
    )

    return env_analysis


## Functions


@pytest.fixture
def linear_func():
    """Create a linear function based on a list of points. The function
    represents y = x and may be used on different tests.

    Returns
    -------
    Function
        A linear function representing y = x.
    """
    return Function(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
    )


@pytest.fixture
def linearly_interpolated_func():
    """Create a linearly interpolated function based on a list of points.

    Returns
    -------
    Function
        Piece-wise linearly interpolated, with constant extrapolation
    """
    return Function(
        [[0, 0], [1, 7], [2, -3], [3, -1], [4, 3]],
        interpolation="spline",
        extrapolation="constant",
    )


@pytest.fixture
def spline_interpolated_func():
    """Create a spline interpolated function based on a list of points.

    Returns
    -------
    Function
        Spline interpolated, with natural extrapolation
    """
    return Function(
        [[0, 0], [1, 7], [2, -3], [3, -1], [4, 3]],
        interpolation="spline",
        extrapolation="natural",
    )


@pytest.fixture
def func_from_csv():
    """Create a function based on a csv file. The csv file contains the
    coordinates of the E473 airfoil at 10e6 degrees, but anything else could be
    used here as long as it is a csv file.

    Returns
    -------
    rocketpy.Function
        A function based on a csv file.
    """
    func = Function(
        source="tests/fixtures/airfoils/e473-10e6-degrees.csv",
    )
    return func


@pytest.fixture
def func_2d_from_csv():
    """Create a 2d function based on a csv file.

    Returns
    -------
    rocketpy.Function
        A function based on a csv file.
    """
    # Do not define any of the optional parameters so that the tests can check
    # if the defaults are being used correctly.
    func = Function(
        source="tests/fixtures/function/2d.csv",
    )
    return func
