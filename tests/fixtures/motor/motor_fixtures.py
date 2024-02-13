import pytest

from rocketpy import GenericMotor, SolidMotor

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
def cesaroni_m1670_shifted():  # old name: solid_motor
    """Create a simple object of the SolidMotor class to be used in the tests.
    This is the same motor that has been used in the getting started guide for
    years. The difference relies in the thrust_source, which was shifted for
    testing purposes.

    Returns
    -------
    rocketpy.SolidMotor
        A simple object of the SolidMotor class
    """
    example_motor = SolidMotor(
        thrust_source="tests/fixtures/motor/Cesaroni_M1670_shifted.eng",
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
        reshape_thrust_curve=(5, 3000),
    )
    return example_motor


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