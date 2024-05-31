import pytest

from rocketpy import GenericMotor

# Fixtures
## Motors and rockets


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


@pytest.fixture
def generic_motor_cesaroni_M1520():
    """Defines a Cesaroni M1520 motor for the Prometheus rocket using the
    GenericMotor class.

    Returns
    -------
    GenericMotor
        The Cesaroni M1520 motor for the Prometheus rocket.
    """
    return GenericMotor(
        # burn specs: https://www.thrustcurve.org/simfiles/5f4294d20002e900000006b1/
        thrust_source="data/motors/cesaroni/Cesaroni_7579M1520-P.eng",
        burn_time=4.897,
        propellant_initial_mass=3.737,
        dry_mass=2.981,
        # casing specs: Pro98 3G Gen2 casing
        chamber_radius=0.064,
        chamber_height=0.548,
        chamber_position=0.274,
        nozzle_radius=0.027,
    )
