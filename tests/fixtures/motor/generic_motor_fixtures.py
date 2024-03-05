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
