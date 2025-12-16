import pytest

from rocketpy import HybridMotor


@pytest.fixture
def hybrid_motor(oxidizer_tank):
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

    motor.add_tank(oxidizer_tank, position=0.3)

    return motor
