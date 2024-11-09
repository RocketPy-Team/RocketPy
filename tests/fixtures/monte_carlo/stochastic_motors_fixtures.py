"""This module contains fixtures for the stochastic motors tests."""

import pytest

from rocketpy.mathutils.function import Function
from rocketpy.stochastic import StochasticGenericMotor, StochasticSolidMotor


@pytest.fixture
def stochastic_solid_motor(cesaroni_m1670):
    """A Stochastic Solid Motor fixture for the Cesaroni M1670 motor.

    Parameters
    ----------
    cesaroni_m1670 : SolidMotor
        This is another fixture.

    Returns
    -------
    StochasticSolidMotor
        The stochastic solid motor object.
    """
    return StochasticSolidMotor(
        solid_motor=cesaroni_m1670,
        thrust_source=[
            "data/motors/cesaroni/Cesaroni_M1670.eng",
            [[0, 6000], [1, 6000], [2, 6000], [3, 6000], [4, 6000]],
            Function([[0, 6000], [1, 6000], [2, 6000], [3, 6000], [4, 6000]]),
        ],
        burn_out_time=(4, 0.1),
        grains_center_of_mass_position=0.001,
        grain_density=50,
        grain_separation=1 / 1000,
        grain_initial_height=1 / 1000,
        grain_initial_inner_radius=0.375 / 1000,
        grain_outer_radius=0.375 / 1000,
        total_impulse=(6500, 1000),
        throat_radius=0.5 / 1000,
        nozzle_radius=0.5 / 1000,
        nozzle_position=0.001,
    )


@pytest.fixture
def stochastic_generic_motor(generic_motor):
    """A Stochastic Generic Motor fixture

    Parameters
    ----------
    generic_motor : GenericMotor
        This is another fixture.

    Returns
    -------
    StochasticGenericMotor
        The stochastic generic motor object.
    """
    return StochasticGenericMotor(
        generic_motor,
        thrust_source=None,
        total_impulse=None,
        burn_start_time=None,
        burn_out_time=None,
        propellant_initial_mass=None,
        dry_mass=None,
        dry_inertia_11=None,
        dry_inertia_22=None,
        dry_inertia_33=None,
        dry_inertia_12=None,
        dry_inertia_13=None,
        dry_inertia_23=None,
        chamber_radius=None,
        chamber_height=(0.5, 0.005),
        chamber_position=None,
        nozzle_radius=None,
        nozzle_position=None,
        center_of_dry_mass_position=None,
    )
