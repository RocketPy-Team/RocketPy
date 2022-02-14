# -*- coding: utf-8 -*-

__author__ = "Franz Masatoshi Yuri"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"


def compute_CdS_from_drop_test(
    terminal_velocity, rocket_mass, air_density=1.225, gravity=9.80665
):
    """Returns the parachute's CdS calculated through its final speed, air
    density in the landing point, the rocket's mass and the force of gravity
    in the landing point.

    Parameters
    ----------
    terminal_velocity : float
        Rocket's speed in m/s when landing.
    rocket_mass : float
        Rocket's mass in kg.
    air_density : float, optional
        Air density, in kg/m^3, right before the rocket lands. Default value is 1.225.
    gravity : float, optional
        Gravitational acceleration experienced by the rocket and parachute during
        descent in m/s^2. Default value is the standard gravity, 9.80665.

    Returns
    -------
    CdS : float
        Number equal to drag coefficient times reference area for parachute.

    """

    CdS = 2 * rocket_mass * gravity / ((terminal_velocity**2) * air_density)
    return CdS
