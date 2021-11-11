# -*- coding: utf-8 -*-

__author__ = "Franz Masatoshi Yuri"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"



def compute_CdS_from_drop_test(terminal_velocity, rocket_mass, air_density=1.225, gravity=9.80665):

    """Returns the parachute's Cds calculated through its final speed, air
    density in the landing point, the rocket's mass and the force of gravity
    in the landing point.

    Parameters
    ----------
    final_speed : float
        Rocket's speed in m/s^2 when landing.
    final_air_density : float, optional
        Air density, in kg/m^3, right before the rocket lands. Default value is 1.225.

    rocket_mass : float
        Rocket's mass in kg.
    gravity : float
        Gravitational acceleration experienced by the rocket and parachute during descent in m/s^2. Default value is the standard gravity, 9.80665.


    Returns
    -------
    cds : float
        Number equal to drag coefficient times reference area for parachute.

    """

    cds = 2 * rocket_mass * gravity / ((final_speed ** 2) * final_air_density)
    return cds
