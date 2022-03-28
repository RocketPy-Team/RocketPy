# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np

from .Function import Function
from .Environment import Environment
__author__ = "Franz Masatoshi Yuri, Lucas Kierulff Balabram"
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


def calculateEquilibriumAltitude(mass, CdS, z0, v0 = 0, env = None, eps = 1e-3, seeGraphs = None):

    def check_constant(f, eps):
        for i in range(len(f) - 2):
            if abs(f[i + 2] - f[i + 1]) < eps and abs(f[i + 1] - f[i]) < eps:
                return i
        
        return None

    g = -9.80665
    if env == None:
        environment = Environment(
    railLength=5.0,
    latitude=0,
    longitude=0,
    elevation=1000,
    date=(2020, 3, 4, 12)
) 
    else:
        environment = env


    def du(u, z):
        return (u[1], g + environment.density(z) * (u[1] ** 2) * CdS / (2 * mass))

    u0 = [z0, v0]

    ts = np.arange(0, 30, 0.05)
    # must improve the timesteps 
    us = odeint(du, u0, ts)

    constant_index = check_constant(us[:,1], eps)

    if constant_index is not None:
        infos = {'time':ts[constant_index], 'height':us[constant_index,0], 'velocity':us[constant_index,1]}

    if seeGraphs:
        fig1 = plt.figure()
        plt.plot(ts, us[:,0])
        plt.title("Height")
        if constant_index is not None:
            plt.scatter(ts[constant_index], us[constant_index,0], color="red")

        plt.show()

        fig2 = plt.figure()
        plt.plot(ts, us[:,1])
        if constant_index is not None:
            plt.scatter(ts[constant_index], us[constant_index,1], color="red")
        plt.title("Velocity")
        plt.show()

    return infos
