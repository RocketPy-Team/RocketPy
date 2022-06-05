# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
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


def calculateEquilibriumAltitude(
    rocket_mass, CdS, z0, v0=0, env=None, eps=1e-3, seeGraphs=True
):
    """Returns a dictionary containing the time, height and velocoty of the
    system rocket-parachute in which the terminal velocity is reached.


    Parameters
    ----------
    rocket_mass : float
        Rocket's mass in kg.
    CdS : float
        Number equal to drag coefficient times reference area for parachute.
    z0 : float
        Initial height of the rocket in meters.
    v0 : float, optional
        Rocket's initial speed in m/s.
    env : Environment, optional
        Environmental conditions at the time of the launch.
    eps : float, optional
        acceptable error in meters.
    seeGraphs : boolean, optional
        True if you want to see time vs height and time vs speed graphs,
        False otherwise.


    Returns
    -------
    infos : dictionary
        Dictionary containing the values for time, height and speed of
        the rocket when it reaches terminal velocity.
    """

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
            date=(2020, 3, 4, 12),
        )
    else:
        environment = env

    def du(z, u):
        return (
            u[1],
            g + environment.density(z) * ((u[1]) ** 2) * CdS / (2 * rocket_mass),
        )

    u0 = [z0, v0]

    ts = np.arange(0, 30, 0.05)
    # TODO: Improve the timesteps
    us = solve_ivp(du, (0, 30), u0, vectorized=True, method="LSODA", max_step=0.4)
    # TODO: Check if the odeint worked
    constant_index = check_constant(us.y[1], eps)

    if constant_index is not None:
        infos = {
            "time": us.t[constant_index],
            "height": us.y[0][constant_index],
            "velocity": us.y[1][constant_index],
        }
    # TODO: Otherwise further simulations should be made with more time

    if seeGraphs:
        fig1 = plt.figure()
        plt.plot(us.t, us.y[0])
        plt.title("Height")
        if constant_index is not None:
            plt.scatter(us.t[constant_index], us.y[0][constant_index], color="red")

        plt.show()

        fig2 = plt.figure()
        plt.plot(us.t, us.y[1])
        if constant_index is not None:
            plt.scatter(us.t[constant_index], us.y[1][constant_index], color="red")
        plt.title("Velocity")
        plt.show()
    print(infos)
    return infos
