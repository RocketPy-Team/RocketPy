# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from .Environment import Environment

__author__ = "Franz Masatoshi Yuri, Lucas Kierulff Balabram, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


def compute_CdS_from_drop_test(
    terminal_velocity, rocket_mass, air_density=1.225, g=9.80665
):
    """Returns the parachute's CdS calculated through its final speed, air
    density in the landing point, the rocket's mass and the force of gravity
    in the landing point.

    Parameters
    ----------
    terminal_velocity : float
        Rocket's speed in m/s when landing.
    rocket_mass : float
        Rocket's dry mass in kg.
    air_density : float, optional
        Air density, in kg/m^3, right before the rocket lands. Default value is 1.225.
    g : float, optional
        Gravitational acceleration experienced by the rocket and parachute during
        descent in m/s^2. Default value is the standard gravity, 9.80665.

    Returns
    -------
    CdS : float
        Number equal to drag coefficient times reference area for parachute.

    """

    return 2 * rocket_mass * g / ((terminal_velocity**2) * air_density)


def calculateEquilibriumAltitude(
    rocket_mass, CdS, z0, v0=0, env=None, eps=1e-3, max_step=0.1, seeGraphs=True, g=9.80665, estimated_final_time=10
):
    """Returns a dictionary containing the time, height and velocity of the
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
        Rocket's initial speed in m/s. Must be negative
    env : Environment, optional
        Environmental conditions at the time of the launch.
    eps : float, optional
        acceptable error in meters.
    max_step: float, optional
        maximum allowed time step size to solve the integration
    seeGraphs : boolean, optional
        True if you want to see time vs height and time vs speed graphs,
        False otherwise.
    g : float, optional
        Gravitational acceleration experienced by the rocket and parachute during
        descent in m/s^2. Default value is the standard gravity, 9.80665.
    estimated_final_time: float, optional
        Estimative of how much time (in seconds) will spend until vertical terminal velocity is reached. Must be positive. Default is 10. It can affect the final result if the value is not high enough. Increase the estimative in case the final solution is not founded.

    Returns
    -------
    final_sol : dictionary
        Dictionary containing the values for time, height and speed of
        the rocket when it reaches terminal velocity.
    """
    final_sol = {}

    if not v0 < 0:
        print("Please set a valid negative value for v0")
        return None

    # TODO: Improve docs
    def check_constant(f, eps):
        """_summary_

        Args:
            f (_type_): _description_
            eps (_type_): _description_

        Returns:
            _type_: _description_
        """
        for i in range(len(f) - 2):
            if abs(f[i + 2] - f[i + 1]) < eps and abs(f[i + 1] - f[i]) < eps:
                return i
        return None

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

    # TODO: Improve docs
    def du(z, u):
        """_summary_

        Args:
            z (_type_): _description_
            u (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            u[1],
            - g + environment.density(z) *
            ((u[1]) ** 2) * CdS / (2 * rocket_mass),
        )

    u0 = [z0, v0]

    us = solve_ivp(fun=du,
                   t_span=(0, estimated_final_time),
                   y0=u0,
                   vectorized=True,
                   method="LSODA",
                   max_step=max_step
                   )

    constant_index = check_constant(us.y[1], eps)

    # TODO: Improve docs by explaining what is happening below with constant_index
    if constant_index is not None:
        final_sol = {
            "time": us.t[constant_index],
            "height": us.y[0][constant_index],
            "velocity": us.y[1][constant_index],
        }

    # TODO: Convert result from solve_ivp to Function objects
    if seeGraphs:
        fig1 = plt.figure(figsize=(4, 3))
        plt.plot(us.t, us.y[0], label="Height", color="blue")
        plt.title("Height (m) x time (s)")
        plt.xlim(0, max(us.t))
        plt.ylim(min(us.y[0]), z0)
        plt.xlabel("Time (s)")
        plt.ylabel("Height (m)")
        if constant_index is not None:
            plt.scatter(us.t[constant_index], us.y[0]
                        [constant_index], color="red", label="Terminal Velocity is reached")
        plt.legend()
        plt.show()

        fig2 = plt.figure(figsize=(4, 3))
        plt.plot(us.t, us.y[1], label="Velocity", color="blue")
        if constant_index is not None:
            plt.scatter(us.t[constant_index], us.y[1]
                        [constant_index], color="red", label="Terminal Velocity is reached")
        plt.title("Vertical velocity x time (s)")
        plt.xlim(0, max(us.t))
        plt.ylim(min(us.y[1])-2, 0)
        plt.xlabel("Time (s)")
        plt.ylabel("Vertical velocity (m)")
        plt.legend()
        plt.show()

    return final_sol
