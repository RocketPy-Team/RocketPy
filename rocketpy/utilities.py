# -*- coding: utf-8 -*-
__author__ = "Franz Masatoshi Yuri, Lucas Kierulff Balabram, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from .Environment import Environment
from .EnvironmentAnalysis import EnvironmentAnalysis
from .Function import Function


# TODO: Needs tests
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


# TODO: Needs tests


def calculateEquilibriumAltitude(
    rocket_mass,
    CdS,
    z0,
    v0=0,
    env=None,
    eps=1e-3,
    max_step=0.1,
    seeGraphs=True,
    g=9.80665,
    estimated_final_time=10,
):
    """Returns a dictionary containing the time, altitude and velocity of the
    system rocket-parachute in which the terminal velocity is reached.


    Parameters
    ----------
    rocket_mass : float
        Rocket's mass in kg.
    CdS : float
        Number equal to drag coefficient times reference area for parachute.
    z0 : float
        Initial altitude of the rocket in meters.
    v0 : float, optional
        Rocket's initial speed in m/s. Must be negative
    env : Environment, optional
        Environmental conditions at the time of the launch.
    eps : float, optional
        acceptable error in meters.
    max_step: float, optional
        maximum allowed time step size to solve the integration
    seeGraphs : boolean, optional
        True if you want to see time vs altitude and time vs speed graphs,
        False otherwise.
    g : float, optional
        Gravitational acceleration experienced by the rocket and parachute during
        descent in m/s^2. Default value is the standard gravity, 9.80665.
    estimated_final_time: float, optional
        Estimative of how much time (in seconds) will spend until vertical terminal
        velocity is reached. Must be positive. Default is 10. It can affect the final
        result if the value is not high enough. Increase the estimative in case the
        final solution is not founded.


    Returns
    -------
    altitudeFunction: Function
        Altitude as a function of time. Always a Function object.
    velocityFunction:
        Vertical velocity as a function of time. Always a Function object.
    final_sol : dictionary
        Dictionary containing the values for time, altitude and speed of
        the rocket when it reaches terminal velocity.
    """
    final_sol = {}

    if not v0 < 0:
        print("Please set a valid negative value for v0")
        return None

    # TODO: Improve docs
    def check_constant(f, eps):
        """_summary_

        Parameters
        ----------
        f : array, list
            _description_
        eps : float
            _description_

        Returns
        -------
        int, None
            _description_
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

        Parameters
        ----------
        z : float
            _description_
        u : float
            velocity, in m/s, at a given z altitude

        Returns
        -------
        float
            _description_
        """
        return (
            u[1],
            -g + environment.density(z) * ((u[1]) ** 2) * CdS / (2 * rocket_mass),
        )

    u0 = [z0, v0]

    us = solve_ivp(
        fun=du,
        t_span=(0, estimated_final_time),
        y0=u0,
        vectorized=True,
        method="LSODA",
        max_step=max_step,
    )

    constant_index = check_constant(us.y[1], eps)

    # TODO: Improve docs by explaining what is happening below with constant_index
    if constant_index is not None:
        final_sol = {
            "time": us.t[constant_index],
            "altitude": us.y[0][constant_index],
            "velocity": us.y[1][constant_index],
        }

    altitudeFunction = Function(
        source=np.array(list(zip(us.t, us.y[0])), dtype=np.float64),
        inputs="Time (s)",
        outputs="Altitude (m)",
        interpolation="linear",
    )

    velocityFunction = Function(
        source=np.array(list(zip(us.t, us.y[1])), dtype=np.float64),
        inputs="Time (s)",
        outputs="Vertical Velocity (m/s)",
        interpolation="linear",
    )

    if seeGraphs:
        altitudeFunction()
        velocityFunction()

    return altitudeFunction, velocityFunction, final_sol


def compareEnvironments(env1, env2, names=["env1", "env2"]):
    """Compares two environments and plots everything.
    Useful when comparing Environment Analysis results against forecasts.

    Parameters
    ----------
    env1 : Environment or EnvironmentAnalysis
        Environment to compare with.
    env2: Environment or EnvironmentAnalysis
        Environment to compare with.

    Returns
    -------
    diff: Dict
        Dictionary with the differences.
    """

    # Raise TypeError if env1 or env2 are not Environment nor EnvironmentAnalysis
    if not isinstance(env1, Environment) and not isinstance(env1, EnvironmentAnalysis):
        raise TypeError("env1 must be an Environment or EnvironmentAnalysis object")

    if not isinstance(env2, Environment) and not isinstance(env2, EnvironmentAnalysis):
        raise TypeError("env2 must be an Environment or EnvironmentAnalysis object")

    # If instances are Environment Analysis, get the equivalent environment
    if isinstance(env1, EnvironmentAnalysis):
        env1.process_temperature_profile_over_average_day()
        env1.process_pressure_profile_over_average_day()
        env1.process_wind_velocity_x_profile_over_average_day()
        env1.process_wind_velocity_y_profile_over_average_day()
        print()
    if isinstance(env2, EnvironmentAnalysis):
        env2.process_temperature_profile_over_average_day()
        env2.process_pressure_profile_over_average_day()
        env2.process_wind_velocity_x_profile_over_average_day()
        env2.process_wind_velocity_y_profile_over_average_day()
        print()

    # Plot graphs
    print("\n\nAtmospheric Model Plots")
    # Create height grid
    grid = np.linspace(env1.elevation, env1.maxExpectedHeight)

    # Create figure
    plt.figure(figsize=(9, 9))

    # Create wind speed and wind direction subplot
    ax1 = plt.subplot(221)
    ax1.plot(
        [env1.windSpeed(i) for i in grid],
        grid,
        "#ff7f0e",
        label="Speed of Sound " + names[0],
    )
    ax1.plot(
        [env2.windSpeed(i) for i in grid],
        grid,
        "#ff7f0e",
        label="Speed of Sound " + names[1],
    )
    ax1.set_xlabel("Wind Speed (m/s)", color="#ff7f0e")
    ax1.tick_params("x", colors="#ff7f0e")
    # ax1up = ax1.twiny()
    # ax1up.plot(
    #     [env1.windDirection(i) for i in grid],
    #     grid,
    #     color="#1f77b4",
    #     label="Density "+names[0],
    # )
    # ax1up.plot(
    #     [env2.windDirection(i) for i in grid],
    #     grid,
    #     color="#1f77b4",
    #     label="Density "+names[1],
    # )
    # ax1up.set_xlabel("Wind Direction (°)", color="#1f77b4")
    # ax1up.tick_params("x", colors="#1f77b4")
    # ax1up.set_xlim(0, 360)
    ax1.set_ylabel("Height Above Sea Level (m)")
    ax1.grid(True)

    # # Create density and speed of sound subplot
    # ax2 = plt.subplot(222)
    # ax2.plot(
    #     [env1.speedOfSound(i) for i in grid],
    #     grid,
    #     "#ff7f0e",
    #     label="Speed of Sound",
    # )
    # ax2.set_xlabel("Speed of Sound (m/s)", color="#ff7f0e")
    # ax2.tick_params("x", colors="#ff7f0e")
    # ax2up = ax2.twiny()
    # ax2up.plot([env1.density(i) for i in grid], grid, color="#1f77b4", label="Density")
    # ax2up.set_xlabel("Density (kg/m³)", color="#1f77b4")
    # ax2up.tick_params("x", colors="#1f77b4")
    # ax2.set_ylabel("Height Above Sea Level (m)")
    # ax2.grid(True)

    # Create wind u and wind v subplot
    ax3 = plt.subplot(223)
    ax3.plot([env1.windVelocityX(i) for i in grid], grid, label="Wind U " + names[0])
    ax3.plot([env1.windVelocityY(i) for i in grid], grid, label="Wind V " + names[0])
    ax3.plot([env2.windVelocityX(i) for i in grid], grid, label="Wind U " + names[1])
    ax3.plot([env2.windVelocityY(i) for i in grid], grid, label="Wind V " + names[1])
    ax3.legend(loc="best").set_draggable(True)
    ax3.set_ylabel("Height Above Sea Level (m)")
    ax3.set_xlabel("Wind Speed (m/s)")
    ax3.grid(True)

    # Create pressure and temperature subplot
    ax4 = plt.subplot(224)
    ax4.plot([env1.pressure(i) / 100 for i in grid], grid, "#ff7f0e", label="Pressure")
    ax4.set_xlabel("Pressure (hPa)", color="#ff7f0e")
    ax4.tick_params("x", colors="#ff7f0e")
    ax4up = ax4.twiny()
    ax4up.plot(
        [env1.temperature(i) for i in grid],
        grid,
        color="#1f77b4",
        label="Temperature",
    )
    ax4up.set_xlabel("Temperature (K)", color="#1f77b4")
    ax4up.tick_params("x", colors="#1f77b4")
    ax4.set_ylabel("Height Above Sea Level (m)")
    ax4.grid(True)

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.show()

    return None  # diff
