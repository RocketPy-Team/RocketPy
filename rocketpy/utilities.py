# -*- coding: utf-8 -*-
__author__ = "Franz Masatoshi Yuri, Lucas Kierulff Balabram, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .Environment import Environment
from .Function import Function

# Parachutes related functions

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


def compareTrajectories(
    trajectory_list,
    names=None,
    legend=True,
):
    """Creates a trajectory plot combining the trajectories listed.
    This function was created based two source-codes:
    - Mateus Stano: https://github.com/RocketPy-Team/Hackathon_2020/pull/123
    - Dyllon Preston: https://github.com/Dyllon-P/MBS-Template/blob/main/MBS.py

    Parameters
    ----------
    trajectory_list : list, array
        List of trajectories. Must be in the form of [trajectory_1, trajectory_2, ..., trajectory_n]
        where each element is a list with the arrays regarding positions in x, y and z [x, y, z].
        The trajectories must be in the same reference frame. The z coordinate must be referenced
        to the ground or to the sea level, but it is important that all trajectories are passed
        in the same reference.
    names : list, optional
        List of strings with the name of each trajectory inputted. The names must be in
        the same order as the trajectories in trajectory_list. If no names are passed, the
        trajectories will be named as "Trajectory 1", "Trajectory 2", ..., "Trajectory n".
    legend : boolean, optional
        Whether legend will or will not be plotted. Default is True

    Returns
    -------
    None

    """
    # TODO: Allow the user to catch different planes (xy, yz, xz) from the main plot (this can be done in a separate function)
    # TODO: Allow the user to set the colors or color style
    # TODO: Allow the user to set the line style

    # Initialize variables
    maxX, maxY, maxZ, minX, minY, minZ, maxXY, minXY = 0, 0, 0, 0, 0, 0, 0, 0

    names = (
        [("Trajectory " + str(i + 1)) for i in range(len(trajectory_list))]
        if names == None
        else names
    )

    # Create the figure
    fig1 = plt.figure(figsize=(9, 9))
    ax1 = plt.subplot(111, projection="3d")

    # Iterate through trajectories
    for i, trajectory in enumerate(trajectory_list):

        x, y, z = trajectory

        # Find max/min values for each component
        maxX = max(x) if max(x) > maxX else maxX
        maxY = max(y) if max(y) > maxY else maxY
        maxZ = max(z) if max(z) > maxZ else maxZ
        minX = min(x) if min(x) > minX else minX
        minY = min(x) if min(x) > minX else minX
        minZ = min(z) if min(z) > minZ else minZ
        maxXY = max(maxX, maxY) if max(maxX, maxY) > maxXY else maxXY
        minXY = min(minX, minY) if min(minX, minY) > minXY else minXY

        # Add Trajectory as a plot in main figure
        ax1.plot(x, y, z, linewidth="2", label=names[i])

    # Plot settings
    ax1.scatter(0, 0, 0)
    ax1.set_xlabel("X - East (m)")
    ax1.set_ylabel("Y - North (m)")
    ax1.set_zlabel("Z - Altitude (m)")
    ax1.set_title("Flight Trajectories Comparison")
    ax1.set_zlim3d([minZ, maxZ])
    ax1.set_ylim3d([minXY, maxXY])
    ax1.set_xlim3d([minXY, maxXY])
    ax1.view_init(15, 45)
    if legend:
        plt.legend()
    plt.show()

    return None


def compareFlightTrajectories(
    flight_list,
    names=None,
    legend=True,
):
    """Creates a trajectory plot that is the combination of the trajectories of
    the Flight objects passed via a Python list.

    Parameters
    ----------
    flight_list : list, array
        List of FLight objects. The flights must be in the same reference frame.
    names : list, optional
        List of strings with the name of each trajectory inputted. The names must be in
        the same order as the trajectories in flight_list
    legend : boolean, optional
        Whether legend will or will not be included. Default is True

    Returns
    -------
    None

    """
    # TODO: Allow the user to catch different planes (xy, yz, xz) from the main plot
    # TODO: Allow the user to set the colors or color style
    # TODO: Allow the user to set the line style

    # Iterate through Flight objects and create a list of trajectories
    trajectory_list = []
    for flight in flight_list:

        # Check post process
        if flight.postProcessed is False:
            flight.postProcess()

        # Get trajectories
        x = flight.x[:, 1]
        y = flight.y[:, 1]
        z = flight.z[:, 1] - flight.env.elevation
        trajectory_list.append([x, y, z])

    # Call compareTrajectories function to do the hard work
    compareTrajectories(trajectory_list, names, legend)

    return None


def compareAllInfo(flight_list, names=None):
    """Creates a plot with the altitude, velocity and acceleration of the
    Flight objects passed via a Python list.

    Parameters
    ----------
    flight_list : list, array
        List of FLight objects. The flights must be in the same reference frame.
    names : list, optional
        List of strings with the name of each trajectory inputted. The names must be in
        the same order as the trajectories in flight_list

    Returns
    -------
    None

    """
    return None


def create_dispersion_dictionary(filename):
    """Creates a dictionary with the rocket data provided by a .csv file.
    File should be organized in four columns: attribute_class, parameter_name,
    mean_value, standard_deviation. The first row should be the header.
    It is advised to use ";" as separator, but "," should work on most of cases.

    Parameters
    ----------
    filename : string
        String with the path to the .csv file.

    Returns
    -------
    dictionary
        Dictionary with all rocket data to be used in dispersion analysis.
    """
    try:
        file = np.genfromtxt(
            filename, usecols=(1, 2, 3), skip_header=1, delimiter=";", dtype=str
        )
    except:
        print(
            "Error: The delimiter should be ';'. Using ',' instead, be aware that some resources might not work as expected. Please consider changing the delimiter to ';'."
        )
        file = np.genfromtxt(
            filename, usecols=(1, 2, 3), skip_header=1, delimiter=",", dtype=str
        )
    analysis_parameters = dict()
    for row in file:
        if row[0] != "":
            if row[2] == "":
                try:
                    analysis_parameters[row[0].strip()] = float(row[1])
                except:
                    analysis_parameters[row[0].strip()] = eval(row[1])
            else:
                try:
                    analysis_parameters[row[0].strip()] = (float(row[1]), float(row[2]))
                except:
                    analysis_parameters[row[0].strip()] = ""
    return analysis_parameters
