# -*- coding: utf-8 -*-
__author__ = "Franz Masatoshi Yuri, Lucas Kierulff Balabram, Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"
import math
import simplekml
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib.patches import Ellipse

import numpy as np
from scipy.integrate import solve_ivp

from .Environment import Environment
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

    return 2 * rocket_mass * g / ((terminal_velocity ** 2) * air_density)


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


def haversine(lat0, lon0, distance, bearing, R=6.3781 * (10 ** 6)):
    """ returns a tuple with new latitude and longitude
    considering 1 cm or less to be indifferent
    
    Parameters
    ----------
    lat0 : float
        Rocket's latitude of launch in degrees.
    lon0 : float
        Rocket's longitude of launch in degrees.
    distance : float
        New distance from launching point in meters.
    bearing : float
        Azimuth from launching point in degrees.
    R : float, optional
        Earth radius. Default value is 6.3781e10.

    Returns
    -------
    coordinates : tuple
        New coordinates expressed by a tuple in 
    format(new latitude, new lungitude), in degrees.

    """

    lat1 = np.deg2rad(lat0)
    lon1 = np.deg2rad(lon0)

    if abs(distance * math.sin(bearing)) < 1e-2:
        lat2 = lat1
    else:
        lat2 = np.rad2deg(
            math.asin(
                math.sin(lat1) * math.cos(distance / R)
                + math.cos(lat1) * math.sin(distance / R) * math.cos(bearing)
            )
        )
    if abs(distance * math.cos(bearing)) < 1e-2:
        lon2 = lon1
    else:
        lon2 = np.rad2deg(
            (
                lon1
                + math.atan2(
                    math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                    math.cos(distance / R) - math.sin(lat1) * math.sin(lat2),
                )
            )
        )
    coordinates = (lat2, lon2)
    return coordinates


def exportElipsesToKML(impact_ellipses, filename, origin_lat, origin_lon):
    """Generates a KML file with the ellipses on the impact point.
    Parameters
    ----------
    impact_ellipses : matplolib.patches.Ellipse
        Contains ellipse details for the plot. 
    filename : String
        Name to the KML exported file.
    origin_lat : float
        Latitute degrees of the Ellipse center.
    origin_lon : float
        Longitudeorigin_lat : float
        Latitute degrees of the Ellipse center. degrees of the Ellipse center.
    """

    outputs = []

    for impactEll in impact_ellipses:
        # Get ellipse path points
        center = impactEll.get_center()
        width = impactEll.get_width()
        height = impactEll.get_height()
        angle = np.deg2rad(impactEll.get_angle())
        print("angle", np.rad2deg(angle))
        points = []

        resolution = 1000
        for i in range(resolution):
            x = width / 2 * math.cos(2 * np.pi * i / resolution)
            y = height / 2 * math.sin(2 * np.pi * i / resolution)
            x_rot = center[0] + x * math.cos(angle) - y * math.sin(angle)
            y_rot = center[1] + x * math.sin(angle) + y * math.cos(angle)
            points.append((x_rot, y_rot))
        # points = impactEll.get_verts()
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], "r-")

        # Convert path points to latlon
        lat_lon_points = []
        for point in points:
            x = point[0]
            y = point[1]

            # Convert to distance and bearing
            d = math.sqrt((x ** 2 + y ** 2))
            brng = math.atan2(x, y)
            # Convert to lat lon
            lat_lon_points.append(haversine(origin_lat, origin_lon, d, brng))

        # Export string
        outputs.append(lat_lon_points)
    plt.show()

    # Prepare data to KML file
    kml_data = []
    for i in range(len(outputs)):
        temp = []
        for j in range(len(outputs[i])):
            temp.append((outputs[i][j][1], outputs[i][j][0]))  # log, lat
        kml_data.append(temp)

    # Export to KML
    kml = simplekml.Kml()

    for i in range(len(outputs)):
        mult_ell = kml.newmultigeometry(name="σ" + str(i + 1))
        mult_ell.newpolygon(
            outerboundaryis=kml_data[i],
            innerboundaryis=kml_data[i],
            name="Ellipse " + str(i),
        )
        # Setting ellipse style
        mult_ell.tessellate = 1
        mult_ell.visibility = 1
        # mult_ell.innerboundaryis = kml_data
        # mult_ell.outerboundaryis = kml_data
        mult_ell.style.linestyle.color = simplekml.Color.black
        mult_ell.style.linestyle.width = 3
        mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
            100, simplekml.Color.blue
        )

    kml.save(filename)

    # ellipse = kml.newpolygon(name="Ellipse")
    # kml.save(filename)
