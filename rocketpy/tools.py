"""The module rocketpy.tools contains a set of functions that are used
throughout the rocketpy package. These functions are not specific to any
particular class or module, and are used to perform general tasks that are
required by multiple classes or modules. These functions can be modified or
expanded to suit the needs of other modules and may present breaking changes
between minor versions if necessary, although this will be always avoided.
"""

import functools
import importlib
import importlib.metadata
import math
import re
import time
from bisect import bisect_left

import numpy as np
import pytz
from cftime import num2pydate
from matplotlib.patches import Ellipse
from packaging import version as packaging_version
import json

# Mapping of module name and the name of the package that should be installed
INSTALL_MAPPING = {"IPython": "ipython"}


def tuple_handler(value):
    """Transforms the input value into a tuple that
    represents a range. If the input is an input or float,
    the output is a tuple from zero to the input value. If
    the input is a tuple or list, the output is a tuple with
    the same range.

    Parameters
    ----------
    value : int, float, tuple, list
        Input value.

    Returns
    -------
    tuple
        Tuple that represents the inputted range.
    """
    if isinstance(value, (int, float)):
        return (0, value)
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (0, value[0])
        elif len(value) == 2:
            return tuple(value)
        else:
            raise ValueError("value must be a list or tuple of length 1 or 2.")


def calculate_cubic_hermite_coefficients(x0, x1, y0, yp0, y1, yp1):
    """Calculate the coefficients of a cubic Hermite interpolation function.
    The function is defined as ax**3 + bx**2 + cx + d.

    Parameters
    ----------
    x0 : float
        Position of the first point.
    x1 : float
        Position of the second point.
    y0 : float
        Value of the function evaluated at the first point.
    yp0 : float
        Value of the derivative of the function evaluated at the first
        point.
    y1 : float
        Value of the function evaluated at the second point.
    yp1 : float
        Value of the derivative of the function evaluated at the second
        point.

    Returns
    -------
    tuple[float, float, float, float]
        The coefficients of the cubic Hermite interpolation function.
    """
    dx = x1 - x0
    d = float(y0)
    c = float(yp0)
    b = float((3 * y1 - yp1 * dx - 2 * c * dx - 3 * d) / (dx**2))
    a = float(-(2 * y1 - yp1 * dx - c * dx - 2 * d) / (dx**3))
    return a, b, c, d


def find_roots_cubic_function(a, b, c, d):
    """Calculate the roots of a cubic function using Cardano's method.

    This method applies Cardano's method to find the roots of a cubic
    function of the form ax^3 + bx^2 + cx + d. The roots may be complex
    numbers.

    Parameters
    ----------
    a : float
        Coefficient of the cubic term (x^3).
    b : float
        Coefficient of the quadratic term (x^2).
    c : float
        Coefficient of the linear term (x).
    d : float
        Constant term.

    Returns
    -------
    tuple[complex, complex, complex]
        A tuple containing the real and complex roots of the cubic function.
        Note that the roots may be complex numbers. The roots are ordered
        in the tuple as x1, x2, x3.

    References
    ----------
    - Cardano's method: https://en.wikipedia.org/wiki/Cubic_function#Cardano's_method

    Examples
    --------
    >>> from rocketpy.tools import find_roots_cubic_function

    First we define the coefficients of the function ax**3 + bx**2 + cx + d
    >>> a, b, c, d = 1, -3, -1, 3
    >>> x1, x2, x3 = find_roots_cubic_function(a, b, c, d)
    >>> x1, x2, x3
    ((-1+0j), (3+7.401486830834377e-17j), (1-1.4802973661668753e-16j))

    To get the real part of the roots, use the real attribute of the complex
    number.
    >>> x1.real, x2.real, x3.real
    (-1.0, 3.0, 1.0)
    """
    delta_0 = b**2 - 3 * a * c
    delta_1 = 2 * b**3 - 9 * a * b * c + 27 * d * a**2
    c1 = ((delta_1 + (delta_1**2 - 4 * delta_0**3) ** (0.5)) / 2) ** (1 / 3)

    c2_0 = c1
    x1 = -(1 / (3 * a)) * (b + c2_0 + delta_0 / c2_0)

    c2_1 = c1 * (-1 / 2 + 1j * (3**0.5) / 2) ** 1
    x2 = -(1 / (3 * a)) * (b + c2_1 + delta_0 / c2_1)

    c2_2 = c1 * (-1 / 2 + 1j * (3**0.5) / 2) ** 2
    x3 = -(1 / (3 * a)) * (b + c2_2 + delta_0 / c2_2)

    return x1, x2, x3


def find_root_linear_interpolation(x0, x1, y0, y1, y):
    """Calculate the root of a linear interpolation function.

    This method calculates the root of a linear interpolation function
    given two points (x0, y0) and (x1, y1) and a value y. The function
    is defined as y = m*x + c.

    Parameters
    ----------
    x0 : float
        Position of the first point.
    x1 : float
        Position of the second point.
    y0 : float
        Value of the function evaluated at the first point.
    y1 : float
        Value of the function evaluated at the second point.
    y : float
        Value of the function at the desired point.

    Returns
    -------
    float
        The root of the linear interpolation function. This represents the
        value of x at which the function evaluates to y.

    Examples
    --------
    >>> from rocketpy.tools import find_root_linear_interpolation
    >>> x0, x1, y0, y1, y = 0, 1, 0, 1, 0.5
    >>> x = find_root_linear_interpolation(x0, x1, y0, y1, y)
    >>> x
    0.5
    """
    m = (y1 - y0) / (x1 - x0)
    c = y0 - m * x0
    return (y - c) / m


def bilinear_interpolation(x, y, x1, x2, y1, y2, z11, z12, z21, z22):
    """Bilinear interpolation. It considers the values of the four points
    around the point to be interpolated and returns the interpolated value.
    Made with a lot of help from GitHub Copilot.

    Parameters
    ----------
    x : float
        x coordinate to which the value will be interpolated.
    y : float
        y coordinate to which the value will be interpolated.
    x1 : float
        x coordinate of the first point.
    x2 : float
        x coordinate of the second point.
    y1 : float
        y coordinate of the first point.
    y2 : float
        y coordinate of the second point.
    z11 : float
        Value at the first point.
    z12 : float
        Value at the second point.
    z21 : float
        Value at the third point.
    z22 : float
        Value at the fourth point.

    Returns
    -------
    float
        Interpolated value.

    Examples
    --------
    >>> from rocketpy.tools import bilinear_interpolation
    >>> bilinear_interpolation(0.5, 0.5, 0, 1, 0, 1, 0, 1, 1, 0)
    0.5
    """
    return (
        z11 * (x2 - x) * (y2 - y)
        + z21 * (x - x1) * (y2 - y)
        + z12 * (x2 - x) * (y - y1)
        + z22 * (x - x1) * (y - y1)
    ) / ((x2 - x1) * (y2 - y1))


def get_distribution(distribution_function_name):
    """Sets the distribution function to be used in the monte carlo analysis.

    Parameters
    ----------
    distribution_function_name : string
        The type of distribution to be used in the analysis. It can be
        'uniform', 'normal', 'lognormal', etc.

    Returns
    -------
    np.random distribution function
        The distribution function to be used in the analysis.
    """
    distributions = {
        "normal": np.random.normal,
        "binomial": np.random.binomial,
        "chisquare": np.random.chisquare,
        "exponential": np.random.exponential,
        "gamma": np.random.gamma,
        "gumbel": np.random.gumbel,
        "laplace": np.random.laplace,
        "logistic": np.random.logistic,
        "poisson": np.random.poisson,
        "uniform": np.random.uniform,
        "wald": np.random.wald,
    }
    try:
        return distributions[distribution_function_name]
    except KeyError as e:
        raise ValueError(
            f"Distribution function '{distribution_function_name}' not found, "
            + "please use one of the following np.random distribution function:"
            + '\n\t"normal"'
            + '\n\t"binomial"'
            + '\n\t"chisquare"'
            + '\n\t"exponential"'
            + '\n\t"geometric"'
            + '\n\t"gamma"'
            + '\n\t"gumbel"'
            + '\n\t"laplace"'
            + '\n\t"logistic"'
            + '\n\t"poisson"'
            + '\n\t"uniform"'
            + '\n\t"wald"\n'
        ) from e


def haversine(lat0, lon0, lat1, lon1, earth_radius=6.3781e6):
    """Returns the distance between two points in meters.
    The points are defined by their latitude and longitude coordinates.

    Parameters
    ----------
    lat0 : float
        Latitude of the first point, in degrees.
    lon0 : float
        Longitude of the first point, in degrees.
    lat1 : float
        Latitude of the second point, in degrees.
    lon1 : float
        Longitude of the second point, in degrees.
    earth_radius : float, optional
        Earth's radius in meters. Default value is 6.3781e6.

    Returns
    -------
    float
        Distance between the two points in meters.

    """
    lat0_rad = math.radians(lat0)
    lat1_rad = math.radians(lat1)
    delta_lat_rad = math.radians(lat1 - lat0)
    delta_lon_rad = math.radians(lon1 - lon0)

    a = (
        math.sin(delta_lat_rad / 2) ** 2
        + math.cos(lat0_rad) * math.cos(lat1_rad) * math.sin(delta_lon_rad / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


def inverted_haversine(lat0, lon0, distance, bearing, earth_radius=6.3781e6):
    """Returns a tuple with new latitude and longitude coordinates considering
    a displacement of a given distance in a given direction (bearing compass)
    starting from a point defined by (lat0, lon0). This is the opposite of
    Haversine function.

    Parameters
    ----------
    lat0 : float
        Origin latitude coordinate, in degrees.
    lon0 : float
        Origin longitude coordinate, in degrees.
    distance : float
        Distance from the origin point, in meters.
    bearing : float
        Azimuth (or bearing compass) from the origin point, in degrees.
    earth_radius : float, optional
        Earth radius, in meters. Default value is 6.3781e6.
        See the Environment.calculateEarthRadius() function for more accuracy.

    Returns
    -------
    lat1 : float
        New latitude coordinate, in degrees.
    lon1 : float
        New longitude coordinate, in degrees.

    """

    # Convert coordinates to radians
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    # Apply inverted Haversine formula
    lat1_rad = math.asin(
        math.sin(lat0_rad) * math.cos(distance / earth_radius)
        + math.cos(lat0_rad) * math.sin(distance / earth_radius) * math.cos(bearing)
    )

    lon1_rad = lon0_rad + math.atan2(
        math.sin(bearing) * math.sin(distance / earth_radius) * math.cos(lat0_rad),
        math.cos(distance / earth_radius) - math.sin(lat0_rad) * math.sin(lat1_rad),
    )

    # Convert back to degrees and then return
    lat1_deg = np.rad2deg(lat1_rad)
    lon1_deg = np.rad2deg(lon1_rad)

    return lat1_deg, lon1_deg


# Functions for monte carlo analysis
# pylint: disable=too-many-statements
def generate_monte_carlo_ellipses(results):
    """A function to create apogee and impact ellipses from the monte carlo
    analysis results.

    Parameters
    ----------
    results : dict
        A dictionary containing the results of the monte carlo analysis. It
        should contain the following keys:
        - apogeeX: an array containing the x coordinates of the apogee
        - apogeeY: an array containing the y coordinates of the apogee
        - xImpact: an array containing the x coordinates of the impact
        - yImpact: an array containing the y coordinates of the impact

    Returns
    -------
    apogee_ellipse : list[Ellipse]
        A list of ellipse objects representing the apogee ellipses.
    impact_ellipse : list[Ellipse]
        A list of ellipse objects representing the impact ellipses.
    apogeeX : np.array
        An array containing the x coordinates of the apogee ellipse.
    apogeeY : np.array
        An array containing the y coordinates of the apogee ellipse.
    impactX : np.array
        An array containing the x coordinates of the impact ellipse.
    impactY : np.array
        An array containing the y coordinates of the impact ellipse.
    """

    # Retrieve monte carlo data por apogee and impact XY position
    try:
        apogee_x = np.array(results["apogee_x"])
        apogee_y = np.array(results["apogee_y"])
    except KeyError:
        print("No apogee data found. Skipping apogee ellipses.")
        apogee_x = np.array([])
        apogee_y = np.array([])
    try:
        impact_x = np.array(results["x_impact"])
        impact_y = np.array(results["y_impact"])
    except KeyError:
        print("No impact data found. Skipping impact ellipses.")
        impact_x = np.array([])
        impact_y = np.array([])

    # Define function to calculate Eigenvalues
    def eigsorted(cov):
        # Calculate eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        # Order eigenvalues and eigenvectors in descending order
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def calculate_ellipses(list_x, list_y):
        # Calculate covariance matrix
        cov = np.cov(list_x, list_y)
        # Calculate eigenvalues and eigenvectors
        vals, vecs = eigsorted(cov)
        # Calculate ellipse angle and width/height
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * np.sqrt(vals)
        return theta, w, h

    def create_ellipse_objects(x, y, n, w, h, theta, rgb):
        """Create a list of matplotlib.patches.Ellipse objects.

        Parameters
        ----------
        x : list or np.array
            List of x coordinates.
        y : list or np.array
            List of y coordinates.
        n : int
            Number of ellipses to create. It represents the number of confidence
            intervals to be used. For example, n=3 will create 3 ellipses with
            1, 2 and 3 standard deviations.
        w : float
            Width of the ellipse.
        h : float
            Height of the ellipse.
        theta : float
            Angle of the ellipse.
        rgb : tuple
            Tuple containing the color of the ellipse in RGB format. For example,
            (0, 0, 1) will create a blue ellipse.

        Returns
        -------
        list
            List of matplotlib.patches.Ellipse objects.
        """
        ell_list = [None] * n
        for j in range(n):
            ell = Ellipse(
                xy=(np.mean(x), np.mean(y)),
                width=w,
                height=h,
                angle=theta,
                color="black",
            )
            ell.set_facecolor(rgb)
            ell_list[j] = ell
        return ell_list

    # Calculate error ellipses for impact and apogee
    impact_theta, impact_w, impact_h = calculate_ellipses(impact_x, impact_y)
    apogee_theta, apogee_w, apogee_h = calculate_ellipses(apogee_x, apogee_y)

    # Draw error ellipses for impact
    impact_ellipses = create_ellipse_objects(
        impact_x, impact_y, 3, impact_w, impact_h, impact_theta, (0, 0, 1, 0.2)
    )

    apogee_ellipses = create_ellipse_objects(
        apogee_x, apogee_y, 3, apogee_w, apogee_h, apogee_theta, (0, 1, 0, 0.2)
    )

    return impact_ellipses, apogee_ellipses, apogee_x, apogee_y, impact_x, impact_y


def generate_monte_carlo_ellipses_coordinates(
    ellipses, origin_lat, origin_lon, resolution=100
):
    """Generate a list of latitude and longitude points for each ellipse in
    ellipses.

    Parameters
    ----------
    ellipses : list
        List of matplotlib.patches.Ellipse objects.
    origin_lat : float
        Latitude of the origin of the coordinate system.
    origin_lon : float
        Longitude of the origin of the coordinate system.
    resolution : int, optional
        Number of points to generate for each ellipse, by default 100

    Returns
    -------
    list
        List of lists of tuples containing the latitude and longitude of each
        point in each ellipse.
    """
    outputs = [None] * len(ellipses)

    for index, ell in enumerate(ellipses):
        # Get ellipse path points
        center = ell.get_center()
        width = ell.get_width()
        height = ell.get_height()
        angle = np.deg2rad(ell.get_angle())
        points = lat_lon_points = [None] * resolution

        # Generate ellipse path points (in a Cartesian coordinate system)
        for i in range(resolution):
            x = width / 2 * math.cos(2 * np.pi * i / resolution)
            y = height / 2 * math.sin(2 * np.pi * i / resolution)
            x_rot = center[0] + x * math.cos(angle) - y * math.sin(angle)
            y_rot = center[1] + x * math.sin(angle) + y * math.cos(angle)
            points[i] = (x_rot, y_rot)
        points = np.array(points)

        # Convert path points to lat/lon
        for point in points:
            x, y = point
            # Convert to distance and bearing
            d = math.sqrt((x**2 + y**2))
            bearing = math.atan2(
                x, y
            )  # math.atan2 returns the angle in the range [-pi, pi]

            lat_lon_points[i] = inverted_haversine(
                origin_lat, origin_lon, d, bearing, earth_radius=6.3781e6
            )

        outputs[index] = lat_lon_points
    return outputs


def load_monte_carlo_data(
    input_filename: str,
    output_filename: str,
    parameters_list: list[str],
    target_variables_list: list[str],
):
    """Reads MonteCarlo simulation data file and builds parameters and flight
    variables matrices from specified

    Parameters
    ----------
    input_filename : str
        Input file exported by MonteCarlo class. Each line is a
        sample unit described by a dictionary where keys are parameters names
        and the values are the sampled parameters values.

    output_filename : str
        Output file exported by MonteCarlo.simulate function. Each line is a
        sample unit described by a dictionary where keys are target variables
        names and the values are the obtained values from the flight simulation.

    parameters_list : list[str]
        List of parameters whose values will be extracted.

    target_variables_list : list[str]
        List of target variables whose values will be extracted.

    Returns
    -------
    parameters_matrix: np.matrix
        Numpy matrix contaning input parameters values. Each column correspond
        to a parameter in the same order specified by 'parameters_list' input.

    target_variables_matrix: np.matrix
        Numpy matrix contaning target variables values. Each column correspond
        to a target variable in the same order specified by 'target_variables_list'
        input.
    """
    number_of_samples_parameters = 0
    number_of_samples_variables = 0

    # Auxiliary function that unnests dictionary
    def unnest_dict(x):
        new_dict = {}
        for key, value in x.items():
            # the nested dictionary is inside a list
            if isinstance(x[key], list):
                # sometimes the object inside the list is another list
                # we must skip these cases
                if isinstance(value[0], dict):
                    inner_dict = unnest_dict(value[0])
                    inner_dict = {
                        key + "_" + inner_key: inner_value
                        for inner_key, inner_value in inner_dict.items()
                    }
                    new_dict.update(inner_dict)
            else:
                new_dict.update({key: value})

        return new_dict

    parameters_samples = {parameter: [] for parameter in parameters_list}
    with open(input_filename, "r") as parameters_file:
        for line in parameters_file.readlines():
            number_of_samples_parameters += 1

            parameters_dict = json.loads(line)
            parameters_dict = unnest_dict(parameters_dict)
            for parameter in parameters_list:
                try:
                    value = parameters_dict[parameter]
                except Exception:
                    raise Exception(
                        f"Parameter {parameter} was not found in {input_filename}!"
                    )
                parameters_samples[parameter].append(value)

    target_variables_samples = {variable: [] for variable in target_variables_list}
    with open(output_filename, "r") as target_variables_file:
        for line in target_variables_file.readlines():
            number_of_samples_variables += 1
            target_variables_dict = json.loads(line)
            for variable in target_variables_list:
                try:
                    value = target_variables_dict[variable]
                except Exception:
                    raise Exception(
                        f"Variable {variable} was not found in {output_filename}!"
                    )
                target_variables_samples[variable].append(value)

    if number_of_samples_parameters != number_of_samples_variables:
        raise ValueError(
            "Number of samples for parameters does not match the number of samples for target variables!"
        )

    n_samples = number_of_samples_variables
    n_parameters = len(parameters_list)
    n_variables = len(target_variables_list)
    parameters_matrix = np.empty((n_samples, n_parameters))
    target_variables_matrix = np.empty((n_samples, n_variables))

    for i in range(n_parameters):
        parameter = parameters_list[i]
        parameters_matrix[:, i] = parameters_samples[parameter]

    for i in range(n_variables):
        target_variable = target_variables_list[i]
        target_variables_matrix[:, i] = target_variables_samples[target_variable]

    return parameters_matrix, target_variables_matrix


def find_two_closest_integers(number):
    """Find the two closest integer factors of a number.

    Parameters
    ----------
    number: int

    Returns
    -------
    tuple
        Two closest integer factors of the number.

    Examples
    --------
    >>> from rocketpy.tools import find_two_closest_integers
    >>> find_two_closest_integers(10)
    (2, 5)
    >>> find_two_closest_integers(12)
    (3, 4)
    >>> find_two_closest_integers(13)
    (1, 13)
    >>> find_two_closest_integers(150)
    (10, 15)
    """
    number_sqrt = number**0.5
    if isinstance(number_sqrt, int):
        return number_sqrt, number_sqrt
    else:
        guess = int(number_sqrt)
        while True:
            if number % guess == 0:
                return guess, number // guess
            else:
                guess -= 1


def time_num_to_date_string(time_num, units, timezone, calendar="gregorian"):
    """Convert time number (usually hours before a certain date) into two
    strings: one for the date (example: 2022.04.31) and one for the hour
    (example: 14). See cftime.num2date for details on units and calendar.
    Automatically converts time number from UTC to local time zone based on
    lat, lon coordinates. This function was created originally for the
    EnvironmentAnalysis class.

    Parameters
    ----------
    time_num : float
        Time number to be converted.
    units : str
        Units of the time number. See cftime.num2date for details.
    timezone : pytz.timezone
        Timezone to which the time number will be converted. See
        pytz.timezone for details.
    calendar : str, optional
        Calendar to be used. See cftime.num2date for details.

    Returns
    -------
    date_string : str
        Date string.
    hour_string : str
        Hour string.
    date_time : datetime.datetime
        Datetime object.
    """
    date_time_utc = num2pydate(time_num, units, calendar=calendar)
    date_time_utc = date_time_utc.replace(tzinfo=pytz.UTC)
    date_time = date_time_utc.astimezone(timezone)
    date_string = f"{date_time.year}.{date_time.month}.{date_time.day}"
    hour_string = f"{date_time.hour}"
    return date_string, hour_string, date_time


def geopotential_height_to_geometric_height(geopotential_height, radius=63781370.0):
    """Converts geopotential height to geometric height.

    Parameters
    ----------
    geopotential_height : float
        Geopotential height in meters. This vertical coordinate, referenced to
        Earth's mean sea level, accounts for variations in gravity with altitude
        and latitude.
    radius : float, optional
        The Earth's radius in meters, defaulting to 6378137.0.

    Returns
    -------
    geometric_height : float
        Geometric height in meters.

    Examples
    --------
    >>> from rocketpy.tools import geopotential_height_to_geometric_height
    >>> geopotential_height_to_geometric_height(0)
    0.0
    >>> geopotential_height_to_geometric_height(10000)
    10001.568101798659
    >>> geopotential_height_to_geometric_height(20000)
    20006.2733909262
    """
    return radius * geopotential_height / (radius - geopotential_height)


def geopotential_to_height_asl(geopotential, radius=63781370, g=9.80665):
    """Compute height above sea level from geopotential.

    Source: https://en.wikipedia.org/wiki/Geopotential

    Parameters
    ----------
    geopotential : float
        Geopotential in m^2/s^2. It is the geopotential value at a given
        pressure level, to be converted to height above sea level.
    radius : float, optional
        Earth radius in m. Default is 63781370 m.
    g : float, optional
        Gravity acceleration in m/s^2. Default is 9.80665 m/s^2.

    Returns
    -------
    geopotential_to_height_asl : float
        Height above sea level in m

    Examples
    --------
    >>> from rocketpy.tools import geopotential_to_height_asl
    >>> geopotential_to_height_asl(0)
    0.0
    >>> geopotential_to_height_asl(100000)
    10198.792680243916
    >>> geopotential_to_height_asl(200000)
    20400.84750449947
    """
    geopotential_height = geopotential / g
    return geopotential_height_to_geometric_height(geopotential_height, radius)


def geopotential_to_height_agl(geopotential, elevation, radius=63781370, g=9.80665):
    """Compute height above ground level from geopotential and elevation.

    Parameters
    ----------
    geopotential : float
        Geopotential in m^2/s^2. It is the geopotential value at a given
        pressure level, to be converted to height above ground level.
    elevation : float
        Surface elevation in m
    radius : float, optional
        Earth radius in m. Default is 63781370 m.
    g : float, optional
        Gravity acceleration in m/s^2. Default is 9.80665 m/s^2.

    Returns
    -------
    height_above_ground_level : float
        Height above ground level in m

    Examples
    --------
    >>> from rocketpy.tools import geopotential_to_height_agl
    >>> geopotential_to_height_agl(0, 0)
    0.0
    >>> geopotential_to_height_agl(100000, 0)
    10198.792680243916
    >>> geopotential_to_height_agl(100000, 1000)
    9198.792680243916
    """
    return geopotential_to_height_asl(geopotential, radius, g) - elevation


def find_closest(ordered_sequence, value):
    """Find the index of the closest value to a given value within an ordered
    sequence.

    Parameters
    ----------
    ordered_sequence : list
        A sequence of values that is ordered from smallest to largest.
    value : float
        The value to which you want to find the closest value.

    Returns
    -------
    index : int
        The index of the closest value to the given value within the ordered
        sequence. If the given value is lower than the first value in the
        sequence, then 0 is returned. If the given value is greater than the
        last value in the sequence, then the index of the last value in the
        sequence is returned.

    Examples
    --------
    >>> from rocketpy.tools import find_closest
    >>> find_closest([1, 2, 3, 4, 5], 0)
    0
    >>> find_closest([1, 2, 3, 4, 5], 1.5)
    0
    >>> find_closest([1, 2, 3, 4, 5], 2.0)
    1
    >>> find_closest([1, 2, 3, 4, 5], 2.8)
    2
    >>> find_closest([1, 2, 3, 4, 5], 4.9)
    4
    >>> find_closest([1, 2, 3, 4, 5], 5.5)
    4
    >>> find_closest([], 10)
    0
    """
    pivot_index = bisect_left(ordered_sequence, value)
    if pivot_index == 0:
        return pivot_index
    if pivot_index == len(ordered_sequence):
        return pivot_index - 1

    smaller, greater = ordered_sequence[pivot_index - 1], ordered_sequence[pivot_index]

    return pivot_index - 1 if value - smaller <= greater - value else pivot_index


def import_optional_dependency(name):
    """Import an optional dependency. If the dependency is not installed, an
    ImportError is raised. This function is based on the implementation found in
    pandas repository:
    github.com/pandas-dev/pandas/blob/main/pandas/compat/_optional.py

    Parameters
    ----------
    name : str
        The name of the module to import. Can be used to import submodules too.
        The name will be used as an argument to importlib.import_module method.

    Examples:
    ---------
    >>> from rocketpy.tools import import_optional_dependency
    >>> matplotlib = import_optional_dependency("matplotlib")
    >>> matplotlib.__name__
    'matplotlib'
    >>> plt = import_optional_dependency("matplotlib.pyplot")
    >>> plt.__name__
    'matplotlib.pyplot'
    """
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        module_name = name.split(".")[0]
        package_name = INSTALL_MAPPING.get(module_name, module_name)
        raise ImportError(
            f"{package_name} is an optional dependency and is not installed.\n"
            + f"\t\tUse 'pip install {package_name}' to install it or "
            + "'pip install rocketpy[all]' to install all optional dependencies."
        ) from exc
    return module


def check_requirement_version(module_name, version):
    """This function tests if a module is installed and if the version is
    correct. If the module is not installed, an ImportError is raised. If the
    version is not correct, an error is raised.

    Parameters
    ----------
    module_name : str
        The name of the module to be tested.
    version : str
        The version of the module that is required. The string must start with
        one of the following operators: ">", "<", ">=", "<=", "==", "!=".

    Example:
    --------
    >>> from rocketpy.tools import check_requirement_version
    >>> check_requirement_version("numpy", ">=1.0.0")
    True
    >>> check_requirement_version("matplotlib", ">=3.0")
    True
    """
    operators = [">=", "<=", "==", ">", "<", "!="]
    # separator the operator from the version number
    operator, v_number = re.match(f"({'|'.join(operators)})(.*)", version).groups()

    if operator not in operators:
        raise ValueError(
            f"Version must start with one of the following operators: {operators}"
        )
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"{module_name} is not installed. You can install it by running "
            + f"'pip install {module_name}'"
        )
    installed_version = packaging_version.parse(importlib.metadata.version(module_name))
    required_version = packaging_version.parse(v_number)
    if installed_version < required_version:
        raise ImportError(
            f"{module_name} version is {installed_version}, which is not correct"
            + f". A version {version} is required. You can install a correct "
            + f"version by running 'pip install {module_name}{version}'"
        )
    return True


def exponential_backoff(max_attempts, base_delay=1, max_delay=60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    if i == max_attempts - 1:
                        raise e from None
                    delay = min(delay * 2, max_delay)
                    time.sleep(delay)

        return wrapper

    return decorator


def parallel_axis_theorem_from_com(com_inertia_moment, mass, distance):
    """Calculates the moment of inertia of a object relative to a new axis using
    the parallel axis theorem. The new axis is parallel to and at a distance
    'distance' from the original axis, which *must* passes through the object's
    center of mass.

    Parameters
    ----------
    com_inertia_moment : float
        Moment of inertia relative to the center of mass of the object.
    mass : float
        Mass of the object.
    distance : float
        Perpendicular distance between the original and new axis.

    Returns
    -------
    float
        Moment of inertia relative to the new axis.

    References
    ----------
    https://en.wikipedia.org/wiki/Parallel_axis_theorem
    """
    return com_inertia_moment + mass * distance**2


# Flight
def quaternions_to_precession(e0, e1, e2, e3):
    """Calculates the Precession angle

    Parameters
    ----------
    e0 : float
        Euler parameter 0, must be between -1 and 1
    e1 : float
        Euler parameter 1, must be between -1 and 1
    e2 : float
        Euler parameter 2, must be between -1 and 1
    e3 : float
        Euler parameter 3, must be between -1 and 1

    Returns
    -------
    float
        Euler Precession angle in degrees

    References
    ----------
    Baruh, Haim. Analytical dynamics
    """
    # minus sign in e2 and e1 is due to changing from 3-1-3 to 3-2-3 convention
    return (180 / np.pi) * (np.arctan2(e3, e0) + np.arctan2(-e2, -e1))


def quaternions_to_spin(e0, e1, e2, e3):
    """Calculates the Spin angle from quaternions.

    Parameters
    ----------
    e0 : float
        Euler parameter 0, must be between -1 and 1
    e1 : float
        Euler parameter 1, must be between -1 and 1
    e2 : float
        Euler parameter 2, must be between -1 and 1
    e3 : float
        Euler parameter 3, must be between -1 and 1

    Returns
    -------
    float
        Euler Spin angle in degrees

    References
    ----------
    Baruh, Haim. Analytical dynamics
    """
    # minus sign in e2 and e1 is due to changing from 3-1-3 to 3-2-3 convention
    return (180 / np.pi) * (np.arctan2(e3, e0) - np.arctan2(-e2, -e1))


def quaternions_to_nutation(e1, e2):
    """Calculates the Nutation angle from quaternions.

    Parameters
    ----------
    e1 : float
        Euler parameter 1, must be between -1 and 1
    e2 : float
        Euler parameter 2, must be between -1 and 1

    Returns
    -------
    float
        Euler Nutation angle in degrees

    References
    ----------
    Baruh, Haim. Analytical dynamics
    """
    # we are changing from 3-1-3 to 3-2-3 conventions
    return (180 / np.pi) * 2 * np.arcsin(-((e1**2 + e2**2) ** 0.5))


def euler_angles_to_euler_parameters(phi, theta, psi):
    """Convert 3-1-3 Euler Angles to Euler Parameters (quaternions).

    Parameters
    ----------
    phi : float
        Rotation angle around the z-axis (in radians). Represents the precession angle.
    theta : float
        Rotation angle around the x-axis (in radians). Represents the nutation angle.
    psi : float
        Rotation angle around the z-axis (in radians). Represents the spin angle.


    Returns
    -------
    tuple[float, float, float, float]
        The Euler parameters or quaternions (e0, e1, e2, e3)

    References
    ----------
    https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
    """
    e0 = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.sin(
        phi / 2
    ) * np.cos(theta / 2) * np.sin(psi / 2)
    e1 = np.cos(phi / 2) * np.cos(psi / 2) * np.sin(theta / 2) + np.sin(
        phi / 2
    ) * np.sin(theta / 2) * np.sin(psi / 2)
    e2 = np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2) - np.sin(
        phi / 2
    ) * np.cos(psi / 2) * np.sin(theta / 2)
    e3 = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) + np.cos(
        theta / 2
    ) * np.cos(psi / 2) * np.sin(phi / 2)
    return e0, e1, e2, e3


if __name__ == "__main__":
    import doctest

    res = doctest.testmod()
    if res.failed < 1:
        print(f"All the {res.attempted} tests passed!")
    else:
        print(f"{res.failed} out of {res.attempted} tests failed.")
