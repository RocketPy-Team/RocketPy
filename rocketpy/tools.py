"""The module rocketpy.tools contains a set of functions that are used
throughout the rocketpy package. These functions are not specific to any
particular class or module, and are used to perform general tasks that are
required by multiple classes or modules. These functions can be modified or
expanded to suit the needs of other modules and may present breaking changes
between minor versions if necessary, although this will be always avoided.
"""

import base64
import functools
import importlib
import importlib.metadata
import json
import math
import re
import time
import warnings
from bisect import bisect_left

import dill
import matplotlib.pyplot as plt
import numpy as np
import pytz
from cftime import num2pydate
from matplotlib.patches import Ellipse
from packaging import version as packaging_version

# Mapping of module name and the name of the package that should be installed
INSTALL_MAPPING = {"IPython": "ipython"}


def deprecated(reason=None, version=None, alternative=None):
    """
    Decorator to mark functions or methods as deprecated.

    This decorator issues a DeprecationWarning when the decorated function
    is called, indicating that it will be removed in future versions.

    Parameters
    ----------
    reason : str, optional
        Custom deprecation message. If not provided, a default message will be used.
    version : str, optional
        Version when the function will be removed. If provided, it will be
        included in the warning message.
    alternative : str, optional
        Name of the alternative function/method that should be used instead.
        If provided, it will be included in the warning message.

    Returns
    -------
    callable
        The decorated function with deprecation warning functionality.

    Examples
    --------
    >>> @deprecated(reason="This function is obsolete", version="v2.0.0",
    ...             alternative="new_function")
    ... def old_function():
    ...     return "old result"

    >>> @deprecated()
    ... def another_old_function():
    ...     return "result"
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build the deprecation message
            if reason:
                message = reason
            else:
                message = f"The function `{func.__name__}` is deprecated"

            if version:
                message += f" and will be removed in {version}"

            if alternative:
                message += f". Use `{alternative}` instead"

            message += "."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def tuple_handler(value):
    """Transforms the input value into a tuple that represents a range. If the
    input is an int or float, the output is a tuple from zero to the input
    value. If the input is a tuple or list, the output is a tuple with the same
    range.

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
    >>> import cmath

    First we define the coefficients of the function ax**3 + bx**2 + cx + d
    >>> a, b, c, d = 1, -3, -1, 3
    >>> x1, x2, x3 = find_roots_cubic_function(a, b, c, d)
    >>> cmath.isclose(x1, (-1+0j))
    True

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


def get_distribution(distribution_function_name, random_number_generator=None):
    """Sets the distribution function to be used in the monte carlo analysis.

    Parameters
    ----------
    distribution_function_name : string
        The type of distribution to be used in the analysis. It can be
        'uniform', 'normal', 'lognormal', etc.
    random_number_generator : np.random.Generator, optional
        The random number generator to be used. If None, the default generator
        ``numpy.random.default_rng`` is used.

    Returns
    -------
    np.random distribution function
        The distribution function to be used in the analysis.
    """
    if random_number_generator is None:
        random_number_generator = np.random.default_rng()

    # Dictionary mapping distribution names to RNG methods
    distributions = {
        "normal": random_number_generator.normal,
        "binomial": random_number_generator.binomial,
        "chisquare": random_number_generator.chisquare,
        "exponential": random_number_generator.exponential,
        "gamma": random_number_generator.gamma,
        "gumbel": random_number_generator.gumbel,
        "laplace": random_number_generator.laplace,
        "logistic": random_number_generator.logistic,
        "poisson": random_number_generator.poisson,
        "uniform": random_number_generator.uniform,
        "wald": random_number_generator.wald,
    }
    try:
        return distributions[distribution_function_name]
    except KeyError as e:  # pragma: no cover
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
        + math.cos(lat0_rad)
        * math.sin(distance / earth_radius)
        * math.cos(math.radians(bearing))
    )

    lon1_rad = lon0_rad + math.atan2(
        math.sin(math.radians(bearing))
        * math.sin(distance / earth_radius)
        * math.cos(lat0_rad),
        math.cos(distance / earth_radius) - math.sin(lat0_rad) * math.sin(lat1_rad),
    )

    # Convert back to degrees and then return
    lat1_deg = np.rad2deg(lat1_rad)
    lon1_deg = np.rad2deg(lon1_rad)

    return lat1_deg, lon1_deg


def mercator_to_wgs84(x, y, earth_radius=6.3781e6):
    """Convert Web Mercator (EPSG:3857) coordinates to WGS84 (EPSG:4326) coordinates.

    This function converts coordinates from Web Mercator projection to WGS84
    latitude/longitude without requiring the pyproj dependency.

    Parameters
    ----------
    x : float
        X coordinate in Web Mercator projection (meters).
    y : float
        Y coordinate in Web Mercator projection (meters).
    earth_radius : float, optional
        Earth's radius in meters. Default value is 6.3781e6.

    Returns
    -------
    tuple[float, float]
        A tuple containing (latitude, longitude) in degrees.

    """
    lon = x / earth_radius * 180.0 / math.pi
    lat = (2 * math.atan(math.exp(y / earth_radius)) - math.pi / 2.0) * 180.0 / math.pi
    return lat, lon


def convert_local_extent_to_wgs84(
    local_extent, origin_lat, origin_lon, earth_radius=6.3781e6
):
    """Convert local extent to geographic extent (latitude/longitude bounding box).

    This function converts a local extent (bounding box in meters relative to an
    origin point) to a geographic extent (bounding box in latitude/longitude degrees).

    Parameters
    ----------
    local_extent : list[float]
        Local extent [x_min, x_max, y_min, y_max] in meters relative to the origin.
    origin_lat : float
        Origin latitude in degrees.
    origin_lon : float
        Origin longitude in degrees.
    earth_radius : float, optional
        Earth's radius in meters. Default value is 6.3781e6.

    Returns
    -------
    tuple[float, float, float, float]
        Geographic extent (west, south, east, north) in degrees.
    """
    x_min, x_max, y_min, y_max = local_extent
    corners_xy = [
        (x_min, y_min),  # Bottom-Left
        (x_min, y_max),  # Top-Left
        (x_max, y_min),  # Bottom-Right
        (x_max, y_max),  # Top-Right
    ]
    req_lats, req_lons = [], []

    for x, y in corners_xy:
        dist = (x**2 + y**2) ** 0.5
        # Calculate bearing: 0 is North (Y), 90 is East (X)
        bearing = np.degrees(np.arctan2(x, y))
        lat, lon = inverted_haversine(
            origin_lat, origin_lon, dist, bearing, earth_radius
        )
        req_lats.append(lat)
        req_lons.append(lon)

    return min(req_lons), min(req_lats), max(req_lons), max(req_lats)


def convert_mercator_extent_to_local(
    mercator_extent, origin_lat, origin_lon, earth_radius=6.3781e6
):
    """Convert Mercator extent to local extent (coordinates relative to origin).

    This function converts a geographic extent from Web Mercator coordinates to a
    local extent (bounding box in meters relative to an origin point).

    Parameters
    ----------
    mercator_extent : list[float]
        Mercator extent [minX, maxX, minY, maxY] in meters.
    origin_lat : float
        Origin latitude in degrees.
    origin_lon : float
        Origin longitude in degrees.
    earth_radius : float, optional
        Earth's radius in meters. Default value is 6.3781e6.

    Returns
    -------
    list[float]
        Local extent [x_min, x_max, y_min, y_max] in meters relative to the origin.
    """
    # Convert corners of the fetched image from Mercator to WGS84
    bg_lat_min, bg_lon_min = mercator_to_wgs84(
        mercator_extent[0], mercator_extent[2], earth_radius
    )  # Bottom-Left
    bg_lat_max, bg_lon_max = mercator_to_wgs84(
        mercator_extent[1], mercator_extent[3], earth_radius
    )  # Top-Right

    # Calculate X/Y meters relative to origin (lat0, lon0) using haversine
    # X = Distance along longitude (East-West)
    # Y = Distance along latitude (North-South)

    # Calculate X min (Left)
    x_min = haversine(origin_lat, origin_lon, origin_lat, bg_lon_min, earth_radius)
    if bg_lon_min < origin_lon:
        x_min = -x_min

    # Calculate X max (Right)
    x_max = haversine(origin_lat, origin_lon, origin_lat, bg_lon_max, earth_radius)
    if bg_lon_max < origin_lon:
        x_max = -x_max

    # Calculate Y min (Bottom)
    y_min = haversine(origin_lat, origin_lon, bg_lat_min, origin_lon, earth_radius)
    if bg_lat_min < origin_lat:
        y_min = -y_min

    # Calculate Y max (Top)
    y_max = haversine(origin_lat, origin_lon, bg_lat_max, origin_lon, earth_radius)
    if bg_lat_max < origin_lat:
        y_max = -y_max

    return [x_min, x_max, y_min, y_max]


# Functions for monte carlo analysis
def sort_eigenvalues(cov):
    # Calculate eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    # Order eigenvalues and eigenvectors in descending order
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def calculate_confidence_ellipse(list_x, list_y, n_std=3):
    """Given a list of x and y coordinates, calculate the confidence ellipse
    parameters (theta, width, height) for a given number of standard deviations.
    """
    covariance_matrix = np.cov(list_x, list_y)
    eigenvalues, eigenvectors = sort_eigenvalues(covariance_matrix)
    theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    return theta, width, height


def create_matplotlib_ellipse(x, y, w, h, theta, rgb, opacity):
    """Create a matplotlib.patches.Ellipse object.

    Parameters
    ----------
    x : list or np.array
        List of x coordinates.
    y : list or np.array
        List of y coordinates.
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
    matplotlib.patches.Ellipse
        One matplotlib.patches.Ellipse objects.
    """

    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=w,
        height=h,
        angle=theta,
        color="black",
    )
    ell.set_facecolor(rgb)
    ell.set_alpha(opacity)
    return ell


def generate_monte_carlo_ellipses(
    apogee_x=np.array([]),
    apogee_y=np.array([]),
    impact_x=np.array([]),
    impact_y=np.array([]),
    n_apogee=[1, 2, 3],
    n_impact=[1, 2, 3],
    apogee_rgb=(0, 1, 0),
    impact_rgb=(0, 0, 1),
    opacity=0.2,
):  # pylint: disable=dangerous-default-value
    """Function to generate Monte Carlo ellipses for apogee and impact points.

    Parameters
    ----------
    apogee_x : np.ndarray, optional
        Array of x-coordinates for apogee points, by default np.array([])
    apogee_y : np.ndarray, optional
        Array of y-coordinates for apogee points, by default np.array([])
    impact_x : np.ndarray, optional
        Array of x-coordinates for impact points, by default np.array([])
    impact_y : np.ndarray, optional
        Array of y-coordinates for impact points, by default np.array([])
    n_apogee : list, optional
        List of integers representing the number of standard deviations for
        apogee ellipses, by default [1, 2, 3]
    n_impact : list, optional
        List of integers representing the number of standard deviations for
        impact ellipses, by default [1, 2, 3]
    apogee_rgb : tuple, optional
        RGB color tuple for apogee ellipses, by default (0, 1, 0).
    impact_rgb : tuple, optional
        RGB color tuple for impact ellipses, by default (0, 0, 1).
    opacity : float, optional
        The alpha parameter for the solid face of the ellipses, by default 0.2

    Returns
    -------
    tuple[list[matplotlib.patches.Ellipse], list[matplotlib.patches.Ellipse]]
        A tuple containing two lists:
        - List of matplotlib.patches.Ellipse objects for apogee ellipses.
        - List of matplotlib.patches.Ellipse objects for impact ellipses.
    """

    # Calculate error ellipses for impact and apogee
    apogee_ellipses = []
    for i in n_apogee:
        theta, width, height = calculate_confidence_ellipse(apogee_x, apogee_y, n_std=i)
        apogee_ellipses.append(
            create_matplotlib_ellipse(
                apogee_x, apogee_y, width, height, theta, apogee_rgb, opacity
            )
        )

    # Draw error ellipses for impact
    impact_ellipses = []
    for i in n_impact:
        theta, width, height = calculate_confidence_ellipse(impact_x, impact_y, n_std=i)
        impact_ellipses.append(
            create_matplotlib_ellipse(
                impact_x, impact_y, width, height, theta, impact_rgb, opacity
            )
        )

    return impact_ellipses, apogee_ellipses


def generate_monte_carlo_ellipses_coordinates(
    ellipses, origin_lat, origin_lon, resolution=100
):
    """Generate a list of latitude and longitude points for each ellipse in
    ellipses.

    Parameters
    ----------
    ellipses : list[matplotlib.patches.Ellipse]
        List of matplotlib.patches.Ellipse objects.
    origin_lat : float
        Latitude of the origin of the coordinate system.
    origin_lon : float
        Longitude of the origin of the coordinate system.
    resolution : int, optional
        Number of points to generate for each ellipse, by default 100

    Returns
    -------
    list[list[tuple[float, float]]]
        List of lists of tuples containing the latitude and longitude of each
        point in each ellipse.
    """
    return [
        __convert_to_lat_lon(
            __generate_ellipse_points(ell, resolution), origin_lat, origin_lon
        )
        for ell in ellipses
    ]


def __convert_to_lat_lon(points: list, origin_lat: float, origin_lon: float):
    return [
        inverted_haversine(
            origin_lat,
            origin_lon,
            math.sqrt(x**2 + y**2),
            math.degrees(math.atan2(x, y)),
            earth_radius=6.3781e6,
        )
        for x, y in points
    ]


def __generate_ellipse_points(ellipse, resolution: int):
    center = ellipse.get_center()
    width = ellipse.get_width()
    height = ellipse.get_height()
    angle = np.deg2rad(ellipse.get_angle())

    points = [
        (
            center[0]
            + (width / 2 * math.cos(2 * np.pi * i / resolution)) * math.cos(angle)
            - (height / 2 * math.sin(2 * np.pi * i / resolution)) * math.sin(angle),
            center[1]
            + (width / 2 * math.cos(2 * np.pi * i / resolution)) * math.sin(angle)
            + (height / 2 * math.sin(2 * np.pi * i / resolution)) * math.cos(angle),
        )
        for i in range(resolution)
    ]
    return np.array(points)


def flatten_dict(original_dict):
    """Flatten a dictionary for easy handling of nested variables

    This function is mainly used for handling data in sensitivity analysis
    and in the MRS.

    Parameters
    ----------
    original_dict : dict
        A dictionary possibly containing nested variables. This means that
        a key might contain another dictionary inside of it.

    Returns
    -------
    flatted_dict : dict
        The flatted dictionary which, ideally, should not contain nested
        variables. All nested information should be available directly in
        the first level (access by key). Variables that were available
        inside the first level retain their original key name. Variables
        that were nested are created by appending the name of the outer
        key used to access it concatenated with a '_' and the key name
        of the variable.
    """
    flatted_dict = {}
    for key, value in original_dict.items():
        # the nested dictionary is inside a list
        if isinstance(original_dict[key], list):
            for inner_item in value:
                if isinstance(inner_item, dict):
                    inner_dict = flatten_dict(inner_item)
                    sep_str = "_"
                    if "name" in inner_dict:
                        sep_str = "_" + inner_dict["name"] + "_"
                    inner_dict = {
                        key + sep_str + inner_key: inner_value
                        for inner_key, inner_value in inner_dict.items()
                    }
                    flatted_dict.update(inner_dict)
        else:
            flatted_dict.update({key: value})

    return flatted_dict


def load_monte_carlo_data(
    input_filename,
    output_filename,
    parameters_list,
    target_variables_list,
):  # pylint: disable=too-many-statements
    """Reads MonteCarlo simulation data file and builds parameters and flight
    variables matrices

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
        Numpy matrix containing input parameters values. Each column correspond
        to a parameter in the same order specified by 'parameters_list' input.
    target_variables_matrix: np.matrix
        Numpy matrix containing target variables values. Each column correspond
        to a target variable in the same order specified by 'target_variables_list'
        input.
    """
    number_of_samples_parameters = 0
    number_of_samples_variables = 0

    parameters_samples = {parameter: [] for parameter in parameters_list}
    with open(input_filename, "r") as parameters_file:
        for line in parameters_file.readlines():
            number_of_samples_parameters += 1

            parameters_dict = json.loads(line)
            parameters_dict = flatten_dict(parameters_dict)
            for parameter in parameters_list:
                try:
                    value = parameters_dict[parameter]
                except KeyError as e:
                    raise KeyError(
                        f"Parameter {parameter} was not found in {input_filename}!"
                    ) from e
                parameters_samples[parameter].append(value)

    target_variables_samples = {variable: [] for variable in target_variables_list}
    with open(output_filename, "r") as target_variables_file:
        for line in target_variables_file.readlines():
            number_of_samples_variables += 1
            target_variables_dict = json.loads(line)
            for variable in target_variables_list:
                try:
                    value = target_variables_dict[variable]
                except KeyError as e:
                    raise KeyError(
                        f"Variable {variable} was not found in {output_filename}!"
                    ) from e
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

    for i, parameter in enumerate(parameters_list):
        parameters_matrix[:, i] = parameters_samples[parameter]

    for i, target_variable in enumerate(target_variables_list):
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
    except ImportError as exc:  # pragma: no cover
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
                # pylint: disable=broad-except
                except Exception as e:  # pragma: no cover
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


def normalize_quaternions(quaternions):
    """Normalizes the quaternions (Euler parameters) to have unit magnitude.

    Parameters
    ----------
    quaternions : tuple
        Tuple containing the Euler parameters e0, e1, e2, e3

    Returns
    -------
    tuple
        Tuple containing the Euler parameters e0, e1, e2, e3
    """
    q_w, q_x, q_y, q_z = quaternions
    q_norm = (q_w**2 + q_x**2 + q_y**2 + q_z**2) ** 0.5
    if q_norm == 0:
        return 1, 0, 0, 0
    return q_w / q_norm, q_x / q_norm, q_y / q_norm, q_z / q_norm


def euler313_to_quaternions(phi, theta, psi):
    """Convert 3-1-3 Euler angles to Euler parameters (quaternions).

    Parameters
    ----------
    phi : float
        Rotation angle around the z-axis (in radians). Represents the precession
        angle or the roll angle.
    theta : float
        Rotation angle around the x-axis (in radians). Represents the nutation
        angle or the pitch angle.
    psi : float
        Rotation angle around the z-axis (in radians). Represents the spin angle
        or the roll angle.

    Returns
    -------
    tuple[float, float, float, float]
        The Euler parameters or quaternions (e0, e1, e2, e3)

    References
    ----------
    https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
    """
    cphi = np.cos(phi / 2)
    sphi = np.sin(phi / 2)
    ctheta = np.cos(theta / 2)
    stheta = np.sin(theta / 2)
    cpsi = np.cos(psi / 2)
    spsi = np.sin(psi / 2)
    e0 = cphi * ctheta * cpsi - sphi * ctheta * spsi
    e1 = cphi * cpsi * stheta + sphi * stheta * spsi
    e2 = cphi * stheta * spsi - sphi * cpsi * stheta
    e3 = cphi * ctheta * spsi + ctheta * cpsi * sphi
    return e0, e1, e2, e3


def get_matplotlib_supported_file_endings():
    """Gets the file endings supported by matplotlib.

    Returns
    -------
    list[str]
        List of file endings prepended with a dot
    """
    # Get matplotlib's supported file ending and return them (without descriptions, hence only keys)
    filetypes = plt.gcf().canvas.get_supported_filetypes().keys()

    # Ensure the dot is included in the filetype endings
    filetypes = ["." + filetype for filetype in filetypes]

    return filetypes


def to_hex_encode(obj, encoder=base64.b85encode):
    """Converts an object to hex representation using dill.

    Parameters
    ----------
    obj : object
        Object to be converted to hex.
    encoder : callable, optional
        Function to encode the bytes. Default is base64.b85encode.

    Returns
    -------
    bytes
        Object converted to bytes.
    """
    return encoder(dill.dumps(obj)).hex()


def from_hex_decode(obj_bytes, decoder=base64.b85decode):
    """Converts an object from hex representation using dill.

    Parameters
    ----------
    obj_bytes : str
        Hex string to be converted to object.
    decoder : callable, optional
        Function to decode the bytes. Default is base64.b85decode.

    Returns
    -------
    object
        Object converted from bytes.
    """
    return dill.loads(decoder(bytes.fromhex(obj_bytes)))


def find_obj_from_hash(obj, hash_, depth_limit=None):
    """Searches the object (and its children) for
    an object whose '__rpy_hash' field has a particular hash value.

    Parameters
    ----------
    obj : object
        Object to search.
    hash_ : int
        Hash value to search for in the '__rpy_hash' field.
    depth_limit : int, optional
        Maximum depth to search recursively. If None, no limit.

    Returns
    -------
    object
        The object whose '__rpy_hash' matches ``hash_``, or None if not found.
    """

    stack = [(obj, 0)]
    while stack:
        current_obj, current_depth = stack.pop()
        if depth_limit is not None and current_depth > depth_limit:
            continue

        if getattr(current_obj, "__rpy_hash", None) == hash_:
            return current_obj

        if isinstance(current_obj, dict):
            stack.extend((v, current_depth + 1) for v in current_obj.values())

        elif isinstance(current_obj, (list, tuple, set)):
            stack.extend((item, current_depth + 1) for item in current_obj)

    return None


if __name__ == "__main__":  # pragma: no cover
    import doctest

    res = doctest.testmod()
    if res.failed < 1:
        print(f"All the {res.attempted} tests passed!")
    else:
        print(f"{res.failed} out of {res.attempted} tests failed.")
