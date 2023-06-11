# In this file we define functions that are used in the rocketpy package.
# Note that these functions can not depend on any other rocketpy module.
# If they depend on another module, they should be moved to that module or to
# the utilities module.

_NOT_FOUND = object()

import math

import numpy as np
from matplotlib.patches import Ellipse


class cached_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        cache = instance.__dict__
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            cache[self.attrname] = val
        return val


def get_distribution(distribution_function_name):
    """Sets the distribution function to be used in the dispersion analysis.

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
    except KeyError:
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
        )


def Haversine(lat0, lon0, lat1, lon1, eRadius=6.3781e6):
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
    eRadius : float, optional
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

    return eRadius * c


def invertedHaversine(lat0, lon0, distance, bearing, eRadius=6.3781e6):
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
    eRadius : float, optional
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
        math.sin(lat0_rad) * math.cos(distance / eRadius)
        + math.cos(lat0_rad) * math.sin(distance / eRadius) * math.cos(bearing)
    )

    lon1_rad = lon0_rad + math.atan2(
        math.sin(bearing) * math.sin(distance / eRadius) * math.cos(lat0_rad),
        math.cos(distance / eRadius) - math.sin(lat0_rad) * math.sin(lat1_rad),
    )

    # Convert back to degrees and then return
    lat1_deg = np.rad2deg(lat1_rad)
    lon1_deg = np.rad2deg(lon1_rad)

    return lat1_deg, lon1_deg


def decimalDegreesToArcSeconds(angle):
    """Function to convert an angle in decimal degrees to deg/min/sec.
     Converts (°) to (° ' ")

    Parameters
    ----------
    angle : float
        The angle that you need convert to deg/min/sec. Must be given in
        decimal degrees.

    Returns
    -------
    deg: float
        The degrees.
    min: float
        The arc minutes. 1 arc-minute = (1/60)*degree
    sec: float
        The arc Seconds. 1 arc-second = (1/3600)*degree
    """

    if angle < 0:
        signal = -1
    else:
        signal = 1

    deg = (signal * angle) // 1
    min = abs(signal * angle - deg) * 60 // 1
    sec = abs((signal * angle - deg) * 60 - min) * 60

    return deg, min, sec


# Functions for dispersion analysis
def generate_dispersion_ellipses(results):
    """A function to create apogee and impact ellipses from the dispersion
    results.

    Parameters
    ----------
    results : dict
        A dictionary containing the results of the dispersion analysis. It should
        contain the following keys:
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

    # Retrieve dispersion data por apogee and impact XY position
    try:
        apogee_x = np.array(results["apogeeX"])
        apogee_y = np.array(results["apogeeY"])
    except KeyError:
        print("No apogee data found. Skipping apogee ellipses.")
        apogee_x = np.array([])
        apogee_y = np.array([])
    try:
        impact_x = np.array(results["xImpact"])
        impact_y = np.array(results["yImpact"])
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
    impactTheta, impactW, impactH = calculate_ellipses(impact_x, impact_y)
    apogeeTheta, apogeeW, apogeeH = calculate_ellipses(apogee_x, apogee_y)

    # Draw error ellipses for impact
    impact_ellipses = create_ellipse_objects(
        impact_x, impact_y, 3, impactW, impactH, impactTheta, (0, 0, 1, 0.2)
    )

    apogee_ellipses = create_ellipse_objects(
        apogee_x, apogee_y, 3, apogeeW, apogeeH, apogeeTheta, (0, 1, 0, 0.2)
    )

    return impact_ellipses, apogee_ellipses, apogee_x, apogee_y, impact_x, impact_y


def generate_dispersion_ellipses_coordinates(
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

            lat_lon_points[i] = invertedHaversine(
                origin_lat, origin_lon, d, bearing, eRadius=6.3781e6
            )

        outputs[index] = lat_lon_points
    return outputs
