# In this file we define functions that are used in the rocketpy package.
# Note that these functions can not depend on any other rocketpy module.
# If they depend on another module, they should be moved to that module or to
# the utilities module.

_NOT_FOUND = object()

import math

import numpy as np
from numpy.random import *


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


# TODO: discuss if this is in the right place
def get_distribution(distribuition_function_name):
    """Sets the distribution function to be used in the dispersion analysis.

    Parameters
    ----------
    distribuition_function_name : string
        The type of distribution to be used in the analysis. It can be
        'uniform', 'normal', 'lognormal', etc.

    Returns
    -------
    np.random distribution function
        The distribution function to be used in the analysis.
    """
    distributions = {
        "normal": normal,
        "beta": beta,
        "binomial": binomial,
        "chisquare": chisquare,
        "dirichlet": dirichlet,
        "exponential": exponential,
        "f": f,
        "gamma": gamma,
        "geometric": geometric,
        "gumbel": gumbel,
        "hypergeometric": hypergeometric,
        "laplace": laplace,
        "logistic": logistic,
        "lognormal": lognormal,
        "logseries": logseries,
        "multinomial": multinomial,
        "multivariate_normal": multivariate_normal,
        "negative_binomial": negative_binomial,
        "noncentral_chisquare": noncentral_chisquare,
        "noncentral_f": noncentral_f,
        "pareto": pareto,
        "poisson": poisson,
        "power": power,
        "rayleigh": rayleigh,
        "standard_cauchy": standard_cauchy,
        "standard_exponential": standard_exponential,
        "standard_gamma": standard_gamma,
        "standard_normal": standard_normal,
        "standard_t": standard_t,
        "triangular": triangular,
        "uneliform": uniform,
        "vonmises": vonmises,
        "wald": wald,
        "weibull": weibull,
        "zipf": zipf,
    }
    try:
        return distributions[distribuition_function_name]
    except KeyError:
        raise ValueError(
            f"Distribution function '{distribuition_function_name}' not found, please use a np.random distribution."
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
