_NOT_FOUND = object()

import numpy as np
import pytz
from cftime import num2pydate


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
    """
    return (
        z11 * (x2 - x) * (y2 - y)
        + z21 * (x - x1) * (y2 - y)
        + z12 * (x2 - x) * (y - y1)
        + z22 * (x - x1) * (y - y1)
    ) / ((x2 - x1) * (y2 - y1))


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
    Automatically converts time number from UTC to local timezone based on
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
    >>> geopotential_to_height_asl(0)
    0.0
    >>> geopotential_to_height_asl(100000)
    849.5
    >>> geopotential_to_height_asl(200000)
    1699.0
    """
    geopotential_height = geopotential / g
    return radius * geopotential_height / (radius - geopotential_height)


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
    >>> geopotential_to_height_agl(0, 0)
    0.0
    >>> geopotential_to_height_agl(100000, 0)
    849.5
    >>> geopotential_to_height_agl(100000, 1000)
    1849.5
    """
    return geopotential_to_height_asl(geopotential, radius, g) - elevation
