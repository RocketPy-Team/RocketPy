import importlib
import importlib.metadata
import re
from bisect import bisect_left

import pytz
from cftime import num2pydate
from packaging import version as packaging_version

_NOT_FOUND = object()

# Mapping of module name and the name of the package that should be installed
INSTALL_MAPPING = {"IPython": "ipython"}


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
    >>> from rocketpy.tools import geopotential_to_height_asl
    >>> geopotential_to_height_asl(0)
    0.0
    >>> geopotential_to_height_asl(100000)
    10198.792680243916
    >>> geopotential_to_height_asl(200000)
    20400.84750449947
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


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()
    if results.failed < 1:
        print(f"All the {results.attempted} tests passed!")
    else:
        print(f"{results.failed} out of {results.attempted} tests failed.")
