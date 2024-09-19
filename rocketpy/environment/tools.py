""""This module contains auxiliary functions for helping with the Environment
classes operations. The functions mainly deal with wind calculations and
interpolation of data from netCDF4 datasets. As this is a recent addition to
the library (introduced in version 1.5.0), some functions may be modified in the
future to improve their performance and usability.
"""

import bisect
import warnings

import netCDF4
import numpy as np

from rocketpy.tools import bilinear_interpolation

## Wind data functions


def calculate_wind_heading(u, v):
    """Calculates the wind heading from the u and v components of the wind.

    Parameters
    ----------
    u : float
        The velocity of the wind in the u (or x) direction. It can be either
        positive or negative values.
    v : float
        The velocity of the wind in the v (or y) direction. It can be either
        positive or negative values.

    Returns
    -------
    float
        The wind heading in degrees, ranging from 0 to 360 degrees.

    Examples
    --------
    >>> from rocketpy.environment.tools import calculate_wind_heading
    >>> calculate_wind_heading(1, 0)
    np.float64(90.0)
    >>> calculate_wind_heading(0, 1)
    np.float64(0.0)
    >>> calculate_wind_heading(3, 3)
    np.float64(45.0)
    >>> calculate_wind_heading(-3, 3)
    np.float64(315.0)
    """
    return np.degrees(np.arctan2(u, v)) % 360


def convert_wind_heading_to_direction(wind_heading):
    """Converts wind heading to wind direction. The wind direction is the
    direction from which the wind is coming from, while the wind heading is the
    direction to which the wind is blowing to.

    Parameters
    ----------
    wind_heading : float
        The wind heading in degrees, ranging from 0 to 360 degrees.

    Returns
    -------
    float
        The wind direction in degrees, ranging from 0 to 360 degrees.
    """
    return (wind_heading - 180) % 360


def calculate_wind_speed(u, v, w=0.0):
    """Calculates the wind speed from the u, v, and w components of the wind.

    Parameters
    ----------
    u : float
        The velocity of the wind in the u (or x) direction. It can be either
        positive or negative values.
    v : float
        The velocity of the wind in the v (or y) direction. It can be either
        positive or negative values.
    w : float
        The velocity of the wind in the w (or z) direction. It can be either
        positive or negative values.

    Returns
    -------
    float
        The wind speed in m/s.

    Examples
    --------
    >>> from rocketpy.environment.tools import calculate_wind_speed
    >>> calculate_wind_speed(1, 0, 0)
    np.float64(1.0)
    >>> calculate_wind_speed(0, 1, 0)
    np.float64(1.0)
    >>> calculate_wind_speed(0, 0, 1)
    np.float64(1.0)
    >>> calculate_wind_speed(3, 4, 0)
    np.float64(5.0)

    The third component of the wind is optional, and if not provided, it is
    assumed to be zero.

    >>> calculate_wind_speed(3, 4)
    np.float64(5.0)
    >>> calculate_wind_speed(3, 4, 0)
    np.float64(5.0)
    """
    return np.sqrt(u**2 + v**2 + w**2)


## These functions are meant to be used with netcdf4 datasets


def get_pressure_levels_from_file(data, dictionary):
    """Extracts pressure levels from a netCDF4 dataset and converts them to Pa.

    Parameters
    ----------
    data : netCDF4.Dataset
        The netCDF4 dataset containing the pressure level data.
    dictionary : dict
        A dictionary mapping variable names to dataset keys.

    Returns
    -------
    numpy.ndarray
        An array of pressure levels in Pa.

    Raises
    ------
    ValueError
        If the pressure levels cannot be read from the file.
    """
    try:
        # Convert mbar to Pa
        levels = 100 * data.variables[dictionary["level"]][:]
    except KeyError as e:
        raise ValueError(
            "Unable to read pressure levels from file. Check file and dictionary."
        ) from e
    return levels


def mask_and_clean_dataset(*args):
    """Masks and cleans a dataset by removing rows with masked values.

    Parameters
    ----------
    *args : numpy.ma.MaskedArray
        Variable number of masked arrays to be cleaned.

    Returns
    -------
    numpy.ma.MaskedArray
        A cleaned array with rows containing masked values removed.
    """
    data_array = np.ma.column_stack(list(args))

    # Remove lines with masked content
    if np.any(data_array.mask):
        data_array = np.ma.compress_rows(data_array)
        warnings.warn(
            "Some values were missing from this weather dataset, therefore, "
            "certain pressure levels were removed."
        )

    return data_array


def find_longitude_index(longitude, lon_list):
    """Finds the index of the given longitude in a list of longitudes.

    Parameters
    ----------
    longitude : float
        The longitude to find in the list.
    lon_list : list of float
        The list of longitudes.

    Returns
    -------
    tuple
        A tuple containing the adjusted longitude and its index in the list.

    Raises
    ------
    ValueError
        If the longitude is not within the range covered by the list.
    """
    # Determine if file uses -180 to 180 or 0 to 360
    if lon_list[0] < 0 or lon_list[-1] < 0:
        # Convert input to -180 - 180
        lon = longitude if longitude < 180 else -180 + longitude % 180
    else:
        # Convert input to 0 - 360
        lon = longitude % 360
    # Check if reversed or sorted
    if lon_list[0] < lon_list[-1]:
        # Deal with sorted lon_list
        lon_index = bisect.bisect(lon_list, lon)
    else:
        # Deal with reversed lon_list
        lon_list.reverse()
        lon_index = len(lon_list) - bisect.bisect_left(lon_list, lon)
        lon_list.reverse()
    # Take care of longitude value equal to maximum longitude in the grid
    if lon_index == len(lon_list) and lon_list[lon_index - 1] == lon:
        lon_index = lon_index - 1
    # Check if longitude value is inside the grid
    if lon_index == 0 or lon_index == len(lon_list):
        raise ValueError(
            f"Longitude {lon} not inside region covered by file, which is "
            f"from {lon_list[0]} to {lon_list[-1]}."
        )

    return lon, lon_index


def find_latitude_index(latitude, lat_list):
    """Finds the index of the given latitude in a list of latitudes.

    Parameters
    ----------
    latitude : float
        The latitude to find in the list.
    lat_list : list of float
        The list of latitudes.

    Returns
    -------
    tuple
        A tuple containing the latitude and its index in the list.

    Raises
    ------
    ValueError
        If the latitude is not within the range covered by the list.
    """
    # Check if reversed or sorted
    if lat_list[0] < lat_list[-1]:
        # Deal with sorted lat_list
        lat_index = bisect.bisect(lat_list, latitude)
    else:
        # Deal with reversed lat_list
        lat_list.reverse()
        lat_index = len(lat_list) - bisect.bisect_left(lat_list, latitude)
        lat_list.reverse()
    # Take care of latitude value equal to maximum longitude in the grid
    if lat_index == len(lat_list) and lat_list[lat_index - 1] == latitude:
        lat_index = lat_index - 1
    # Check if latitude value is inside the grid
    if lat_index == 0 or lat_index == len(lat_list):
        raise ValueError(
            f"Latitude {latitude} not inside region covered by file, "
            f"which is from {lat_list[0]} to {lat_list[-1]}."
        )
    return latitude, lat_index


def find_time_index(datetime_date, time_array):
    """Finds the index of the given datetime in a netCDF4 time array.

    Parameters
    ----------
    datetime_date : datetime.datetime
        The datetime to find in the array.
    time_array : netCDF4.Variable
        The netCDF4 time array.

    Returns
    -------
    int
        The index of the datetime in the time array.

    Raises
    ------
    ValueError
        If the datetime is not within the range covered by the time array.
    ValueError
        If the exact datetime is not available and the nearest datetime is used instead.
    """
    time_index = netCDF4.date2index(
        datetime_date, time_array, calendar="gregorian", select="nearest"
    )
    # Convert times do dates and numbers
    input_time_num = netCDF4.date2num(
        datetime_date, time_array.units, calendar="gregorian"
    )
    file_time_num = time_array[time_index]
    file_time_date = netCDF4.num2date(
        time_array[time_index], time_array.units, calendar="gregorian"
    )
    # Check if time is inside range supplied by file
    if time_index == 0 and input_time_num < file_time_num:
        raise ValueError(
            f"The chosen launch time '{datetime_date.strftime('%Y-%m-%d-%H:')} UTC' is"
            " not available in the provided file. Please choose a time within the range"
            " of the file, which starts at "
            f"'{file_time_date.strftime('%Y-%m-%d-%H')} UTC'."
        )
    elif time_index == len(time_array) - 1 and input_time_num > file_time_num:
        raise ValueError(
            "Chosen launch time is not available in the provided file, "
            f"which ends at {file_time_date}."
        )
    # Check if time is exactly equal to one in the file
    if input_time_num != file_time_num:
        warnings.warn(
            "Exact chosen launch time is not available in the provided file, "
            f"using {file_time_date} UTC instead."
        )

    return time_index


def get_elevation_data_from_dataset(
    dictionary, data, time_index, lat_index, lon_index, x, y, x1, x2, y1, y2
):
    """Retrieves elevation data from a netCDF4 dataset and applies bilinear
    interpolation.

    Parameters
    ----------
    dictionary : dict
        A dictionary mapping variable names to dataset keys.
    data : netCDF4.Dataset
        The netCDF4 dataset containing the elevation data.
    time_index : int
        The time index for the data.
    lat_index : int
        The latitude index for the data.
    lon_index : int
        The longitude index for the data.
    x : float
        The x-coordinate of the point to be interpolated.
    y : float
        The y-coordinate of the point to be interpolated.
    x1 : float
        The x-coordinate of the first reference point.
    x2 : float
        The x-coordinate of the second reference point.
    y1 : float
        The y-coordinate of the first reference point.
    y2 : float
        The y-coordinate of the second reference point.

    Returns
    -------
    float
        The interpolated elevation value at the point (x, y).

    Raises
    ------
    ValueError
        If the elevation data cannot be read from the file.
    """
    try:
        elevations = data.variables[dictionary["surface_geopotential_height"]][
            time_index, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
        ]
    except KeyError as e:
        raise ValueError(
            "Unable to read surface elevation data. Check file and dictionary."
        ) from e
    return bilinear_interpolation(
        x,
        y,
        x1,
        x2,
        y1,
        y2,
        elevations[0, 0],
        elevations[0, 1],
        elevations[1, 0],
        elevations[1, 1],
    )


def get_initial_date_from_time_array(time_array, units=None):
    """Returns a datetime object representing the first time in the time array.

    Parameters
    ----------
    time_array : netCDF4.Variable
        The netCDF4 time array.
    units : str, optional
        The time units, by default None.

    Returns
    -------
    datetime.datetime
        A datetime object representing the first time in the time array.
    """
    units = units or time_array.units
    return netCDF4.num2date(time_array[0], units, calendar="gregorian")


def get_final_date_from_time_array(time_array, units=None):
    """Returns a datetime object representing the last time in the time array.

    Parameters
    ----------
    time_array : netCDF4.Variable
        The netCDF4 time array.
    units : str, optional
        The time units, by default None.

    Returns
    -------
    datetime.datetime
        A datetime object representing the last time in the time array.
    """
    units = units if units is not None else time_array.units
    return netCDF4.num2date(time_array[-1], units, calendar="gregorian")


def get_interval_date_from_time_array(time_array, units=None):
    """Returns the interval between two times in the time array in hours.

    Parameters
    ----------
    time_array : netCDF4.Variable
        The netCDF4 time array.
    units : str, optional
        The time units, by default None. If None is set, the units from the
        time array are used.

    Returns
    -------
    int
        The interval in hours between two times in the time array.
    """
    units = units or time_array.units
    return netCDF4.num2date(
        (time_array[-1] - time_array[0]) / (len(time_array) - 1),
        units,
        calendar="gregorian",
    ).hour


# Geodesic conversions functions


def geodesic_to_utm(
    lat, lon, semi_major_axis=6378137.0, flattening=1 / 298.257223563
):  # pylint: disable=too-many-locals,too-many-statements
    # NOTE: already documented in the Environment class.
    # TODO: deprecated the static method from the environment class, use only this one.

    # Calculate the central meridian of UTM zone
    if lon != 0:
        signal = lon / abs(lon)
        if signal > 0:
            aux = lon - 3
            aux = aux * signal
            div = aux // 6
            lon_mc = div * 6 + 3
            EW = "E"  # pylint: disable=invalid-name
        else:
            aux = lon + 3
            aux = aux * signal
            div = aux // 6
            lon_mc = (div * 6 + 3) * signal
            EW = "W"  # pylint: disable=invalid-name
    else:
        lon_mc = 3
        EW = "W|E"  # pylint: disable=invalid-name

    # Evaluate the hemisphere and determine the N coordinate at the Equator
    if lat < 0:
        N0 = 10000000
        hemis = "S"
    else:
        N0 = 0
        hemis = "N"

    # Convert the input lat and lon to radians
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    lon_mc = lon_mc * np.pi / 180

    # Evaluate reference parameters
    K0 = 1 - 1 / 2500
    e2 = 2 * flattening - flattening**2
    e2lin = e2 / (1 - e2)

    # Evaluate auxiliary parameters
    A = e2 * e2
    B = A * e2
    C = np.sin(2 * lat)
    D = np.sin(4 * lat)
    E = np.sin(6 * lat)
    F = (1 - e2 / 4 - 3 * A / 64 - 5 * B / 256) * lat
    G = (3 * e2 / 8 + 3 * A / 32 + 45 * B / 1024) * C
    H = (15 * A / 256 + 45 * B / 1024) * D
    aux_i = (35 * B / 3072) * E

    # Evaluate other reference parameters
    n = semi_major_axis / ((1 - e2 * (np.sin(lat) ** 2)) ** 0.5)
    t = np.tan(lat) ** 2
    c = e2lin * (np.cos(lat) ** 2)
    ag = (lon - lon_mc) * np.cos(lat)
    m = semi_major_axis * (F - G + H - aux_i)

    # Evaluate new auxiliary parameters
    J = (1 - t + c) * ag * ag * ag / 6
    K = (5 - 18 * t + t * t + 72 * c - 58 * e2lin) * (ag**5) / 120
    L = (5 - t + 9 * c + 4 * c * c) * ag * ag * ag * ag / 24
    M = (61 - 58 * t + t * t + 600 * c - 330 * e2lin) * (ag**6) / 720

    # Evaluate the final coordinates
    x = 500000 + K0 * n * (ag + J + K)
    y = N0 + K0 * (m + n * np.tan(lat) * (ag * ag / 2 + L + M))

    # Convert the output lat and lon to degrees
    lat = lat * 180 / np.pi
    lon = lon * 180 / np.pi
    lon_mc = lon_mc * 180 / np.pi

    # Calculate the UTM zone number
    utm_zone = int((lon_mc + 183) / 6)

    # Calculate the UTM zone letter
    letters = "CDEFGHJKLMNPQRSTUVWXX"
    utm_letter = letters[int(80 + lat) >> 3]

    return x, y, utm_zone, utm_letter, hemis, EW


def utm_to_geodesic(  # pylint: disable=too-many-locals,too-many-statements
    x, y, utm_zone, hemis, semi_major_axis=6378137.0, flattening=1 / 298.257223563
):
    # NOTE: already documented in the Environment class.
    # TODO: deprecate the static method from the environment class, use only this one.

    if hemis == "N":
        y = y + 10000000

    # Calculate the Central Meridian from the UTM zone number
    central_meridian = utm_zone * 6 - 183  # degrees

    # Calculate reference values
    K0 = 1 - 1 / 2500
    e2 = 2 * flattening - flattening**2
    e2lin = e2 / (1 - e2)
    e1 = (1 - (1 - e2) ** 0.5) / (1 + (1 - e2) ** 0.5)

    # Calculate auxiliary values
    A = e2 * e2
    B = A * e2
    C = e1 * e1
    D = e1 * C
    E = e1 * D

    m = (y - 10000000) / K0
    mi = m / (semi_major_axis * (1 - e2 / 4 - 3 * A / 64 - 5 * B / 256))

    # Calculate other auxiliary values
    F = (3 * e1 / 2 - 27 * D / 32) * np.sin(2 * mi)
    G = (21 * C / 16 - 55 * E / 32) * np.sin(4 * mi)
    H = (151 * D / 96) * np.sin(6 * mi)

    lat1 = mi + F + G + H
    c1 = e2lin * (np.cos(lat1) ** 2)
    t1 = np.tan(lat1) ** 2
    n1 = semi_major_axis / ((1 - e2 * (np.sin(lat1) ** 2)) ** 0.5)
    quoc = (1 - e2 * np.sin(lat1) * np.sin(lat1)) ** 3
    r1 = semi_major_axis * (1 - e2) / (quoc**0.5)
    d = (x - 500000) / (n1 * K0)

    # Calculate other auxiliary values
    aux_i = (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * e2lin) * d * d * d * d / 24
    J = (
        (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * e2lin - 3 * c1 * c1)
        * (d**6)
        / 720
    )
    K = d - (1 + 2 * t1 + c1) * d * d * d / 6
    L = (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * e2lin + 24 * t1 * t1) * (d**5) / 120

    # Finally calculate the coordinates in lat/lot
    lat = lat1 - (n1 * np.tan(lat1) / r1) * (d * d / 2 - aux_i + J)
    lon = central_meridian * np.pi / 180 + (K + L) / np.cos(lat1)

    # Convert final lat/lon to Degrees
    lat = lat * 180 / np.pi
    lon = lon * 180 / np.pi

    return lat, lon


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()
    if results.failed < 1:
        print(f"All the {results.attempted} tests passed!")
    else:
        print(f"{results.failed} out of {results.attempted} tests failed.")
