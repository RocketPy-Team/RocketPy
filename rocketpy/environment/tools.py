"""This module contains auxiliary functions for helping with the Environment
classes operations. The functions mainly deal with wind calculations and
interpolation of data from netCDF4 datasets. As this is a recent addition to
the library (introduced in version 1.5.0), some functions may be modified in the
future to improve their performance and usability.
"""

import math
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


def geodesic_to_lambert_conformal(lat, lon, projection_variable, x_units="m"):
    """Convert geodesic coordinates to Lambert conformal projected coordinates.

    Parameters
    ----------
    lat : float
        Latitude in degrees, ranging from -90 to 90
    lon : float
        Longitude in degrees, ranging from -180 to 180.
    projection_variable : netCDF4.Variable
        Projection variable containing Lambert conformal metadata.
    x_units : str, optional
        Units used by the dataset x coordinate. Supported values are meters
        and kilometers. Default is "m".

    Returns
    -------
    tuple[float, float]
        Projected coordinates ``(x, y)`` in the same units as ``x_units``.
    """
    lat_radians = math.radians(lat)
    lon_radians = math.radians(lon % 360)

    lat_origin = math.radians(float(projection_variable.latitude_of_projection_origin))
    lon_origin = math.radians(float(projection_variable.longitude_of_central_meridian))

    standard_parallel = projection_variable.standard_parallel
    if np.ndim(standard_parallel) == 0:
        standard_parallels = [float(standard_parallel)]
    else:
        standard_parallels = np.asarray(standard_parallel, dtype=float).tolist()

    if len(standard_parallels) >= 2:
        phi_1 = math.radians(standard_parallels[0])
        phi_2 = math.radians(standard_parallels[1])
        n = math.log(math.cos(phi_1) / math.cos(phi_2)) / math.log(
            math.tan(math.pi / 4 + phi_2 / 2) / math.tan(math.pi / 4 + phi_1 / 2)
        )
    else:
        phi_1 = math.radians(standard_parallels[0])
        n = math.sin(phi_1)

    earth_radius = float(getattr(projection_variable, "earth_radius", 6371229.0))
    f_const = (math.cos(phi_1) * math.tan(math.pi / 4 + phi_1 / 2) ** n) / n

    rho = earth_radius * f_const / (math.tan(math.pi / 4 + lat_radians / 2) ** n)
    rho_origin = earth_radius * f_const / (math.tan(math.pi / 4 + lat_origin / 2) ** n)
    theta = n * (lon_radians - lon_origin)

    x_meters = rho * math.sin(theta)
    y_meters = rho_origin - rho * math.cos(theta)

    if str(x_units).lower().startswith("km"):
        return x_meters / 1000.0, y_meters / 1000.0
    return x_meters, y_meters


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


def _normalize_longitude_value(longitude, lon_start, lon_end):
    """Normalize longitude based on grid format [-180, 180] or [0, 360].

    Parameters
    ----------
    longitude : float
        The longitude to normalize.
    lon_start : float
        The first longitude value in the grid.
    lon_end : float
        The last longitude value in the grid.

    Returns
    -------
    float
        The normalized longitude value.
    """
    # Determine if file uses geographic longitudes in [-180, 180] or [0, 360].
    # Do not remap projected x coordinates.
    is_geographic_longitude = abs(lon_start) <= 360 and abs(lon_end) <= 360
    if is_geographic_longitude:
        if lon_start < 0 or lon_end < 0:
            return longitude if longitude < 180 else -180 + longitude % 180
        return longitude % 360
    return longitude


def _binary_search_coordinate_index(target_value, coord_list, is_ascending):
    """Find insertion index for target value using binary search.

    Parameters
    ----------
    target_value : float
        The coordinate value to locate.
    coord_list : list of float
        The list of coordinate values.
    is_ascending : bool
        Whether the coordinate list is in ascending order.

    Returns
    -------
    int
        The insertion index such that coord_list[index-1] and coord_list[index]
        bracket the target value.
    """
    low = 0
    high = len(coord_list)
    while low < high:
        mid = (low + high) // 2
        mid_value = float(coord_list[mid])
        if (mid_value < target_value) if is_ascending else (mid_value > target_value):
            low = mid + 1
        else:
            high = mid
    return low


def _adjust_boundary_coordinate_index(index, coord_list, coord_value):
    """Adjust index for exact matches at grid boundaries.

    Parameters
    ----------
    index : int
        The current index from binary search.
    coord_list : list of float
        The list of coordinate values.
    coord_value : float
        The coordinate value being matched.

    Returns
    -------
    int
        The adjusted index after boundary handling.
    """
    coord_len = len(coord_list)
    if index == 0 and math.isclose(float(coord_list[0]), coord_value):
        return 1
    if index == coord_len and float(coord_list[coord_len - 1]) == coord_value:
        return index - 1
    return index


def _validate_coordinate_index_in_range(
    index, coord_len, coord_start, coord_end, coord_name
):
    """Validate that coordinate index is within valid interpolation range.

    Parameters
    ----------
    index : int
        The coordinate index to validate.
    coord_len : int
        The length of the coordinate list.
    coord_start : float
        The first coordinate value in the grid.
    coord_end : float
        The last coordinate value in the grid.
    coord_name : str
        The name of the coordinate (e.g., "Longitude", "Latitude").

    Raises
    ------
    ValueError
        If the index is out of valid range (0 or coord_len).
    """
    if index in (0, coord_len):
        raise ValueError(
            f"{coord_name} not inside region covered by file, which is "
            f"from {coord_start} to {coord_end}."
        )


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
    lon_len = len(lon_list)
    lon_start = float(lon_list[0])
    lon_end = float(lon_list[lon_len - 1])

    lon = _normalize_longitude_value(longitude, lon_start, lon_end)
    is_ascending = lon_start < lon_end

    lon_index = _binary_search_coordinate_index(lon, lon_list, is_ascending)
    lon_index = _adjust_boundary_coordinate_index(lon_index, lon_list, lon)

    _validate_coordinate_index_in_range(
        lon_index, lon_len, lon_start, lon_end, "Longitude"
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
    lat_len = len(lat_list)
    lat_start = float(lat_list[0])
    lat_end = float(lat_list[lat_len - 1])
    is_ascending = lat_start < lat_end

    lat_index = _binary_search_coordinate_index(latitude, lat_list, is_ascending)
    lat_index = _adjust_boundary_coordinate_index(lat_index, lat_list, latitude)

    _validate_coordinate_index_in_range(
        lat_index, lat_len, lat_start, lat_end, "Latitude"
    )

    return latitude, lat_index


def find_time_index(datetime_date, time_array):  # pylint: disable=too-many-statements
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
    time_len = len(time_array)
    time_units = time_array.units
    input_time_num = netCDF4.date2num(datetime_date, time_units, calendar="gregorian")

    first_time_num = float(time_array[0])
    last_time_num = float(time_array[time_len - 1])
    is_ascending = first_time_num <= last_time_num

    # Binary search nearest index using scalar probing only.
    low = 0
    high = time_len
    while low < high:
        mid = (low + high) // 2
        mid_time_num = float(time_array[mid])
        if (
            (mid_time_num < input_time_num)
            if is_ascending
            else (mid_time_num > input_time_num)
        ):
            low = mid + 1
        else:
            high = mid

    right_index = min(max(low, 0), time_len - 1)
    left_index = min(max(right_index - 1, 0), time_len - 1)

    right_time_num = float(time_array[right_index])
    left_time_num = float(time_array[left_index])
    if abs(input_time_num - left_time_num) <= abs(right_time_num - input_time_num):
        time_index = left_index
        file_time_num = left_time_num
    else:
        time_index = right_index
        file_time_num = right_time_num

    file_time_date = netCDF4.num2date(file_time_num, time_units, calendar="gregorian")

    # Check if time is inside range supplied by file
    if time_index == 0 and (
        (is_ascending and input_time_num < file_time_num)
        or (not is_ascending and input_time_num > file_time_num)
    ):
        raise ValueError(
            f"The chosen launch time '{datetime_date.strftime('%Y-%m-%d-%H:')} UTC' is"
            " not available in the provided file. Please choose a time within the range"
            " of the file, which starts at "
            f"'{file_time_date.strftime('%Y-%m-%d-%H')} UTC'."
        )
    elif time_index == time_len - 1 and (
        (is_ascending and input_time_num > file_time_num)
        or (not is_ascending and input_time_num < file_time_num)
    ):
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


def geodesic_to_utm(lat, lon, semi_major_axis=6378137.0, flattening=1 / 298.257223563):  # pylint: disable=too-many-locals,too-many-statements
    """Function which converts geodetic coordinates, i.e. lat/lon, to UTM
    projection coordinates. Can be used only for latitudes between -80.00°
    and 84.00°

    Parameters
    ----------
    lat : float
        The latitude coordinates of the point of analysis, must be contained
        between -80.00° and 84.00°
    lon : float
        The longitude coordinates of the point of analysis, must be
        contained between -180.00° and 180.00°
    semi_major_axis : float
        The semi-major axis of the ellipsoid used to represent the Earth,
        must be given in meters (default is 6,378,137.0 m, which corresponds
        to the WGS84 ellipsoid)
    flattening : float
        The flattening of the ellipsoid used to represent the Earth, usually
        between 1/250 and 1/150 (default is 1/298.257223563, which
        corresponds to the WGS84 ellipsoid)

    Returns
    -------
    x : float
        East coordinate, always positive
    y : float
        North coordinate, always positive
    utm_zone : int
        The number of the UTM zone of the point of analysis, can vary
        between 1 and 60
    utm_letter : string
        The letter of the UTM zone of the point of analysis, can vary
        between C and X, omitting the letters "I" and "O"
    hemis : string
        Returns "S" for southern hemisphere and "N" for Northern hemisphere
    EW : string
        Returns "W" for western hemisphere and "E" for eastern hemisphere
    """
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
    """Function to convert UTM coordinates to geodesic coordinates
    (i.e. latitude and longitude).

    Parameters
    ----------
    x : float
        East UTM coordinate in meters
    y : float
        North UTM coordinate in meters
    utm_zone : int
        The number of the UTM zone of the point of analysis, can vary
        between 1 and 60
    hemis : string
        Equals to "S" for southern hemisphere and "N" for Northern
        hemisphere
    semi_major_axis : float
        The semi-major axis of the ellipsoid used to represent the Earth,
        must be given in meters (default is 6,378,137.0 m, which corresponds
        to the WGS84 ellipsoid)
    flattening : float
        The flattening of the ellipsoid used to represent the Earth, usually
        between 1/250 and 1/150 (default is 1/298.257223563, which
        corresponds to the WGS84 ellipsoid)

    Returns
    -------
    lat : float
        latitude of the analyzed point
    lon : float
        latitude of the analyzed point
    """
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


if __name__ == "__main__":  # pragma: no cover
    import doctest

    results = doctest.testmod()
    if results.failed < 1:
        print(f"All the {results.attempted} tests passed!")
    else:
        print(f"{results.failed} out of {results.attempted} tests failed.")
