import bisect
import warnings

import netCDF4
import numpy as np

from rocketpy.tools import bilinear_interpolation

## These functions are meant to be used with netcdf4 datasets


def mask_and_clean_dataset(*args):
    data_array = np.ma.column_stack(list(args))

    # Remove lines with masked content
    if np.any(data_array.mask):
        data_array = np.ma.compress_rows(data_array)
        warnings.warn(
            "Some values were missing from this weather dataset, therefore, "
            "certain pressure levels were removed."
        )

    return data_array


def apply_bilinear_interpolation(x, y, x1, x2, y1, y2, data):
    return bilinear_interpolation(
        x,
        y,
        x1,
        x2,
        y1,
        y2,
        data[:, 0, 0],
        data[:, 0, 1],
        data[:, 1, 0],
        data[:, 1, 1],
    )


def apply_bilinear_interpolation_ensemble(x, y, x1, x2, y1, y2, data):
    return bilinear_interpolation(
        x,
        y,
        x1,
        x2,
        y1,
        y2,
        data[:, :, 0, 0],
        data[:, :, 0, 1],
        data[:, :, 1, 0],
        data[:, :, 1, 1],
    )


def find_longitude_index(longitude, lon_list):
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
            "Chosen launch time is not available in the provided file, "
            f"which starts at {file_time_date}."
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
    try:
        elevations = data.variables[dictionary["surface_geopotential_height"]][
            time_index, (lat_index - 1, lat_index), (lon_index - 1, lon_index)
        ]
    except:
        raise ValueError(
            "Unable to read surface elevation data. Check file and dictionary."
        )
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
