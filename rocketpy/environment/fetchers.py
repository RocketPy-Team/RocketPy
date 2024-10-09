"""This module contains auxiliary functions for fetching data from various
third-party APIs. As this is a recent module (introduced in v1.2.0), some
functions may be changed without notice in future feature releases.
"""

import re
import time
import warnings
from datetime import datetime, timedelta, timezone

import netCDF4
import requests

from rocketpy.tools import exponential_backoff


@exponential_backoff(max_attempts=3, base_delay=1, max_delay=60)
def fetch_open_elevation(lat, lon):
    """Fetches elevation data from the Open-Elevation API at a given latitude
    and longitude.

    Parameters
    ----------
    lat : float
        The latitude of the location.
    lon : float
        The longitude of the location.

    Returns
    -------
    float
        The elevation at the given latitude and longitude in meters.

    Raises
    ------
    RuntimeError
        If there is a problem reaching the Open-Elevation API servers.
    """
    print(f"Fetching elevation from open-elevation.com for lat={lat}, lon={lon}...")
    request_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(request_url)
        results = response.json()["results"]
        return results[0]["elevation"]
    except (
        requests.exceptions.RequestException,
        requests.exceptions.JSONDecodeError,
    ) as e:
        raise RuntimeError("Unable to reach Open-Elevation API servers.") from e


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_atmospheric_data_from_windy(lat, lon, model):
    """Fetches atmospheric data from Windy.com API for a given latitude and
    longitude, using a specific model.

    Parameters
    ----------
    lat : float
        The latitude of the location.
    lon : float
        The longitude of the location.
    model : str
        The atmospheric model to use. Options are: ecmwf, GFS, ICON or ICONEU.

    Returns
    -------
    dict
        A dictionary containing the atmospheric data retrieved from the API.
    """
    model = model.lower()
    if model[-1] == "u":  # case iconEu
        model = "".join([model[:4], model[4].upper(), model[5:]])

    url = (
        f"https://node.windy.com/forecast/meteogram/{model}/{lat}/{lon}/"
        "?step=undefined"
    )

    try:
        response = requests.get(url).json()
        if "data" not in response.keys():
            raise ValueError(
                f"Could not get a valid response for '{model}' from Windy. "
                "Check if the coordinates are set inside the model's domain."
            )
    except requests.exceptions.RequestException as e:
        if model == "iconEu":
            raise ValueError(
                "Could not get a valid response for Icon-EU from Windy. "
                "Check if the coordinates are set inside Europe."
            ) from e

    return response


def fetch_gfs_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest GFS (Global Forecast System) dataset from the NOAA's
    GrADS data server using the OpenDAP protocol.

    Parameters
    ----------
    max_attempts : int, optional
        The maximum number of attempts to fetch the dataset. Default is 10.
    base_delay : int, optional
        The base delay in seconds between attempts. Default is 2.

    Returns
    -------
    netCDF4.Dataset
        The GFS dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for GFS.
    """
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    # TODO: the code below is trying to determine the hour of the latest available
    # forecast by trial and error. This is not the best way to do it. We should
    # actually check the NOAA website for the latest forecast time. Refactor needed.
    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=6)  # GFS updates every 6 hours
        file_url = (
            f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs"
            f"{time_attempt.year:04d}{time_attempt.month:02d}"
            f"{time_attempt.day:02d}/"
            f"gfs_0p25_{6 * (time_attempt.hour // 6):02d}z"
        )
        try:
            # Attempts to create a dataset from the file using OpenDAP protocol.
            dataset = netCDF4.Dataset(file_url)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(base_delay**attempt_count)

    if dataset is None:
        raise RuntimeError(
            "Unable to load latest weather data for GFS through " + file_url
        )


def fetch_nam_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest NAM (North American Mesoscale) dataset from the NOAA's
    GrADS data server using the OpenDAP protocol.

    Parameters
    ----------
    max_attempts : int, optional
        The maximum number of attempts to fetch the dataset. Default is 10.
    base_delay : int, optional
        The base delay in seconds between attempts. Default is 2.

    Returns
    -------
    netCDF4.Dataset
        The NAM dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for NAM.
    """
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=6)  # NAM updates every 6 hours
        file = (
            f"https://nomads.ncep.noaa.gov/dods/nam/nam{time_attempt.year:04d}"
            f"{time_attempt.month:02d}{time_attempt.day:02d}/"
            f"nam_conusnest_{6 * (time_attempt.hour // 6):02d}z"
        )
        try:
            # Attempts to create a dataset from the file using OpenDAP protocol.
            dataset = netCDF4.Dataset(file)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(base_delay**attempt_count)

    if dataset is None:
        raise RuntimeError("Unable to load latest weather data for NAM through " + file)


def fetch_rap_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest RAP (Rapid Refresh) dataset from the NOAA's GrADS data
    server using the OpenDAP protocol.

    Parameters
    ----------
    max_attempts : int, optional
        The maximum number of attempts to fetch the dataset. Default is 10.
    base_delay : int, optional
        The base delay in seconds between attempts. Default is 2.

    Returns
    -------
    netCDF4.Dataset
        The RAP dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for RAP.
    """
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=1)  # RAP updates every hour
        file = (
            f"https://nomads.ncep.noaa.gov/dods/rap/rap{time_attempt.year:04d}"
            f"{time_attempt.month:02d}{time_attempt.day:02d}/"
            f"rap_{time_attempt.hour:02d}z"
        )
        try:
            # Attempts to create a dataset from the file using OpenDAP protocol.
            dataset = netCDF4.Dataset(file)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(base_delay**attempt_count)

    if dataset is None:
        raise RuntimeError("Unable to load latest weather data for RAP through " + file)


def fetch_hiresw_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest HiResW (High-Resolution Window) dataset from the NOAA's
    GrADS data server using the OpenDAP protocol.

    Parameters
    ----------
    max_attempts : int, optional
        The maximum number of attempts to fetch the dataset. Default is 10.
    base_delay : int, optional
        The base delay in seconds between attempts. Default is 2.

    Returns
    -------
    netCDF4.Dataset
        The HiResW dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for HiResW.
    """
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    today = datetime.now(tz=timezone.utc)
    date_info = (today.year, today.month, today.day, 12)  # Hour given in UTC time

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=12)
        date_info = (
            time_attempt.year,
            time_attempt.month,
            time_attempt.day,
            12,
        )  # Hour given in UTC time
        date_string = f"{date_info[0]:04d}{date_info[1]:02d}{date_info[2]:02d}"
        file = (
            f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/"
            "hiresw_conusarw_12z"
        )
        try:
            # Attempts to create a dataset from the file using OpenDAP protocol.
            dataset = netCDF4.Dataset(file)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(base_delay**attempt_count)

    if dataset is None:
        raise RuntimeError(
            "Unable to load latest weather data for HiResW through " + file
        )


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_wyoming_sounding(file):
    """Fetches sounding data from a specified file using the Wyoming Weather
    Web.

    Parameters
    ----------
    file : str
        The URL of the file to fetch.

    Returns
    -------
    str
        The content of the fetched file.

    Raises
    ------
    ImportError
        If unable to load the specified file.
    ValueError
        If the response indicates the specified station or date is invalid.
    ValueError
        If the response indicates the output format is invalid.
    """
    response = requests.get(file)
    if response.status_code != 200:
        raise ImportError(f"Unable to load {file}.")  # pragma: no cover
    if len(re.findall("Can't get .+ Observations at", response.text)):
        raise ValueError(
            re.findall("Can't get .+ Observations at .+", response.text)[0]
            + " Check station number and date."
        )
    if response.text == "Invalid OUTPUT: specified\n":
        raise ValueError(
            "Invalid OUTPUT: specified. Make sure the output is Text: List."
        )
    return response


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_noaaruc_sounding(file):
    """Fetches sounding data from a specified file using the NOAA RUC soundings.

    Parameters
    ----------
    file : str
        The URL of the file to fetch.

    Returns
    -------
    str
        The content of the fetched file.

    Raises
    ------
    ImportError
        If unable to load the specified file or the file content is too short.
    """
    warnings.warn(
        "The NOAA RUC soundings are deprecated since September 30th, 2024. "
        "This method will be removed in version 1.8.0.",
        DeprecationWarning,
    )
    return file


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_gefs_ensemble():
    """Fetches the latest GEFS (Global Ensemble Forecast System) dataset from
    the NOAA's GrADS data server using the OpenDAP protocol.

    Returns
    -------
    netCDF4.Dataset
        The GEFS dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for GEFS.
    """
    time_attempt = datetime.now(tz=timezone.utc)
    success = False
    attempt_count = 0
    while not success and attempt_count < 10:
        time_attempt -= timedelta(hours=6 * attempt_count)  # GEFS updates every 6 hours
        file = (
            f"https://nomads.ncep.noaa.gov/dods/gens_bc/gens"
            f"{time_attempt.year:04d}{time_attempt.month:02d}"
            f"{time_attempt.day:02d}/"
            f"gep_all_{6 * (time_attempt.hour // 6):02d}z"
        )
        try:
            dataset = netCDF4.Dataset(file)
            success = True
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(2**attempt_count)
    if not success:
        raise RuntimeError(
            "Unable to load latest weather data for GEFS through " + file
        )


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_cmc_ensemble():
    """Fetches the latest CMC (Canadian Meteorological Centre) ensemble dataset
    from the NOAA's GrADS data server using the OpenDAP protocol.

    Returns
    -------
    netCDF4.Dataset
        The CMC ensemble dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for CMC.
    """
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    success = False
    attempt_count = 0
    while not success and attempt_count < 10:
        time_attempt -= timedelta(
            hours=12 * attempt_count
        )  # CMC updates every 12 hours
        file = (
            f"https://nomads.ncep.noaa.gov/dods/cmcens/"
            f"cmcens{time_attempt.year:04d}{time_attempt.month:02d}"
            f"{time_attempt.day:02d}/"
            f"cmcensspr_{12 * (time_attempt.hour // 12):02d}z"
        )
        try:
            dataset = netCDF4.Dataset(file)
            success = True
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(2**attempt_count)
    if not success:
        raise RuntimeError("Unable to load latest weather data for CMC through " + file)
