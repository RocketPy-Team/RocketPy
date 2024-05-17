# NOTE: any function in this file may be changed without notice in future versions
# Auxiliary functions - Fetching Data from 3rd party APIs

import re
import time
from datetime import datetime, timedelta, timezone

import netCDF4
import requests

from rocketpy.tools import exponential_backoff


@exponential_backoff(max_attempts=3, base_delay=1, max_delay=60)
def fetch_open_elevation(lat, lon):
    print("Fetching elevation from open-elevation.com...")
    request_url = (
        "https://api.open-elevation.com/api/v1/lookup?locations" f"={lat},{lon}"
    )
    try:
        response = requests.get(request_url)
    except requests.exceptions.RequestException as e:
        raise RuntimeError("Unable to reach Open-Elevation API servers.") from e
    results = response.json()["results"]
    return results[0]["elevation"]


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_atmospheric_data_from_windy(lat, lon, model):
    model = model.lower()
    if model[-1] == "u":  # case iconEu
        model = "".join([model[:4], model[4].upper(), model[4 + 1 :]])
    url = (
        f"https://node.windy.com/forecast/meteogram/{model}/"
        f"{lat}/{lon}/?step=undefined"
    )
    try:
        response = requests.get(url).json()
    except Exception as e:
        if model == "iconEu":
            raise ValueError(
                "Could not get a valid response for Icon-EU from Windy. "
                "Check if the coordinates are set inside Europe."
            ) from e
    return response


def fetch_gfs_file_return_dataset(max_attempts=10, base_delay=2):
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=6)
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
            time.sleep(base_delay * attempt_count)

    if dataset is None:
        raise RuntimeError(
            "Unable to load latest weather data for GFS through " + file_url
        )


def fetch_nam_file_return_dataset(max_attempts=10, base_delay=2):
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=6)
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
            time.sleep(base_delay * attempt_count)

    if dataset is None:
        raise RuntimeError("Unable to load latest weather data for NAM through " + file)


def fetch_rap_file_return_dataset(max_attempts=10, base_delay=2):
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=6)
        file = "https://nomads.ncep.noaa.gov/dods/rap/rap{:04d}{:02d}{:02d}/rap_{:02d}z".format(
            time_attempt.year,
            time_attempt.month,
            time_attempt.day,
            time_attempt.hour,
        )
        try:
            # Attempts to create a dataset from the file using OpenDAP protocol.
            dataset = netCDF4.Dataset(file)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(base_delay * attempt_count)


def fetch_hiresw_file_return_dataset(max_attempts=10, base_delay=2):
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
        file = f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/hiresw_conusarw_12z"
        try:
            # Attempts to create a dataset from the file using OpenDAP protocol.
            dataset = netCDF4.Dataset(file)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(base_delay * attempt_count)

    if dataset is None:
        raise RuntimeError(
            "Unable to load latest weather data for HiResW through " + file
        )


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_wyoming_sounding(file):
    response = requests.get(file)
    if response.status_code != 200:
        raise ImportError(f"Unable to load {file}.")
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
    response = requests.get(file)
    if response.status_code != 200 or len(response.text) < 10:
        raise ImportError("Unable to load " + file + ".")
    return response


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_gefs_ensemble():
    time_attempt = datetime.now(tz=timezone.utc)
    success = False
    attempt_count = 0
    while not success and attempt_count < 10:
        time_attempt -= timedelta(hours=6 * attempt_count)
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
    if not success:
        raise RuntimeError(
            "Unable to load latest weather data for GEFS through " + file
        )


@exponential_backoff(max_attempts=5, base_delay=2, max_delay=60)
def fetch_cmc_ensemble():
    # Attempt to get latest forecast
    time_attempt = datetime.now(tz=timezone.utc)
    success = False
    attempt_count = 0
    while not success and attempt_count < 10:
        time_attempt -= timedelta(hours=12 * attempt_count)
        file = (
            f"https://nomads.ncep.noaa.gov/dods/cmcens/"
            f"cmcens{time_attempt.year:04d}{time_attempt.month:02d}"
            f"{time_attempt.day:02d}/"
            f"cmcens_all_{12 * (time_attempt.hour // 12):02d}z"
        )
        try:
            dataset = netCDF4.Dataset(file)
            success = True
            return dataset
        except OSError:
            attempt_count += 1
    if not success:
        raise RuntimeError("Unable to load latest weather data for CMC through " + file)
