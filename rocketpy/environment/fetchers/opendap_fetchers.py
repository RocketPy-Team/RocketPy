"""Fetch weather datasets using OPeNDAP protocol (NOAA, UCAR, CMC, GEFS)."""

import time
from datetime import datetime, timedelta, timezone

import netCDF4

from rocketpy.environment.fetchers.base import MAX_RETRY_DELAY_SECONDS
from rocketpy.tools import exponential_backoff


def fetch_gfs_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest GFS (Global Forecast System) dataset from the UCAR
    THREDDS data server using the OPeNDAP protocol.

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
    file_url = (
        "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"
    )
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            return netCDF4.Dataset(file_url)
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    raise RuntimeError("Unable to load latest weather data for GFS through " + file_url)


def fetch_nam_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest NAM (North American Mesoscale) dataset from the UCAR
    THREDDS data server using the OPeNDAP protocol.

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
    file_url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NAM/CONUS_12km/Best"
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            return netCDF4.Dataset(file_url)
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    raise RuntimeError("Unable to load latest weather data for NAM through " + file_url)


def fetch_rap_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest RAP (Rapid Refresh) dataset from the UCAR THREDDS
    data server using the OPeNDAP protocol.

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
    file_url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/RAP/CONUS_13km/Best"
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            return netCDF4.Dataset(file_url)
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    raise RuntimeError("Unable to load latest weather data for RAP through " + file_url)


def fetch_hrrr_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest HRRR (High-Resolution Rapid Refresh) dataset from
    the NOAA's GrADS data server using the OpenDAP protocol.

    Parameters
    ----------
    max_attempts : int, optional
        The maximum number of attempts to fetch the dataset. Default is 10.
    base_delay : int, optional
        The base delay in seconds between attempts. Default is 2.

    Returns
    -------
    netCDF4.Dataset
        The HRRR dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for HRRR.
    """
    file_url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/HRRR/CONUS_2p5km/Best"
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            return netCDF4.Dataset(file_url)
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    raise RuntimeError(
        "Unable to load latest weather data for HRRR through " + file_url
    )


def fetch_aigfs_file_return_dataset(max_attempts=10, base_delay=2):
    """Fetches the latest AIGFS (Artificial Intelligence GFS) dataset from
    the NOAA's GrADS data server using the OpenDAP protocol.

    Parameters
    ----------
    max_attempts : int, optional
        The maximum number of attempts to fetch the dataset. Default is 10.
    base_delay : int, optional
        The base delay in seconds between attempts. Default is 2.

    Returns
    -------
    netCDF4.Dataset
        The AIGFS dataset.

    Raises
    ------
    RuntimeError
        If unable to load the latest weather data for AIGFS.
    """
    file_url = (
        "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/AIGFS/Global_0p25deg/Best"
    )
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            return netCDF4.Dataset(file_url)
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    raise RuntimeError(
        "Unable to load latest weather data for AIGFS through " + file_url
    )


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
    time_attempt = datetime.now(tz=timezone.utc)
    attempt_count = 0
    dataset = None

    today = datetime.now(tz=timezone.utc)
    date_info = (today.year, today.month, today.day, 12)

    while attempt_count < max_attempts:
        time_attempt -= timedelta(hours=12)
        date_info = (
            time_attempt.year,
            time_attempt.month,
            time_attempt.day,
            12,
        )
        date_string = f"{date_info[0]:04d}{date_info[1]:02d}{date_info[2]:02d}"
        file = (
            f"https://nomads.ncep.noaa.gov/dods/hiresw/hiresw{date_string}/"
            "hiresw_conusarw_12z"
        )
        try:
            dataset = netCDF4.Dataset(file)
            return dataset
        except OSError:
            attempt_count += 1
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

    if dataset is None:
        raise RuntimeError(
            "Unable to load latest weather data for HiResW through " + file
        )


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
            time.sleep(min(2**attempt_count, MAX_RETRY_DELAY_SECONDS))
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
    time_attempt = datetime.now(tz=timezone.utc)
    success = False
    attempt_count = 0
    while not success and attempt_count < 10:
        time_attempt -= timedelta(hours=12 * attempt_count)
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
            time.sleep(min(2**attempt_count, MAX_RETRY_DELAY_SECONDS))
    if not success:
        raise RuntimeError("Unable to load latest weather data for CMC through " + file)
