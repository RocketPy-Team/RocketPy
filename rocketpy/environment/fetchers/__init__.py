"""This module contains auxiliary functions and classes for fetching data from
various third-party APIs.
"""

import time

import netCDF4
import requests

from rocketpy.environment.fetchers.base import (
    MAX_RETRY_DELAY_SECONDS,
    logger,
)
from rocketpy.environment.fetchers.elevation_fetcher import fetch_open_elevation
from rocketpy.environment.fetchers.meteomatics_fetcher import (
    METEOMATICS_BASE_URL,
    METEOMATICS_LOGIN_URL,
    METEOMATICS_TIMEOUT_SECONDS,
    MeteomaticsFetcher,
    fetch_atmospheric_data_from_meteomatics,
    fetch_meteomatics_token,
)
from rocketpy.environment.fetchers.opendap_fetchers import (
    fetch_aigfs_file_return_dataset,
    fetch_cmc_ensemble,
    fetch_gefs_ensemble,
    fetch_gfs_file_return_dataset,
    fetch_hiresw_file_return_dataset,
    fetch_hrrr_file_return_dataset,
    fetch_nam_file_return_dataset,
    fetch_rap_file_return_dataset,
)
from rocketpy.environment.fetchers.windy_fetcher import (
    fetch_atmospheric_data_from_windy,
)
from rocketpy.environment.fetchers.wyoming_fetcher import (
    fetch_wyoming_sounding,
)

__all__ = [
    "MAX_RETRY_DELAY_SECONDS",
    "METEOMATICS_BASE_URL",
    "METEOMATICS_LOGIN_URL",
    "METEOMATICS_TIMEOUT_SECONDS",
    "MeteomaticsFetcher",
    "fetch_aigfs_file_return_dataset",
    "fetch_atmospheric_data_from_meteomatics",
    "fetch_atmospheric_data_from_windy",
    "fetch_cmc_ensemble",
    "fetch_gefs_ensemble",
    "fetch_gfs_file_return_dataset",
    "fetch_hiresw_file_return_dataset",
    "fetch_hrrr_file_return_dataset",
    "fetch_meteomatics_token",
    "fetch_nam_file_return_dataset",
    "fetch_open_elevation",
    "fetch_rap_file_return_dataset",
    "fetch_wyoming_sounding",
    "logger",
    "netCDF4",
    "requests",
    "time",
]
