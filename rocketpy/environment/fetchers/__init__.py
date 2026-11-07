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
from rocketpy.environment.fetchers.open_meteo_fetcher import (
    OPEN_METEO_ENSEMBLE_MODELS,
    OPEN_METEO_ENSEMBLE_URL,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_START_DATE,
    OPEN_METEO_HISTORICAL_URL,
    OPEN_METEO_PRESSURE_LEVELS,
    OPEN_METEO_TIMEOUT_SECONDS,
    build_hourly_variables,
    fetch_open_meteo_ensemble,
    fetch_open_meteo_forecast,
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
    "OPEN_METEO_ENSEMBLE_MODELS",
    "OPEN_METEO_ENSEMBLE_URL",
    "OPEN_METEO_FORECAST_URL",
    "OPEN_METEO_HISTORICAL_START_DATE",
    "OPEN_METEO_HISTORICAL_URL",
    "OPEN_METEO_PRESSURE_LEVELS",
    "OPEN_METEO_TIMEOUT_SECONDS",
    "MeteomaticsFetcher",
    "build_hourly_variables",
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
    "fetch_open_meteo_ensemble",
    "fetch_open_meteo_forecast",
    "fetch_rap_file_return_dataset",
    "fetch_wyoming_sounding",
    "logger",
    "netCDF4",
    "requests",
    "time",
]
