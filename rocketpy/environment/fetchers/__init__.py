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
from rocketpy.environment.fetchers.noaa_catalog import (
    FORECAST_MODELS_CATALOG_URL,
    NOAA_MODEL_COLLECTIONS,
    THREDDS_OPENDAP_ROOT,
    build_noaa_opendap_url,
    collection_catalog_url,
    fetch_latest_noaa_dataset,
    get_latest_noaa_dataset_identifier,
    get_latest_noaa_opendap_url,
    list_noaa_atmosphere_datasets,
    list_noaa_dataset_identifiers,
    resolve_noaa_collection_path,
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
    "FORECAST_MODELS_CATALOG_URL",
    "METEOMATICS_BASE_URL",
    "METEOMATICS_LOGIN_URL",
    "METEOMATICS_TIMEOUT_SECONDS",
    "NOAA_MODEL_COLLECTIONS",
    "OPEN_METEO_ENSEMBLE_MODELS",
    "OPEN_METEO_ENSEMBLE_URL",
    "OPEN_METEO_FORECAST_URL",
    "OPEN_METEO_HISTORICAL_START_DATE",
    "OPEN_METEO_HISTORICAL_URL",
    "OPEN_METEO_PRESSURE_LEVELS",
    "OPEN_METEO_TIMEOUT_SECONDS",
    "THREDDS_OPENDAP_ROOT",
    "MeteomaticsFetcher",
    "build_hourly_variables",
    "build_noaa_opendap_url",
    "collection_catalog_url",
    "fetch_aigfs_file_return_dataset",
    "fetch_atmospheric_data_from_meteomatics",
    "fetch_atmospheric_data_from_windy",
    "fetch_cmc_ensemble",
    "fetch_gefs_ensemble",
    "fetch_gfs_file_return_dataset",
    "fetch_hiresw_file_return_dataset",
    "fetch_hrrr_file_return_dataset",
    "fetch_latest_noaa_dataset",
    "fetch_meteomatics_token",
    "fetch_nam_file_return_dataset",
    "fetch_open_elevation",
    "fetch_open_meteo_ensemble",
    "fetch_open_meteo_forecast",
    "fetch_rap_file_return_dataset",
    "fetch_wyoming_sounding",
    "get_latest_noaa_dataset_identifier",
    "get_latest_noaa_opendap_url",
    "list_noaa_atmosphere_datasets",
    "list_noaa_dataset_identifiers",
    "logger",
    "netCDF4",
    "requests",
    "resolve_noaa_collection_path",
    "time",
]
