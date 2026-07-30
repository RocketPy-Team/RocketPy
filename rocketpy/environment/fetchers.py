"""This module contains auxiliary functions for fetching data from various
third-party APIs. As this is a recent module (introduced in v1.2.0), some
functions may be changed without notice in future feature releases.
"""

import base64
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import netCDF4
import numpy as np
import requests

from rocketpy.tools import exponential_backoff

logger = logging.getLogger(__name__)

MAX_RETRY_DELAY_SECONDS = 600

METEOMATICS_BASE_URL = "https://api.meteomatics.com"
METEOMATICS_LOGIN_URL = "https://login.meteomatics.com/api/v1/token"
METEOMATICS_TIMEOUT_SECONDS = 30


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
    logger.debug(
        "Fetching elevation from open-elevation.com for lat=%s, lon=%s", lat, lon
    )
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
        f"https://node.windy.com/forecast/meteogram/{model}/{lat}/{lon}/?step=undefined"
    )

    try:
        response = requests.get(url).json()
        if "data" not in response.keys():  # pragma: no cover
            raise ValueError(
                f"Could not get a valid response for '{model}' from Windy. "
                "Check if the coordinates are set inside the model's domain."
            )
    except requests.exceptions.RequestException as e:  # pragma: no cover
        if model == "iconEu":
            raise ValueError(
                "Could not get a valid response for Icon-EU from Windy. "
                "Check if the coordinates are set inside Europe."
            ) from e

    return response


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
            time.sleep(min(base_delay**attempt_count, MAX_RETRY_DELAY_SECONDS))

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
    if response.status_code != 200:  # pragma: no cover
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
            time.sleep(min(2**attempt_count, MAX_RETRY_DELAY_SECONDS))
    if not success:
        raise RuntimeError("Unable to load latest weather data for CMC through " + file)


@exponential_backoff(max_attempts=3, base_delay=1, max_delay=60)
def _meteomatics_get(url, headers=None, params=None):
    """Performs a single Meteomatics GET request, retrying transient failures.

    Connection-level errors (and server-side 5xx responses) raise and are
    retried by the decorator. Client-side 4xx responses are returned as-is so
    the caller can turn them into an actionable, non-retried error, since
    retrying a deterministic 4xx only wastes time and API quota.
    """
    response = requests.get(
        url, headers=headers, params=params, timeout=METEOMATICS_TIMEOUT_SECONDS
    )
    if response.status_code >= 500:
        response.raise_for_status()
    return response


def _meteomatics_request_json(url, endpoint, headers=None, params=None):
    """Queries a Meteomatics endpoint and returns its parsed JSON body.

    Parameters
    ----------
    url : str
        The endpoint address to query.
    endpoint : str
        Human-readable name of the endpoint (e.g. ``"login service"``), used to
        build the error messages.
    headers : dict, optional
        Headers to send with the request.
    params : dict, optional
        Query parameters to send with the request.

    Returns
    -------
    dict
        The parsed JSON body of the response.

    Raises
    ------
    RuntimeError
        If the endpoint cannot be reached, rejects the credentials, returns an
        error status, or returns a malformed (non-JSON) body. Client-side (4xx)
        errors are definitive and are not retried.
    """
    try:
        response = _meteomatics_get(url, headers=headers, params=params)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Unable to reach the Meteomatics {endpoint}. Please try again later."
        ) from e
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Meteomatics rejected the credentials (HTTP {response.status_code}). "
            "Check your username and password."
        )
    if not response.ok:
        raise RuntimeError(
            f"Meteomatics {endpoint} request failed "
            f"(HTTP {response.status_code}). {response.text[:300]}".strip()
        )
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise RuntimeError(
            f"Meteomatics {endpoint} returned a malformed (non-JSON) response."
        ) from e


def fetch_meteomatics_token(username, password):
    """Requests a short-lived access token from the Meteomatics login service.

    The Meteomatics API authenticates with a personal ``username`` and
    ``password``. Instead of sending the credentials on every request, a token
    is generated once and reused for the handful of requests needed to build a
    single ``Environment``.

    Parameters
    ----------
    username : str
        The Meteomatics account username.
    password : str
        The Meteomatics account password.

    Returns
    -------
    str
        The access token to be used as the ``access_token`` query parameter in
        subsequent data requests.

    Raises
    ------
    RuntimeError
        If the login service cannot be reached, rejects the credentials, or
        does not return a token.
    """
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    payload = _meteomatics_request_json(
        METEOMATICS_LOGIN_URL,
        "login service",
        headers={"Authorization": f"Basic {encoded_credentials}"},
    )
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(
            "Meteomatics login service did not return an access token. "
            "Check your username and password."
        )
    logger.info("Meteomatics access token generated successfully.")
    return token


def _build_meteomatics_parameters(
    min_altitude, max_altitude, wind_resolution, temperature_pressure_resolution
):
    """Builds the Meteomatics height-level parameters to query.

    Wind components are sampled on a finer altitude grid than temperature and
    pressure, since the wind profile is usually the most variable one.

    Parameters
    ----------
    min_altitude : float
        Lowest altitude above ground level (in meters) to query.
    max_altitude : float
        Highest altitude above ground level (in meters) to query.
    wind_resolution : int
        Number of altitude levels used for the wind components.
    temperature_pressure_resolution : int
        Number of altitude levels used for temperature and pressure.

    Returns
    -------
    dict
        Maps each parameter string, in the ``"<variable>_<height>m:<unit>"``
        format, to the ``(profile name, height)`` pair it carries. Keeping this
        mapping spares the caller from parsing the parameter strings back.
    """

    def levels(resolution):
        # Round to integer meters and drop duplicates that rounding may
        # introduce for narrow bands, so we never request (and pay for) the
        # same height twice.
        return np.unique(
            np.linspace(min_altitude, max_altitude, resolution).round().astype(int)
        )

    grids = [
        (
            levels(wind_resolution),
            [("wind_speed_u", "ms", "wind_u"), ("wind_speed_v", "ms", "wind_v")],
        ),
        (
            levels(temperature_pressure_resolution),
            [("t", "K", "temperature"), ("pressure", "Pa", "pressure")],
        ),
    ]
    return {
        f"{var}_{height}m:{unit}": (profile, int(height))
        for heights, variables in grids
        for height in heights
        for var, unit, profile in variables
    }


def _extract_meteomatics_json(data):
    """Extracts (parameter, value) pairs from a Meteomatics JSON response.

    Only the first coordinate and first date of each parameter are used, since
    the query is always issued for a single location and a single instant.

    Parameters
    ----------
    data : dict
        The JSON payload returned by the Meteomatics data endpoint.

    Returns
    -------
    list of tuple
        A list of ``(parameter, value)`` tuples.

    Raises
    ------
    RuntimeError
        If the payload does not have the expected Meteomatics structure.
    """
    try:
        return [
            (entry["parameter"], entry["coordinates"][0]["dates"][0]["value"])
            for entry in data["data"]
        ]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            "Unexpected Meteomatics response structure; could not extract the "
            "requested data."
        ) from e


def fetch_atmospheric_data_from_meteomatics(
    username,
    password,
    latitude,
    longitude,
    date,
    model="mix",
    min_altitude=10,
    max_altitude=12000,
    wind_resolution=20,
    temperature_pressure_resolution=10,
    query_limit=10,
):
    """Fetches a vertical atmospheric profile from the Meteomatics API.

    The data is retrieved for a single location and instant, sampling
    temperature, pressure and both wind components at several altitudes above
    ground level. To respect the account's per-request parameter limit, the
    parameters are split into groups that are queried separately.

    Parameters
    ----------
    username : str
        The Meteomatics account username.
    password : str
        The Meteomatics account password.
    latitude : float
        Latitude of the launch site, in degrees.
    longitude : float
        Longitude of the launch site, in degrees.
    date : datetime.datetime
        The instant to query. It is formatted according to the Meteomatics
        date-time specification (``%Y-%m-%dT%H:%M:%SZ``).
    model : str, optional
        The Meteomatics weather model to use. Default is ``"mix"``. Your
        account may not have access to every model. See
        https://www.meteomatics.com/en/api/request/optional-parameters/data-source/
    min_altitude : float, optional
        Lowest altitude above ground level (in meters) to query. Default is 10.
    max_altitude : float, optional
        Highest altitude above ground level (in meters) to query. Default is
        12000. The API returns an error if the requested altitude is outside
        the range supported by the chosen model.
    wind_resolution : int, optional
        Number of altitude levels used for the wind components. Default is 20.
    temperature_pressure_resolution : int, optional
        Number of altitude levels used for temperature and pressure. Default is
        10.
    query_limit : int, optional
        Maximum number of parameters requested at once. Parameters are grouped
        accordingly to work around the account's per-request limit. Default is
        10. See https://api.meteomatics.com/user_stats for your own limits.

    Returns
    -------
    dict
        A dictionary with the keys ``"temperature"``, ``"pressure"``,
        ``"wind_u"`` and ``"wind_v"``. Each value is a dictionary mapping the
        altitude above ground level (in meters) to the corresponding value, in
        SI units (K, Pa, m/s and m/s respectively).

    Raises
    ------
    RuntimeError
        If authentication fails, the API cannot be reached, returns an error
        status, or returns a malformed response.
    ValueError
        If the altitude range is invalid or the response contains an
        unrecognized parameter.
    """
    if min_altitude < 0:
        raise ValueError(
            "min_altitude must be non-negative (heights are above ground level)."
        )
    if max_altitude <= min_altitude:
        raise ValueError("max_altitude must be greater than min_altitude.")

    token = fetch_meteomatics_token(username, password)

    date_string = date.strftime("%Y-%m-%dT%H:%M:%SZ")
    parameter_map = _build_meteomatics_parameters(
        min_altitude, max_altitude, wind_resolution, temperature_pressure_resolution
    )
    parameters = list(parameter_map)
    parameter_groups = [
        parameters[i : i + query_limit] for i in range(0, len(parameters), query_limit)
    ]

    profiles = {
        "temperature": {},
        "pressure": {},
        "wind_u": {},
        "wind_v": {},
    }

    for index, parameter_group in enumerate(parameter_groups):
        logger.info(
            "Fetching Meteomatics data for group %d/%d.",
            index + 1,
            len(parameter_groups),
        )
        parameters_str = ",".join(parameter_group)
        base_url = (
            f"{METEOMATICS_BASE_URL}/{date_string}/{parameters_str}/"
            f"{latitude},{longitude}/json"
        )
        query_params = {"model": model, "access_token": token}
        data = _meteomatics_request_json(base_url, "data API", params=query_params)

        for parameter, value in _extract_meteomatics_json(data):
            try:
                profile, height = parameter_map[parameter]
            except KeyError as e:
                raise ValueError(
                    f"Unrecognized Meteomatics parameter '{parameter}'."
                ) from e
            profiles[profile][height] = value

    return profiles
