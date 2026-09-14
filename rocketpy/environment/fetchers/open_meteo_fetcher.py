"""Fetch weather data from the Open-Meteo API.

Open-Meteo (https://open-meteo.com/) serves pressure-level forecasts, past
forecasts and ensemble forecasts as plain JSON over HTTPS, with no API key and
no heavy NetCDF/OPeNDAP dependency. This module wraps the three endpoints
RocketPy needs and returns their raw JSON bodies.
"""

import warnings
from datetime import datetime, timedelta, timezone

import requests

from rocketpy.environment.fetchers.base import logger
from rocketpy.tools import exponential_backoff

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPEN_METEO_TIMEOUT_SECONDS = 60

# Pressure levels (hPa) that Open-Meteo publishes for its pressure-level
# variables. Not every model resolves every level; levels that come back empty
# are dropped while parsing rather than requested conditionally, because the
# per-model coverage is not advertised by the API.
OPEN_METEO_PRESSURE_LEVELS = (
    1000,
    975,
    950,
    925,
    900,
    850,
    800,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    70,
    50,
    30,
)

# Per-level variables requested for every query. Open-Meteo reports wind as
# speed/direction rather than the u/v components RocketPy uses internally, so
# the conversion happens in the Environment parsing step.
OPEN_METEO_LEVEL_VARIABLES = (
    "temperature",
    "geopotential_height",
    "wind_speed",
    "wind_direction",
)

# Ensemble models known to publish the full set of pressure-level variables
# RocketPy needs (temperature, geopotential height and both wind fields). The
# other ensemble models answer with HTTP 200 but are unusable: gfs025,
# icon_global and bom_access_global_ensemble return nulls at every level, while
# gem_global serves temperature and geopotential height but no winds at all.
# Rejecting them up front avoids an opaque "no data" failure later on.
OPEN_METEO_ENSEMBLE_MODELS = ("gfs05", "ecmwf_ifs025")

# Earliest date the historical-forecast archive covers at pressure levels.
# Earlier dates still answer with HTTP 200, but every value is null, so warning
# is the only way for the user to tell an unsupported date from bad weather
# data. Probed against the live API: 2021-03-15 comes back empty while
# 2021-03-23 is complete, so the cutoff sits between them.
#
# Note that Open-Meteo's ERA5 archive endpoint (archive-api.open-meteo.com) is
# *not* used here: it serves surface variables only, with no pressure-level
# data at all, so it cannot produce a vertical profile.
OPEN_METEO_HISTORICAL_START_DATE = datetime(2021, 4, 1, tzinfo=timezone.utc)


def build_hourly_variables(levels=OPEN_METEO_PRESSURE_LEVELS):
    """Builds the comma-separated ``hourly`` query parameter for Open-Meteo.

    Parameters
    ----------
    levels : sequence of int, optional
        Pressure levels, in hPa, to request. Defaults to
        :data:`OPEN_METEO_PRESSURE_LEVELS`.

    Returns
    -------
    str
        The value to pass as the ``hourly`` query parameter, e.g.
        ``"temperature_1000hPa,geopotential_height_1000hPa,..."``.

    Examples
    --------
    >>> from rocketpy.environment.fetchers.open_meteo_fetcher import (
    ...     build_hourly_variables,
    ... )
    >>> build_hourly_variables(levels=[500])
    'temperature_500hPa,geopotential_height_500hPa,wind_speed_500hPa,wind_direction_500hPa'
    """
    return ",".join(
        f"{variable}_{level}hPa"
        for level in levels
        for variable in OPEN_METEO_LEVEL_VARIABLES
    )


@exponential_backoff(max_attempts=3, base_delay=1, max_delay=60)
def _get_json(url, params):
    """Performs a single Open-Meteo GET request and returns its parsed body.

    Connection errors and server-side 5xx responses raise so the decorator
    retries them. Client-side 4xx responses are definitive (a malformed query
    or an out-of-range date) and are turned into an actionable error by the
    caller instead of being retried.
    """
    response = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT_SECONDS)
    if response.status_code >= 500:  # pragma: no cover
        response.raise_for_status()
    return response


def _request(url, params, endpoint):
    """Queries an Open-Meteo endpoint and returns its parsed JSON body.

    Parameters
    ----------
    url : str
        The endpoint address to query.
    params : dict
        Query parameters to send with the request.
    endpoint : str
        Human-readable endpoint name (e.g. ``"forecast"``), used in error
        messages.

    Returns
    -------
    dict
        The parsed JSON body of the response.

    Raises
    ------
    RuntimeError
        If the endpoint cannot be reached, rejects the query, or returns a
        malformed (non-JSON) body.
    """
    try:
        response = _get_json(url, params)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Unable to reach the Open-Meteo {endpoint} API. Please try again later."
        ) from e

    try:
        payload = response.json()
    except ValueError as e:
        raise RuntimeError(
            f"The Open-Meteo {endpoint} API returned a malformed (non-JSON) "
            "response. Please try again later."
        ) from e

    # Open-Meteo reports query errors as {"error": true, "reason": "..."},
    # which is far more specific than the bare status code.
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(
            f"The Open-Meteo {endpoint} API rejected the request: "
            f"{payload.get('reason', 'no reason given')}"
        )
    if not response.ok:  # pragma: no cover
        raise RuntimeError(
            f"The Open-Meteo {endpoint} API request failed with status "
            f"{response.status_code}."
        )
    if "hourly" not in payload:
        raise RuntimeError(
            f"The Open-Meteo {endpoint} API response did not contain any hourly "
            "data. Please try again later."
        )

    return payload


def fetch_open_meteo_forecast(latitude, longitude, model="best_match", date=None):
    """Fetches a pressure-level forecast from the Open-Meteo API.

    Requests are routed to the historical-forecast endpoint when ``date`` lies
    in the past, and to the regular forecast endpoint otherwise.

    Parameters
    ----------
    latitude : float
        The latitude of the location, in degrees.
    longitude : float
        The longitude of the location, in degrees.
    model : str, optional
        The Open-Meteo weather model to query, such as ``"best_match"`` (the
        default), ``"gfs_seamless"``, ``"ecmwf_ifs025"`` or ``"icon_seamless"``.
        See https://open-meteo.com/en/docs for the full list.
    date : datetime.datetime, optional
        The launch date and time. Used to pick the endpoint and, for past
        dates, to bound the queried period. When None, the regular forecast
        endpoint is queried.

    Returns
    -------
    dict
        The parsed JSON body returned by the API.

    Raises
    ------
    RuntimeError
        If the API cannot be reached or returns no usable data.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": build_hourly_variables(),
        "models": model,
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }

    if _is_past_date(date):
        _warn_if_before_archive_start(date)
        # The archive is indexed by calendar day, so a one-day pad on each side
        # guarantees the launch hour is inside the returned range regardless of
        # the local-time offset.
        params["start_date"] = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        params["end_date"] = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(
            "Launch date %s is in the past; querying the Open-Meteo "
            "historical-forecast API.",
            date,
        )
        return _request(OPEN_METEO_HISTORICAL_URL, params, "historical forecast")

    return _request(OPEN_METEO_FORECAST_URL, params, "forecast")


def fetch_open_meteo_ensemble(latitude, longitude, model="gfs05", date=None):
    """Fetches a pressure-level ensemble forecast from the Open-Meteo API.

    Parameters
    ----------
    latitude : float
        The latitude of the location, in degrees.
    longitude : float
        The longitude of the location, in degrees.
    model : str, optional
        The Open-Meteo ensemble model to query. Default is ``"gfs05"``. Only
        the models in :data:`OPEN_METEO_ENSEMBLE_MODELS` publish complete
        pressure-level data.
    date : datetime.datetime, optional
        The launch date and time. Past dates are queried against the
        historical-forecast window of the ensemble endpoint.

    Returns
    -------
    dict
        The parsed JSON body returned by the API.

    Raises
    ------
    ValueError
        If ``model`` is not known to publish complete pressure-level data.
    RuntimeError
        If the API cannot be reached or returns no usable data.
    """
    if model not in OPEN_METEO_ENSEMBLE_MODELS:
        raise ValueError(
            f"Invalid Open-Meteo ensemble model '{model}'. Only "
            f"{' and '.join(OPEN_METEO_ENSEMBLE_MODELS)} publish the complete "
            "set of pressure-level variables (temperature, geopotential height "
            "and winds) that RocketPy requires to build an atmospheric profile."
        )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": build_hourly_variables(),
        "models": model,
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }

    if _is_past_date(date):
        params["start_date"] = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        params["end_date"] = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    return _request(OPEN_METEO_ENSEMBLE_URL, params, "ensemble")


def _is_past_date(date):
    """Returns True when ``date`` is far enough in the past that the regular
    forecast endpoint would no longer cover it.

    Open-Meteo's forecast endpoint keeps a couple of past days available, so
    only dates before that window need the historical-forecast archive.
    """
    if date is None:
        return False
    return _as_utc(date) < _utc_now() - timedelta(days=1)


def _as_utc(date):
    """Returns ``date`` as an aware UTC datetime, assuming UTC when naive."""
    return date if date.tzinfo is not None else date.replace(tzinfo=timezone.utc)


def _warn_if_before_archive_start(date):
    """Warns when ``date`` precedes the historical archive coverage.

    Open-Meteo answers such requests with HTTP 200 and null values at every
    pressure level, so without this warning the user would only see a generic
    "not enough usable pressure levels" error and no hint that the date itself
    is the problem.
    """
    if _as_utc(date) >= OPEN_METEO_HISTORICAL_START_DATE:
        return

    warnings.warn(
        f"The requested launch date ({date:%Y-%m-%d}) precedes Open-Meteo's "
        "historical-forecast archive, which starts around "
        f"{OPEN_METEO_HISTORICAL_START_DATE:%B %Y}. The API will most likely "
        "return no pressure-level data for it. Consider using the 'reanalysis' "
        "or 'wyoming_sounding' atmospheric models for earlier dates.",
        UserWarning,
        stacklevel=3,
    )


def _utc_now():
    """Returns the current UTC time. Wrapped in a helper so tests can patch it."""

    return datetime.now(timezone.utc)
