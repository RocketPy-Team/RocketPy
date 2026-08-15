"""Fetch elevation data from third-party APIs."""

import requests

from rocketpy.environment.fetchers.base import logger
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
