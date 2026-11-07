"""Fetch weather data from the Meteomatics API."""

import base64
from datetime import timezone

import numpy as np
import requests

from rocketpy.environment.fetchers.base import logger
from rocketpy.tools import exponential_backoff

METEOMATICS_BASE_URL = "https://api.meteomatics.com"
METEOMATICS_LOGIN_URL = "https://login.meteomatics.com/api/v1/token"
METEOMATICS_TIMEOUT_SECONDS = 30


class MeteomaticsFetcher:
    """Fetcher class to authenticate and query vertical atmospheric profiles
    from the Meteomatics API.
    """

    @staticmethod
    @exponential_backoff(max_attempts=3, base_delay=1, max_delay=60)
    def _get(url, headers=None, params=None):
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

    @classmethod
    def _request_json(cls, url, endpoint, headers=None, params=None):
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
            response = cls._get(url, headers=headers, params=params)
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

    @classmethod
    def fetch_token(cls, username, password):
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
        payload = cls._request_json(
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

    @staticmethod
    def _build_parameters(
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

    @staticmethod
    def _validate_sampling(
        min_altitude,
        max_altitude,
        wind_resolution,
        temperature_pressure_resolution,
        query_limit,
    ):
        """Validates the sampling arguments before any request is issued.

        Catching these here keeps a degenerate input from reaching ``linspace`` or
        ``range``, where it would surface as an opaque low-level error (or as an
        empty request) after the account has already been charged for the login.

        Raises
        ------
        ValueError
            If the altitude range, the resolutions or the query limit are invalid.
        """
        if min_altitude < 0:
            raise ValueError(
                "min_altitude must be non-negative (heights are above ground level)."
            )
        if max_altitude <= min_altitude:
            raise ValueError("max_altitude must be greater than min_altitude.")
        if wind_resolution < 2 or temperature_pressure_resolution < 2:
            raise ValueError(
                "wind_resolution and temperature_pressure_resolution must be at least "
                "2: a single altitude level is not enough to define a profile."
            )
        if query_limit < 1:
            raise ValueError("query_limit must be at least 1.")

    @staticmethod
    def _extract_json(data):
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

    @classmethod
    def fetch_atmospheric_data(
        cls,
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
            date-time specification (``%Y-%m-%dT%H:%M:%SZ``). Timezone-aware
            datetimes are converted to UTC; naive ones are assumed to be UTC
            already.
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
            If the altitude range, the resolutions or the query limit are invalid,
            or if the response contains an unrecognized parameter.
        """
        cls._validate_sampling(
            min_altitude,
            max_altitude,
            wind_resolution,
            temperature_pressure_resolution,
            query_limit,
        )

        token = cls.fetch_token(username, password)

        if date.tzinfo is not None:
            date = date.astimezone(timezone.utc)
        date_string = date.strftime("%Y-%m-%dT%H:%M:%SZ")
        parameter_map = cls._build_parameters(
            min_altitude,
            max_altitude,
            wind_resolution,
            temperature_pressure_resolution,
        )
        parameters = list(parameter_map)
        parameter_groups = [
            parameters[i : i + query_limit]
            for i in range(0, len(parameters), query_limit)
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
            data = cls._request_json(base_url, "data API", params=query_params)

            for parameter, value in cls._extract_json(data):
                try:
                    profile, height = parameter_map[parameter]
                except KeyError as e:
                    raise ValueError(
                        f"Unrecognized Meteomatics parameter '{parameter}'."
                    ) from e
                profiles[profile][height] = value

        return profiles


def fetch_meteomatics_token(username, password):
    """Requests a short-lived access token from the Meteomatics login service."""
    return MeteomaticsFetcher.fetch_token(username, password)


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
    """Fetches a vertical atmospheric profile from the Meteomatics API."""
    return MeteomaticsFetcher.fetch_atmospheric_data(
        username=username,
        password=password,
        latitude=latitude,
        longitude=longitude,
        date=date,
        model=model,
        min_altitude=min_altitude,
        max_altitude=max_altitude,
        wind_resolution=wind_resolution,
        temperature_pressure_resolution=temperature_pressure_resolution,
        query_limit=query_limit,
    )
