"""Fetch upper air sounding data from Wyoming Weather Web."""

import re

import requests

from rocketpy.tools import exponential_backoff


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
