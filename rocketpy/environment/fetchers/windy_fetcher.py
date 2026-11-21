"""Fetch weather data from Windy.com API."""

import requests

from rocketpy.tools import exponential_backoff


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
