Connecting to other APIs
========================

Although we have covered an extensive list of supported atmospheric models,
it is always possible to load data from other APIs.
Indeed, it constitutes a significant part of the flexibility of RocketPy.


With OPeNDAP protocol
----------------------

The OPeNDAP protocol allows you to access remote data in a simple and efficient way.
We have used it to load data from the NOAA's data server in the :ref:`forecast`
section.

In case you want to use a different atmospheric model from NOAA, for example,
you are going to need to provide the URL of corresponding API endpoint and a dictionary
mapping the names of the variables to the names used in the dataset to the 
RocketPy's Environment class.

We use a dictionary because we need to determine the name used by the model of
the following dimensions and variables:

- Time
- Latitude
- Longitude
- Pressure Levels
- Geopotential Height (as a function of Time, Pressure Levels, Latitude and Longitude)
- Surface Geopotential Height (as a function of Time, Latitude and Longitude)
- Wind - U Component (as a function of Time, Pressure Levels, Latitude and Longitude)
- Wind - V Component (as a function of Time, Pressure Levels, Latitude and Longitude)


For example, let's imagine we want to use the HIRESW model from this endpoint: 
`https://nomads.ncep.noaa.gov/dods/hiresw/ <https://nomads.ncep.noaa.gov/dods/hiresw/>`_


Looking through the variable list in the link above, we find the following correspondence:

- Time = "time"
- Latitude = "lat"
- Longitude = "lon"
- Pressure Levels = "lev"
- Geopotential Height = "hgtprs"
- Surface Geopotential Height = "hgtsfc"
- Wind - U Component = "ugrdprs"
- Wind - V Component = "vgrdprs"

Therefore, we can create an environment like this:

.. code-block:: python

    from rocketpy import Environment

    env = Environment()

    name_mapping = {
       "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }

    env.set_atmospheric_model(
        type="forecast",
        file="<your_url_here>",
        dictionary=name_mapping,
    )

.. caution::

    Notice the ``file`` argument were suppressed in the code above. This is because \
    the URL depends on the date you are running the simulation. For example, as \
    it for now, a possible link could be: https://nomads.ncep.noaa.gov/dods/hiresw/hiresw20240803/hiresw_conusfv3_12z \
    (for the 3rd of August, 2024, at 12:00 UTC). \
    You should replace the date in the URL with the date you are running the simulation. \
    Different models may have different URL structures, so be sure to check the \
    documentation of the model you are using.


Without OPeNDAP protocol
-------------------------

On the other hand, one can also load data from APIs that do not support the OPeNDAP protocol.
In these cases, what we recommend is to download the data and then load it as a custom atmosphere.

There are some efforts to natively support other APIs in RocketPy's
Environment class, for example: 

- `Meteomatics <https://www.meteomatics.com/en/weather-api/>`_: `#545 <https://github.com/RocketPy-Team/RocketPy/issues/545>`_
- `Open-Meteo <https://open-meteo.com/>`_: `#520 <https://github.com/RocketPy-Team/RocketPy/issues/520>`_

