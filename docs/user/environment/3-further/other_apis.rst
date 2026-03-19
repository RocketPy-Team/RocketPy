.. _environment_other_apis:

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
- Temperature (as a function of Time, Pressure Levels, Latitude and Longitude)
- Geopotential Height (as a function of Time, Pressure Levels, Latitude and Longitude)
- or Geopotential (as a function of Time, Pressure Levels, Latitude and Longitude)
- Surface Geopotential Height (as a function of Time, Latitude and Longitude)
    (optional)
- Wind - U Component (as a function of Time, Pressure Levels, Latitude and Longitude)
- Wind - V Component (as a function of Time, Pressure Levels, Latitude and Longitude)

Some projected grids also require a ``projection`` key in the mapping.


For example, let's imagine we want to use a forecast model available via an
OPeNDAP endpoint.


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

Built-in mapping dictionaries
-----------------------------

Instead of a custom dictionary, you can pass a built-in mapping name in the
``dictionary`` argument. Common options include:

- ``"ECMWF"``
- ``"ECMWF_v0"``
- ``"NOAA"``
- ``"GFS"``
- ``"NAM"``
- ``"RAP"``
- ``"HIRESW"`` (mapping available; latest-model shortcut currently disabled)
- ``"GEFS"`` (mapping available; latest-model shortcut currently disabled)
- ``"MERRA2"``
- ``"CMC"`` (for compatible datasets loaded explicitly)

What a mapping name means
^^^^^^^^^^^^^^^^^^^^^^^^^

- Base mapping names (for example ``"GFS"``, ``"NAM"`` and ``"RAP"``) map
    RocketPy weather keys to the current default variable naming used by the
    corresponding provider datasets.
- These defaults are aligned with current shortcut workflows (for example,
    THREDDS-backed latest model sources) and may use projected coordinates
    (``x``/``y`` plus ``projection``) depending on the model.

Legacy mapping names
^^^^^^^^^^^^^^^^^^^^

If you are loading archived or older NOMADS-style datasets, use the explicit
legacy aliases:

- ``"GFS_LEGACY"``
- ``"NAM_LEGACY"``
- ``"NOAA_LEGACY"``
- ``"RAP_LEGACY"``
- ``"CMC_LEGACY"``
- ``"GEFS_LEGACY"``
- ``"HIRESW_LEGACY"``
- ``"MERRA2_LEGACY"``

Legacy aliases primarily cover older variable naming patterns such as
``lev``, ``tmpprs``, ``hgtprs``, ``ugrdprs`` and ``vgrdprs``.

.. note::

        Mapping names are case-insensitive. For example,
        ``"gfs_legacy"`` and ``"GFS_LEGACY"`` are equivalent.

For custom dictionaries, the canonical structure is:

.. code-block:: python

    mapping = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",  # optional
        "geopotential_height": "hgtprs",          # or geopotential
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }

.. important::

    Ensemble datasets require an additional key for member selection:
    ``"ensemble": "<your_member_dimension_name>"``.

.. caution::

    The ``file`` argument was intentionally omitted in the example above. This is
    because the URL depends on the provider, dataset, and date you are running
    the simulation. Build the endpoint according to the provider specification
    and always validate that the target service is active before running your
    simulation workflow.


Without OPeNDAP protocol
-------------------------

On the other hand, one can also load data from APIs that do not support the OPeNDAP protocol.
In these cases, what we recommend is to download the data and then load it as a custom atmosphere.

There are some efforts to natively support other APIs in RocketPy's
Environment class, for example: 

- `Meteomatics <https://www.meteomatics.com/en/weather-api/>`_: `#545 <https://github.com/RocketPy-Team/RocketPy/issues/545>`_
- `Open-Meteo <https://open-meteo.com/>`_: `#520 <https://github.com/RocketPy-Team/RocketPy/issues/520>`_
