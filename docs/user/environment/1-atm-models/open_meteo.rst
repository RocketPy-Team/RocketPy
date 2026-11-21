.. _open_meteo:

Open-Meteo
==========

`Open-Meteo <https://open-meteo.com/>`_ is a weather API that serves
pressure-level forecasts, past forecasts and ensemble forecasts as plain JSON
over HTTPS.

It is often the most convenient weather source in RocketPy, because:

- **No API key** is required for non-commercial use.
- **No heavy dependencies**: unlike the :ref:`forecast` and :ref:`reanalysis`
  models, no ``netCDF4``/OPeNDAP download is involved, so requests are quick.
- **No external files**: recent past launches can be reconstructed straight from
  the API, without downloading reanalysis files by hand.
- **Many models in one place**: GFS, ECMWF, ICON, MET Norway, Météo-France, JMA,
  GEM and UKMO are all reachable through the same interface.

.. note::

    Open-Meteo is free for non-commercial use, with a limit on the number of
    daily requests. Please read
    `their terms <https://open-meteo.com/en/terms>`_ before using it, and
    consider their paid plans for heavier or commercial workloads.


Forecasts
---------

Set the atmospheric model to ``open_meteo``. The launch date must be set,
because the vertical profile is taken at the hour closest to it.

.. jupyter-execute::

    from datetime import datetime, timedelta
    from rocketpy import Environment

    tomorrow = datetime.now() + timedelta(days=1)

    env = Environment(
        date=tomorrow,
        latitude=39.3897,
        longitude=-8.28896388889,
    )

    env.set_atmospheric_model(type="open_meteo")

    env.plots.atmospheric_model()

Note that ``elevation`` was never specified above: Open-Meteo reports the
elevation of the grid cell it answered for, and RocketPy uses it to set the
launch site elevation automatically.


Selecting a weather model
^^^^^^^^^^^^^^^^^^^^^^^^^

By default RocketPy asks for ``"best_match"``, which lets Open-Meteo pick the
highest-resolution model available for the requested location. A specific model
can be requested through the ``file`` argument:

.. jupyter-execute::

    env_ecmwf = Environment(
        date=tomorrow,
        latitude=39.3897,
        longitude=-8.28896388889,
    )
    env_ecmwf.set_atmospheric_model(type="open_meteo", file="ecmwf_ifs025")
    env_ecmwf.plots.atmospheric_model()

Frequently useful models are:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Model
      - Description
    * - ``best_match``
      - Open-Meteo picks the best available model for the location (default).
    * - ``gfs_seamless``
      - NOAA GFS, global coverage.
    * - ``ecmwf_ifs025``
      - ECMWF IFS at 0.25°, global coverage.
    * - ``icon_seamless``
      - DWD ICON, global coverage with a higher-resolution European nest.
    * - ``meteofrance_seamless``
      - Météo-France ARPEGE/AROME.
    * - ``gem_seamless``
      - Environment Canada GEM.
    * - ``ukmo_seamless``
      - UK Met Office.

.. seealso::

    The `Open-Meteo documentation <https://open-meteo.com/en/docs>`_ lists every
    model available, along with its resolution and update frequency.

.. important::

    Not every model resolves every pressure level, and coverage varies with
    location. RocketPy silently drops the levels a model does not provide, so
    the resulting profile may reach a lower altitude for some models (for
    instance, ``ecmwf_ifs025`` tops out at 50 hPa while ``gfs_seamless``
    reaches 30 hPa). Check ``env.max_expected_height`` if the ceiling matters
    for your simulation.


Past launches
-------------

When the launch date is in the past, ``open_meteo`` transparently queries
Open-Meteo's historical-forecast archive instead of the live forecast. No extra
argument is needed:

.. jupyter-execute::

    env_past = Environment(
        date=datetime(2024, 1, 10, 12),
        latitude=39.3897,
        longitude=-8.28896388889,
    )
    env_past.set_atmospheric_model(type="open_meteo")
    env_past.plots.atmospheric_model()

This is the quickest way to reconstruct the atmosphere of a past flight, since
it needs neither an external file nor a sounding station nearby. For
comparison, see :ref:`reanalysis` and :ref:`soundings`.

.. important::

    Open-Meteo's historical data is built from its own archived forecast runs
    and only covers pressure levels **from around March 2021 onwards**. Earlier
    dates return no data, and RocketPy warns you when it detects one; use
    :ref:`reanalysis` or :ref:`soundings` for those instead.

.. note::

    Open-Meteo also offers an ERA5 archive endpoint, but it serves surface
    variables only and provides no pressure-level data, so RocketPy does not
    use it: it cannot produce a vertical profile.


Ensemble forecasts
------------------

Open-Meteo also exposes ensemble forecasts, where each member represents a
slightly different evolution of the atmosphere. They are used exactly like the
other :ref:`ensemble_atmosphere` models:

.. jupyter-execute::

    env_ensemble = Environment(
        date=tomorrow,
        latitude=39.3897,
        longitude=-8.28896388889,
    )
    env_ensemble.set_atmospheric_model(type="open_meteo_ensemble", file="gfs05")

    print(f"Number of members: {env_ensemble.num_ensemble_members}")

    env_ensemble.plots.ensemble_member_comparison()

Individual members are activated with
:meth:`rocketpy.Environment.select_ensemble_member`:

.. jupyter-execute::

    env_ensemble.select_ensemble_member(10)
    print(f"Wind speed at 1 km: {env_ensemble.wind_speed(1000):.2f} m/s")

Member ``0`` is the unperturbed control run and is the one selected by default.

Two ensemble models publish the complete set of pressure-level variables that
RocketPy needs:

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Model
      - Members
      - Description
    * - ``gfs05``
      - 31
      - NOAA GEFS at 0.5° (default).
    * - ``ecmwf_ifs025``
      - 51
      - ECMWF ensemble at 0.25°.

The member counts above include the control run, which RocketPy exposes as
member ``0``.

.. important::

    The remaining Open-Meteo ensemble models cannot be used to build a vertical
    profile, so RocketPy rejects them with an explanatory error rather than
    failing later on. ``gfs025``, ``icon_global`` and
    ``bom_access_global_ensemble`` answer successfully but return no
    pressure-level values at all, and ``gem_global`` provides temperature and
    geopotential height but no pressure-level winds.


Further considerations
----------------------

Requests may fail if the API is unreachable or if the daily free-tier limit is
exceeded. RocketPy retries transient failures automatically and raises a
``RuntimeError`` with the reason reported by Open-Meteo when the request cannot
be satisfied.

.. seealso::

    - :ref:`forecast` for OPeNDAP-based forecasts (GFS, NAM, RAP, HRRR).
    - :ref:`reanalysis` for ERA5 and MERRA-2 reanalysis files.
    - :ref:`ensemble_atmosphere` for the ensemble workflow in general.
