.. _forecast:

Forecasts
=========

Weather forecasts can be used to set the atmospheric model in RocketPy.

Here, we will showcase how to import global forecasts such as GFS, as well as
local forecasts like NAM, RAP and HRRR for North America, all available through
OPeNDAP on the `UCAR THREDDS <https://thredds.ucar.edu/>`_ server.
Other generic forecasts can also be imported.

.. .. important::

..     As a rule of thumb, forecasts can only be done for future dates. \
..     If you want to simulate your rocket launch using past data, you should use \
..     :ref:`reanalysis` or :ref:`soundings`.


.. .. _global-forecast-system:

.. Global Forecast System (GFS)
.. ----------------------------

.. GFS is NOAA's global numerical weather prediction model. It provides worldwide
.. atmospheric forecasts and is usually a good default choice when you need broad
.. coverage, consistent availability, and launch planning several days ahead.

.. Using the latest forecast from GFS is simple.
.. Set the atmospheric model to ``forecast`` and specify that GFS is the file you want.
.. Note that since data is downloaded from a remote OPeNDAP server, this line of code can
.. take longer than usual.

.. .. jupyter-execute::

..     from datetime import datetime, timedelta
..     from rocketpy import Environment

..     tomorrow = datetime.now() + timedelta(days=1)
    
..     env_gfs = Environment(date=tomorrow)

..     env_gfs.set_atmospheric_model(type="forecast", file="GFS")

..     env_gfs.plots.atmospheric_model()

.. .. note::

..     The GFS model is updated every 6 hours. It is a global model with a resolution \
..     of 0.25° x 0.25°, capable of forecasting up to 10 days into the future. \
..     For more information, visit the \
..     `GFS overview page <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php>`_.


.. Artificial Intelligence Global Forecast System (AIGFS)
.. ------------------------------------------------------

.. AIGFS is a global AI-based forecast product distributed through the same THREDDS
.. ecosystem used by other RocketPy forecast inputs. It is useful when you want a
.. global forecast alternative to traditional physics-only models.

.. RocketPy supports the latest AIGFS global forecast through THREDDS.

.. .. jupyter-execute::

..     env_aigfs = Environment(date=tomorrow)
..     env_aigfs.set_atmospheric_model(type="forecast", file="AIGFS")
..     env_aigfs.plots.atmospheric_model()

.. .. note::

..     AIGFS is currently available as a global 0.25 degree forecast product on
..     UCAR THREDDS.


.. North American Mesoscale Forecast System (NAM)
.. ----------------------------------------------

.. NAM is a regional forecast model focused on North America. It is best suited
.. for launches inside its coverage area when you want finer regional detail than
.. global models typically provide.

.. You can also request the latest forecasts from NAM.
.. Since this is a regional model for North America, you need to specify latitude
.. and longitude points within North America.
.. We will use **SpacePort America** for this, represented by coordinates
.. 32.988528, -106.975056 (32°59'18.7"N 106°58'30.2"W).

.. .. jupyter-execute::

..     env_nam = Environment(
..         date=tomorrow,
..         latitude=32.988528,
..         longitude=-106.975056,
..     )
..     env_nam.set_atmospheric_model(type="forecast", file="NAM")
..     env_nam.plots.atmospheric_model()

.. .. note::

..     The NAM model is updated every 6 hours, it has a resolution of 5 km (Regional CONUS Nest) \
..     and can forecast up to 21 points spaced by 3 hours, which means it can predict \
..     the weather for the next 63 hours (3 days). For more information, visit the \
..     `NAM overview page <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/nam.php>`_.


.. Rapid Refresh (RAP)
.. -------------------

.. RAP is a short-range, high-frequency regional model for North America. It is
.. especially useful for near-term operations, where fast update cycles are more
.. important than long forecast horizon.

.. The Rapid Refresh (RAP) model is another regional model for North America.
.. It is similar to NAM, but with a higher resolution and a shorter forecast range.
.. The same coordinates for SpacePort America will be used.

.. .. jupyter-execute::

..     now = datetime.now()
..     now_plus_twelve = now + timedelta(hours=12)

..     env_rap = Environment(
..         date=now_plus_twelve,
..         latitude=32.988528,
..         longitude=-106.975056,
..     )
..     env_rap.set_atmospheric_model(type="forecast", file="RAP")
..     env_rap.plots.atmospheric_model()


.. .. note::

..     The RAP model, which succeeded the RUC model on May 1, 2012,  offers a 13 km \
..     horizontal resolution and 50 vertical layers. RAP generates forecasts every \
..     hour for North America and Alaska. 

..     For the CONUS region, RAP graphics are available for the latest 24 hours at \
..     hourly intervals, extending up to 51 hours for the 03, 09, 15, and 21 cycles, \
..     and up to 21 hours for other cycles.

..     For more details, visit: `RAP Model Info <http://rapidrefresh.noaa.gov>`_.

.. High Resolution Window (HIRESW)
.. -------------------------------

.. HIRESW is a convection-allowing, high-resolution regional system designed to
.. resolve local weather structure better than coarser grids. It is most useful
.. for short-range, local analysis where small-scale wind and weather features
.. matter.

.. The High Resolution Window (HIRESW) model is a sophisticated weather forecasting
.. system that operates at a high spatial resolution of approximately 3 km.
.. It utilizes two main dynamical cores: the Advanced Research WRF (WRF-ARW) and
.. the Finite Volume Cubed Sphere (FV3), each designed to enhance the accuracy of
.. weather predictions.

.. .. danger::

..     **HIRESW shortcut unavailable**: ``file="HIRESW"`` is currently disabled in
..     RocketPy because NOMADS OPeNDAP is deactivated for this endpoint.

.. If you have a HIRESW-compatible dataset from another provider (or a local copy),
.. you can still load it explicitly by passing the path/URL in ``file`` and an
.. appropriate mapping in ``dictionary``.


.. High-Resolution Rapid Refresh (HRRR)
.. ------------------------------------

.. HRRR is a high-resolution, short-range forecast model for North America with
.. hourly updates. It is generally best for day-of-launch weather assessment and
.. rapidly changing local conditions.

.. RocketPy supports HRRR through a dedicated THREDDS shortcut.
.. Like NAM and RAP, HRRR is a regional model over North America.

.. If you have a HIRESW-compatible dataset from another provider (or a local copy),
.. you can still load it explicitly by passing the path/URL in ``file`` and an
.. appropriate mapping in ``dictionary``.

..     env_hrrr = Environment(
..         date=now_plus_twelve,
..         latitude=32.988528,
..         longitude=-106.975056,
..     )
..     env_hrrr.set_atmospheric_model(type="forecast", file="HRRR")
..     env_hrrr.plots.atmospheric_model()

.. .. note::

..     HRRR is a high-resolution regional model with approximately 2.5 km grid
..     spacing over CONUS. Availability depends on upstream THREDDS data services.


.. Using Windy Atmosphere
.. ----------------------

.. **Windy.com** is a website that provides weather and atmospheric forecasts for
.. any location worldwide.
.. The same atmospheric predictions and data available on
.. `**windy.com** <https://www.windy.com/>`_ can be used in RocketPy.

.. The following models are accepted:

.. - **ECMWF-HRES**
.. - **GFS**
.. - **ICON-Global**
.. - **ICON-EU** (Europe only)


.. Let's see how to use Windy's data in RocketPy. First, we will set the location
.. to EuRoC's launch area in Portugal.


.. .. jupyter-execute::

..     env_windy = Environment(
..         date=tomorrow,
..         latitude=39.3897,
..         longitude=-8.28896388889,
..     )


.. ECMWF
.. ^^^^^

.. ECMWF (HRES) is a global, high-skill forecast model known for strong
.. medium-range performance. It is often a good choice for mission planning when
.. you need reliable synoptic-scale forecasts several days ahead.

.. We can use the ``ECMWF`` model from Windy.com. 

.. .. jupyter-execute::

..     env_windy.set_atmospheric_model(type="Windy", file="ECMWF")
..     env_windy_ecmwf = env_windy
..     env_windy_ecmwf.plots.atmospheric_model()

.. .. note::

..     The ECMWF model is a global model with a resolution of 9 km. It is updated \
..     every 12 hours and can forecast up to 10 days in advance. To learn more about \
..     the ECMWF model, visit the \
..     `ECMWF website <https://www.ecmwf.int/en/forecasts/datasets/open-data>`_.


.. GFS
.. ^^^

.. Windy's GFS option provides NOAA's global model through Windy's interface. It
.. is a practical baseline for global coverage and for comparing against other
.. models when assessing forecast uncertainty.

.. The ``GFS`` model is also available on Windy.com. This is the same model as
.. described in the :ref:`global-forecast-system` section.

.. .. jupyter-execute::
    
..     env_windy.set_atmospheric_model(type="Windy", file="GFS")
..     env_windy_gfs = env_windy
..     env_windy_gfs.plots.atmospheric_model()


.. ICON
.. ^^^^

.. ICON is DWD's global weather model, available in Windy for broad-scale
.. forecasting. It is useful as an independent global model source to cross-check
.. wind and temperature trends against GFS or ECMWF.

.. The ICON model is a global weather forecasting model already available on Windy.com.

.. .. jupyter-execute::

..     env_windy.set_atmospheric_model(type="Windy", file="ICON")
..     env_windy_icon = env_windy
..     env_windy_icon.plots.atmospheric_model()

.. .. note::

..     The ICON model is a global model with a resolution of 13 km. It is updated \
..     every 6 hours and can forecast up to 7 days in advance. For more information, \
..     visit `here <https://windy.app/blog/what-is-icon-weather-model-forecast.html>`_.

.. ICON-EU
.. ^^^^^^^

.. ICON-EU is the regional European configuration of ICON, with higher spatial
.. detail over Europe than ICON-Global. It is best for European launch sites when
.. regional structure is important.

.. The ICON-EU model is a regional weather forecasting model available on Windy.com.

.. .. code-block:: python

..     env_windy.set_atmospheric_model(type="Windy", file="ICONEU")
..     env_windy_icon_eu = env_windy
..     env_windy_icon_eu.plots.atmospheric_model()

.. .. important::

..     The `ICON-EU` model is only available for Europe.


.. Further considerations
.. -----------------------

.. When using forecasts, it is important to remember that the data is not always \
.. available for the exact time you want. 


.. Also, the servers may be down or may face high traffic.

.. .. seealso::

..     To browse available NCEP model collections on UCAR THREDDS, visit
..     `THREDDS NCEP Catalog <https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.html>`_.
