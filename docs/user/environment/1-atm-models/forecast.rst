.. _forecast:

Forecasts
=========

Weather forecasts can be used to set the atmospheric model in RocketPy.

Here, we will showcase how to import global forecasts such as GFS, as well as
local forecasts like NAM and RAP for North America, all available through
OPeNDAP on the `NOAA's NCEP NOMADS <http://nomads.ncep.noaa.gov/>`_ website.
Other generic forecasts can also be imported.

.. important::

    As a rule of thumb, forecasts can only be done for future dates. \
    If you want to simulate your rocket launch using past data, you should use \
    :ref:`reanalysis` or :ref:`soundings`.


.. _global-forecast-system:

Global Forecast System (GFS)
----------------------------

Using the latest forecast from GFS is simple.
Set the atmospheric model to ``forecast`` and specify that GFS is the file you want.
Note that since data is downloaded from the NOMADS server, this line of code can
take longer than usual.

.. jupyter-execute::

    from datetime import datetime, timedelta
    from rocketpy import Environment

    tomorrow = datetime.now() + timedelta(days=1)
    
    env_gfs = Environment(date=tomorrow)

    env_gfs.set_atmospheric_model(type="forecast", file="GFS")

    env_gfs.plots.atmospheric_model()

.. note::

    The GFS model is updated every 6 hours. It is a global model with a resolution \
    of 0.25° x 0.25°, capable of forecasting up to 10 days into the future. \
    For more information, visit the \
    `GFS overview page <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php>`_.


North American Mesoscale Forecast System (NAM)
----------------------------------------------

You can also request the latest forecasts from NAM.
Since this is a regional model for North America, you need to specify latitude
and longitude points within North America.
We will use **SpacePort America** for this, represented by coordinates
32.988528, -106.975056 (32°59'18.7"N 106°58'30.2"W).

.. jupyter-execute::

    env_nam = Environment(
        date=tomorrow,
        latitude=32.988528,
        longitude=-106.975056,
    )
    env_nam.set_atmospheric_model(type="forecast", file="NAM")
    env_nam.plots.atmospheric_model()

.. note::

    The NAM model is updated every 6 hours, it has a resolution of 5 km (Regional CONUS Nest) \
    and can forecast up to 21 points spaced by 3 hours, which means it can predict \
    the weather for the next 63 hours (3 days). For more information, visit the \
    `NAM overview page <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/nam.php>`_.


Rapid Refresh (RAP)
-------------------

The Rapid Refresh (RAP) model is another regional model for North America.
It is similar to NAM, but with a higher resolution and a shorter forecast range.
The same coordinates for SpacePort America will be used.

.. jupyter-execute::

    now = datetime.now()
    now_plus_twelve = now + timedelta(hours=12)

    env_rap = Environment(
        date=now_plus_twelve,
        latitude=32.988528,
        longitude=-106.975056,
    )
    env_rap.set_atmospheric_model(type="forecast", file="RAP")
    env_rap.plots.atmospheric_model()


.. note::

    The RAP model, which succeeded the RUC model on May 1, 2012,  offers a 13 km \
    horizontal resolution and 50 vertical layers. RAP generates forecasts every \
    hour for North America and Alaska. 

    For the CONUS region, RAP graphics are available for the latest 24 hours at \
    hourly intervals, extending up to 51 hours for the 03, 09, 15, and 21 cycles, \
    and up to 21 hours for other cycles.

    For more details, visit: `RAP Model Info <http://rapidrefresh.noaa.gov>`_.

High Resolution Window (HIRESW)
-------------------------------

The High Resolution Window (HIRESW) model is a sophisticated weather forecasting
system that operates at a high spatial resolution of approximately 3 km.
It utilizes two main dynamical cores: the Advanced Research WRF (WRF-ARW) and
the Finite Volume Cubed Sphere (FV3), each designed to enhance the accuracy of
weather predictions.

You can easily set up HIRESW in RocketPy by specifying the date, latitude, and
longitude of your location. Let's use SpacePort America as an example.

.. jupyter-execute::

    env_hiresw = Environment(
        date=tomorrow,
        latitude=32.988528,
        longitude=-106.975056,
    )

    env_hiresw.set_atmospheric_model(
        type="Forecast",
        file="HIRESW",
        dictionary="HIRESW",
    )

    env_hiresw.plots.atmospheric_model()

.. note::

    The HRES model is updated every 12 hours, providing forecasts with a \
    resolution of 3 km. The model can predict weather conditions up to 48 hours \
    in advance. RocketPy uses the CONUS domain with ARW core.


Using Windy Atmosphere
----------------------

**Windy.com** is a website that provides weather and atmospheric forecasts for
any location worldwide.
The same atmospheric predictions and data available on
`**windy.com** <https://www.windy.com/>`_ can be used in RocketPy.

The following models are accepted:

- **ECMWF-HRES**
- **GFS**
- **ICON-Global**
- **ICON-EU** (Europe only)


Let's see how to use Windy's data in RocketPy. First, we will set the location
to EuRoC's launch area in Portugal.


.. jupyter-execute::

    env_windy = Environment(
        date=tomorrow,
        latitude=39.3897,
        longitude=-8.28896388889,
    )


ECMWF
^^^^^

We can use the ``ECMWF`` model from Windy.com. 

.. jupyter-execute::

    env_windy.set_atmospheric_model(type="Windy", file="ECMWF")
    env_windy_ecmwf = env_windy
    env_windy_ecmwf.plots.atmospheric_model()

.. note::

    The ECMWF model is a global model with a resolution of 9 km. It is updated \
    every 12 hours and can forecast up to 10 days in advance. To learn more about \
    the ECMWF model, visit the \
    `ECMWF website <https://www.ecmwf.int/en/forecasts/datasets/open-data>`_.


GFS
^^^

The ``GFS`` model is also available on Windy.com. This is the same model as
described in the :ref:`global-forecast-system` section.

.. jupyter-execute::
    
    env_windy.set_atmospheric_model(type="Windy", file="GFS")
    env_windy_gfs = env_windy
    env_windy_gfs.plots.atmospheric_model()


ICON
^^^^

The ICON model is a global weather forecasting model already available on Windy.com.

.. jupyter-execute::

    env_windy.set_atmospheric_model(type="Windy", file="ICON")
    env_windy_icon = env_windy
    env_windy_icon.plots.atmospheric_model()

.. note::

    The ICON model is a global model with a resolution of 13 km. It is updated \
    every 6 hours and can forecast up to 7 days in advance. For more information, \
    visit `here <https://windy.app/blog/what-is-icon-weather-model-forecast.html>`_.

ICON-EU
^^^^^^^

The ICON-EU model is a regional weather forecasting model available on Windy.com.

.. code-block:: python

    env_windy.set_atmospheric_model(type="Windy", file="ICONEU")
    env_windy_icon_eu = env_windy
    env_windy_icon_eu.plots.atmospheric_model()

.. important::

    The `ICON-EU` model is only available for Europe.


Further considerations
-----------------------

When using forecasts, it is important to remember that the data is not always \
available for the exact time you want. 


Also, the servers may be down or may face high traffic.

.. seealso::

    To see a complete list of available models on the NOAA's NOMADS server, visit
    `NOMADS <https://nomads.ncep.noaa.gov/>`_.

