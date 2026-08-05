.. _soundings:

Soundings
=========

Another option to define atmospheric conditions is to use upper air soundings.
These are profiles of temperature, pressure, humidity, and wind speed and direction
measured by weather balloons or similar devices.


Wyoming Upper Air Soundings
---------------------------

The University of Wyoming - College of Engineering - Department of Atmospheric
Sciences has a comprehensive collection of atmospheric soundings on their website,
accessible `here <https://weather.uwyo.edu/upperair/sounding.shtml>`_.

For this example, we will use the sounding from 83779 SBMT Marte Civ Observations
at 05 Feb 2019, which can be accessed using this URL:
https://weather.uwyo.edu/wsgi/sounding?datetime=2019-02-05%2012:00:00&id=83779&type=TEXT:LIST

.. important::

    The University of Wyoming discontinued the legacy
    ``weather.uwyo.edu/cgi-bin/sounding`` endpoint. URLs in that old format no
    longer work; use the new ``weather.uwyo.edu/wsgi/sounding`` format shown
    above, which RocketPy supports since v1.13.0.


Initialize a new Environment instance:

.. note::

    The example below is shown as static code (it is not executed during the
    documentation build) because it requires a live network request to an
    external University of Wyoming service. Run it locally to fetch the
    sounding and see the resulting atmospheric plots.

.. code-block:: python

    from rocketpy import Environment

    url = "https://weather.uwyo.edu/wsgi/sounding?datetime=2019-02-05%2012:00:00&id=83779&type=TEXT:LIST"

    env = Environment()
    env.set_atmospheric_model(type="wyoming_sounding", file=url)
    env.plots.atmospheric_model()

.. note::

    The ``wyoming_sounding`` does not require the ``date`` parameter to be set, \
    as the data is already provided in the URL.


NOAA's Ruc Soundings
--------------------

.. important::

    From September 30th, 2024, this model is no longer available since NOAA has \
    discontinued the Ruc Soundings public service. The following message is \
    displayed on the website: \
    "On Monday, September 30, a number of legacy websites were permanently removed. \
    These sites were no longer being maintained and did not meet security and \
    design requirements mandated by NOAA. They were intended for research \
    purposes and are not designed for operational use, such as for commercial \
    purposes or the safety of life and property."

Another option for upper air soundings is `NOAA's Ruc Soundings <https://rucsoundings.noaa.gov/>`_.
This service allows users to download virtual soundings from numerical weather
prediction models such as GFS, RAP, and NAM, and also real soundings from the
Integrated Global Radiosonde Archive (IGRA).

These options can be retrieved as a text file in GSD format. However,
RocketPy no longer provides a dedicated ``set_atmospheric_model`` type for
NOAA RUC Soundings, since NOAA has discontinued the OPENDAP service.

.. note::
    
    Select ROABs as the initial data source, specify the station through its \
    WMO-ID, and opt for the ASCII (GSD format) button.

If you need to use RUC-sounding-like data in RocketPy, convert it to one of the
supported workflows:

- Use :ref:`custom_atmosphere` after parsing the text data.
- Use :ref:`reanalysis` or :ref:`forecast` with NetCDF/OPeNDAP sources.

.. note::

    The leading `r` in the URL string is used to indicate a raw string, which \
    is useful when dealing with backslashes in URLs.