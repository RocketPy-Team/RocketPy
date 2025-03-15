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
accessible `here <http://weather.uwyo.edu/upperair/sounding.html>`_.

For this example, we will use the sounding from 83779 SBMT Marte Civ Observations
at 04 Feb 2019, which can be accessed using this URL:
http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779


Initialize a new Environment instance:

.. jupyter-execute::

    from rocketpy import Environment

    url = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"

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

These options can be retrieved as a text file in GSD format.
By generating such a file through the link above, the file's URL can be used to
import the atmospheric data into RocketPy.

We will use the same sounding station as we did for the Wyoming Soundings.

.. note::
    
    Select ROABs as the initial data source, specify the station through its \
    WMO-ID, and opt for the ASCII (GSD format) button.

Initialize a new Environment instance:

.. code-block:: python

    url = r"https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest"

    env = Environment()
    env.set_atmospheric_model(type="NOAARucSounding", file=url)
    env.plots.atmospheric_model()

.. note::

    The leading `r` in the URL string is used to indicate a raw string, which \
    is useful when dealing with backslashes in URLs.



