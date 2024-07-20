Topography
==========

We will show how RocketPy interacts with Earth Topography.
We mainly will use data provided by the NASADEM Merged DEM Global 1 arc second nc.

NASADEM is a digital elevation model based on the Shuttle Radar Topography Mission (SRTM),
a collaboration between NASA and the National Geospatial-Intelligence Agency (NGA),
as well as participation from the German and Italian space agencies.

.. seealso::

    You can understand more about NASADEM by reading their documentation at \
    `NASA's Earth data Search <https://lpdaac.usgs.gov/products/nasadem_hgtv001/>`_.


This is a first step forward stopping consider Earth as a constant plane better
results when we are flying next to mountains or valleys

Initialization
--------------

First of all, we import the Environment class: 

.. jupyter-execute::

    from rocketpy import Environment

For example, let's set an Environment consider a fictional launch at Switzerland.
First we need to set the basic information about our Environment object

.. jupyter-execute::

    env = Environment(latitude=46.90479, longitude=8.07575, datum="WGS84")

.. note::
    
    Notice that the datum argument is used only for the converting from geodesic \
    (i.e. lat/lon) to UTM coordinate system.

Set the topographic profile
---------------------------

Now we finally set our topography

.. jupyter-execute::

    env.set_topographic_profile(
        type="NASADEM_HGT",
        file="../data/sites/switzerland/NASADEM_NC_n46e008.nc",
        dictionary="netCDF4",
        crs=None,
    )


Find the launch site elevation
------------------------------

Once we defined the topographic profile, we can find the launch site elevation

.. jupyter-execute::

    elevation = env.get_elevation_from_topographic_profile(env.latitude, env.longitude)
    print(f"The elevation at latitude {env.latitude} and longitude {env.longitude} is {elevation} meters.")

And finally set the elevation to the Environment object:

.. jupyter-execute::
    
    env.set_elevation(elevation)

Visualize information
---------------------

Now we can see the elevation that we have set, as well as other important
attributes of our Environment object. We do that by running the
:meth:`rocketpy.Environment.info` method:

.. jupyter-execute::
    
    env.prints.launch_site_details()


