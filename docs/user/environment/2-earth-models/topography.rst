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


Including the topography provides more accurate results when flying next to mountains or valleys.

Initialization
--------------

Let us consider a fictional launch in Switzerland. First, we set the basic information about our Environment object

.. jupyter-execute::

    from rocketpy import Environment
    env = Environment(latitude=46.90479, longitude=8.07575, datum="WGS84")

.. note::
    
    The datum argument is used only for converting from geodesic \
    (i.e. lat/lon) to the UTM coordinate system.

Set the topographic profile
---------------------------

Now we set our topography

.. jupyter-execute::

    env.set_topographic_profile(
        type="NASADEM_HGT",
        file="../data/sites/switzerland/NASADEM_NC_n46e008.nc",
        dictionary="netCDF4",
        crs=None,
    )


Find the launch site elevation
------------------------------

Once the topographic profile is defined, we can find the launch site elevation

.. jupyter-execute::

    elevation = env.get_elevation_from_topographic_profile(env.latitude, env.longitude)
    print(f"The elevation at latitude {env.latitude} and longitude {env.longitude} is {elevation} meters.")

And finally set the elevation to the Environment object:

.. jupyter-execute::
    
    env.set_elevation(elevation)

Visualize information
---------------------

To check the elevation we have set, as well as other important
attributes of our Environment object, we run the
:meth:`rocketpy.Environment.info` method:

.. jupyter-execute::
    
    env.prints.launch_site_details()


