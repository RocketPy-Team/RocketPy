.. _reanalysis:

Reanalysis
==========

Climate reanalyses combine past observations with models to generate consistent
time series of multiple climate variables. They are among the most-used datasets
in the geophysical sciences, providing a comprehensive description of the
observed climate as it has evolved over recent decades on 3D grids at sub-daily
intervals
(`Climate Reanalysis <https://climate.copernicus.eu/climate-reanalysis>`_).

Reanalysis data can be used to set up the environment in RocketPy.
One common reanalysis dataset is the ERA5.

ERA5
----

ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate,
covering the period from January 1940 to the present. Produced by the Copernicus
Climate Change Service (C3S) at ECMWF, ERA5 provides hourly estimates of numerous
atmospheric, land, and oceanic climate variables. The data cover the Earth on a
31km grid and resolve the atmosphere using 137 levels from the surface up to a
height of 80km. ERA5 includes information about uncertainties for all variables
at reduced spatial and temporal resolutions
(`Climate Reanalysis <https://climate.copernicus.eu/climate-reanalysis>`_).

Downloading the Data
^^^^^^^^^^^^^^^^^^^^

ERA5 data can be downloaded from the
`ECMWF's Climate Data Store <https://cds.climate.copernicus.eu/#!/home>`_.

#. Go to the `ERA5 hourly data on pressure levels from 1940 to present <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form>`_ page.
#. On "Product Type", select "Reanalysis". Optionally, you can select "Ensemble" to download ensemble data.
#. On "Variable", select the variables you want to download. RocketPy requires the following variables: "Geopotential", "Temperature", "U-component of wind" and "V-component of wind"
#. On "Pressure Level", select the pressure levels you want to download. You can convert the pressure levels to altitude using online calculators.
#. Select the date range you want to download, which are described by the "Year", "Month", "Day" and "Time" fields. 
#. Select the latitude and longitude boundaries of the area you want to download.
#. Download the file as a NetCDF file.

.. danger::

    If you select a long time range, many variables and a really large area, \
    the download may take a long time and the file may be too large to be \
    processed by RocketPy. It is recommended that you download only the \
    necessary data.

MERRA-2
-------

The Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA-2) is a NASA atmospheric reanalysis for the satellite era using the Goddard Earth Observing System, Version 5 (GEOS-5) with its Atmospheric Data Assimilation System (ADAS).

You can download these files from the `NASA GES DISC <https://disc.gsfc.nasa.gov/>`_.

To use MERRA-2 data in RocketPy, you generally need the **Assimilated Meteorological Fields** collection (specifically the 3D Pressure Level data, usually named ``inst3_3d_asm_Np``). Note that MERRA-2 files typically use the ``.nc4`` extension (NetCDF-4), which is fully supported by RocketPy.

You can load these files using the ``dictionary="MERRA2"`` argument:

.. code-block:: python

    env.set_atmospheric_model(
        type="Reanalysis",
        file="MERRA2_400.inst3_3d_asm_Np.20230620.nc4",
        dictionary="MERRA2"
    )

RocketPy automatically handles the unit conversion for MERRA-2's surface geopotential (energy) to geometric height (meters).


Setting the Environment
^^^^^^^^^^^^^^^^^^^^^^^

To set up the environment, use the following Python code.
Remember that you have to download the data first. The date and location must
be within the range of the data.


.. code-block:: python

    from rocketpy import Environment

    env_era5 = Environment(
        date=(2018, 10, 15, 12),
        latitude=39.389700,
        longitude=-8.288964,
        elevation=113,
    )
    
    filename = "../data/weather/EuroC_pressure_levels_reanalysis_2001-2021.nc"

    env_era5.set_atmospheric_model(
        type="Reanalysis",
        file=filename,
        dictionary="ECMWF",
    )

    env_era5.plots.atmospheric_model()


.. _reanalysis_ensemble:

Ensemble
--------

Ensemble reanalysis data, which is a set of reanalysis data with many ensemble
members, can also be used. Set the ``type`` parameter to ``Ensemble`` and
provide the path to the file containing the data:


.. code-block:: python

    from rocketpy import Environment

    env_reanalysis = Environment(
        date=(2019, 8, 10, 21),
        latitude=-23.363611,
        longitude=-48.011389,
        elevation=668,
    )
    
    env_reanalysis.set_atmospheric_model(
        type="Ensemble",
        file="../data/weather/LASC2019_TATUI_reanalysis_ensemble.nc",
        dictionary="ECMWF",
    )

    env_reanalysis.plots.atmospheric_model()

.. important::

    To use Ensemble from reanalysis data, you must have included the "Ensemble" \
    option when downloading the data from the ECMWF's Climate Data Store.

