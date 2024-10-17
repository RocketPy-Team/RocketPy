Elevation
=========

The elevation of the launch site is particularly important for
determining the rocket's interaction with the atmosphere and Earth's gravitational field.

Known elevation
---------------

If the elevation, measured in meters above sea level, is known, it can be used
to initialize an Environment class instance as follows:

.. jupyter-execute::

    from rocketpy import Environment
    
    env = Environment(elevation=110)

You can change the elevation at any time by using the following method:

.. jupyter-execute::

    env.set_elevation(120)

Using Open-Elevation API
------------------------

Fortunately, there are alternatives to find an approximate value of the
elevation if the latitude and longitude values are known.

One very useful and handy option is to use
`Open-Elevation <https://open-elevation.com/>`_, a free and open-source
elevation API.
It is integrated with RocketPy and can be used as follows.

First, initialize a new Environment:

.. code-block:: python

    env = Environment(
        date=(2019, 2, 10, 18),
        latitude=-21.960641,
        longitude=-47.482122
    )

Then, set the elevation using Open-Elevation:

.. code-block:: python

    env.set_elevation("Open-Elevation")

To get information from the Environment, use the following method:

.. code-block:: python

    env.prints.launch_site_details()


Using Atmospheric Models
------------------------

One option is to use the elevation data supplied by some atmospheric models.
Since elevation data is crucial for numerical weather prediction, some weather
models make elevation data available together with other variables.
This will be covered in the :ref:`atmospheric_models` section.

