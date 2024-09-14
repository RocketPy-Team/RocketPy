Location
========

Setting up the location is simple using the :meth:`rocketpy.Environment.set_location` method.
This method allows you to specify the latitude and longitude of the launch site
and automatically refreshes the atmospheric conditions if a location-dependent
model is being used.

.. jupyter-execute::

    from rocketpy import Environment

    env = Environment()

    env.set_location(latitude=32.988528, longitude=-106.975056)

    env.prints.launch_site_details()


.. important::

    If you use the :meth:`rocketpy.Environment.set_location` method again, the atmospheric \
    conditions will be automatically refreshed. You can update the location at any time.
