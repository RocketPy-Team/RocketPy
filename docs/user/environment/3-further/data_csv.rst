.. _loading-environment-data-from-csv:

Loading data from a .csv file
=============================

In this example, we will load the data from a .csv file to set up the environment.

As you might already know, an atmospheric model in RocketPy is defined by a
pressure, temperature, height (or geopotential height) and wind profiles.
What we will need to do is to create a .csv file with these profiles and load
it into the environment using a :ref:`custom_atmosphere` model.

Create a .csv file
------------------

This step is not mandatory, you can create the .csv file with any text editor
or spreadsheet software. The file should have the following structure:

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    
    from rocketpy import Environment, Function


    # Create a .csv file with the profiles
    data = {
        'height': [0, 1000, 2000, 3000, 4000], # m
        'pressure': [101325, 89876, 79508, 70122, 61653], # Pa
        'temperature': [288.15, 281.65, 275.15, 268.65, 262.15], # K
        'wind_u': [0, 1.0, 3, -0.5, -1], # m/s
        'wind_v': [0, 3.0, 2.2, 5.8, 10], # m/s
    }

    df = pd.DataFrame(data)
    df.to_csv('atmosphere.csv', index=False)

    df.head()

.. important::

    To avoid errors and misinterpretations, RocketPy expects all the values to \
    be in SI units. Of course you can import the data in different units and \
    convert them to SI units before saving the .csv file.


Load the .csv file
------------------

Now we will load the .csv file into the environment.

.. jupyter-execute::

    # Load the .csv file into the environment
    df = pd.read_csv('atmosphere.csv')

    # Create Function objects to represent the profiles
    pressure_func = Function(np.column_stack([df['height'], df['pressure']]))
    temperature_func = Function(np.column_stack([df['height'], df['temperature']]))
    wind_u_func = Function(np.column_stack([df['height'], df['wind_u']]))
    wind_v_func = Function(np.column_stack([df['height'], df['wind_v']]))

    # Set up the environment
    env_csv = Environment()
    env_csv.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=pressure_func,
        temperature=temperature_func,
        wind_u=wind_u_func,
        wind_v=wind_v_func,
    )

    # Plot the atmospheric model
    env_csv.plots.atmospheric_model()

.. jupyter-execute::
    :hide-code:

    # remove the .csv file
    import os
    os.remove('atmosphere.csv')
