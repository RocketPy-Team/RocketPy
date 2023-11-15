.. _functionusage::

Function Class Usage
====================

The ``rocketpy.Function`` class in RocketPy is an auxiliary module that allows for easy manipulation of datasets and Python functions. Some of the class features are data interpolation, extrapolation, algebra and plotting.

The basic steps to create a ``Function`` are as follows:

1. Define a data source: a dataset (e.g. x,y coordinates) or a function that maps a dataset to another (e.g. f(x) = x^2);
2. Construct a ``Function`` object with this dataset as ``source``;
3. Use the ``Function`` features as needed: add datasets, integrate at a point, make a scatter plot and much more.

These basic steps are detailed in this guide.

1. Define a Data Source
-----------------------

The ``Function`` class supports a wide variety of data sources:

Datasets
~~~~~~~~

- ``list`` or ``numpy.ndarray``: a list of datapoints that maps input values to an output. For instance, we can define a dataset that follows the function f(x) = x^2:

.. jupyter-execute::
    from rocketpy import Function

    # Source dataset [(x0, y0), (x1, y1), ...]
    source = [
        (-3, 9), (-2, 4), (-1, 1), 
        (0, 0), (1, 1), (2, 4), 
        (3, 9)
        ]

    # Create a Function object with this dataset
    f = Function(source)

    # Print the source to see the dataset
    print(f.source)

    # Plot the source with standard spline interpolation
    f.plot()

The dataset can be defined as a higher dimensional array (more than one input maps to an output), where each row is a datapoint. For example, let us define a dataset that follows the plane z = x + y:

.. jupyter-execute::
    # Source dataset [(x0, y0, z0), (x1, y1, z1), ...]
    source = [
        (-1, -1, -2), (-1, 0, -1), (-1, 1, 0), 
        (0, -1, -1), (0, 0, 0), (0, 1, 1), 
        (1, -1, 0), (1, 0, 1), (1, 1, 2)
        ]

    # Create a Function object with this dataset
    f = Function(source)

    # Print the source to see the dataset
    print(f.source)

    # Plot the source with standard 2d shepard interpolation
    f.plot()

.. important::
    The ``Function`` class only supports interpolation and extrapolation of type ``shepard`` for datasets higher than one dimension (more than one input). 

- ``string``: a csv file path that contains a dataset structured so that each line is a datapoint: the last column is the output and the previous columns are the inputs;

.. jupyter-execute::
    # Create a csv and save with pandas
    import pandas as pd

    df = pd.DataFrame({
        'x': [-3, -2, -1, 0, 1, 2, 3],
        'y': [9, 4, 1, 0, 1, 4, 9],
        })
    df.to_csv('source.csv', index=False)
    #pd.read_csv('source.csv')

Having the csv file, we can define a ``Function`` object with it:

.. jupyter-execute::
    # Create a Function object with this dataset
    #f = Function('source.csv')

    # One may even delete the csv file
    #import os
    #os.remove('source.csv')

    # Print the source to see the dataset
    #print(f.source)

.. note::
    A header in the csv file is optional, but if present must be in a string like format, i.e. beginning and ending with quotation marks.


.. note::
    The ``Function`` class plots only supports one or two dimensional inputs.






