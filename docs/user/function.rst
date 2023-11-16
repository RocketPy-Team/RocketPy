.. _functionusage::

Function Class Usage
====================

The :class:`rocketpy.mathutils.Function` class in RocketPy is an auxiliary module that allows for easy manipulation of datasets and Python functions. Some of the class features are data interpolation, extrapolation, algebra and plotting.

The basic steps to create a ``Function`` are as follows:

1. Define a data source: a dataset (e.g. x,y coordinates) or a function that maps a dataset to a value (e.g. f(x) = x^2);
2. Construct a ``Function`` object with this dataset as ``source``;
3. Use the ``Function`` features as needed: add datasets, integrate at a point, make a scatter plot and much more.

These basic steps are detailed in this guide.

1. Define a Data Source
-----------------------

Datasets
~~~~~~~~

The ``Function`` class supports a wide variety of dataset sources:

List or Numpy Array
^^^^^^^^^^^^^^^^^^^

A ``list`` or ``numpy.ndarray`` of datapoints that maps input values to an output can be used as a ``Function`` source. For instance, we can define a dataset that follows the function f(x) = x^2:

.. jupyter-execute::

    from rocketpy.mathutils import Function

    # Source dataset [(x0, y0), (x1, y1), ...]
    source = [
        (-3, 9), (-2, 4), (-1, 1), 
        (0, 0), (1, 1), (2, 4), 
        (3, 9)
        ]

    # Create a Function object with this dataset
    f = Function(source, "x", "y")

One may print the source attribute from the ``Function`` object to check the inputed dataset.

.. jupyter-execute::

    # Print the source to see the dataset
    print(f.source)

Furthermore, in order to visualize the dataset, one may use the ``plot`` method from the ``Function`` object:

.. jupyter-execute::

    # Plot the source with standard spline interpolation
    f.plot()

|

The dataset can be defined as a *multidimensional* array (more than one input maps to an output), where each row is a datapoint. For example, let us define a dataset that follows the plane z = x + y:

.. jupyter-execute::

    # Source dataset [(x0, y0, z0), (x1, y1, z1), ...]
    source = [
        (-1, -1, -2), (-1, 0, -1), (-1, 1, 0), 
        (0, -1, -1), (0, 0, 0), (0, 1, 1), 
        (1, -1, 0), (1, 0, 1), (1, 1, 2)
        ]

    # Create a Function object with this dataset
    f = Function(source, ["x", "y"], "z")

One may print the source attribute from the ``Function`` object to check the inputed dataset.

.. jupyter-execute::

    print(f.source)

Two dimensional plots are also supported, therefore this datasource can be plotted as follows:

.. jupyter-execute::

    # Plot the source with standard 2d shepard interpolation
    f.plot()

.. important::
    The ``Function`` class only supports interpolation ``shepard`` and extrapolation ``natural`` for datasets higher than one dimension (more than one input). 

CSV File
^^^^^^^^ 

A CSV file path can be passed as ``string`` to the ``Function`` source. The file must contain a dataset structured so that each line is a datapoint: the last column is the output and the previous columns are the inputs.

.. jupyter-execute::

    # Create a csv and save with pandas
    import pandas as pd

    df = pd.DataFrame({
        '"x"': [-3, -2, -1, 0, 1, 2, 3],
        '"y"': [9, 4, 1, 0, 1, 4, 9],
        })
    df.to_csv('source.csv', index=False)
    pd.read_csv('source.csv')

|

Having the csv file, we can define a ``Function`` object with it:

.. jupyter-execute::

    # Create a Function object with this dataset
    f = Function('source.csv')

    # One may even delete the csv file
    import os
    os.remove('source.csv')

    # Print the source to see the dataset
    print(f.source)

.. note::
    A header in the csv file is optional, but if present must be in a string like format, i.e. beginning and ending with quotation marks.

Function Map
~~~~~~~~~~~~

A Python function that maps a set of parameters to a result can be used as a ``Function`` source. For instance, we can define a function that maps x to f(x) = sin(x):

.. jupyter-execute::

    import numpy as np
    from rocketpy.mathutils import Function

    # Define source function
    def source_func(x):
        return np.sin(x)

    # Create a Function from source
    f = Function(source_func)

The result of this operation is a ``Function`` object that wraps the source function and features many functionalities, such as plotting.

Constant Functions
^^^^^^^^^^^^^^^^^^

A special case of the python function source is the definition of a constant ``Function``. The class supports a convenient shortcut to ease the definition of a constant source:

.. jupyter-execute::

    # Constant function
    f = Function(1.5)

    print(f(0))
    

.. note::
    This shortcut is completely equivalent to defining a Python constant function as the source:

    .. jupyter-input::
        def const_source(_):
            return 1.5

        g = Function(const_source)


2. Building your Function
-------------------------

In this section we are going to delve deeper on ``Function`` creation and its parameters:

- source: the ``Function`` datasource. We have explored this parameter in the section above;
- inputs: a list of strings containing each input variable name. If the source only has one input, may be abbreviated as a string (e.g. "speed (m/s)");
- outputs: a list of strings containing each output variable name. If the source only has one output, may be abbreviated as a string (e.g. "total energy (J)");
- interpolation: a string that is the interpolation method to be used if the source is a dataset. Defaults to ``spline``;
- extrapolation: a string that is the extrapolation method to be used if the source is a dataset. Defaults to ``constant``;
- title: the title to be shown in the plots.

With these in mind, let us create a more concrete example so that each of these parameters usefulness is explored.::

    Suppose we have a particle named Bob

.. seealso::
    Check out more about the constructor parameters and other functionalities in the :class:`rocketpy.mathutils.Function` documentation.

.. note::
    The ``Function`` class plots only supports one or two dimensional inputs.
