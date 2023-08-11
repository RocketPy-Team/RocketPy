Requirements
============

Usually, you can skip this part if you have an updated version of Python running on a normal computer.
But read further if this is not your case!

Python Version
--------------

RocketPy supports Python 3.8 and above.
Sorry, there are currently no plans to support earlier versions.
If you really need to run RocketPy on Python 3.7 or earlier, feel free to submit an issue and we will see what we can do!

Required Packages
-----------------

The following packages are needed in order to run RocketPy:

- requests
- Numpy >= 1.13
- Scipy >= 1.0
- Matplotlib >= 3.0
- netCDF4 >= 1.6.4
- windrose >= 1.6.8
- requests
- pytz
- simplekml

All of these packages, are automatically installed when RocketPy is installed using either ``pip`` or ``conda``.
However, in case the user wants to install these packages manually, they can do so by following the instructions bellow.

Installing Required Packages Using ``pip``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The packages needed can be installed via ``pip`` by running the following lines of code in your preferred terminal, assuming pip is added to the PATH:

.. code-block:: shell

    pip install "numpy>=1.13" 
    pip install "scipy>=1.0"
    pip install "matplotlib>=3.0"
    pip install "netCDF4>=1.6.4"
    pip install requests
    pip install pytz
    pip install simplekml

Installing Required Packages Using ``conda``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numpy, Scipy, Matplotlib and requests come with Anaconda, but Scipy might need updating.
The nedCDF4 package can be installed if there is interest in importing weather data from netCDF files.
To update Scipy and install netCDF4 using Conda, the following code is used:

.. code-block:: shell

    conda install "scipy>=1.0"
    conda install -c anaconda "netcdf4>=1.6.4"


Optional Packages
-----------------

The EnvironmentAnalysis class requires a few extra packages to be installed.
In case you want to use this class, you will need to install the following packages:

- `timezonefinder` : to allow for automatic timezone detection,
- `windrose` : to allow for windrose plots,
- `ipywidgets` : to allow for GIFs generation,
- `jsonpickle` : to allow for saving and loading of class instances.

You can install all these packages by simply running the following lines in your preferred terminal:

.. code-block:: shell

    pip install rocketpy[env_analysis]


Alternatively, you can instal all extra packages by running the following line in your preferred terminal:

.. code-block:: shell

    pip install rocketpy[all]
    

Useful Packages
---------------

Although `Jupyter Notebooks <http://jupyter.org/>`_ are by no means required to run RocketPy, they can be a handy tool!
All of are examples are written using Jupyter Notebooks so that you can follow along easily.
They already come with Anaconda builds, but can also be installed separately using pip:

.. code-block:: shell

    pip install jupyter
