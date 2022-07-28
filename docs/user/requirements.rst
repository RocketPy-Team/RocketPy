Requirements
============

Usually, you can skip this part if you have an updated version of Python running on a normal computer.
But read further if this is not your case!

Python Version
--------------

RocketPy was made to run on Python 3.6+.
Sorry, there are currently no plans to support earlier versions.
If you really need to run RocketPy on Python 3.5 or earlier, feel free to submit an issue and we will see what we can do!

Required Packages
-----------------

The following packages are needed in order to run RocketPy:

- requests
- Numpy >= 1.0
- Scipy >= 1.0
- Matplotlib >= 3.0
- netCDF4 >= 1.4 (optional, requires Cython)
- windrose >= 1.6.8
- requests
- pytz
- timezonefinder
- simplekml
- ipywidgets >= 7.6.3

 
All of these packages, with the exception of netCDF4, should be automatically installed when RocketPy is installed using either ``pip`` or ``conda``.
However, in case the user wants to install these packages manually, they can do so by following the instructions bellow.

Installing Required Packages Using ``pip``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The packages needed can be installed via ``pip`` by running the following lines of code in your preferred terminal, assuming pip is added to the PATH:

.. code-block:: shell

    pip install "numpy>=1.0" 
    pip install "scipy>=1.0"
    pip install "matplotlib>=3.0"
    pip install "netCDF4>=1.4"
    pip install "windrose >= 1.6.8"
    pip install "ipywidgets>=7.6.3"
    pip install requests
    pip install pytz
    pip install timezonefinder
    pip install simplekml

Installing Required Packages Using ``conda``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numpy, Scipy, Matplotlib and requests come with Anaconda, but Scipy might need updating.
The nedCDF4 package can be installed if there is interest in importing weather data from netCDF files.
To update Scipy and install netCDF4 using Conda, the following code is used:

.. code-block:: shell

    conda install "scipy>=1.0"
    conda install -c anaconda "netcdf4>=1.4"

Useful Packages
---------------

Although `Jupyter Notebooks <http://jupyter.org/>`_ are by no means required to run RocketPy, they can be a handy tool!
All of are examples are written using Jupyter Notebooks so that you can follow along easily.
They already come with Anaconda builds, but can also be installed separately using pip:

.. code-block:: shell

    pip install jupyter
