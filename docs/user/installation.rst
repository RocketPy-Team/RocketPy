Installation & Requirements
===========================

Installation
------------

Quick Install Using ``pip``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get a copy of RocketPy's latest stable version using ``pip``, just open up your terminal and run:

.. code-block:: shell

    pip install rocketpy

If you don't see any error messages, you are all set!

If you want to choose a specific version to guarantee compatibility, you may instead run:

.. code-block:: shell

    pip install rocketpy==1.10.0


Optional Installation Method: ``conda``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, RocketPy can also be installed using ``conda`` and the `Conda-Forge <https://conda-forge.org/>`_ channel.
Just open your Anaconda terminal and run:

.. code-block:: shell

    conda install -c conda-forge rocketpy


Optional Installation Method: from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to download RocketPy from source, you may do so either by:

- Downloading it from `RocketPy's GitHub <https://github.com/RocketPy-Team/RocketPy>`_ page and unzipping the downloaded folder.
- Or cloning it to a desired directory using git:

.. code-block:: shell

    git clone https://github.com/RocketPy-Team/RocketPy.git

Once you are done downloading/cloning RocketPy's repository, you can install it by opening up a terminal inside the repository's folder on your computer and running:

.. code-block:: shell

    python -m pip install .


Development version
-------------------

Using ``pip``
^^^^^^^^^^^^^

RocketPy is being actively developed, which means we have stable versions and development versions.
All methods above will install a stable version to your computer.
If you want to get the latest development version, you also can!
And it's as simple as using ``pip``.
Just open up your terminal and run:

.. code-block:: shell

    pip install git+https://github.com/RocketPy-Team/RocketPy.git@develop


Cloning RocketPy's Repo
^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can clone RocketPy's repository, check out the branch named ``develop`` and proceed with:

.. code-block:: shell

    python -m pip install -e .


Requirements
------------

Python Version
^^^^^^^^^^^^^^

RocketPy supports Python 3.10 and above.
Sorry, there are currently no plans to support earlier versions.
If you really need to run RocketPy on Python 3.8 or earlier, feel free to submit an issue and we will see what we can do!

Required Packages
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

Although `Jupyter Notebooks <http://jupyter.org/>`_ are by no means required to run RocketPy, they can be a handy tool!
All of are examples are written using Jupyter Notebooks so that you can follow along easily.
They already come with Anaconda builds, but can also be installed separately using pip:

.. code-block:: shell

    pip install jupyter
