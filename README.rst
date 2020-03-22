RocketPy
========

RocketPy is a trajectory simulation for High-Power Rocketry built by
`Projeto Jupiter <https://www.facebook.com/ProjetoJupiter/>`__. The code
is written as a `Python <http://www.python.org>`__ library and allows
for a complete 6 degrees of freedom simulation of a rocket's flight
trajectory, including high fidelity variable mass effects as well as
descent under parachutes. Weather conditions, such as wind profile, can
be imported from sophisticated datasets, allowing for realistic
scenarios. Furthermore, the implementation facilitates complex
simulations, such as multi-stage rockets, design and trajectory
optimization and dispersion analysis.

Previewing
----------

You can preview RocketPy's main functionalities by browsing through a
`sample
notebook <https://mybinder.org/v2/gh/giovaniceotto/RocketPy/master?filepath=docs%2notebooks%2Fgetting_started.ipynb>`__!

Then, you can read the *Getting Started* section to get your own copy!

Getting Started
---------------

These instructions will get you a copy of RocketPy up and running on
your local machine.

Prerequisites
~~~~~~~~~~~~~

The following is needed in order to run RocketPy:

-  Python >= 3.0
-  Numpy >= 1.0
-  Scipy >= 1.0
-  Matplotlib >= 3.0
-  netCDF4 >= 1.4 (optional, requires Cython)

The first 4 prerequisites come with Anaconda, but Scipy might need
updating. The nedCDF4 package can be installed if there is interest in
importing weather data from netCDF files. To update Scipy and install
netCDF4 using Conda, the following code is used:

::

    $ conda install "scipy>=1.0"
    $ conda install -c anaconda "netcdf4>=1.4"

Alternatively, if you only have Python 3.X installed, the packages
needed can be installed using pip:

::

    $ pip install "numpy>=1.0"
    $ pip install "scipy>=1.0"
    $ pip install "matplotlib>=3.0"
    $ pip install "netCDF4>=1.4"

Although `Jupyter Notebooks <http://jupyter.org/>`__ are by no means
required to run RocketPy, they are strongly recommend. They already come
with Anaconda builds, but can also be installed separately using pip:

::

    $ pip install jupyter

Installation
~~~~~~~~~~~~

To get a copy of RocketPy, just run:

::

    $ pip install "rocketpyalpha"

Alternatively, you may want to downloaded from sorce:

-  Download it from `RocketPy's
   GitHub <https://github.com/giovaniceotto/RocketPy>`__ page

   -  Unzip the folder and you are ready to go

-  Or clone it to a desired directory using git:

   -  ``$ git clone https://github.com/giovaniceotto/RocketPy.git``

The repository comes with the following content:

-  Files
-  README.md
-  LICENSE.md
-  setup.py
-  Folders
-  rocketpyalpha - Python Library
-  data - Input data for the simulation, such as motor thrust curves and
   wind profiles.
-  disp - Example of dispersion analysis, but needs to be refined.
-  docs - Documentation available about the physics models used.
-  nbks - Main python library, some example notebooks and other random
   files which will soon be cleaned up.

The RockeyPy library can then be installed by running:

::

    $ python setup.py install 

Running Your First Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to run your first rocket trajectory simulation using RocketPy,
you can start a Jupyter Notebook and navigate to the ***nbks*** folder.
Open ***Getting Started - Examples.ipynb*** and you are ready to go.

Otherwise, you may want to create your own script or your own notebook
using RocketPy. To do this, let's see how to use RocketPy's four main
classes:

-  Environment - Keeps data related to weather.
-  SolidMotor - Keeps data related to solid motors. Hybrid motor suport
   is coming in the next weeks.
-  Rocket - Keeps data related to a rocket.
-  Flight - Runs the simulation and keeps the results.

A typical workflow starts with importing these classes from RocketPy:

.. code:: python

    from rocketpy import Environment, Rocket, SolidMotor, Flight

Then create an Environment object. To learn more about it, you can use:

.. code:: python

    help(Environment)

A sample code is:

.. code:: python

    Env = Environment(
        railLength=5.2,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        date=(2020, 3, 4, 12) # Tomorrow's date in year, month, day, hour UTC format
    ) 

    Env.setAtmosphericModel(type='Forecast', file='GFS')

This can be followed up by starting a Solid Motor object. To get help on
it, just use:

.. code:: python

    help(SolidMotor)

A sample Motor object can be created by the following code:

.. code:: python

    Pro75M1670 = SolidMotor(
        thrustSource="../data/motors/Cesaroni_M1670.eng",
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5/1000,
        grainDensity=1815,
        grainOuterRadius=33/1000,
        grainInitialInnerRadius=15/1000,
        grainInitialHeight=120/1000,
        nozzleRadius=33/1000,
        throatRadius=11/1000,
        interpolationMethod='linear'
    )

With a Solid Motor defined, you are ready to create your Rocket object.
As you may have guessed, to get help on it, use:

.. code:: python

    help(Rocket)

A sample code to create a Rocket is:

.. code:: python

    Calisto = Rocket(
        motor=Pro75M1670,
        radius=127/2000,
        mass=19.197-2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag='../data/calisto/powerOffDragCurve.csv',
        powerOnDrag='../data/calisto/powerOnDragCurve.csv'
    )

    Calisto.setRailButtons([0.2, -0.5])

    NoseCone = Calisto.addNose(length=0.55829, kind="vonKarman", distanceToCM=0.71971)

    FinSet = Calisto.addFins(4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956)

    Tail = Calisto.addTail(topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656)

You may want to add parachutes to your rocket as well:

.. code:: python

    def drogueTrigger(p, y):
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        return True if y[5] < 0 and y[2] < 800 else False

    Main = Calisto.addParachute('Main',
                                CdS=10.0,
                                trigger=mainTrigger, 
                                samplingRate=105,
                                lag=1.5,
                                noise=(0, 8.3, 0.5))

    Drogue = Calisto.addParachute('Drogue',
                                  CdS=1.0,
                                  trigger=drogueTrigger, 
                                  samplingRate=105,
                                  lag=1.5,
                                  noise=(0, 8.3, 0.5))

Finally, you can create a Flight object to simulate your trajectory. To
get help on the Flight class, use:

.. code:: python

    help(Flight)

To actually create a Flight object, use:

.. code:: python

    TestFlight = Flight(rocket=Calisto, environment=Env, inclination=85, heading=0)

Once the TestFlight object is created, your simulation is done! Use the
following code to get a summary of the results:

.. code:: python

    TestFlight.info()

To seel all available results, use:

.. code:: python

    TestFlight.allInfo()

Built With
----------

-  `Numpy <http://www.numpy.org/>`__
-  `Scipy <https://www.scipy.org/>`__
-  `Matplotlib <https://matplotlib.org/>`__
-  `netCDF4 <https://github.com/Unidata/netcdf4-python>`__

Contributing
------------

Please read
`CONTRIBUTING.md <https://github.com/giovaniceotto/RocketPy/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us. - ***Still working on this!***

Versioning
----------

***Still working on this!***

Authors
-------

-  **Giovani Hidalgo Ceotto**

See also the list of
`contributors <https://github.com/giovaniceotto/RocketPy/contributors>`__
who participated in this project.

License
-------

This project is licensed under the MIT License - see the
`LICENSE.md <https://github.com/giovaniceotto/RocketPy/blob/master/LICENSE>`__
file for details

Acknowledgments
---------------

***Still working on this!***
