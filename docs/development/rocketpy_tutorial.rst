========================
Introduction to RocketPy
========================

This tutorial part shows how to open rocketpy files and run a simple simulation.

Opening rocketpy folder
=======================

Go into the cloned repository folder by typing on a terminal

.. code-block:: console

    cd <rocketpy directory>

Open your preference editor by typing on a terminal

.. code-block:: console
    
    <editor name> .

For example, to open VS Code type on terminal

.. code-block:: console
    
    code .

Alternatively, you can open the folder directly through your editor's interface.

Preparing directory for code editing
====================================

You may create a testing file in any directory, but you must remember that they should not be included in the commits and pull requests unless they are part of the solution proposed. With that in mind, we suggest the creation of a folder with all testing files, so they can be added on .gitignore file (this file contains the name of all the files and folders that should not be added to the commits). To create the folder, type on the terminal:

.. code-block:: console

    mkdir <folder name>

And to add it on .gitignore type:

.. code-block:: console
    
    echo <folder name>/ >> .gitignore

It is important to remember that all files inside this folder will not be included in commits so, if it is important to the solution, do not add them inside it.

Running a simulation with RocketPy
==================================

Importing the RocketPy files
----------------------------

First, create a python (or .ipynb) file to make the simulation. To ensure you are using the local files and not the files as a python package (if you installed the library via pip for example), add 

.. code-block:: python

    import sys
    sys.path.append('../') # if you are using a notebook
    sys.path.append('../rocketpy') # if you are using a script

Alternatively you can use the following command to pip install the local library:

.. code-block:: console
    
    pip install -e .  

Import the classes that will be used, in case:

.. code-block:: python
    
    from rocketpy import Environment, SolidMotor, Rocket, Flight, Function

If it is the first time you are using rocketpy and you do not have all auxiliary libraries installed, you could use the command:

.. code-block:: python

    pip install -r </path/to/requirements.txt>

For example, if you are in rocketpy folder, just type

.. code-block:: python

    pip install -r requirements.txt

Creating an Environment
-----------------------

Here we create the environment that will be used in the simulation. It contains information about local pressure, wind direction and intensity, etc.

.. code-block:: python

    Env = Environment(railLength=5.2, latitude=32.990254, longitude=-106.974998, elevation=1400)

RocketPy can use local files via the Ensemble method or forecasts through netCDF4 library. Assuming we are using forecast, first we set the simulated data with:

.. code-block:: python

    import datetime
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    Env.setDate((tomorrow.year, tomorrow.month, tomorrow.day, 12))  # Hour given in UTC time

Then we set the atmospheric model, in case, GFS forecast:

.. code-block:: python

    Env.setAtmosphericModel(type="Forecast", file="GFS")

Weather forecast data can be visualized through

``Env.info()`` or ``Env.allInfo()``

Creating the motor that boosts the rocket
-----------------------------------------

Now you need to create the motor. For example, we will use a solid motor called Pro75M1670, but other configurations are also possible, they are actually part of the challenges. The motor class contains information about the thrust curve and uses some geometric parameters to calculate the mass variation in time, total thrust, etc.

.. code-block:: python

    Pro75M1670 = SolidMotor(
        thrustSource="../data/motors/Cesaroni_M1670.eng",
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        interpolationMethod="linear",
    )

Motor data can be visualized through

``Pro75M1670.info()`` or ``Pro75M1670.allInfo()``


Creating the rocket
-------------------

The Rocket class contains all information about the rocket that is necessary to the simulation, that includes the motor, mass, inertias, aerodynamic surfaces, parachutes, etc. The first step is to initialize the class with the vital data

.. code-block:: python

    Calisto = Rocket(
        motor=Pro75M1670,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag="../../data/calisto/powerOffDragCurve.csv",
        powerOnDrag="../../data/calisto/powerOnDragCurve.csv",
    )

Then the rail buttons must be set

.. code-block:: python
    
    Calisto.setRailButtons([0.2, -0.5])

In sequence, the aerodynamic surfaces must be set. If a lift curve for the fin set is not specified, it is assumed that they behave according to a linearized model with a coefficient calculated with Barrowman's theory. In the example, a nosecone, one finset and one tail were added (could be more)

.. code-block:: python

    NoseCone = Calisto.addNose(length=0.55829, kind="vonKarman", distanceToCM=0.71971)

    FinSet = Calisto.addFins(4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956)

    Tail = Calisto.addTail(topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656)

If you are considering the parachutes in the simulation, they also have to be added to the class. A trigger function must be supplied to trigger the parachutes. Currently, the pressure `(p)` and the state-space variables `(y)` are necessary inputs for the function. The state-space contains information about the rocket's position and velocities (translation and rotation). For example:

.. code-block:: python

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False


    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 + 1400 m (+1400 due to surface elevation).
        return True if y[5] < 0 and y[2] < 800 + 1400 else False

Having the trigger functions defined, the parachute must be added to the rocket:

.. code-block:: python

    Main = Calisto.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = Calisto.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

Simulating the flight
--------------------

Finally, the flight can be simulated with the provided data. The rocket and environment classes are supplied as inputs, such as the rail inclination and heading angle.

.. code-block:: python

    TestFlight = Flight(rocket=Calisto, environment=Env, inclination=85, heading=0)

Flight data can be retrieved through:

``TestFlight.info()`` or ``TestFlight.allInfo()``

This function prints a comprehensive ammount of flight data and graphs but, if you want to access one specific variable, for exemple Z position, this may be achieved by `TestFlight.z`. If you insert `TestFlight.z()` the graph of the function will be plotted. This and other features can be found in the documentation of the `Function` class, which is a class that allows data to be treated as actual handle functions. The documentation of each variable used in the class can be found on `Flight.py` file.

Further questions
=================

RocketPy's classes documentation can be accessed in code via `help(<name of the class>)` command. For example, to access Flight class parameters type

.. code-block:: python

    help(Flight)

More documentation materials can be found at [read the docs](https://docs.rocketpy.org/en/latest/?badge=latest). It can also be found on RocketPy's GitHub page on the badge "docs".