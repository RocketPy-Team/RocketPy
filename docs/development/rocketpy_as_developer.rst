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

You may create a testing file in any directory, but you must remember that they should not be included in the commits and pull requests unless they are part of the proposed solution.
With that in mind, we suggest you to create a folder with all testing files, so they can be added in the .gitignore file, which contains the name of all the files and folders that will not be added to the commits. To create the folder, type on the terminal:

.. code-block:: console

    mkdir <folder name>

And, to add it on .gitignore, type:

.. code-block:: console
    
    echo <folder name>/ >> .gitignore

It is important to remember that all the files inside this folder will not be included in any commit so, if it is important to the solution, do not add them inside it.

Running a simulation with RocketPy
==================================

Importing the RocketPy files
----------------------------

First, create a python (or .ipynb) file to make the simulation.
To ensure you are using the local files and not the files as a python package (if you installed the library via pip for example), add 

.. code-block:: python

    pip install -e .

Alternatively you can use the following command to pip install the local library:

.. code-block:: console
    
    import sys
    sys.path.append('../') # if you are using a notebook
    sys.path.append('../rocketpy') # if you are using a script

Import the classes that will be used, in case:

.. code-block:: python
    
    from rocketpy import Environment, SolidMotor, Rocket, Flight, Function

If it is the first time you are using rocketpy and you do not have all required libraries installed, you could use the command:

.. code-block:: python

    pip install -r </path/to/requirements.txt>

Alternatively, if you are in rocketpy folder, just type

.. code-block:: python

    pip install -r requirements.txt

Creating an Environment
-----------------------

Here we create the environment object that will be used in the simulation.
It contains information about the local pressure profile, temperature, speed of sound, wind direction and intensity, etc.

.. code-block:: python

    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

RocketPy can use local files via the Ensemble method or meteorological forecasts through OpenDAP protocol. 
To work with environment files, it will be very important ensuring tha that you have the netCDF4 library installed.
Assuming we are using forecast, first we set the simulated data with:

.. code-block:: python

    import datetime
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))  # Hour given in UTC time

Then we set the atmospheric model, in this case, GFS forecast:

.. code-block:: python

    env.set_atmospheric_model(type="Forecast", file="GFS")

Weather forecast data can be visualized through two info methods.

``env.info()`` or ``env.all_info()``

Creating the motor that boosts the rocket
-----------------------------------------

Now we need to create the motor. 
For example, we will use a solid motor called Pro75M1670, but other configurations are also possible.
The motor class contains information about the thrust curve and uses some geometric parameters to calculate the mass variation over time, as well as the total thrust and other important outputs.

.. code-block:: python

    Pro75M1670 = SolidMotor(
        thrust_source="../data/motors/Cesaroni_M1670.eng", #copy here the path to the thrust source file
        burn_time=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
    )

Motor data can be visualized through the following methods:

``Pro75M1670.info()`` or ``Pro75M1670.all_info()``


Creating the rocket
-------------------

The Rocket class contains all information about the rocket that are necessary to the simulation, including the motor, rocket mass and inertia, aerodynamic surfaces, parachutes, etc.
The first step is to initialize the class with the vital data:

.. code-block:: python

    calisto = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia_i=6.60,
        inertia_z=0.0351,
        power_off_drag="../../data/calisto/powerOffDragCurve.csv",
        power_on_drag="../../data/calisto/powerOnDragCurve.csv",
        center_of_dry_mass_position=0,
        coordinate_system_orientation="tail_to_nose",
    )

    calisto.add_motor(Pro75M1670, position=-1.255)

Then the rail buttons must be set:

.. code-block:: python
    
    calisto.set_rail_buttons(0.2, -0.5)

In sequence, the aerodynamic surfaces must be set.
If a lift curve for the fin set is not specified, it is assumed that they behave according to a linearized model with a coefficient calculated with Barrowman's theory.
In the example, a nosecone, one fin set and one tail were added, but each case can be designed differently.

.. code-block:: python

    nosecone = calisto.add_nose(length=0.55829, kind="vonKarman", position=0.71971 + 0.55829)

    fin_set = calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.040,
        span=0.100,
        position=-1.04956,
        cant_angle=0,
        radius=None,
        airfoil=None,
    )

    tail = calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

If you are considering the parachutes in the simulation, they also have to be added to the rocket object.
A trigger function must be supplied to trigger the parachutes.
Currently, the pressure `(p)`, the height above ground level considering noise `(h)`, and the state-space variables `(y)` are necessary inputs for the function.
The state-space contains information about the rocket's position and velocities (translation and rotation).
For example:

.. code-block:: python

    def drogue_trigger(p, h, y):
        # p = pressure considering parachute noise signal
        # h = height above ground level considering parachute noise signal
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]

        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False


    def main_trigger(p, h, y):
        # p = pressure considering parachute noise signal
        # h = height above ground level considering parachute noise signal
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]

        # activate main when vz < 0 m/s and z < 800 m
        return True if y[5] < 0 and h < 800 else False

After having the trigger functions defined, the parachute must be added to the rocket:

.. code-block:: python

    Main = calisto.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=main_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = calisto.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger=drogue_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

Simulating the flight
---------------------

Finally, the flight can be simulated with the provided data.
The rocket and environment classes are supplied as inputs, as well as the rail length, inclination and heading angle.

.. code-block:: python

    test_flight = Flight(rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0)

Flight data can be retrieved through:

``test_flight.info()`` or ``test_flight.all_info()``

This function plots a comprehensive amount of flight data and graphs but, if you want to access one specific variable, for example Z position, this may be achieved by `test_flight.z`.
If you insert `test_flight.z()` the graph of the function will be plotted.
This and other features can be found in the documentation of the `Function` class, which allows data to be treated in an easier way.
The documentation of each variable used in the class can be found on `Flight.py` file.

Further considerations
======================

RocketPy's classes documentation can be accessed in code via `help(<name of the class>)` command.
For example, to access Flight class parameters, you can use:

.. code-block:: python

    help(Flight)

More documentation materials can be found at `read the docs <https://docs.rocketpy.org/en/latest/?badge=latest>`_.
It can also be found on RocketPy's GitHub page on the badge "docs".