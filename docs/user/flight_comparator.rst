Flight Comparator
=================

This example demonstrates how to use the RocketPy ``FlightComparator`` class.
This class is designed to compare a RocketPy simulation against external data sources,
such as:

- Real flight data (avionics logs, altimeter CSVs)
- Simulations from other software (OpenRocket, RASAero)
- Theoretical models or manual calculations

Unlike ``CompareFlights`` (which compares multiple RocketPy simulations), ``FlightComparator``
specifically handles the challenge of aligning different time steps and calculating
error metrics (RMSE, MAE, etc.) between a "Reference" simulation and "External" data.

Importing classes
-----------------

We will start by importing the necessary classes and modules:

.. jupyter-execute::

    import numpy as np
    from rocketpy import Environment, Rocket, SolidMotor, Flight

Create Simulation (Reference)
-----------------------------

First, let's create the standard RocketPy simulation that will serve as our "Ground Truth"
or reference model. This follows the standard setup.

.. jupyter-execute::

    # 1. Setup Environment
    env = Environment(date=(2022, 10, 1, 12), latitude=32.990254, longitude=-106.974998, elevation=1400)
    env.set_atmospheric_model(type='standard_atmosphere')

    # 2. Setup Motor
    Pro75M1670 = SolidMotor(
        thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
        burn_time=3.9,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        grains_center_of_mass_position=0.33,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
    )

    # 3. Setup Rocket
    calisto = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="../data/calisto/powerOffDragCurve.csv",
        power_on_drag="../data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    calisto.set_rail_buttons(0.0818, -0.618, 45)
    calisto.add_motor(Pro75M1670, position=-1.255)

    # Add aerodynamic surfaces
    nosecone = calisto.add_nose(length=0.55829, kind="vonKarman", position=0.71971)
    fin_set = calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.040,
        span=0.100,
        position=-1.04956,
        cant_angle=0.5,
        airfoil=("../data/calisto/fins/NACA0012-radians.txt", "radians"),
    )
    tail = calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

    # 4. Simulate
    flight = Flight(rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0)


Importing External Data
-----------------------

In this example, external data is generated synthetically, but in practice you
would typically load it from CSV or similar sources and pass it to
``FlightComparator.add_data`` or ``Flight.compare`` as dictionaries of
``(time_array, data_array)``.

The data format required is a dictionary where keys are variable names (e.g. 'z', 'vz')
and values are tuples of ``(time_array, data_array)``.

.. jupyter-execute::

    # Generate fake sensor data (Simulation + Noise + Drift)
    sensor_time = np.linspace(0, flight.t_final, 200) # Lower frequency than simulation
    
    # Altitude with some error
    sensor_alt = flight.z(sensor_time) * 0.95 + np.random.normal(0, 5, 200)
    
    # Velocity with some noise
    sensor_vz = flight.vz(sensor_time) + np.random.normal(0, 2, 200)

    # Prepare the dictionary
    flight_data = {
        "altitude": (sensor_time, sensor_alt),
        "vz": (sensor_time, sensor_vz)
    }


Start the Comparison
--------------------

We can initialize the comparison directly from the Flight object using the helper method.

.. jupyter-execute::

    # Initialize and add data in one step
    comparator = flight.compare(flight_data, label="Altimeter Log")


Comparison Summary
------------------

To get a quick overview of how accurate your simulation was compared to the data,
use the summary method. This prints error metrics (MAE, RMSE) and compares key events
like Apogee and Max Velocity.

.. jupyter-execute::

    # Access key event metrics programmatically
    key_events = comparator.compare_key_events()
    key_events  # dict with apogee, max velocity, impact velocity comparisons


Visualizing the Difference
--------------------------

You can plot specific variables to see the trajectory and the residuals (error) over time.

.. jupyter-execute::

    # Compare Altitude
    comparator.compare("altitude")

.. jupyter-execute::

    # Compare Vertical Velocity
    comparator.compare("vz")


Comparing 2D Trajectories
-------------------------

If you have spatial data (e.g. GPS coordinates), you can visualize the flight path deviation.
Here we add some fake X-position data to demonstrate.
Coordinates are plotted in the inertial frame used by Flight, where x is East, y is North and z is Up.


.. jupyter-execute::

    # Add GPS data (Drifting further East than simulated)
    gps_x = flight.x(sensor_time) + np.linspace(0, 200, 200) 
    
    # Add this new source to the existing comparator
    comparator.add_data("GPS Log", {
        "x": (sensor_time, gps_x),
        "z": (sensor_time, sensor_alt)
    })

    # Plot X vs Z trajectory
    comparator.trajectories_2d(plane="xz")


Comparing Multiple Sources
--------------------------

The ``FlightComparator`` can handle multiple datasets at once (e.g. comparing against OpenRocket AND RasAero).

.. jupyter-execute::

    # Add another "simulation" source
    openrocket_alt = flight.z(sensor_time) * 1.05 # Over-predicts by 5%
    
    comparator.add_data("OpenRocket", {
        "altitude": (sensor_time, openrocket_alt)
    })

    # Compare all of them on one plot
    comparator.compare("altitude")