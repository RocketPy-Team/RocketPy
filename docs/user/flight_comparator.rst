Flight Comparator
=================

This example demonstrates how to use the RocketPy ``FlightComparator`` class to
compare a Flight simulation against external data sources.

Users must explicitly create a `FlightComparator` instance.


This class is designed to compare a RocketPy Flight simulation against external
data sources, such as:

- Real flight data (avionics logs, altimeter CSVs)
- Simulations from other software (OpenRocket, RASAero)
- Theoretical models or manual calculations

Unlike ``CompareFlights`` (which compares multiple RocketPy simulations),
``FlightComparator`` specifically handles the challenge of aligning different
time steps and calculating error metrics (RMSE, MAE, etc.) between a
"Reference" simulation and "External" data.

Importing classes
-----------------

We will start by importing the necessary classes and modules:

.. jupyter-execute::

   import numpy as np

   from rocketpy import Environment, Rocket, SolidMotor, Flight
   from rocketpy.simulation import FlightComparator, FlightDataImporter

Create Simulation (Reference)
-----------------------------

First, let's create the standard RocketPy simulation that will serve as our
"Ground Truth" or reference model. This follows the standard setup.

.. jupyter-execute::

   # 1. Setup Environment
   env = Environment(
       date=(2022, 10, 1, 12),
       latitude=32.990254,
       longitude=-106.974998,
       elevation=1400,
   )
   env.set_atmospheric_model(type="standard_atmosphere")

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
       top_radius=0.0635,
       bottom_radius=0.0435,
       length=0.060,
       position=-1.194656,
   )

    # 4. Simulate
    flight = Flight(
    rocket=calisto,
    environment=env,
    rail_length=5.2,
    inclination=85,
    heading=0,
    )

    # 5. Create FlightComparator instance
    comparator = FlightComparator(flight)

Adding Another Flight Object
----------------------------

You can compare against another RocketPy Flight simulation directly:

.. jupyter-execute::

    # Create a second simulation with slightly different parameters
    flight2 = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.0,  # Slightly shorter rail
        inclination=85,
        heading=0,
    )

    # Add the second flight directly
    comparator.add_data("Alternative Sim", flight2)

    print(f"Added variables from flight2: {list(comparator.data_sources['Alternative Sim'].keys())}")

Importing External Data (dict)
------------------------------

The primary data format expected by ``FlightComparator.add_data`` is a dictionary
where keys are variable names (e.g. ``"z"``, ``"vz"``, ``"altitude"``) and values
are either:

- A RocketPy ``Function`` object, or
- A tuple of ``(time_array, data_array)``.

Let's create some synthetic external data to compare against our reference
simulation:

.. jupyter-execute::

    # Generate synthetic external data with realistic noise
    time_external = np.linspace(0, flight.t_final, 80)  # Different time steps
    external_altitude = flight.z(time_external) + np.random.normal(0, 3, 80)  # 3m noise
    external_velocity = flight.vz(time_external) + np.random.normal(0, 0.5, 80)  # 0.5 m/s noise

    # Add the external data to our comparator
    comparator.add_data(
        "External Simulator", 
        {
            "altitude": (time_external, external_altitude),
            "vz": (time_external, external_velocity),
        }
    )

Running Comparisons
-------------------

Now we can run the various comparison methods:

.. jupyter-execute::

    # Generate summary with key events
    comparator.summary()

    # Compare specific variable
    comparator.compare("altitude")

    # Compare all available variables
    comparator.all()

    # Plot 2D trajectory
    comparator.trajectories_2d(plane="xz")
