.. _flightcomparator:

Flight Comparator
=================

The :class:`rocketpy.simulation.FlightComparator` class enables users to compare
a RocketPy Flight simulation against external data sources, providing error
metrics and visualization tools for validation and analysis.

.. seealso::

    For comparing multiple RocketPy simulations together, see the
    :doc:`Compare Flights </user/compare_flights>` guide.

Overview
--------

This class is designed to compare a RocketPy Flight simulation against external
data sources, such as:

- **Real flight data** - Avionics logs, altimeter CSVs, GPS tracking data
- **Other simulators** - OpenRocket, RASAero, or custom simulation tools
- **Theoretical models** - Analytical solutions or manual calculations

Unlike :class:`rocketpy.plots.compare.CompareFlights` (which compares multiple
RocketPy simulations), ``FlightComparator`` specifically handles the challenge
of aligning different time steps and calculating error metrics (RMSE, MAE, etc.)
between a "Reference" simulation and "External" data.

Key Features
------------

- Automatic time-step alignment between different data sources
- Error metric calculations (RMSE, MAE, etc.)
- Side-by-side visualization of flight variables
- Support for multiple external data sources
- Direct comparison with other RocketPy Flight objects


Creating a Reference Simulation
--------------------------------

First, import the necessary classes and modules:

Importing Classes
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import numpy as np
   from rocketpy import Environment, Rocket, SolidMotor, Flight
   from rocketpy.simulation import FlightComparator, FlightDataImporter

Setting Up the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's create the standard RocketPy simulation that will serve as our
reference model. This follows the same pattern as in :ref:`firstsimulation`.

.. jupyter-execute::

   # Create Environment (Spaceport America, NM)
   env = Environment(
       date=(2022, 10, 1, 12),
       latitude=32.990254,
       longitude=-106.974998,
       elevation=1400,
   )
   env.set_atmospheric_model(type="standard_atmosphere")

Setting Up the Motor
~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Create a Cesaroni M1670 Solid Motor
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

Setting Up the Rocket
~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Create Calisto rocket
   calisto = Rocket(
       radius=127 / 2000,
       mass=19.197 - 2.956,
       inertia=(6.321, 6.321, 0.034),
       power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
       power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
       center_of_mass_without_motor=0,
       coordinate_system_orientation="tail_to_nose",
   )

   # Add rail buttons
   calisto.set_rail_buttons(0.0818, -0.618, 45)

   # Add motor to rocket
   calisto.add_motor(Pro75M1670, position=-1.255)

   # Add aerodynamic surfaces
   nosecone = calisto.add_nose(
       length=0.55829,
       kind="vonKarman",
       position=0.71971,
   )

   fin_set = calisto.add_trapezoidal_fins(
       n=4,
       root_chord=0.120,
       tip_chord=0.040,
       span=0.100,
       position=-1.04956,
       cant_angle=0.5,
       airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
   )

   tail = calisto.add_tail(
       top_radius=0.0635,
       bottom_radius=0.0435,
       length=0.060,
       position=-1.194656,
   )

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Create and run flight simulation
   flight = Flight(
       rocket=calisto,
       environment=env,
       rail_length=5.2,
       inclination=85,
       heading=0,
   )

Creating the FlightComparator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Initialize FlightComparator with reference flight
   comparator = FlightComparator(flight)


Adding Comparison Data
----------------------

Comparing with Another Flight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can compare against another RocketPy :class:`rocketpy.Flight` object directly:

.. jupyter-execute::

    # Create a second simulation with slightly different parameters
    flight2 = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.0,  # Slightly shorter rail
        inclination=85,
        heading=0,
    )

    # Add the second flight to comparator
    comparator.add_data("Alternative Sim", flight2)

Comparing with External Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary data format expected by ``FlightComparator.add_data`` is a dictionary
where keys are variable names (e.g., ``"z"``, ``"vz"``, ``"altitude"``) and values
are either:

- A RocketPy :class:`rocketpy.Function` object, or
- A tuple of ``(time_array, data_array)``

Let's create some synthetic external data to demonstrate the comparison process:

.. jupyter-execute::

    # Generate synthetic external data with realistic noise
    time_external = np.linspace(0, flight.t_final, 80)  # Different time steps
    external_altitude = flight.z(time_external) + np.random.normal(0, 3, 80)  # Add 3 m noise
    external_velocity = flight.vz(time_external) + np.random.normal(0, 0.5, 80)  # Add 0.5 m/s noise

    # Add external data to comparator
    comparator.add_data(
        "External Simulator",
        {
            "altitude": (time_external, external_altitude),
            "vz": (time_external, external_velocity),
        }
    )

.. tip::
    External data does not need to have the same time steps as the reference
    simulation. The ``FlightComparator`` automatically handles interpolation
    and alignment for comparison.


Analyzing Comparison Results
-----------------------------

Summary Report
~~~~~~~~~~~~~~

Generate a comprehensive summary of the comparison:

.. jupyter-execute::

    # Display comparison summary with key flight events
    comparator.summary()

Comparing Specific Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can compare individual flight variables:

.. jupyter-execute::

    # Compare altitude data across all sources
    comparator.compare("altitude")

The comparison plot shows the reference simulation alongside all external data
sources, making it easy to identify discrepancies and validate the simulation.

Comparing All Variables
~~~~~~~~~~~~~~~~~~~~~~~~

To get a complete overview, compare all available variables at once:

.. jupyter-execute::

    # Generate plots for all common variables
    comparator.all()

Trajectory Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize 2D trajectory projections for spatial comparison:

.. jupyter-execute::

    # Plot trajectory in the XZ plane (side view)
    comparator.trajectories_2d(plane="xz")

.. seealso::

    For detailed API documentation and additional methods, see
    :class:`rocketpy.simulation.FlightComparator` in the API reference.
