Compare Flights
===============

This example demonstrates how to use the rocketpy ``CompareFlights`` class.
This class has many applications, including the comparison of different flight
setups for a single rocket, the simulation of deployable systems, and the
multi-stage rocket analysis.

Importing classes
-----------------

We will start by importing the necessary classes and modules:

.. jupyter-execute::

      from rocketpy.plots.compare import CompareFlights
      from rocketpy import Environment, Flight, Rocket, SolidMotor
      from datetime import datetime, timedelta


Create Environment, Motor and Rocket
------------------------------------

First, let's create the environment, motor and rocket objects.
This is done following the same steps as in the :ref:`firstsimulation` example.

.. jupyter-execute::

      after_tomorrow = datetime.now() + timedelta(days=2)
      env = Environment(latitude=-23, longitude=-49, date=after_tomorrow)
      env.set_atmospheric_model(type="Forecast", file="GFS")

      cesaroni_motor = SolidMotor(
          thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
          dry_mass=1.815,
          dry_inertia=(0.125, 0.125, 0.002),
          nozzle_radius=33 / 1000,
          grain_number=5,
          grain_density=1815,
          grain_outer_radius=33 / 1000,
          grain_initial_inner_radius=15 / 1000,
          grain_initial_height=120 / 1000,
          grain_separation=5 / 1000,
          grains_center_of_mass_position=0.397,
          center_of_dry_mass_position=0.317,
          nozzle_position=0,
          burn_time=3.9,
          throat_radius=11 / 1000,
          coordinate_system_orientation="nozzle_to_combustion_chamber",
      )

      calisto = Rocket(
          radius=127 / 2000,
          mass=14.426,
          inertia=(6.321, 6.321, 0.034),
          power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
          power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
          center_of_mass_without_motor=0,
          coordinate_system_orientation="tail_to_nose",
      )

      calisto.set_rail_buttons(
          upper_button_position=0.0818,
          lower_button_position=-0.618,
          angular_position=45,
      )

      calisto.add_motor(cesaroni_motor, position=-1.255)

      nosecone = calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)

      fin_set = calisto.add_trapezoidal_fins(
          n=4,
          root_chord=0.120,
          tip_chord=0.060,
          span=0.110,
          position=-1.04956,
          cant_angle=0.5,
          airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
      )

      tail = calisto.add_tail(
          top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
      )

      main_chute = calisto.add_parachute(
          "Main",
          cd_s=10.0,
          trigger=800,
          sampling_rate=105,
          lag=1.5,
          noise=(0, 8.3, 0.5),
      )

      drogue_chute = calisto.add_parachute(
          "Drogue",
          cd_s=1.0,
          trigger="apogee",
          sampling_rate=105,
          lag=1.5,
          noise=(0, 8.3, 0.5),
      )

Creating the Flight objects
---------------------------

Now we can create different flights varying the launch angle and the rail inclination:

.. jupyter-execute::

      inclinations = [85, 75]
      headings = [90, 135]
      flights = []

      for heading in headings:
          for inclination in inclinations:
              flight = Flight(
                  environment=env,
                  rocket=calisto,
                  rail_length=5.2,
                  inclination=inclination,
                  heading=heading,
                  name=f"Incl {inclination} Head {heading}",
              )
              flights.append(flight)


We can easily visualize the number of flights created:

.. jupyter-execute::

      print("Number of flights: ", len(flights))

Start the comparison
--------------------

It is easy to initialize the ``CompareFlights`` object:

.. jupyter-execute::

      comparison = CompareFlights(flights)


After the initialization, we can use different methods to plot the results in a comparative way.
To see a full description of the available methods, you can check the :ref:`compareflights` documentation.

Plotting results one by one
----------------------------

The flights results are divided into different methods, so we can plot them one by one.
This is practical when we want to focus on a specific aspect of the flights.

.. jupyter-execute::

      comparison.trajectories_3d(legend=True)

.. jupyter-execute::

      comparison.positions()

.. jupyter-execute::

      comparison.trajectories_2d(plane="xy", legend=True)

.. jupyter-execute::

      comparison.velocities()

.. jupyter-execute::

      comparison.stream_velocities()

.. jupyter-execute::

      comparison.accelerations()

.. jupyter-execute::

      comparison.angular_velocities()

.. jupyter-execute::

      comparison.angular_accelerations()

.. jupyter-execute::

      comparison.attitude_angles()

.. jupyter-execute::

      comparison.euler_angles()

.. jupyter-execute::

      comparison.quaternions()

.. jupyter-execute::

      comparison.angles_of_attack()

.. jupyter-execute::

      comparison.aerodynamic_forces()

.. jupyter-execute::

      comparison.aerodynamic_moments()

.. jupyter-execute::

      comparison.fluid_mechanics()

.. jupyter-execute::

      comparison.energies()

.. jupyter-execute::

      comparison.powers()


Plotting using the ``all`` method
---------------------------------

Alternatively, we can plot the results altogether by calling one simple method:

.. jupyter-execute::

      # commented to avoid long output
      # comparison.all()
