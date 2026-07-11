Deployable Payload
==================

Here we try to demonstrate how to use RocketPy to simulate a flight of a rocket
that contains a deployable payload.

Let's start by importing the rocketpy classes we will use.

.. jupyter-execute::

      from rocketpy import Environment, SolidMotor, Rocket, Flight


Creating Environment
--------------------

We will set an ``Environment`` object to the Spaceport America, in New Mexico, USA.

.. jupyter-execute::

      env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

To get weather data from the GFS forecast, available online, we run the following lines.

.. seealso::

   See `Environment Class Usage <environment/environment_class_usage.ipynb>`__
   for more information on how to use the Environment class.

.. jupyter-execute::

      import datetime

      tomorrow = datetime.date.today() + datetime.timedelta(days=1)

      env.set_date(
          (tomorrow.year, tomorrow.month, tomorrow.day, 12)
      )  # Hour given in UTC time

      env.set_atmospheric_model(type="Forecast", file="GFS")
      env.max_expected_height = 8000

.. jupyter-execute::

      env.info()

Creating a Motor
----------------

A solid rocket motor is used in this case, the Cesaroni Pro75 M1670.

.. jupyter-execute::

      Pro75M1670 = SolidMotor(
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

.. jupyter-execute::

      Pro75M1670.info()

Simulating the 1st Flight: ascending phase
------------------------------------------

Let's start to simulate our rocket's flight. We will use the Environment and Motor objects we created before.

We will assume that the payload is ejected at apogee, however, this can be modified if needed.

We start by defining the value of each relevant mass, ensuring they are correct before continuing.

.. seealso::

   See :ref:`First Simulation <firstsimulation>` example for more details on how to simulate a rocket flight.

.. jupyter-execute::

      # 14.426 is the mass of the rocket including the payload but without the motor
      payload_mass = 4.5  # in kg
      rocket_mass = 14.426 - payload_mass  # in kg

      print(
          "Rocket Mass Without Motor: {:.4} kg (with Payload)".format(
              rocket_mass + payload_mass
          )
      )
      print("Loaded Motor Mass: {:.4} kg".format(Pro75M1670.total_mass(0)))
      print("Payload Mass: {:.4} kg".format(payload_mass))
      print(
          "Fully loaded Rocket Mass: {:.4} kg".format(
              rocket_mass + Pro75M1670.total_mass(0) + payload_mass
          )
      )

Then we define our rocket, including the payload mass.

.. jupyter-execute::

      rocket_with_payload = Rocket(
          radius=127 / 2000,
          mass=rocket_mass + rocket_mass,
          inertia=(6.321, 6.321, 0.034),
          power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
          power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
          center_of_mass_without_motor=0,
          coordinate_system_orientation="tail_to_nose",
      )

      rocket_with_payload.add_motor(Pro75M1670, position=-1.255)

      rocket_with_payload.set_rail_buttons(
          upper_button_position=0.0818,
          lower_button_position=-0.618,
          angular_position=45,
      )

      rocket_with_payload.add_nose(length=0.55829, kind="von karman", position=1.278)

      rocket_with_payload.add_trapezoidal_fins(
          n=4,
          root_chord=0.120,
          tip_chord=0.060,
          span=0.110,
          position=-1.04956,
          cant_angle=0.5,
      )

      rocket_with_payload.add_tail(
          top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
      )

.. jupyter-execute::

      rocket_with_payload.info()

Finally we create the flight simulation of this rocket, but stopping at apogee

.. jupyter-execute::

      flight_with_payload = Flight(
          rocket=rocket_with_payload,
          environment=env,
          rail_length=5.2,
          inclination=85,
          heading=25,
          terminate_on_apogee=True,
          name="Rocket Flight With Payload",
      )

Simulate the 2nd Flight: Rocket Without Payload
------------------------------------------------

Now we will simulate the second flight stage, which is the landing phase of our Rocket.
Here we will consider that the payload was ejected at the apogee of the first stage.
Therefore we should be careful with the value of its mass.

.. jupyter-execute::

      rocket_without_payload = Rocket(
          radius=127 / 2000,
          mass=rocket_mass,
          inertia=(6.321, 6.321, 0.034),
          power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
          power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
          center_of_mass_without_motor=0,
          coordinate_system_orientation="tail_to_nose",
      )


      # Define Parachutes for the rocket
      main_chute = rocket_without_payload.add_parachute(
          "Main",
          cd_s=7.2,
          trigger=800,
          sampling_rate=105,
          lag=1.5,
          noise=(0, 8.3, 0.5),
      )

      drogue_chute = rocket_without_payload.add_parachute(
          "Drogue",
          cd_s=0.72,
          trigger="apogee",
          sampling_rate=105,
          lag=1.5,
          noise=(0, 8.3, 0.5),
      )

.. jupyter-execute::

      rocket_without_payload.info()

The line ``initial_solution=flight_with_payload`` will make the simulation start
from the end of the first stage.

This will simulate our rocket with its payload ejected, after reaching apogee.

.. jupyter-execute::

      flight_without_payload = Flight(
          rocket=rocket_without_payload,
          environment=env,
          rail_length=5.2,  # does not matter since the flight is starting at apogee
          inclination=0,
          heading=0,
          initial_solution=flight_with_payload,
          name="Rocket Flight Without Payload",
      )

Simulating the 3rd Flight: Payload
----------------------------------

Here we will simulate the payload flight, which is the third flight stage of our Rocket.
The Payload will be ejected at the apogee of the first stage.
Here, it will be modeled as a "dummy" rocket, which does not have any aerodynamic
surfaces to stabilize it, nor a motor that ignites. It does, however, have parachutes.

.. jupyter-execute::

      # Define the "Payload Rocket"

      payload_rocket = Rocket(
          radius=127 / 2000,
          mass=payload_mass,
          inertia=(0.1, 0.1, 0.001),
          power_off_drag=0.5,
          power_on_drag=0.5,
          center_of_mass_without_motor=0,
      )

      payload_drogue = payload_rocket.add_parachute(
          "Drogue",
          cd_s=0.35,
          trigger="apogee",
          sampling_rate=105,
          lag=1.5,
          noise=(0, 8.3, 0.5),
      )

      payload_main = payload_rocket.add_parachute(
          "Main",
          cd_s=4.0,
          trigger=800,
          sampling_rate=105,
          lag=1.5,
          noise=(0, 8.3, 0.5),
      )

.. important::

   The magic line ``initialSolution=RocketFlight1`` will make the
   simulation start from the end of the first flight.

.. jupyter-execute::

      payload_flight = Flight(
          rocket=payload_rocket,
          environment=env,
          rail_length=5.2,  # does not matter since the flight is starting at apogee
          inclination=0,
          heading=0,
          initial_solution=flight_with_payload,
          name="PayloadFlight",
      )

Plotting results
----------------

We need to import the ``CompareFlights`` class from the ``rocketpy.plots.compare`` module.

.. jupyter-execute::

      from rocketpy.plots.compare import CompareFlights

Then we create the ``comparison`` object, an instance of ``CompareFlights`` class

.. jupyter-execute::

      comparison = CompareFlights(
          [flight_with_payload, flight_without_payload, payload_flight]
      )

Finally we can plot different aspects of the comparison object.

.. jupyter-execute::

      comparison.trajectories_3d(legend=True)

.. jupyter-execute::

      comparison.positions()

.. jupyter-execute::

      comparison.velocities()

.. jupyter-execute::

      comparison.accelerations()

.. jupyter-execute::

      comparison.aerodynamic_forces()

.. jupyter-execute::

      comparison.aerodynamic_moments()

.. jupyter-execute::

      comparison.angles_of_attack()
