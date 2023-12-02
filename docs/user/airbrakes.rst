Air Brakes
==========

Air brakes are commonly used in rocketry to slow down a rocket's ascent. They 
are usually deployed to make sure that the rocket reaches a certain altitude.

Lets make a simple air brakes example. We will use the same model as in the
:ref:`First Simulation <firstsimulation>` example, but we will add a simple air 
brakes model.

Setting Up The Simulation
-------------------------

First, lets define everything we need for the simulation up to the rocket:

.. jupyter-execute::

    from rocketpy import Environment, SolidMotor, Rocket, Flight

    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

    Pro75M1670 = SolidMotor(
        thrust_source="../data/motors/Cesaroni_M1670.eng",
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
        power_off_drag="../data/calisto/powerOffDragCurve.csv",
        power_on_drag="../data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rail_buttons = calisto.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.618,
        angular_position=45,
    )

    calisto.add_motor(Pro75M1670, position=-1.255)

    nose_cone = calisto.add_nose(
        length=0.55829, kind="vonKarman", position=1.278
    )

    fin_set = calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        position=-1.04956,
        cant_angle=0.5,
        airfoil=("../data/calisto/NACA0012-radians.csv","radians"),
    )

    tail = calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

Setting Up the Air Brakes
-------------------------

Now we can get started!

To create an air brakes model, we essentially need to define the following:

- The air brakes' **drag coefficient** as a function of the air brakes' 
  ``deployed level`` and of the ``Mach number``. This can be done by defining 
  by inputting the directory of a ``CSV`` file containing the drag coefficient 
  as a function of the  air brakes' deployed level and of the ``Mach number``.
  The ``CSV`` file must have three columns: the first column must be the air
  brakes' deployed level, the second column must be the ``Mach number``, and
  the third column must be the drag coefficient.

- The **controller function**, which takes in as argument information about the
  simulation up to the current time step, and the ``AirBrakes`` instance being 
  defined, and sets the desired air brakes' deployed level. The air brakes'
  deployed level must be between 0 and 1, and must be set using the
  ``set_deployed_level`` method of the ``AirBrakes`` instance being controlled.
  Inside this function, any controller logic, filters, and apogee prediction 
  can be implemented.

- The **sampling rate** of the controller function, in seconds. This is the time
  between each call of the controller function, in simulation time. Must be 
  given in Hertz.

Defining the Controller Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lets start by defining a very simple controller function.

The ``controller_function`` must take in the following arguments, in this
order:

- ``time``: The current time of the simulation, in seconds.
- ``sampling_rate``: The sampling rate of the controller function, in seconds.
- ``state``: The state of the simulation at the current time step. The state 
  is a list containing the following values, in this order:

  - ``x``: The x position of the rocket, in meters.
  - ``y``: The y position of the rocket, in meters.
  - ``z``: The z position of the rocket, in meters.
  - ``v_x``: The x component of the velocity of the rocket, in meters per 
      second.
  - ``v_y``: The y component of the velocity of the rocket, in meters per 
      second.
  - ``v_z``: The z component of the velocity of the rocket, in meters per 
      second.
  - ``e0``: The first component of the quaternion representing the rotation 
      of the rocket.
  - ``e1``: The second component of the quaternion representing the rotation 
      of the rocket.
  - ``e2``: The third component of the quaternion representing the rotation 
      of the rocket.
  - ``e3``: The fourth component of the quaternion representing the rotation 
      of the rocket.
  - ``w_x``: The x component of the angular velocity of the rocket, in 
      radians per second.
  - ``w_y``: The y component of the angular velocity of the rocket, in 
      radians per second.
  - ``w_z``: The z component of the angular velocity of the rocket, in 
      radians per second.
- ``state_history``: A list containing the state of the simulation at every 
  time step up to the current time step. The state of the simulation at the 
  previous time step is the last element of the list.
- ``air_brakes``: The ``AirBrakes`` instance being controlled.
    
Our example ``controller_function`` will deploy the air brakes when the rocket
reaches 1500 meters above the ground. The deployed level will be function of the
vertical velocity at the current time step and of the vertical velocity at the
previous time step.

Also, the controller function will check for the burnout of the rocket and only 
deploy the air brakes if the rocket has reached burnout. 

Then a limitation for the speed of the air brakes will be set. The air brakes
will not be able to deploy faster than 0.2 percentage per second.

Lets define the controller function:

.. jupyter-execute::

    def controller_function(time, sampling_rate, state, state_history, air_brakes):
        # state = [x, y, z, v_x, v_y, v_z, e0, e1, e2, e3, w_x, w_y, w_z]
        z = state[2]
        vz = state[5]

        # Get previous state from state_history
        previous_state = state_history[-1]
        previous_vz = previous_state[5]

        # Check if the rocket has reached burnout
        if time > Pro75M1670.burn_out_time:
            # If below 1500 meters, air_brakes are not deployed
            if z < 1500 + env.elevation:
                air_brakes.set_deployed_level(0)

            # Else calculate the deployed level
            else:
                new_deployed_level = (
                    air_brakes.deployed_level + 0.1 * vz + 0.01 * previous_vz**2
                )

                # Limiting the speed of the air_brakes to 0.1 per second
                # Since this function is called every 1/sampling_rate seconds
                # the max change in deployed level per call is 0.1/sampling_rate
                if new_deployed_level > air_brakes.deployed_level + 0.2 / sampling_rate:
                    new_deployed_level = air_brakes.deployed_level + 0.2 / sampling_rate
                elif new_deployed_level < air_brakes.deployed_level - 0.2 / sampling_rate:
                    new_deployed_level = air_brakes.deployed_level - 0.2 / sampling_rate
                else:
                    new_deployed_level = air_brakes.deployed_level

                air_brakes.set_deployed_level(new_deployed_level)

.. note::

    - The code inside the ``controller_function`` can be as complex as needed.
      Anything can be implemented inside the function, including filters,
      apogee prediction, and any controller logic.

    - The ``air_brakes`` instance ``deployed_level`` is clamped between 0 and 1.
      This means that the deployed level will never be set to a value lower than
      0 or higher than 1. If you want to disable this feature, set ``clamp`` to
      ``False`` when defining the air brakes.
    
    - The ``controller_function`` can also be defined in a separate file and
      imported into the simulation script. This includes importing a ``c`` or 
      ``cpp`` code into Python.


Defining the Drag Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now lets define the drag coefficient as a function of the air brakes' deployed 
level and of the Mach number. We will import the data from a CSV file. 

The CSV file must have three columns: the first column must be the air brakes' 
deployed level, the second column must be the Mach number, and the third column
must be the drag coefficient.

Alternatively, the drag coefficient can be defined as a function of the air
brakes' deployed level and of the Mach number. This function must take in the
air brakes' deployed level and the Mach number as arguments, and must return the
drag coefficient.

.. note::

    At deployed level 0, the drag coefficient will always be set to 0, 
    regardless of the input curve. This means that the simulation considers that at
    a deployed level of 0, the air brakes are completely retracted and do not 
    contribute to the drag of the rocket.

Part of the data from the CSV can be seen in the code block below.

.. code-block::

    deployed_level, mach, cd
    0.0, 0.0, 0.0
    0.1, 0.0, 0.0
    0.1, 0.2, 0.0
    0.1, 0.3, 0.01
    0.1, 0.4, 0.005
    0.1, 0.5, 0.006
    0.1, 0.6, 0.018
    0.1, 0.7, 0.012
    0.1, 0.8, 0.014
    0.5, 0.1, 0.051
    0.5, 0.2, 0.051
    0.5, 0.3, 0.065
    0.5, 0.4, 0.061
    0.5, 0.5, 0.067
    0.5, 0.6, 0.083
    0.5, 0.7, 0.08
    0.5, 0.8, 0.085
    1.0, 0.1, 0.32
    1.0, 0.2, 0.225
    1.0, 0.3, 0.225
    1.0, 0.4, 0.21
    1.0, 0.5, 0.19
    1.0, 0.6, 0.22
    1.0, 0.7, 0.21
    1.0, 0.8, 0.218

Adding the Air Brakes to the Rocket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we can add the air brakes to the rocket. 

We will set the ``reference_area`` to ``None``. This means that the reference
area for the calculation of the drag force from the coefficient will be the 
rocket's reference area (the area of the cross section of the rocket). If we
wanted to set a different reference area, we would set ``reference_area`` to 
the desired value.

Also, we will set ``clamp`` to ``True``. This means that the deployed level will
be clamped between 0 and 1. This means that the deployed level will never be set
to a value lower than 0 or higher than 1. This can alter the behavior of the
controller function. If you want to disable this feature, set ``clamp`` to
``False``.

.. jupyter-execute::

    air_brakes, controller = calisto.add_air_brakes(
        drag_coefficient_curve="../data/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=100,
        reference_area=None,
        clamp=True,
        name="AirBrakes",
        controller_name="AirBrakes Controller",
    )

    air_brakes.all_info()

.. seealso::

    For more information on the :class:`rocketpy.AirBrakes` class 
    initialization, see  :class:`rocketpy.AirBrakes.__init__` section.

Simulating a Flight
-------------------

To simulate the air brakes successfully, we must set ``time_overshoot`` to
``False``. This way the simulation will run at the time step defined by our 
controller sampling rate. Be aware that this will make the simulation **much** 
run slower.

Also, we are terminating the simulation at apogee, by setting 
``terminate_at_apogee`` to ``True``. This way the simulation will stop when the 
rocket reaches apogee, and we will save some time.

.. jupyter-execute::

    test_flight = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True
    )

Analyzing the Results
---------------------

Now we can see some plots from our air brakes:

.. jupyter-execute::

    air_brakes.deployed_level_by_time.plot(force_data=True)
    air_brakes.drag_coefficient_by_time.plot(force_data=True)

.. seealso::

    For more information on the :class:`rocketpy.AirBrakes` class attributes, 
    see :class:`rocketpy.AirBrakes` section.

And of course, the simulation results:

.. jupyter-execute::

    test_flight.altitude()
    test_flight.vz()

.. jupyter-execute::

    test_flight.all_info()

