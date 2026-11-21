.. _airbrakes:

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

    from rocketpy import AirBrakes, Controller, Environment, SolidMotor, Rocket, Flight

    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

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

    calisto = Rocket(
        radius=127 / 2000,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
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
        airfoil=("../data/airfoils/NACA0012-radians.txt","radians"),
    )

    tail = calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

Setting Up the Air Brakes
-------------------------

Now we can get started!

To create an air brakes model, we essentially need to define the following:

- The air brakes' **drag coefficient** as a function of the air brakes'
  **deployment level** and of the **Mach number**. This can be done through
  a ``CSV`` file which must have three columns: the first column is the air brakes'
  **deployment level**, the second column is the **Mach number**, and the third
  column is the **drag coefficient** added to rocket due to the air brakes at that
  specific deployment level and Mach number.

- The **controller function**, which takes in as argument information about the
  simulation up to the current time step and sets the desired air brakes'
  deployment level. The air brakes' deployment level must be between 0 and 1,
  and is set by writing ``context.controls.air_brakes.deployment_level``.
  Inside this function, any controller logic, filters, and apogee prediction
  can be implemented.

- The **sampling rate** of the controller function, in Hertz (Hz): how many
  times per second of simulation time the controller function is called.

Defining the Controller Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lets start by defining a very simple controller function.

The ``controller_function`` takes one argument, ``context``, which holds what
the flight computer knows at the current time step:

- ``context.time``: the current simulation time, in seconds.
- ``context.sampling_rate``: the rate at which the controller function is
  called, in Hertz (Hz).
- ``context.state``: the state of the rocket, read by name: ``state.z``
  (altitude above sea level, in meters), ``state.vx``, ``state.vy``,
  ``state.vz`` (velocity, in m/s), ``state.e0`` to ``state.e3`` (the
  quaternion of the attitude) and ``state.w1`` to ``state.w3`` (angular
  velocity, in rad/s).
- ``context.controls``: the control inputs of the rocket, such as
  ``context.controls.air_brakes.deployment_level``.
- ``context.record(...)``: keeps named values with their time, to look at
  after the flight.
- ``context.memory``: a dictionary the controller keeps from one call to the
  next, for what the next call needs (a filter state, an integral, a
  counter). It starts again from its initial values at every flight.
- ``context.environment``, ``context.rocket``, ``context.sensors`` and
  ``context.height_agl``, among others. See :ref:`eventusage` for the full
  list.

Our example ``controller_function`` will deploy the air brakes when the rocket
reaches 1500 meters above the ground. The deployment level will be function of the
vertical velocity at the current time step and of the vertical velocity at the
previous time step.

To read the vertical velocity from the previous controller call, use
``context.previous_state``. That is the state from the previous run of
*this controller*, so it is always exactly one control step back. Reading the
flight solution instead would give you the last stored trajectory point, which
advances on the solver's schedule rather than the controller's.

Also, the controller function will check for the burnout of the rocket's motor
and only deploy the air brakes if the rocket has reached burnout.

Then, a limitation for the opening/closing speed of the air brakes will be set.
The air brakes deployment level will not be able to change faster than 20% per
second, in our case.

Lets define the controller function:

.. jupyter-execute::

    def controller_function(context):
        time = context.time
        state = context.state  # the states, by name
        sampling_rate = context.sampling_rate
        motor = context.rocket.motor
        environment = context.environment
        air_brakes = context.controls.air_brakes  # the air brakes' controls
        altitude_ASL = state.z
        altitude_AGL = context.height_agl
        vx, vy, vz = state.vx, state.vy, state.vz

        # The deployment level in force since the previous call
        deployment_level = air_brakes.deployment_level

        # Get winds in x and y directions
        wind_x, wind_y = environment.wind_velocity_x(altitude_ASL), environment.wind_velocity_y(altitude_ASL)

        # Calculate Mach number
        free_stream_speed = (
            (wind_x - vx) ** 2 + (wind_y - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        mach_number = free_stream_speed / environment.speed_of_sound(altitude_ASL)

        # Get the state from the previous controller call
        previous_state = context.previous_state
        previous_vz = previous_state.vz if previous_state is not None else vz

        # Check if the rocket has reached burnout
        if time < motor.burn_out_time:
            deployment_level = 0

        # If below 1500 meters above ground level, air_brakes are not deployed
        if altitude_AGL < 1500:
            deployment_level = 0

        # Else calculate the deployment level
        else:
            # Controller logic
            new_deployment_level = deployment_level + 0.1 * vz + 0.01 * previous_vz**2

            # Limiting the speed of the air_brakes to 0.2 per second
            # Since this function is called every 1/sampling_rate seconds
            # the max change in deployment level per call is 0.2/sampling_rate
            max_change = 0.2 / sampling_rate
            lower_bound = deployment_level - max_change
            upper_bound = deployment_level + max_change
            deployment_level = min(max(new_deployment_level, lower_bound), upper_bound)

        air_brakes.deployment_level = deployment_level

        # Keep the Mach number with its time, to look at after the flight
        context.record(mach_number=mach_number)

.. note::

    - The ``context`` gives access to atmospheric conditions through
      ``context.environment`` (wind, temperature, pressure, elevation), such
      as ``context.environment.wind_velocity_x(altitude)`` or
      ``context.environment.speed_of_sound(altitude)``, without relying on
      global variables.

    - The positional form of v1.13,
      ``controller_function(time, sampling_rate, state, state_history,
      observed_variables, air_brakes[, sensors[, environment]])``, still
      works but is deprecated and will be removed in v1.16. Write the
      function as ``controller_function(context)`` instead.

    - The code inside the ``controller_function`` can be as complex as needed.
      Anything can be implemented inside the function, including filters,
      apogee prediction, and any controller logic.

    - The deployment level written to
      ``context.controls.air_brakes.deployment_level`` is kept between 0
      and 1: a value outside that range is moved to the nearest end. To put
      no limit on it, set ``clamp`` to ``False`` when defining the air
      brakes. The control input is named
      after the air brakes, in snake_case: ``name="AirBrakes"`` (the default)
      gives ``context.controls.air_brakes``, and ``name="Main Brakes"`` gives
      ``context.controls.main_brakes``, or as given,
      ``context.controls["Main Brakes"]``. To skip the name, use the air brakes
      returned by ``rocket.add_air_brakes``:
      ``context.controls[air_brakes].deployment_level``.

    - What the ``controller_function`` passes to ``context.record`` is kept
      at every call and can be read after the flight in ``flight.records``.
      Each record is a list of ``(time, value)`` pairs, with the value as it
      was given. This is useful for debugging and for plotting the results.

    - The ``controller_function`` can also be defined in a separate file and
      imported into the simulation script. This includes importing a ``c`` or
      ``cpp`` code into Python.


Defining the Drag Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now lets define the drag coefficient as a function of the air brakes' deployment
level and of the Mach number. We will import the data from a CSV file.

The CSV file must have three columns: the first column must be the air brakes'
deployment level, the second column must be the Mach number, and the third column
must be the drag coefficient.

Alternatively, the drag coefficient can be defined as a function of the air
brakes' deployment level and of the Mach number. This function must take in the
air brakes' deployment level and the Mach number as arguments, and must return the
drag coefficient.

.. note::

    At deployment level 0, the drag coefficient will always be set to 0,
    regardless of the input curve. This means that the simulation considers that
    at a deployment level of 0, the air brakes are completely retracted and do not
    contribute to the drag of the rocket.

Part of the data from the CSV can be seen in the code block below.

.. code-block::

    deployment_level, mach, cd
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

.. note::
  The air brakes' drag coefficient curve can be of the air brakes alone or of
  the whole rocket at each deployment level. This is set by the
  ``override_rocket_drag`` argument.

  With ``override_rocket_drag=False`` (the default) the curve is of the air
  brakes alone, and their drag is added to the rocket's.

  With ``override_rocket_drag=True`` the curve is of the whole rocket, as
  measured in a wind tunnel or computed for the complete rocket at several
  deployment levels. The air brakes then add the difference between the curve
  and its value at deployment level 0, at the same Mach number, to the
  rocket's own drag. The curve must include deployment level 0, and the
  rocket's ``power_off_drag`` should be those retracted values: the total drag
  is then the curve. A warning is shown when the two differ by more than 5%.

  The whole-rocket curve is taken as the rocket with its motor off. The rocket
  keeps switching between its two drag curves, and open air brakes add the
  same difference to either one:

  - Motor off: ``power_off_drag`` + (curve at the deployment level − curve at
    deployment level 0)
  - Motor burning: ``power_on_drag`` + the same difference

  To have the air brakes add no drag while the motor burns, build the
  :class:`rocketpy.AirBrakes` with ``active_during="power_off"``.

  In both cases retracted air brakes add no drag, and the rocket flies with its
  own drag curves.

Adding the Air Brakes to the Rocket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The air brakes are an aerodynamic surface with one control, and the controller
is a :class:`rocketpy.Controller` that writes it. Build each and add both to
the rocket, as for any other controllable:

- ``reference_area`` is the area the drag coefficient was made with. Here it
  is the rocket's cross section, ``calisto.area``.
- ``clamp=True`` (the default) keeps the deployment level between 0 and 1: a
  controller asking for more gets the limit. With ``clamp=False`` there is no
  limit and the drag curve is extrapolated.
- ``positions`` is where the air brakes are on the rocket, in meters, in the
  rocket's coordinate system, as for any surface. The drag acts along the
  rocket's axis, so the position only matters while the rocket turns.

``air_brakes.info()`` prints the settings and the drag coefficient at a few
deployment levels and Mach numbers; ``air_brakes.plots.drag_coefficient()``
draws the curve, one line per deployment level.

.. jupyter-execute::

    air_brakes = AirBrakes(
        drag_coefficient_curve="../data/rockets/calisto/air_brakes_cd.csv",
        reference_area=calisto.area,
        clamp=True,
        override_rocket_drag=False,
        name="Air Brakes",
    )
    calisto.add_surfaces(air_brakes, positions=-0.5)  # where they are, in meters

    controller = Controller(
        controller_function,
        sampling_rate=10,
        name="AirBrakes Controller",
    )
    calisto.add_controller(controller)

    air_brakes.all_info()

``rocket.add_air_brakes`` does the same in one call, building the air brakes
and the controller together. It is kept for scripts written that way:

.. code-block:: python

    air_brakes = calisto.add_air_brakes(
        drag_coefficient_curve="../data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        name="Air Brakes",
        position=-0.5,
    )

.. seealso::

    For more information on the :class:`rocketpy.AirBrakes` class
    initialization, see  :class:`rocketpy.AirBrakes.__init__` section.

Simulating a Flight
-------------------

We will be terminating the simulation at apogee, by setting
``terminate_on_apogee`` to ``True``. This way the simulation will stop when the
rocket reaches apogee, and we will save some time.

.. jupyter-execute::

    test_flight = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

Analyzing the Results
---------------------

Now we can create some plots to analyze the results. The deployment level
the air brakes flew with is stored with the flight:
``flight.solution.controls.air_brakes.deployment_level`` is the array of
time and value, and ``flight.controls.air_brakes.deployment_level`` the same
as a function of time. What the ``controller_function`` recorded is in
``flight.records``, named after the controller, exactly as it was recorded:
the default ``controller_name="AirBrakes Controller"`` gives
``flight.records["AirBrakes Controller"].mach_number``, a list of
``(time, value)`` pairs.

.. jupyter-execute::

    test_flight.plots.controls()

    deployment_level = test_flight.controls["Air Brakes"].deployment_level
    mach_number = test_flight.records["AirBrakes Controller"].mach_number
    print(f"Deployment level at 10 s: {deployment_level(10):.3f}")
    time, mach = mach_number[-1]
    print(f"Last Mach number recorded, at {time:.1f} s: {mach:.3f}")

The drag the air brakes made during the flight, from the air they met and the
deployment level they flew with, next to the rocket's total drag:

.. jupyter-execute::

    air_brakes.plots.drag(test_flight)

.. seealso::

    For more information on the :class:`rocketpy.AirBrakes` class attributes,
    see :class:`rocketpy.AirBrakes` section.

And of course, we should check some of the simulation results:

.. jupyter-execute::

    test_flight.prints.burn_out_conditions()
    test_flight.prints.apogee_conditions()
    test_flight.altitude()
    test_flight.vz()