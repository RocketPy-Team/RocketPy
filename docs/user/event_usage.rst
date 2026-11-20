.. _eventusage:

Event Usage
===========

This page explains how to build and use :class:`rocketpy.Event` objects in a
RocketPy simulation. It focuses on practical usage: how to write a custom
event, how the trigger and callback flow works, and which configuration
options are useful in common simulation workflows.

Events are the main way RocketPy reacts to conditions during a flight.
They can:

- detect milestones such as liftoff, rail exit, apogee, motor burnout, or
  landing,
- modify the simulation by requesting new phases, new derivatives, or
  controller changes,
- schedule or remove other events,
- support exact-time solving when a trigger should fire between solver steps,
- automatically enable or disable themselves based on time or custom logic.

If you are looking for the lower-level simulation loop that consumes events and
time nodes, see :doc:`../technical/simulation_loop`.

.. important::
  **Performance Considerations**

  Adding events to a flight simulation can impact computational performance,
  especially in the following scenarios:

  - **Large sampling rates**: Events with very small sampling intervals
    (e.g., 0.01 seconds) will be evaluated more frequently, increasing overhead.
  - **Exact-time solving**: Each time an event with ``exact_time_function``
    triggers, RocketPy searches the solver step for the exact moment, working out
    the event context again at every moment it tries. This adds cost every time
    the event triggers.
  - **Callbacks with expensive computations**: Callback functions that perform
    heavy calculations (large matrix operations, I/O, external API calls) directly
    affect flight simulation speed. Try to keep callbacks lightweight or cache results.

  As a general practice, use events judiciously and test performance with your
  specific number of events, derivatives, and payload callbacks.

.. Hidden setup block for environment, motor, and rocket
.. jupyter-execute::
   :hide-code:

    import numpy as np
    from rocketpy import Environment, SolidMotor, Rocket, Flight, Event

    # Minimal simulation setup using the same public classes as the rest of RocketPy.
    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=0)
    
    motor = SolidMotor(
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
    
    rocket = Rocket(
        radius=127 / 2000,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    
    rocket.add_motor(motor, position=-1.255)


.. _eventcontext:

Parameters for Callback and Trigger Functions
---------------------------------------------

At the most basic level, an event has two callables:

- ``trigger(context)`` returns ``True`` when the event should fire.
- ``callback(context)`` performs the requested action.

The same ``context`` is passed to both callables. It holds the simulation state
and objects at the moment of the check, each read as ``context.<name>``, for
example ``context.time`` or ``context.state.vz``. Some of its values are
only worked out when a function reads them, and are then kept for the rest of
the solver step, so reading a value you do not use costs nothing.

.. tip::

    To have your code editor list these names as you type, tell it what
    ``context`` is:

    .. code-block:: python

        from rocketpy import EventContext

        def trigger(context: EventContext):
            return context.state.vz < 0

The following values are available:

**Simulation time and state:**

- ``time`` (float): The current simulation time in seconds. For an event with
  an ``exact_time_function``, the callback receives the exact moment the event
  happened, and every other value below is given at that moment too.
- ``state``: The rocket's states at this moment. Read one by its name:

  - ``context.state.x``, ``.y``, ``.z``: position, in meters. ``x`` points
    East, ``y`` points North and ``z`` is the altitude above sea level.
  - ``context.state.vx``, ``.vy``, ``.vz``: velocity along the same
    directions, in m/s. ``vz`` is positive while the rocket climbs.
  - ``context.state.e0``, ``.e1``, ``.e2``, ``.e3``: the quaternion that
    describes the rocket's orientation.
  - ``context.state.w1``, ``.w2``, ``.w3``: angular velocity around the
    rocket's own x, y and z axes, in rad/s.

  These thirteen are available in every flight phase. They can also be read by
  position, in the order above, so ``context.state[5]`` is the same number as
  ``context.state.vz`` and ``context.state[3:6]`` is the velocity vector.

  A flight phase may follow states of its own that are not among the thirteen.
  Read those by name in the same way, while the flight is in that phase. For a
  state the phase calls ``my_state``::

      value = context.state.my_state

  Asking for a name the current phase does not have raises an error that lists
  the names it does have. ``context.state.names`` gives the same list.

  For arithmetic on all thirteen values at once, use the plain array
  ``context.state.values``.
- ``height_agl`` (float): Height of the rocket above ground level
  in meters, computed as ``state.z - env.elevation``. Always present.
- ``state_dot``: The derivative of each state with respect to time. The
  thirteen standard ones are read by the same names as in
  :class:`rocketpy.Flight`:

  - ``context.state_dot.vx``, ``.vy``, ``.vz``: velocity, in m/s.
  - ``context.state_dot.ax``, ``.ay``, ``.az``: acceleration, in m/s².
  - ``context.state_dot.e0_dot``, ``.e1_dot``, ``.e2_dot``, ``.e3_dot``: time
    derivative of the quaternion.
  - ``context.state_dot.alpha1``, ``.alpha2``, ``.alpha3``: angular
    acceleration around the rocket's own x, y and z axes, in rad/s².

  The time derivative of a state a flight phase follows on its own is read by
  that state's name followed by ``_dot``, so for a state the phase calls
  ``my_state`` it is ``context.state_dot.my_state_dot``. Worked out when first
  read, by evaluating the equations of motion once more.
- The air around the rocket at this moment, each worked out when first
  read: ``context.pressure`` (Pa), ``context.density`` (kg/m³),
  ``context.speed_of_sound`` (m/s), ``context.mach``,
  ``context.wind_velocity`` (m/s, Earth frame), ``context.stream_velocity``
  (m/s, in the rocket's body frame) and ``context.stream_speed`` (m/s). They
  are the values a controllable's methods receive as ``conditions`` inside the
  equations of motion (:class:`rocketpy.control.FlightConditions`), also
  available as one object, ``context.conditions``, to hand on.
- ``previous_state``: The ``state`` from the previous time this same event was
  checked, or ``None`` on the first check. It is read like ``state``, as
  ``context.previous_state.vz``.

  It keeps the names of the flight phase it was taken in. On the first check
  after the flight enters a new phase, a state only the earlier phase followed
  can still be read from it, while a state only the new phase follows cannot.
- ``previous_time`` (float or ``None``): The ``time`` of that previous check,
  in seconds, or ``None`` on the first check.

  Use these two to fire on a value *crossing* a threshold rather than merely
  being past it. They are always consecutive checks of this same event, so a
  crossing cannot slip between them::

      def trigger(context):
          previous = context.previous_state
          if previous is None:  # the first check has nothing to compare against
              return False
          # vz turned negative
          return previous.vz > 0 >= context.state.vz

  This is the value this event saw at its own previous check.
- ``dt`` (float or ``None``): The time since this event was last checked, in
  seconds: ``time - previous_time``. For a controller, that is the time since
  its previous run::

      def controller_function(context):
          error = context.state.vz - 50
          context.memory["integral"] = (
              context.memory.get("integral", 0.0) + error * context.dt
          )

  On the first check there is no previous one, so ``dt`` is
  ``1 / sampling_rate``, the time a run normally covers. An event without a
  ``sampling_rate``, checked at every solver step, gets ``None`` there.

**Simulation objects:**

- ``rocket`` (:class:`rocketpy.Rocket`): The Rocket object being simulated.
- ``environment`` (:class:`rocketpy.Environment`): The Environment conditions
  for the flight, such as ``context.environment.wind_velocity_x(z)``.
- ``phase_name`` (str): The flight phase being flown: ``"initial_phase"``
  from the start (on the rail), ``"free_flight"`` after leaving the rail, and
  ``"<parachute name>_parachute_descent"`` under a parachute.

Read the rocket and the environment, but never change them: the next flight
flies the same objects, and the flight cannot see such a change, so it would
not act from the right moment or be undone.

**Event and sensor data:**

- ``event`` (:class:`rocketpy.Event`): A reference to the Event object itself,
  allowing access to ``commands`` and other event state.
- ``memory`` (dict): The event's own dictionary, kept from one check to the
  next (the same as ``context.event.memory``). See below.
- ``commands``: The requests this event makes to the flight (the same as
  ``context.event.commands``). See below.
- ``sensors``: The latest reading of each sensor attached to the rocket. Find
  a sensor by its name as given or by that name in snake_case as an
  attribute, then read a channel: ``context.sensors["Barometer"].pressure`` or
  ``context.sensors.barometer.pressure``. Every sensor reads once on the pad 
  when the flight starts, so there is always a reading. See :ref:`sensorsusage`.
- ``sampling_rate`` (float or None): The sampling rate of the event in Hz, or
  ``None`` for a continuous event.

**Control inputs:**

- ``controls``: The flight's control inputs. The controls of each controllable
  of the rocket, such as air brakes, :class:`rocketpy.ControlFins` or a
  surface built with ``controls``, written as
  ``context.controls.air_brakes.deployment_level = 0.5``. The controllable is
  its name in snake_case. A name that is not a plain word is written as given,
  ``context.controls["Main Brakes"]``, and a control whose name is built in a
  loop as ``context.controls.canards[fin]``. ``list(context.controls)`` gives
  the names of the controllables, and ``rocket.control_names`` lists every
  control input as ``<name in snake_case>.<control>``. A value written here is
  kept inside the control's limits once the callback returns,
  and acts from this moment on.

**Sharing values between events:**

- ``outputs``: The values the events of the flight share with each other. An
  event declares its outputs when it is built, writes them under its own name
  in snake_case, and every other event reads them::

      def guidance_function(context):
          context.outputs.guidance.pitch_setpoint = 0.1  # rad

      def attitude_function(context):
          error = context.outputs.guidance.pitch_setpoint - pitch(context)
          context.controls.canards.fin_1 = 2.0 * error

      guidance = Controller(guidance_function, sampling_rate=10,
                            name="Guidance", outputs=("pitch_setpoint",))
      attitude = Controller(attitude_function, sampling_rate=100, name="Attitude")

  An output can hold any value and keeps it until it is written again. Only the
  event that declared an output can write it. Outputs start every flight as
  ``None`` (or at the value given with a dictionary,
  ``outputs={"pitch_setpoint": 0.05}``). After the flight,
  ``flight.event_outputs.guidance.pitch_setpoint`` is the list of
  ``(time, value)`` pairs written. They are the way to share a value between
  events: a value kept in a Python object outside the flight is not reset
  between flights.

**Keeping your own data:**

- The event's own dictionary is ``context.memory``. It is kept from
  one check to the next, so keep what the next call needs there: counters,
  thresholds, a filter state, anything shared between trigger and callback. It
  is restored to its starting values at every flight.
- To see a value after the flight, record it: ``context.record(error=0.4)``
  stores named values at the current time. After the flight,
  ``event.records["error"]`` is the list of ``(time, value)`` pairs recorded,
  with each value exactly as it was given, and ``flight.records`` holds the
  records of every event, by the event's name as given or in snake_case:
  ``flight.records["My Event"].error`` or ``flight.records.my_event.error``. A
  record can hold anything, a number, a vector, a text, a dictionary
  (``context.record(command=[0.1, -0.2], mode="coast")``);
  ``flight.plots.records()`` plots the ones made of numbers.
  Do not add values to the ``context`` passed to your function: it is shared by
  every event checked in the same solver step.

**What your function can do:**

Besides reading values, a trigger or callback can do four things, each
through the context:

- **Set a control input**: ``context.controls.air_brakes.deployment_level = 0.5``.
  The new value is used from this moment on.
- **Keep something for its next call**: ``context.memory["integral"] += error``.
- **Log a value to look at after the flight**: ``context.record(error=error)``.
- **Change how the flight goes on**: ``context.commands``, such as
  ``context.commands.terminate_flight()``, ``set_dynamics(...)``,
  ``start_flight_phase(...)``, adding, enabling or disabling events, and
  switching surfaces on or off. Commands are requests to the flight itself:
  they are queued while your function runs and applied by the flight once it
  returns. See :ref:`event_commands`.

The first three take effect as you write them. Writing a control or a record
is not a command.

Understanding Event Parameters
------------------------------

The Event constructor accepts many parameters. Here we explain each one with
practical examples and demonstrate proper activation.

**callback** (required)
  The function that runs when the event triggers. It takes the ``context``
  as its single argument, and what it returns is ignored. Values to look at
  after the flight are kept with ``context.record(...)``, and read as
  ``event.records``. The callback can also queue commands to modify the
  simulation state through the ``context.commands`` interface. The event's
  ``context.memory`` dictionary lets the callback keep data across triggers.

  .. seealso:: 
    Check the :ref:`event_commands` section for details.

  .. jupyter-execute::

      def my_callback(context):
          context.record(altitude=context.state.z)

      simple_event = Event(
          callback=my_callback, 
          name="Simple event",
          trigger_only_once=False,
          sampling_rate=0.5,
      )

      simple_flight = Flight(
          rocket=rocket,
          environment=env,
          rail_length=5.2,
          inclination=85,
          heading=0,
          max_time=12.0,
          max_time_step=0.1,
          custom_events=[simple_event],
          name="Callback example",
      )
      
      print(f"Flight time = {simple_flight.t:.4f} s\n")
      
      altitude = simple_event.records["altitude"]
      print(f"Altitudes recorded: {[value for _, value in altitude]}\n")
      
      print(simple_event)

**trigger** (optional)
  A callable that returns ``True`` when the event should fire. If ``None``,
  the callback runs unconditionally every time the event is checked: at every
  solver step, or at its ``sampling_rate`` if it has one. Such an event is a
  passive hook, useful for logging or for driving something on a fixed
  schedule.

  .. note::
    The trigger function receives the same ``context`` as the callback, and
    it has all the same functionalities, except that its return value is
    interpreted as a boolean condition instead of it being logged.


  .. tip::
    A list of all triggered times can be accessed with ``event.triggered_times``,
    and the times the event was turned on or off with ``event.enabled_times``
    and ``event.disabled_times``. Like the records, they only hold moments of
    the trajectory that was flown: when the simulation goes back in time, the
    later ones are dropped.


  .. jupyter-execute::

      def simple_trigger(context):
          """Triggers when vertical velocity becomes large (enough)."""
          return context.state.vz > 50  # m/s

      def simple_callback(context):
          context.record(status="event triggered!")

      simple_event = Event(
          callback=simple_callback,
          trigger=simple_trigger,
          name="Simple detector",
          trigger_only_once=False,
      )
      
      simple_flight = Flight(
          rocket=rocket,
          environment=env,
          rail_length=5.2,
          inclination=85,
          heading=0,
          max_time=12.0,
          max_time_step=0.1,
          custom_events=[simple_event],
          name="Trigger example",
      )
      
      print(f"Flight time = {simple_flight.t:.4f} s\n")
      
      print(
          "Event triggered "
          f"{len(simple_event.triggered_times)} time(s)\n"
      )
      
      if simple_event.triggered_times:
          print(f"First trigger time: {simple_event.triggered_times[0]:.4f} s\n")
      
**sampling_rate** (optional)
  Controls how often the event is checked:

  - ``None`` (default): Event is evaluated continuously at every time step.
  - A float (e.g., ``0.1``): Event is sampled every 0.1 seconds.

  .. tip::
    When using continuous events (``sampling_rate=None``), the solver's time 
    stepping is critical for accurate trigger detection. Consider setting 
    ``max_time_step`` and optionally ``min_time_step`` on the Flight 
    initialization to control  the integration step size. Smaller time steps 
    (e.g., ``max_time_step=0.1``) improve the likelihood of catching events
    that occur between larger steps, especially for fast-changing conditions.

  .. jupyter-execute::

      # Continuous event (checked every step)
      continuous_event = Event(
          callback=my_callback,
          name="Continuous checker",
          sampling_rate=None,
          trigger_only_once=False,
      )

      # Discrete event (checked every 0.05 seconds)
      discrete_event = Event(
          callback=my_callback,
          name="Discrete checker",
          sampling_rate=0.5,
          trigger_only_once=False,
      )

      flight = Flight(
          rocket=rocket,
          environment=env,
          rail_length=5.2,
          inclination=85,
          heading=0,
          max_time=12.001,
          max_time_step=0.1,
          custom_events=[continuous_event, discrete_event],
          name="Continuous sampling example",
      )

      print(
          f"Flight time = {flight.t:.4f} s\n"
      )
      
      print(
          "Continuous event: triggered "
          f"{len(continuous_event.triggered_times)} time(s)\n"
      )
      
      print(
          "Discrete event: triggered "
          f"{len(discrete_event.triggered_times)} time(s)"
      )

**memory** (optional)
  The event's own dictionary, kept from one check to the next. Useful for
  counters, thresholds and data shared between trigger and callback. Read and
  write it as ``context.memory`` inside the trigger and callback. It
  is restored to the values given here when the event is reset for a new
  flight.

  .. note::
    ``memory`` holds what the next call needs, not a history: what the
    callback writes there is **not** kept over time nor saved. To see a value
    after the flight, record it with ``context.record`` instead.

  Here the count is what the next call needs, so it lives in ``memory``. The
  time of each call is only wanted afterwards, so it is recorded:

  .. jupyter-execute::

      def counting_callback(context):
          context.memory["count"] += 1
          context.record(count=context.memory["count"])
          return None

      counter_event = Event(
          callback=counting_callback,
          name="Counting event",
          memory={"count": 0},
      )
      
      counter_flight = Flight(
          rocket=rocket,
          environment=env,
          rail_length=5.2,
          inclination=85,
          heading=0,
          max_time=12.0,
          max_time_step=0.1,
          custom_events=[counter_event],
          name="Memory example",
      )

      print(f"Flight time = {counter_flight.t:.4f} s\n")
      print(
          "Event triggered "
          f"{len(counter_event.triggered_times)} time(s)\n"
      )
      print(f"Final event memory: {counter_event.memory}")
      count = counter_event.records["count"]
      print(f"Calls recorded from {count[0][0]:.2f} s to {count[-1][0]:.2f} s")

**disable_on** (optional)
  Automatically disable the event based on a condition. Can be:

  - A string preset: ``"apogee"`` or ``"burnout"``
  - A float/int: simulation time in seconds (e.g., ``120.0`` disables at t=120s)
  - A callable: any callable with the signature ``function(context)`` that 
    returns ``True`` when the event should be disabled.

  .. tip::
    The times when the event is disabled through ``disable_on`` are recorded in
    the list ``event.disabled_times`` for later inspection.

  .. jupyter-execute::

    # Disable at specific time
    time_gated = Event(
        callback=my_callback,
        name="Disabled at t=3s",
        disable_on=3.0,
    )

    # Disable at burnout (preset)
    burnout_gated = Event(
        callback=my_callback,
        name="Disabled at burnout",
        disable_on="burnout",
        sampling_rate=10,
    )

    # Disable via custom condition
    def disable_above_altitude(context):
        # return True to disable when altitude above 1000m AGL
        return context.height_agl > 700.0

    altitude_gated = Event(
        callback=my_callback,
        name="Disabled above 700m",
        disable_on=disable_above_altitude,
    )

    # Run all three gating strategies in a single flight
    gated_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[time_gated, burnout_gated, altitude_gated],
        name="Disable-on example",
    )


    print(f"Flight time = {gated_flight.t:.4f} s\n")

    print(
        "Time-gated event disabled on "
        f"{time_gated.disabled_times[0]:.4f} s\n"
    )

    print(f"Flight burnout time = {gated_flight.rocket.motor.burn_out_time:.4f} s\n")

    print(
        "Burnout-gated event disabled on "
        f"{burnout_gated.disabled_times[0]:.4f} s\n"
    )

    print(
        "Flight reached altitude of 700m at time = "
        f"{gated_flight.altitude.source[np.argmin(np.abs(gated_flight.altitude.source[:, 1] - 700.0)), 0]:.4f} s\n"
    )

    print(
        "Altitude-gated event disabled on "
        f"{altitude_gated.disabled_times[0]:.4f} s"
    )

  .. note::
    The Altitude-gated event in the example above did not disable at the exact
    time the rocket crossed 700m AGL, because ``disable_on`` is only checked when
    the event is evaluated, which for a continuous event is at the end of each
    solver step. ``exact_time_function`` does not change this: it finds the
    exact moment the ``trigger`` becomes true, not the ``disable_on`` condition.
    To act at the precise crossing, write the crossing as the ``trigger``, give
    the event an ``exact_time_function``, and set ``trigger_only_once=True``.

**enable_on** (optional)
  Automatically enable a disabled event based on a condition. Uses the same
  formats as ``disable_on``:

  - String preset: ``"apogee"`` or ``"burnout"``
  - Simulation time threshold
  - Callable predicate

  .. tip::
    The times when the event is enabled through ``enable_on`` are recorded in
    the list ``event.enabled_times`` for later inspection.

  ``enable_on`` is checked before the trigger and ``disable_on`` after the
  callback. So if both conditions are true on the same check, the event is
  enabled, fires, and is disabled again.

  .. jupyter-execute::

    # Re-enable at specific time
    time_enabled = Event(
        callback=my_callback,
        name="Re-enabled at t=2s",
        enabled=False,
        enable_on=2.0,
        sampling_rate=10,
    )

    # Re-enable via custom condition
    def enable_above_altitude(context):
        return context.height_agl > 500.0

    altitude_enabled = Event(
        callback=my_callback,
        name="Enabled above 500m",
        enabled=False,
        enable_on=enable_above_altitude,
        sampling_rate=10,
    )

    # Run both enable strategies in a single flight
    enabled_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[time_enabled, altitude_enabled],
        name="Enable-on example",
    )

    print(f"Flight time = {enabled_flight.t:.4f} s\n")

    print(
        "Time-enabled event: enabled on "
        f"{time_enabled.enabled_times[0]:.4f} s\n"
    )

    print(
        "Flight reached altitude of 500m at time = "
        f"{enabled_flight.altitude.source[np.argmin(np.abs(enabled_flight.altitude.source[:, 1] - 500.0)), 0]:.4f} s\n"
    )

    print(
        "Altitude-gated event disabled on "
        f"{altitude_enabled.enabled_times[0]:.4f} s"
    )

  .. note::
    The Altitude-enabled event in the example above did not enable at the exact
    time when the rocket crossed 500m AGL because the trigger is only evaluated
    at a sampling rate of 10 Hz. So it triggered on the closest evaluation after
    crossing 500m AGL. 

**exact_time_function** and **exact_time_config** (optional)
  Use these parameters to find the exact moment an event happened between two
  solver steps. This matters for instants like apogee or rail exit, which
  usually fall inside a step and would otherwise be placed at the end of it.

  **exact_time_function** receives the same ``context`` as ``trigger`` and
  returns a number that crosses zero at the event. RocketPy searches the last
  solver step for that crossing, working out the context again at each moment
  it tries, so ``state``, ``height_agl`` and the rest are always the values at
  that moment. Once the crossing is found, ``callback``
  runs with the context of that exact moment. For example, to fire exactly when
  the rocket climbs through 500 m above ground level::

      def altitude(context):
          return context.height_agl - 500

  An event about a state a flight phase follows on its own reads it from
  ``state`` in the same way::

      def my_state_crosses_target(context):
          return context.state.my_state - target_value

  ``state_dot`` and ``conditions`` are worked out at each moment tried as well,
  if the function reads them, so each adds cost to every step of the search.

  Exact-time solving is only available for continuous events
  (``sampling_rate=None``). If no crossing is found, the event fires at the
  end of the solver step instead, with a warning that gives the step searched,
  the values of ``exact_time_function`` at its ends, the reason and what to
  check.

  **exact_time_config** is a dictionary choosing how the crossing is searched
  for. Leave it out to use Brent's method with its defaults, which suits
  almost every event. Any key the chosen solver does not accept raises a
  ``ValueError`` when the event is created.

  - **solver** (str, optional): The search method. **Defaults to**
    ``"brentq"``.

    - ``"brentq"``: Brent's method (``scipy.optimize.brentq``) on the solver's
      interpolated state. Evaluates the function several times inside the
      step and needs its value to change sign across the step.
    - ``"linear"``: Draws a straight line between the values at the two ends
      of the step. Cheapest, and exact only when the value changes linearly.
      Needs its value to change sign across the step.
    - ``"cubic_hermite"``: Fits a cubic curve to the values and rates of
      change at the two ends of the step. Needs ``derivative_function``, and
      the curve must reach the target exactly once inside the step.

  - **target** (float, optional): The value to find the crossing of, for every
    solver. **Defaults to 0.0.** The solver finds the moment when
    ``exact_time_function(context) == target``.

  Only for ``"brentq"``:

  - **xtol** (float, optional): Absolute time tolerance, in seconds.
    **Defaults to 1e-12.**
  - **rtol** (float, optional): Relative time tolerance. **Defaults to 1e-8.**
  - **maxiter** (int, optional): Maximum number of iterations.
    **Defaults to 100.**

  Only for ``"cubic_hermite"``:

  - **derivative_function** (callable, **required**): The rate of change of
    ``exact_time_function`` with respect to time. Takes the same ``context``
    and returns a float.

  .. jupyter-execute::

      def altitude_trigger(context):
          """Check when altitude crosses the target value from below."""
          target_altitude = context.memory["target_altitude"]
          return context.height_agl > target_altitude

      def altitude_exact_time_function(context):
          """Return the altitude above ground level at the moment tried."""
          return context.height_agl

      # Exact-time version, which finds the crossing inside the solver step
      exact_time_event = Event(
          callback=my_callback,
          trigger=altitude_trigger,
          exact_time_function=altitude_exact_time_function,
          exact_time_config={"target": 543.21},
          name="Exact-time altitude detector",
          memory={"target_altitude": 543.21},
          trigger_only_once=True,
      )

      # Non-exact-time version (same trigger, no root refinement)
      sampled_altitude_event = Event(
          callback=my_callback,
          trigger=altitude_trigger,
          name="Sampled altitude detector",
          memory={"target_altitude": 543.21},
          trigger_only_once=True,
      )

      # Run both events in the same flight for comparison
      altitude_flight = Flight(
          rocket=rocket,
          environment=env,
          rail_length=5.2,
          inclination=85,
          heading=0,
          max_time=12.0,
          max_time_step=0.1,
          custom_events=[exact_time_event, sampled_altitude_event],
          name="Exact-time altitude example",
      )

      print("Flight reached target altitude at t = "
            f"{altitude_flight.altitude.source[np.argmin(np.abs(altitude_flight.altitude.source[:, 1] - 543.21)), 0]:.4f} s\n"
      )

      print(f"Exact-time event triggered at t = {exact_time_event.triggered_times[0]:.4f} s\n")
      
      print(f"Sampled event triggered at t = {sampled_altitude_event.triggered_times[0]:.4f} s\n")

**trigger_only_once** (optional)
  When ``True``, the event disables itself after the first successful trigger.
  This is useful for one-shot actions such as deployment or separation.

**time_overshootable** (optional)
  Enables overshoot-path evaluation for sampled events. This will is only 
  relevant for events with finite sampling rates (``sampling_rate`` is a float). 
  When ``False``, the Flight simulation will have strict time nodes at multiples
  the sampling rate. When ``True`` (default), the simulation will integrate past
  the next sampling time and go back to evaluate the event at the correct time,
  which allows for far less integration steps and faster simulation.

  .. important::
    There is little reason to set ``time_overshootable=False``. It can be useful
    for debugging or guaranteeing a consistent time-stepping pattern, but it
    will cause the simulation to be many times slower, with no improvement in
    accuracy.

**name** (optional)
  Human-readable identifier used in logs and debugging. If omitted, RocketPy
  uses the default string ``"Event"``.

**enabled** (optional)
  Initial on/off state. Disabled events can be re-enabled through commands or
  via ``enable_on`` parameter.

**verbose** (optional)
  When ``True``, every check of the event is reported by RocketPy's logger at
  the ``INFO`` level, which is shown with ``Flight(..., verbose=True)`` or
  ``rocketpy.enable_logging()``. Useful to see why an event did or did not
  fire.

.. _event_commands:

Using Commands to Modify Simulation State
-----------------------------------------

When a callback needs to change how the flight goes on, it does so through
``context.commands`` (the same as ``context.event.commands``). The callback
queues one or more commands, and ``Flight`` applies them after the callback
returns.

Available commands include:

- ``context.commands.disable() / enable()``: Disable or re-enable the current event.
- ``context.commands.add_event(event)``: Schedule a new event during the simulation.
- ``context.commands.disable_event(event) / enable_event(event)``: Disable or enable another event or controller event.
- ``context.commands.set_dynamics(dynamics, **phase_kwargs)``: Fly the rest of the flight with a different set of equations of motion.
- ``context.commands.start_flight_phase(phase_name=None, lag=0)``: Start a new flight phase.
- ``context.commands.terminate_flight()``: Request to end the flight simulation immediately after the current step.
- ``context.commands.activate_surface(surface) / deactivate_surface(surface)``: Make an aerodynamic surface start or stop producing force.

**context.commands.disable() / enable()**
  Disable the event that is currently running, from now on. ``enable()`` only
  cancels a ``disable()`` made earlier in the same call: a disabled event does
  not run, so it cannot turn itself back on (give it an ``enable_on`` for
  that).

  ``disable()`` and ``disable_on`` do the same thing. Use ``disable_on`` for a
  condition you know when you build the event, such as ``"apogee"`` or a time:
  it is checked for you and saved with the event. Use ``disable()`` when the
  function itself decides to stop, such as a controller that gives up after
  detecting a fault.

  .. jupyter-execute::

    def disable_after_first_hit(context):
        context.commands.disable()

    disable_event = Event(
        callback=disable_after_first_hit,
        trigger=lambda context: context.state.vz > 20,
        name="Self-disabling event",
    )

    disable_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[disable_event],
        name="Disable example",
    )

    print(f"Flight time = {disable_flight.t:.4f} s")
    print(f"Event enabled after run? {disable_event.enabled}")
    print(f"Disabled at: {disable_event.disabled_times}")


**context.commands.add_event(event)**
  Schedule a new event during the simulation. This is useful when one event
  should introduce a follow-up action later in the flight.

  .. jupyter-execute::

    def add_follow_up(context):
        def follow_up_callback(follow_up_context):
            return f"Follow-up event fired at {follow_up_context.time:.2f} s"

        follow_up_event = Event(
            callback=follow_up_callback,
            trigger=lambda event_context: event_context.time >= 8,
            name="Follow-up event",
            sampling_rate=10,
            trigger_only_once=True,
        )
        context.commands.add_event(follow_up_event)

    add_event = Event(
        callback=add_follow_up,
        trigger=lambda context: context.time > 2,
        name="Event adder",
        trigger_only_once=True,
    )

    add_event_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[add_event],
        name="Add event example",
    )

    print(f"Flight time = {add_event_flight.t:.4f} s")
    print(f"Total custom events after run: {len(add_event_flight.custom_events)}")
    print(f"Follow-up event added at: {add_event.triggered_times}")
    print(f"Follow-up event ran at: {add_event_flight.custom_events[1].triggered_times}")

**context.commands.disable_event(event)**
  Disable another event or controller event. This is useful when one trigger
  should shut off a later response.

  .. jupyter-execute::

    target_event = Event(
        callback=lambda context: f"Called at {context.time:.2f} s",
        name="Target event",
        sampling_rate=2,
    )

    def disable_other_event(context):
        context.commands.disable_event(target_event)

    disable_other = Event(
        callback=disable_other_event,
        trigger=lambda context: context.time >= 2,
        name="Disabler",
        sampling_rate=10,
        trigger_only_once=True,
    )

    disable_other_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[target_event, disable_other],
        name="Disable other example",
    )

    print(f"Target event enabled after run? {target_event.enabled}")
    print(f"Target event disabled at: {target_event.disabled_times}")


**context.commands.set_dynamics(dynamics, \*\*phase_kwargs)**
  Fly the rest of the flight with a different set of equations of motion.
  RocketPy ships five of them, importable from ``rocketpy.simulation.helpers``:

  - ``RAIL_DYNAMICS``: still on the launch rail.
  - ``SOLID_PROPULSION_DYNAMICS``: motor burning.
  - ``SIX_DOF_DYNAMICS``: free flight, motor off.
  - ``THREE_DOF_DYNAMICS``: free flight without rotation.
  - ``PARACHUTE_DYNAMICS``: descending under a parachute.

  Extra keyword arguments are passed on to the equations at every step. That is
  how a parachute descent knows which parachute is open:
  ``set_dynamics(PARACHUTE_DYNAMICS, parachute=main)``.

  Writing your own set of equations is possible but internal for now, so it is
  not covered here. Below, a flight switches to a three degree of freedom model
  partway up, and the phases it flew are printed at the end.

  .. jupyter-execute::

    from rocketpy.simulation.helpers import THREE_DOF_DYNAMICS

    def switch_dynamics(context):
        context.commands.set_dynamics(THREE_DOF_DYNAMICS)

    dynamics_event = Event(
        callback=switch_dynamics,
        trigger=lambda context: context.time >= 6,
        name="Dynamics switcher",
        trigger_only_once=True,
    )

    dynamics_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[dynamics_event],
        name="Dynamics example",
    )

    print(f"Flight time = {dynamics_flight.t:.4f} s")
    print(f"Switched dynamics at: {dynamics_event.triggered_times}")
    print("Flight phases: ")
    for _, phase in dynamics_flight.flight_phases:
        print(phase)
    print(dynamics_flight.flight_phases[-1])


**context.commands.start_flight_phase(phase_name=None, lag=0)**
  Start a new flight phase at the trigger, or ``lag`` seconds after it. During
  the lag the flight keeps its current equations of motion, which is how a
  parachute keeps falling freely while it opens. Use it when the callback wants
  the solver restarted at the trigger. To fly the new phase with different
  equations of motion, such as
  those of a deployed parachute, use ``set_dynamics`` instead, for example
  ``context.commands.set_dynamics(PARACHUTE_DYNAMICS, parachute=main)``.

  .. jupyter-execute::

    def start_new_phase(context):
      context.commands.start_flight_phase(phase_name="custom_descent")

    phase_event = Event(
      callback=start_new_phase,
      trigger=lambda context: context.time >= 6,
      name="Phase starter",
      trigger_only_once=True,
      )

    phase_flight = Flight(
      rocket=rocket,
      environment=env,
      rail_length=5.2,
      inclination=85,
      heading=0,
      max_time=12.0,
      max_time_step=0.1,
      custom_events=[phase_event],
      name="Phase example",
    )

    print(f"Flight time = {phase_flight.t:.4f} s")
    print(f"Number of phases: {len(phase_flight.flight_phases)}")
    print("Flight phases: ")
    for _, phase in phase_flight.flight_phases:
        print(phase)
    print(phase_flight.flight_phases[-1])


**context.commands.terminate_flight()**
  End the simulation immediately.

  .. jupyter-execute::

    def stop_on_trigger(context):
        context.commands.terminate_flight()

    stop_event = Event(
        callback=stop_on_trigger,
        trigger=lambda context: context.state.vz <= 0,
        name="Terminate on trigger",
    )

    stop_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=30.0,
        custom_events=[stop_event],
        name="Terminate example",
    )

    print(f"Flight terminated at t = {stop_flight.t:.4f} s")
    print(f"Trigger time: {stop_event.triggered_times[-1]:.4f} s")

**context.commands.activate_surface(surface) / deactivate_surface(surface)**
  Make an aerodynamic surface start or stop producing force from this moment
  on. Use it for a part of the rocket that appears or disappears during the
  flight. The surface must already be one of the rocket's aerodynamic surfaces. A surface that should
  only appear later is built with ``active=False``, so that it starts the
  flight switched off.

  The switch lasts for the rest of this flight only. The surface itself is not
  modified, so another flight of the same rocket starts from its ``active``
  setting again.

  .. code-block:: python

    def switch_off(context):
        context.commands.deactivate_surface(my_surface)

    switch_event = Event(
        callback=switch_off,
        trigger=lambda context: context.state.vz < 0,  # past apogee
        trigger_only_once=True,
        name="Switch surface off",
    )

  .. seealso::
    See :ref:`active_during` for a complete example.

See also
--------

- :doc:`../technical/simulation_loop` for the internal simulation loop.
- :doc:`first_simulation` for a complete RocketPy setup.
- :class:`rocketpy.Flight` for the main simulation driver.
- :class:`rocketpy.Event` for the public event API.