Event Usage
===========

.. _eventusage:

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


Parameters for Callback and Trigger Functions
---------------------------------------------

At the most basic level, an event has two callables:

- ``trigger(context)`` returns ``True`` when the event should fire.
- ``callback(context)`` performs the requested action.

The same ``context`` dictionary is passed to both callables. It holds the
simulation state and objects at the moment of the check. Some of its values are
only worked out when a function reads them, and are then kept for the rest of
the solver step, so reading a value you do not use costs nothing.

The following keys are available:

**Simulation time and state:**

- ``time`` (float): The current simulation time in seconds. For an event with
  an ``exact_time_function``, the callback receives the exact moment the event
  happened, and every other value below is given at that moment too.
- ``state`` (list of float): The state vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``
  where ``(x, y, z)`` is position, ``(vx, vy, vz)`` is velocity,
  ``(e0, e1, e2, e3)`` are quaternion orientation components, and
  ``(wx, wy, wz)`` is angular velocity.
- ``height_agl`` (float): Height of the rocket above ground level
  in meters, computed as ``state[2] - env.elevation``. Always present.
- ``state_dot`` (list of float): The time derivative of state,
  ``[vx, vy, vz, ax, ay, az, e0_dot, e1_dot, e2_dot, e3_dot, wx_dot, wy_dot, wz_dot]``.
  Worked out when first read, by evaluating the equations of motion once more.
- ``phase_state_dot`` (dict): The time derivative of each entry in
  ``phase_state``, under the same names. This is how fast the phase's own
  states are changing, for example a parafoil's turn rate. Comes from the same
  evaluation as ``state_dot``, so reading both costs no more than one.
- ``pressure`` (float): Current atmospheric pressure in Pa at the rocket's
  altitude. Worked out when first read.
- ``previous_state`` (list of float or ``None``): The ``state`` from the
  previous time this same event was checked, or ``None`` on the first check.
- ``previous_phase_state`` (dict or ``None``): The ``phase_state`` from that
  same previous check, or ``None`` on the first check.

  It is also ``None`` on the first check after the flight enters a new phase.
  The states a phase follows are only comparable within that phase, since the
  next one may follow a different set, so a new phase starts a fresh sequence.
  The canonical ``previous_state`` has no such break: every phase can report
  the canonical states, so it carries straight through a phase change.
- ``previous_time`` (float or ``None``): The ``time`` of that previous check,
  in seconds, or ``None`` on the first check.

  Use these two to fire on a value *crossing* a threshold rather than merely
  being past it. They are always consecutive checks of this same event, so a
  crossing cannot slip between them::

      def trigger(context):
          previous = context["previous_state"]
          if previous is None:  # the first check has nothing to compare against
              return False
          return previous[5] > 0 >= context["state"][5]  # vz turned negative

  Do not reach into ``context['flight'].solution`` for this. It holds the stored
  trajectory points, which advance independently of when your event is checked,
  so a crossing can fall between two of them and never be seen. Use the flight
  solution when you want the trajectory itself, not the previous value this
  event saw.
- ``step_size`` (float): How long the simulation has been inside the
  solver step your callback is being evaluated in, in seconds. At the end of a
  step that is the whole step the solver just took; at a point interpolated
  inside a step it is the part of the step up to that point.

  This is a property of the integrator, not of your event. It is **not** the
  time since your callback last ran, and it is **not** ``1 / sampling_rate``: a
  sampled event is checked on its own schedule, which has nothing to do with
  where the solver put its step boundaries, and the solver changes its step size
  throughout the flight. To measure how much time passed between two checks of
  your own event, subtract ``previous_time`` from ``time``.

- ``phase_state`` (dict): The states the current flight phase actually
  integrates, keyed by name. For every phase RocketPy ships this is the same
  thirteen values as ``state``, but a phase is free to follow fewer, or to
  carry states of its own that the canonical thirteen cannot hold, such as a
  parafoil heading or an air brake position. Read those here::

      heading = context["phase_state"]["heading"]

  Prefer ``state`` for anything a phase always has, so your callback keeps
  working when the flight switches to a different set of equations, and reach
  for ``phase_state`` when you are controlling something specific to one phase.

**Simulation objects:**

- ``flight`` (:class:`rocketpy.Flight`): The Flight instance orchestrating 
  the simulation.
- ``rocket`` (:class:`rocketpy.Rocket`): The Rocket object being simulated.
- ``environment`` (:class:`rocketpy.Environment`): The Environment conditions 
  for the flight.
- ``phase``: The flight phase being flown, as listed in
  ``flight.flight_phases``.

**Event and sensor data:**

- ``event`` (:class:`rocketpy.Event`): A reference to the Event object itself,
  allowing access to ``memory``, ``commands``, and other event state.
- ``sensors`` (list): The sensors attached to the rocket. Each sensor has a
  ``measurement`` attribute with the most recent value.
- ``sensors_by_name`` (dict): The same sensors keyed by name (or class name).
  If multiple sensors share the same name, the value is a list.
- ``sampling_rate`` (float or None): The sampling rate of the event in Hz, or
  ``None`` for a continuous event.

**Keeping your own data:**

- The event's own dictionary is ``context["event"].memory``. It is kept from
  one check to the next, so keep counters, thresholds and anything shared
  between trigger and callback there.
  Do not add keys to the ``context`` passed to your function: it is shared by
  every event checked in the same solver step.

Understanding Event Parameters
------------------------------

The Event constructor accepts many parameters. Here we explain each one with
practical examples and demonstrate proper activation.

**callback** (required)
  The function that runs when the event triggers. It takes the ``context``
  dictionary as its single argument, and all of its returns are saved on the ``callback_log`` list for later 
  inspection. The callback can also queue commands to modify the simulation
  state through the ``event.commands`` interface. Access to the
  ``event.memory`` dictionary is also possible, and allows the callback to
  store data across triggers.

  .. seealso:: 
    Check the :ref:`event_commands` section for details.

  .. jupyter-execute::

      def my_callback(context):
          time = context["time"]
          return {"time": time}

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
      
      print(f"Callback log: {simple_event.callback_log}\n")
      
      print(simple_event)

**trigger** (optional)
  A callable that returns ``True`` when the event should fire. If ``None``, the
  event acts as a passive hook and always triggers when called.

  .. note::
    The trigger function receives the same ``context`` as the callback, and
    it has all the same functionalities, except that its return value is
    interpreted as a boolean condition instead of it being logged.


  .. tip::
    A list of all triggered times can be accessed with ``event.triggered_times``.


  .. jupyter-execute::

      def simple_trigger(context):
          """Triggers when vertical velocity becomes large (enough)."""
          return context["state"][5] > 50  # vz > 50 m/s

      def simple_callback(context):
          return {"status": "event triggered!"}

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
  write it as ``context["event"].memory`` inside the trigger and callback. It
  is restored to the values given here when the event is reset for a new
  flight.

  .. note::
    ``memory`` is not logged: what the callback writes there is **not**
    persisted to output logs or files.

  .. jupyter-execute::

      def counting_callback(context):
          event = context["event"]
          event.memory["count"] += 1
          event.memory["last_time"] = context["time"]
          return None

      counter_event = Event(
          callback=counting_callback,
          name="Counting event",
          memory={"count": 0, "last_time": None},
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
        return context["height_agl"] > 700.0

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
        return context["height_agl"] > 500.0

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
  it tries, so ``state``, ``height_agl``, ``phase_state`` and the rest are
  always the values at that moment. Once the crossing is found, ``callback``
  runs with the context of that exact moment. For example, to fire exactly when
  the rocket climbs through 500 m above ground level::

      def altitude(context):
          return context["height_agl"] - 500

  An event about a state the canonical thirteen cannot hold, such as a
  parafoil heading, reads it from ``phase_state`` in the same way::

      def heading_crosses_target(context):
          return context["phase_state"]["heading"] - target_heading

  ``state_dot``, ``phase_state_dot`` and ``pressure`` are worked out at each
  moment tried as well, if the function reads them, so each adds cost to every
  step of the search.

  Exact-time solving is only available for continuous events
  (``sampling_rate=None``). If no crossing is found, RocketPy warns and the
  event fires at the end of the solver step instead.

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
    - ``"cubic_hermite"``: Fits a cubic curve to the values and rates of
      change at the two ends of the step. Needs ``derivative_function``.

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
  - **max_abs_imag** (float, optional): Largest imaginary part a root of the
    cubic may have and still be treated as real. **Defaults to 1e-3.**

  .. jupyter-execute::

      def altitude_trigger(context):
          """Check when altitude crosses the target value from below."""
          target_altitude = context["event"].memory["target_altitude"]
          return context["height_agl"] > target_altitude

      def altitude_exact_time_function(context):
          """Return the altitude above ground level at the moment tried."""
          return context["height_agl"]

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

**changes_dynamics** (optional)
  Signals that the event's callback will change the simulation dynamics or
  parameters that affect the ODE derivative. Examples changing a motor thrust 
  profile, changing a parameter that affects the aerodynamic forces, or any
  object mutation that affects the current rocket configuration.

**name** (optional)
  Human-readable identifier used in logs and debugging. If omitted, RocketPy
  uses the default string ``"Event"``.

**enabled** (optional)
  Initial on/off state. Disabled events can be re-enabled through commands or
  via ``enable_on`` parameter.

**verbose** (optional)
  When ``True``, the event prints a message and stores extra execution logs
  whenever it triggers.

.. _event_commands:

Using Commands to Modify Simulation State
-----------------------------------------

When a callback needs to change the simulation loop state, it does so through
``event.commands``. The callback queues one or more commands, and ``Flight``
applies them after the callback returns.

Available commands include:

- ``event.commands.disable() / enable()``: Disable or re-enable the current event.
- ``event.commands.add_event(event)``: Schedule a new event during the simulation. Not for events with ``changes_dynamics=True``; see the warning below.
- ``event.commands.disable_event(event)``: Disable another event or controller event.
- ``event.commands.set_dynamics(dynamics, **phase_kwargs)``: Fly the rest of the flight with a different set of equations of motion.
- ``event.commands.start_flight_phase(phase_name=None, lag=0)``: Start a new flight phase.
- ``event.commands.terminate_flight()``: Request to end the flight simulation immediately after the current step.

**event.commands.disable() / enable()**
  Disable or re-enable the event that is currently running. This is useful for
  one-shot behaviors or for temporarily muting an event after it fires.

  .. jupyter-execute::

    def disable_after_first_hit(context):
        event = context["event"]
        event.commands.disable()
        return {"action": "disabled self"}

    disable_event = Event(
        callback=disable_after_first_hit,
        trigger=lambda context: context["state"][5] > 20,
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
    print(f"Disable log entries: {disable_event.callback_log}")


**event.commands.add_event(event)**
  Schedule a new event during the simulation. This is useful when one event
  should introduce a follow-up action later in the flight.

  .. jupyter-execute::

    def add_follow_up(context):
        event = context["event"]

        def follow_up_callback(**follow_up_kwargs):
            return f"Follow-up event fired at {follow_up_context['time']:.2f} s"

        follow_up_event = Event(
            callback=follow_up_callback,
            trigger=lambda event_context: event_context["time"] >= 8,
            name="Follow-up event",
            sampling_rate=10,
            trigger_only_once=True,
        )
        event.commands.add_event(follow_up_event)
        return f"Added follow-up event at {context['time']:.2f} s"

    add_event = Event(
        callback=add_follow_up,
        trigger=lambda context: context["time"] > 2,
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
    print(f"First Event log: {add_event.callback_log}")
    print(f"Follow-up Event log: {add_event_flight.custom_events[1].callback_log}")

  .. warning::

     Avoid adding an event with ``changes_dynamics=True`` this way. A flight
     decides before it starts, from the events it was built with, whether to
     record the accelerations, aerodynamic forces and moments and net thrust
     step by step. An event added mid-flight arrives too late for that: the
     trajectory is still right, but those variables are worked out afterwards
     from the rocket as it ends the flight, so what the event changed is
     reported for the whole flight. RocketPy raises a ``UserWarning`` when
     this happens.

     Instead, give the event to the ``Flight`` with ``enabled=False`` and turn
     it on with ``enable_on`` or the ``enable`` command. If the events really
     cannot be written in advance, give the ``Flight`` one disabled placeholder
     with ``changes_dynamics=True``, which makes it record from the start:

     .. code-block:: python

         recording_placeholder = Event(
             callback=lambda context: None,
             trigger=lambda context: False,
             name="Force post-process recording",
             enabled=False,
             changes_dynamics=True,
         )

     Recording costs one extra evaluation of the equations of motion per
     stored step, so only add the placeholder when you will need it.

**event.commands.disable_event(event)**
  Disable another event or controller event. This is useful when one trigger
  should shut off a later response.

  .. jupyter-execute::

    def disable_other_event(context):
        event = context["event"]
        flight = context["flight"]
        target_event = flight.custom_events[0]  # Assuming the target event is the first one added to the flight.
        event.commands.disable_event(target_event)
        return {"action": f"disabled {target_event.name}"}

    target_event = Event(
        callback=lambda context: f"Called at {context['time']:.2f} s",
        name="Target event",
        sampling_rate=2,
    )

    disable_other = Event(
        callback=disable_other_event,
        trigger=lambda context: context["time"]>= 2,
        name="Disabler",
        memory={"target_event": target_event},
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
    print(f"Target event log: {target_event.callback_log}")


**event.commands.set_dynamics(dynamics, \*\*phase_kwargs)**
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

  The event must be created with ``changes_dynamics=True``.

  Writing your own set of equations is possible but internal for now, so it is
  not covered here. Below, a flight switches to a three degree of freedom model
  partway up, and the phases it flew are printed at the end.

  .. jupyter-execute::

    from rocketpy.simulation.helpers import THREE_DOF_DYNAMICS

    def switch_dynamics(context):
        event = context["event"]
        event.commands.set_dynamics(THREE_DOF_DYNAMICS)
        return {"action": "switched to three degrees of freedom"}

    dynamics_event = Event(
        callback=switch_dynamics,
        trigger=lambda context: context["time"] >= 6,
        name="Dynamics switcher",
        trigger_only_once=True,
        changes_dynamics=True,
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
    print(f"Dynamics event log: {dynamics_event.callback_log}")
    print("Flight phases: ")
    for _, phase in dynamics_flight.flight_phases:
        print(phase)
    print(dynamics_flight.flight_phases[-1])


**event.commands.start_flight_phase(phase_name=None, lag=0)**
  Start a new flight phase at the trigger, or ``lag`` seconds after it. During
  the lag the flight keeps its current equations of motion, which is how a
  parachute keeps falling freely while it opens. Use it when the callback wants
  the solver restarted at the trigger. To fly the new phase with different
  equations of motion, such as
  those of a deployed parachute, use ``set_dynamics`` instead, for example
  ``event.commands.set_dynamics(PARACHUTE_DYNAMICS, parachute=main)``.

  .. jupyter-execute::

    def start_new_phase(context):
      event = context["event"]
      event.commands.start_flight_phase(phase_name="custom_descent")
      return {"action": "started new phase"}

    phase_event = Event(
      callback=start_new_phase,
      trigger=lambda context: context["time"] >= 6,
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


**event.commands.terminate_flight()**
  End the simulation immediately.

  .. jupyter-execute::

    def stop_on_trigger(context):
        event = context["event"]
        event.commands.terminate_flight()
        return {
            "action": "termination requested",
            "trigger_time": context["time"],
            "flight_time": context["flight"].t,
        }

    stop_event = Event(
        callback=stop_on_trigger,
        trigger=lambda context: context["state"][5] <= 0,
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
    if stop_event.callback_log and stop_event.callback_log[-1]:
        log = stop_event.callback_log[-1]
        print(f"Trigger time: {log['trigger_time']:.4f} s")
        print(f"Flight time reported by callback: {log['flight_time']:.4f} s")

See also
--------

- :doc:`../technical/simulation_loop` for the internal simulation loop.
- :doc:`first_simulation` for a complete RocketPy setup.
- :class:`rocketpy.Flight` for the main simulation driver.
- :class:`rocketpy.Event` for the public event API.