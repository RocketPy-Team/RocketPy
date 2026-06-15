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
time nodes, see :doc:`../technical/phases_and_nodes`.

.. important::
  **Performance Considerations**

  Adding events to a flight simulation can impact computational performance,
  especially in the following scenarios:

  - **Large sampling rates**: Events with very small sampling intervals
    (e.g., 0.01 seconds) will be evaluated more frequently, increasing overhead.
  - **Exact-time solving**: Events with ``exact_time_function`` require root-finding
    iterations to pinpoint the exact trigger time, which adds computational cost
    per event call.
  - **Callbacks with expensive computations**: Callback functions that perform
    heavy calculations (large matrix operations, I/O, external API calls) directly
    affect flight simulation speed. Try to keep callbacks lightweight or cache results.
  - **The ``needs`` parameter**: By default the simulation computes none of the
    expensive kwargs (``state_dot``, ``pressure``, ``state_history``). If your
    event accesses any of these, declare them via ``needs=['pressure']`` (or
    whichever keys apply) so the runtime computes them. Omitting a required key
    raises a ``KeyError`` with a message telling you exactly which key to add.
    See the ``needs`` parameter description below for details.

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

- ``trigger(**kwargs)`` returns ``True`` when the event should fire.
- ``callback(**kwargs)`` performs the requested action.

The same ``kwargs`` dictionary is passed to both callables. It always includes
the simulation state, objects, and any custom items from the event ``context``.

The following parameters are always or conditionally available:

**Simulation time and state:**

- ``time`` (float): The current simulation time in seconds.
- ``state`` (list of float): The state vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``
  where ``(x, y, z)`` is position, ``(vx, vy, vz)`` is velocity,
  ``(e0, e1, e2, e3)`` are quaternion orientation components, and
  ``(wx, wy, wz)`` is angular velocity.
- ``height_agl`` (float): Height of the rocket above ground level
  in meters, computed as ``state[2] - env.elevation``. Always present.
- ``state_dot`` (list of float, **conditional**): The time derivative of state,
  ``[vx, vy, vz, ax, ay, az, e0_dot, e1_dot, e2_dot, e3_dot, wx_dot, wy_dot, wz_dot]``.
  Only injected when the event declares ``needs=['state_dot']``.
- ``pressure`` (float, **conditional**): Current atmospheric pressure in Pa at
  the rocket's altitude. Only injected when the event declares
  ``needs=['pressure']``.
- ``state_history`` (list of list, **conditional**): History of state vectors
  from the beginning of the simulation up to the current step. Each entry is
  ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``. Only injected when
  the event declares ``needs=['state_history']``.
- ``sampled_time`` (float, optional): If exact-time solving was used, the
  original sampled time before refinement.
- ``sampled_state`` (list of float, optional): If exact-time solving was used,
  the interpolated state at the originally sampled time.
- ``step_size`` (float, optional): Most recent solver step size in seconds.
  Useful for cubic-Hermite exact-time configuration.

**Simulation objects:**

- ``flight`` (:class:`rocketpy.Flight`): The Flight instance orchestrating 
  the simulation.
- ``rocket`` (:class:`rocketpy.Rocket`): The Rocket object being simulated.
- ``environment`` (:class:`rocketpy.Environment`): The Environment conditions 
  for the flight.

**Event and sensor data:**

- ``event`` (:class:`rocketpy.Event`): A reference to the Event object itself,
  allowing access to ``context``, ``commands``, and other event state.
- ``sensors`` (dict): A dictionary mapping sensor names (or class names) to 
  sensor instances. Each sensor has a ``measurement`` attribute with the most 
  recent value. If multiple sensors share the same name, the value is a list.
- ``sampling_rate`` (float or None): The sampling interval of the event in 
  seconds (or ``None`` for continuous events).

**Custom parameters:**

- Any additional key-value pairs defined in the event's ``context`` dictionary 
  are unpacked and passed as separate ``kwargs``.

Understanding Event Parameters
------------------------------

The Event constructor accepts many parameters. Here we explain each one with
practical examples and demonstrate proper activation.

**callback** (required)
  The function that runs when the event triggers. It must accept ``**kwargs``
  and all of its returns are saved on the ``callback_log`` list for later 
  inspection. The callback can also queue commands to modify the simulation
  state through the ``event.commands`` interface. Access to the 
  ``event.context`` dictionary is also possible, and allows the callback to
  store persistent state across triggers.

  .. seealso:: 
    Check the :ref:`event_commands` section for details.

  .. jupyter-execute::

      def my_callback(**kwargs):
          time = kwargs["time"]
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
    The trigger function receives the same ``**kwargs`` as the callback, and
    it has all the same functionalities, except that its return value is
    interpreted as a boolean condition instead of it being logged.


  .. tip::
    A list of all triggered times can be accessed with ``event.triggered_times``.


  .. jupyter-execute::

      def simple_trigger(**kwargs):
          """Triggers when vertical velocity becomes large (enough)."""
          return kwargs["state"][5] > 50  # vz > 50 m/s

      def simple_callback(**kwargs):
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

**context** (optional)
  A dictionary for persistent, mutable event state. Useful for counters,
  thresholds, and shared data between trigger and callback. To access and modify
  the context, use ``event.context`` within the trigger and callback functions.

  .. note::
    The event's ``context`` is not logged, modifications to
    ``event.context`` are not recorded in the ``Event`` class are
    **not** automatically persisted to output logs or files.

  .. jupyter-execute::

      def counting_callback(**kwargs):
          event = kwargs["event"]
          event.context["count"] += 1
          event.context["last_time"] = kwargs["time"]
          return None

      counter_event = Event(
          callback=counting_callback,
          name="Counting event",
          context={"count": 0, "last_time": None},
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
          name="Context example",
      )

      print(f"Flight time = {counter_flight.t:.4f} s\n")
      print(
          "Event triggered "
          f"{len(counter_event.triggered_times)} time(s)\n"
      )
      print(f"Final event context: {counter_event.context}")

**disable_on** (optional)
  Automatically disable the event based on a condition. Can be:

  - A string preset: ``"apogee"`` or ``"burnout"``
  - A float/int: simulation time in seconds (e.g., ``120.0`` disables at t=120s)
  - A callable: any callable with the signature ``function(**kwargs)`` that 
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
    def disable_above_altitude(**kwargs):
        # return True to disable when altitude above 1000m AGL
        return kwargs["height_agl"] > 700.0

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
    The Altitude-gated event in the example above did not disable in the exact
    time when the rocket crossed 700m AGL because the trigger is only evaluated
    at the solver time nodes. If you need to disable at a more precise instant,
    consider using the ``exact_time_function`` parameter.

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
    def enable_above_altitude(**kwargs):
        return kwargs["height_agl"] > 500.0

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
  Use these parameters to refine the event trigger instant to an exact time 
  between solver steps. This is critical for accurately capturing instants
  like apogee or rail exit that can occur mid-step and would otherwise 
  be approximated to the nearest solver time node.

  **exact_time_function** must be a callable with signature 
  ``exact_time_function(state, **kwargs) -> float``. The mandatory ``state``
  argument receives the interpolated solver state vector without time. Any 
  additional keyword arguments are the same context values described in
  `Parameters for Callback and Trigger Functions`_. The function should 
  return a scalar value whose root (zero crossing) defines the event instant.

  .. important::
    Exact-time functions must derive their root quantity directly from the 
    interpolated ``state`` argument. Do not use derived kwargs like
    ``height_agl``. The root solver can only refine roots
    based on values present in the interpolated state.

  **exact_time_config** is a dictionary specifying which solver to use and its 
  parameters. If no configuration is passed or the dictionary is empty, the 
  default ``brentq``-based solver is used with sensible defaults. Only the 
  keys you wish to override need to be provided. Below is a detailed breakdown
  of the keys available in ``exact_time_config``:

  **Solver Selection and Defaults:**

  - **solver** (str, optional): Which root-finding algorithm to use. Options are:

    - ``"linear"``: Linear interpolation between steps.
    - ``"cubic_hermite"``: Cubic Hermite spline interpolation.
    - Omitted/not provided: Falls back to the default ``brentq`` solver from 
      ``scipy.optimize``.

  **Configuration Keys by Solver:**

  Used by the default ``brentq`` solver (when ``solver`` is omitted):

  - **target** (float, optional): The target value for the root that the solver 
    finds. **Defaults to 0.0.** The solver locates the time when 
    ``exact_time_function(state, **kwargs) == target``.
  - **xtol** (float, optional): Absolute tolerance for the root-finder. 
    **Defaults to 1e-12.** Passed as ``xtol`` to ``scipy.optimize.brentq``.

  - **rtol** (float, optional): Relative tolerance for the root-finder.
    **Defaults to 1e-8.** Passed as ``rtol`` to ``scipy.optimize.brentq``.

  - **maxiter** (int, optional): Maximum number of iterations.
    **Defaults to 100.** Passed as ``maxiter`` to ``scipy.optimize.brentq``.

  - Note: Brent's method requires the event function to change sign over the
    bracket (the solver will raise a ``ValueError`` if no sign change is
    detected). The helper raises a user-facing warning when exact-time solving
    fails and falls back to the sampled time.

  Used by the ``"linear"`` solver (when ``solver="linear"``):

  - **target** (float, optional): The target value for the root that the solver 
    finds. **Defaults to 0.0.** The solver locates the time when 
    ``exact_time_function(state, **kwargs) == target``.

  Used by the ``"cubic_hermite"`` solver (when ``solver="cubic_hermite"``):

  - **target** (float, optional): The target value for the root that the solver 
    finds. **Defaults to 0.0.** The solver locates the time when 
    ``exact_time_function(state, **kwargs) == target``.

  - **derivative_function** (callable, **required**): Time derivative of the 
    exact-time function. There is no default; you must provide this. Receives 
    the same parameters as ``exact_time_function`` (``state, **kwargs``). Should 
    return the time derivative as a float.

  - **step_end_function** (callable, optional): Provides a step-size estimate 
    for refining the search interval. **Defaults to None** (uses the solver 
    step duration). Called as
    ``step_end_function(previous_state=..., current_state=..., **context)``.

  - **max_abs_imag** (float, optional): Maximum allowed absolute imaginary part 
    for roots (tolerance for numerical artifacts). **Defaults to 1e-3.**

  .. tip::
    **Most events do not need explicit configuration.** Simply omit 
    ``exact_time_config`` entirely and use the default ``brentq`` solver with
    its sensible defaults (``root_tolerance=1e-8``, ``max_iterations=100``, 
    ``target=0.0``). All three solvers (brentq, linear, cubic_hermite) support
    the ``target`` parameter—by default all find the root at ``target=0.0``.

    Exact-time functions must compute their crossing value from the interpolated
    ``state`` argument. Do not rely on derived kwargs such as
    ``height_agl`` inside ``exact_time_function`` because those
    values are evaluated at the sampled step, not at the refined root time.

    Configure the exact-time solver only if you have specific requirements 
    (e.g., a non-zero target, a non-smooth event function, or strict performance 
    constraints).

  .. jupyter-execute::

      def altitude_trigger(**kwargs):
          """Check when altitude crosses the target value from below."""
          state = kwargs["state"]
          flight = kwargs["flight"]
          target_altitude = kwargs["event"].context["target_altitude"]
          return state[2] - flight.env.elevation > target_altitude

      def altitude_exact_time_function(state, **kwargs):
          """Return altitude above ground from the interpolated state."""
          flight = kwargs["flight"]
          return state[2] - flight.env.elevation

      # Exact-time version with root refinement
      exact_time_event = Event(
          callback=my_callback,
          trigger=altitude_trigger,
          exact_time_function=altitude_exact_time_function,
          exact_time_config={"target": 543.21},
          name="Exact-time altitude detector",
          context={"target_altitude": 543.21},
          trigger_only_once=True,
      )

      # Non-exact-time version (same trigger, no root refinement)
      sampled_altitude_event = Event(
          callback=my_callback,
          trigger=altitude_trigger,
          name="Sampled altitude detector",
          context={"target_altitude": 543.21},
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

**needs** (optional)
  Declares which expensive simulation values the event's trigger and callback
  actually access. Pass a list (or frozenset) containing any combination of
  ``'state_dot'``, ``'pressure'``, and ``'state_history'``.
  When ``None`` (the default), none of the expensive values are computed,
  equivalent to passing an empty list.

  The most expensive value is ``'state_dot'``, which requires a full ODE
  right-hand-side evaluation. ``'state_history'`` is also non-trivial
  (a list copy that grows with simulation time). ``'pressure'`` requires one
  interpolation lookup.

  
.. _event_commands:

Using Commands to Modify Simulation State
-----------------------------------------

When a callback needs to change the simulation loop state, it does so through
``event.commands``. The callback queues one or more commands, and ``Flight``
applies them after the callback returns.

Available commands include:

- ``event.commands.disable() / enable()``: Disable or re-enable the current event.
- ``event.commands.add_event(event)``: Schedule a new event during the simulation.
- ``event.commands.disable_event(event)``: Disable another event or controller event.
- ``event.commands.set_derivative(callable)``: Replace the current phase derivative with a new callable.
- ``event.commands.start_flight_phase(phase_name=None, lag=0, parachute=None)``: Start a new flight phase.
- ``event.commands.terminate_flight()``: Request to end the flight simulation immediately after the current step.

**event.commands.disable() / enable()**
  Disable or re-enable the event that is currently running. This is useful for
  one-shot behaviors or for temporarily muting an event after it fires.

  .. jupyter-execute::

    def disable_after_first_hit(**kwargs):
        event = kwargs["event"]
        event.commands.disable()
        return {"action": "disabled self"}

    disable_event = Event(
        callback=disable_after_first_hit,
        trigger=lambda **kwargs: kwargs["state"][5] > 20,
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

    def add_follow_up(**kwargs):
        event = kwargs["event"]

        def follow_up_callback(**follow_up_kwargs):
            return f"Follow-up event fired at {follow_up_kwargs['time']:.2f} s"

        follow_up_event = Event(
            callback=follow_up_callback,
            trigger=lambda **event_kwargs: event_kwargs["time"] >= 8,
            name="Follow-up event",
            sampling_rate=10,
            trigger_only_once=True,
        )
        event.commands.add_event(follow_up_event)
        return f"Added follow-up event at {kwargs['time']:.2f} s"

    add_event = Event(
        callback=add_follow_up,
        trigger=lambda **kwargs: kwargs["time"] > 2,
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

**event.commands.disable_event(event)**
  Disable another event or controller event. This is useful when one trigger
  should shut off a later response.

  .. jupyter-execute::

    def disable_other_event(**kwargs):
        event = kwargs["event"]
        flight = kwargs["flight"]
        target_event = flight.custom_events[0]  # Assuming the target event is the first one added to the flight.
        event.commands.disable_event(target_event)
        return {"action": f"disabled {target_event.name}"}

    target_event = Event(
        callback=lambda **kwargs: f"Called at {kwargs['time']:.2f} s",
        name="Target event",
        sampling_rate=2,
    )

    disable_other = Event(
        callback=disable_other_event,
        trigger=lambda **kwargs: kwargs["time"]>= 2,
        name="Disabler",
        context={"target_event": target_event},
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

**event.commands.set_derivative(callable)**
  Replace the current phase derivative with a new callable. This is used when
  the flight dynamics should change without starting a separate phase.

  .. jupyter-execute::

    def dummy_derivative(_t, u, *args, **kwargs):
        return np.zeros_like(u)

    def switch_derivative(**kwargs):
        event = kwargs["event"]
        event.commands.set_derivative(dummy_derivative)
        return {"action": "switched derivative"}

    derivative_event = Event(
        callback=switch_derivative,
        trigger=lambda **kwargs: kwargs["time"] >=6,
        name="Derivative switcher",
        trigger_only_once=True,
    )

    derivative_flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=12.0,
        max_time_step=0.1,
        custom_events=[derivative_event],
        name="Derivative example",
    )

    print(f"Flight time = {derivative_flight.t:.4f} s")
    print(f"Derivative event log: {derivative_event.callback_log}")
    print("Flight phases: ")
    for _, phase in derivative_flight.flight_phases:
        print(phase)
    print(derivative_flight.flight_phases[-1])


**event.commands.start_flight_phase(phase_name=None, lag=0, parachute=None)**
  Start a new flight phase. This is the command to use when the callback wants
  to hand control to a different phase model, often after a deployment event.

  .. jupyter-execute::

    def start_new_phase(**kwargs):
      event = kwargs["event"]
      event.commands.start_flight_phase(phase_name="custom_descent")
      return {"action": "started new phase"}

    phase_event = Event(
      callback=start_new_phase,
      trigger=lambda **kwargs: kwargs["time"] >= 6,
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

    def stop_on_trigger(**kwargs):
        event = kwargs["event"]
        event.commands.terminate_flight()
        return {
            "action": "termination requested",
            "trigger_time": kwargs["time"],
            "flight_time": kwargs["flight"].t,
        }

    stop_event = Event(
        callback=stop_on_trigger,
        trigger=lambda **kwargs: kwargs["state"][5] <= 0,
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