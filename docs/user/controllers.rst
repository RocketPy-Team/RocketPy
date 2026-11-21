.. _controllers:

Controllers and Effectors
=========================

A **controller** is a function you write that reads what the rocket is doing
and sets the rocket's **control inputs**: how far the air brakes are deployed,
the angle of each canard, how hard a thruster fires. An **effector** is a part
you describe with a function, which pushes the rocket or turns the motor's
thrust according to its control inputs.

A controlled flight takes three steps:

1. Add to the rocket something that has control inputs: an effector, air
   brakes (:ref:`airbrakes`), control fins (:ref:`controlfins`) or a surface
   with controls (:ref:`genericsurfaces`).
2. Add a controller that writes those control inputs.
3. Fly, and read what the controller did from the flight.

Effectors
---------

An effector is built from a function and a list of control names. Each control
becomes a control input named ``<effector name>.<control>``, with the
effector's name in snake_case: an effector named ``"Wheel"`` with a
control ``command`` gives ``wheel.command``.
``rocket.control_names`` lists every control input of the rocket.

- ``controls``: the names of the controls.
- ``control_limits``: the lowest and highest value of each control. What a
  controller writes is kept inside them.
- ``control_defaults``: the value of each control at the start of the flight
  (0 when not given).

Force effector
~~~~~~~~~~~~~~

A :class:`rocketpy.ForceEffector` applies a force and a moment to the rocket.
Its function receives one value per control, then ``conditions``, and returns
the force in N and the moment in N m, both in the rocket's body axes
(:ref:`rocket_axes`):

.. code-block:: python

    def thruster(firing, conditions):
        # 5 N toward +x while firing
        return [5.0 * firing, 0, 0], [0, 0, 0]

    rcs = ForceEffector(
        thruster,
        controls=("firing",),
        control_limits={"firing": (0, 1)},
        name="RCS",
    )
    rocket.add_effector(rcs, position=1.4)

``position`` is where the force acts along the rocket, in the rocket's
coordinate system, as for a fin. ``angular_position`` (in degrees) and
``radius`` (in m) place it around the tube. The moment that the force causes
about the center of dry mass is added for you, so the function returns only a
moment that comes with no force, like the torque of a reaction wheel. Without
a position, the force acts at the center of dry mass.

Thrust effector
~~~~~~~~~~~~~~~

A :class:`rocketpy.ThrustEffector` changes the motor's thrust. Its function
receives the thrust vector in N and the position of the nozzle exit in m, both
in the body axes, then one value per control, then ``conditions``. It returns
the new thrust and the new nozzle exit position:

.. code-block:: python

    import math
    from rocketpy import ThrustEffector

    def vanes(thrust, nozzle, pitch, conditions):
        # jet vanes turn the thrust by the angle ``pitch`` and keep the exit in place
        turned = [thrust[2] * math.sin(pitch), thrust[1], thrust[2] * math.cos(pitch)]
        return turned, nozzle

    rocket.add_effector(
        ThrustEffector(
            vanes,
            controls=("pitch",),
            control_limits={"pitch": (-0.1, 0.1)},
            name="Vanes",
        )
    )

A gimbal also moves the nozzle exit, so its function returns the moved
position.

The conditions
~~~~~~~~~~~~~~

``conditions`` describes the flight at that moment: ``conditions.time``,
``conditions.state`` (``conditions.state.vz``, ``conditions.state.w3``, ...),
``conditions.height_agl``, ``conditions.pressure``, ``conditions.density``,
``conditions.mach`` and ``conditions.stream_velocity``. See
:class:`rocketpy.control.FlightConditions`.

The controller function
-----------------------

A :class:`rocketpy.Controller` calls your function ``sampling_rate`` times a
second. What the function writes stays in force until the next call. The
function takes one argument, ``context``:

- ``context.time``, and ``context.dt``, the time since the previous call.
- ``context.state``: the rocket's states by name. ``state.x``, ``state.y``,
  ``state.z`` are the position, ``state.vx``, ``state.vy``, ``state.vz`` the
  velocity, ``state.e0`` to ``state.e3`` the attitude quaternion and
  ``state.w1``, ``state.w2``, ``state.w3`` the angular velocity in the body
  axes.
- ``context.state_dot``: the derivative of each state with respect to time,
  ``state_dot.az`` for the vertical acceleration.
- ``context.sensors``: the latest reading of each sensor on the rocket,
  ``context.sensors.accelerometer.az``. Use these instead of ``state`` to fly
  the controller on what a real flight computer measures. See
  :ref:`sensorsusage`.
- ``context.controls``: the control inputs, to read and to write:
  ``context.controls.wheel.command = 1``. When the name is built in a
  loop, write ``context.controls.canards[fin]``.
- ``context.memory``: a dictionary of your own, kept from one call to the
  next: an integral, a filter value, a counter. It starts every flight at the
  ``memory`` given to the controller.
- ``context.record(name=value, ...)``: keep values with their time, to read
  after the flight in ``flight.records``. The function's return value is
  ignored.
- ``context.outputs``: values a controller computes for other controllers to
  read. See below.
- ``context.commands``: change how the flight goes on,
  ``context.commands.disable()`` stops the controller.
- ``context.height_agl``, ``context.pressure``, ``context.mach``,
  ``context.rocket``, ``context.environment``: read them, never change the
  rocket or the environment.

The full list is in :ref:`eventusage`. A controller with memory and records:

.. code-block:: python

    def pi_function(context):
        error = TARGET_RATE - context.state.w3
        integral = context.memory["integral"] + error * context.dt
        context.memory["integral"] = integral
        context.controls.wheel.command = KP * error + KI * integral
        context.record(error=error)

    controller = Controller(
        pi_function,
        sampling_rate=50,
        memory={"integral": 0.0},
        name="Roll PI",
    )

The controller options
~~~~~~~~~~~~~~~~~~~~~~

- ``sampling_rate``: how many times a second the function runs, in Hz.
- ``name``: the name the controller's records and outputs are read under.
- ``memory``: the starting content of ``context.memory``.
- ``enabled``: whether the controller runs from the start of the flight
  (``True`` by default).
- ``enable_on`` and ``disable_on``: when to start and stop it. Each takes
  ``"burnout"``, ``"apogee"`` or another preset of :ref:`eventusage`, a time
  in seconds, or a function of the context that returns ``True``. A controller
  that should wait for ``enable_on`` needs ``enabled=False``.
- ``outputs``: the names of the values it shares with other controllers.

Several controllers run in the order they were added to the rocket.

Sharing values between controllers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A controller writes its outputs, and any other controller reads them, through
``context.outputs``, under the name of the controller that declared them:

.. code-block:: python

    def guidance_function(context):
        context.outputs.guidance.rate_setpoint = 0.5 if context.time < 8 else 0.0

    def attitude_function(context):
        error = context.outputs.guidance.rate_setpoint - context.state.w3
        ...

    guidance = Controller(guidance_function, sampling_rate=10,
                          name="Guidance", outputs=("rate_setpoint",))
    attitude = Controller(attitude_function, sampling_rate=100, name="Attitude")
    rocket.add_controller(guidance)
    rocket.add_controller(attitude)

An output keeps its value until it is written again. It starts every flight as
``None``, or at the value given with a dictionary
(``outputs={"rate_setpoint": 0.2}``).

.. _continuous_controller:

The continuous controller
-------------------------

A :class:`rocketpy.ContinuousController` has no sampling rate: its law acts at
every instant of the flight. Use it to study a control law before choosing a
sampling rate. The law receives the controller's own states, the control
inputs and the conditions:

.. code-block:: python

    from rocketpy import ContinuousController

    def roll_law(states, controls, conditions):
        controls.wheel.command = -2.0 * conditions.state.w3

    rocket.add_controller(ContinuousController(roll_law, name="Roll Law"))

A continuous controller has no ``context``, so no memory, records, outputs or
commands. A value it must keep over time, like an integral, is declared with
``states`` and ``state_derivative``. :ref:`controlfins` flies the same law as
a sampled and as a continuous controller, and shows fins that take time to
reach the angle they are told.

Reading the results
-------------------

.. code-block:: python

    flight.controls.wheel.command          # a control input, as a Function of time
    flight.controls.wheel.command(10.0)    # its value at 10 s
    flight.records.roll_controller.error       # a list of (time, value) pairs
    flight.event_outputs.guidance.rate_setpoint    # a list of (time, value) pairs

    flight.plots.controls()
    flight.plots.records()
    flight.prints.controls()
    flight.prints.controllers()

    controller.info()
    wheel.info()

A flight never changes the rocket: the next flight starts from the default
value of every control input and from each controller's starting memory.

Under a parachute the air brakes, the control fins and the effectors do
nothing: the descent does not depend on any control input. A controller still
runs there and what it writes is kept, but ``flight.controls`` then has the
control input only at the times of the solver's steps, which are far apart
during a descent. To keep a value at every sample use ``context.record``, and
to stop a controller that is no longer needed give it ``disable_on="apogee"``.

.. note::

    In a Monte Carlo study, every sampled rocket flies the same control loop
    as the rocket it is sampled from: its effectors, its controllers and its
    sensors are carried into each sample. The loop itself is not dispersed
    (gains, sampling rates and sensor settings are the same in every sample).

Example: roll control with a reaction wheel
-------------------------------------------

A rocket whose fins are slightly canted rolls all the way up. Here a reaction
wheel stops that roll after burnout, then holds a slow roll, with a gyroscope
as the only measurement. A reaction wheel is a flywheel on the rocket's axis:
when its motor speeds it up one way, the rocket is turned the other way. It
burns nothing, so the rocket's mass does not change, but it can only store so
much angular momentum before it reaches its top speed.

First the rocket, with a ``cant_angle`` on the fins to make it roll:

.. jupyter-execute::

    from copy import deepcopy

    from rocketpy import Environment, SolidMotor, Rocket, Flight

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

    calisto.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.618,
        angular_position=45,
    )

    calisto.add_motor(Pro75M1670, position=-1.255)
    calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)
    calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        position=-1.04956,
        cant_angle=0.02,
    )

    calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

    # A copy without the wheel, to compare with at the end
    calisto_without_wheel = deepcopy(calisto)

|

**The sensor.** The flight computer does not know the true roll rate: it reads
a gyroscope, with noise and a small bias.

.. jupyter-execute::

    from rocketpy import Gyroscope

    gyroscope = Gyroscope(
        sampling_rate=100,
        noise_density=0.01,
        constant_bias=0.02,
        name="Gyroscope",
    )
    calisto.add_sensor(gyroscope, position=0.2)

**The effector.** The wheel turns the rocket about its axis without pushing
it. That is a :class:`rocketpy.ForceEffector` whose function returns no force
and a roll moment. Its one control, ``command``, goes from -1 (full torque one
way) to 1 (full torque the other way):

.. jupyter-execute::

    from rocketpy import ForceEffector

    MAX_TORQUE = 0.1  # N m, the most the wheel's motor gives

    def wheel_torque(command, conditions):
        force = [0, 0, 0]
        moment = [0, 0, MAX_TORQUE * command]  # about the rocket's axis
        return force, moment

    wheel = ForceEffector(
        wheel_torque,
        controls=("command",),
        control_limits={"command": (-1, 1)},
        control_defaults={"command": 0},
        name="Wheel",
    )
    calisto.add_effector(wheel)

    print(calisto.control_names)

**The guidance.** A first controller decides the roll rate the rocket should
have: none until 12 s, then 1 rad/s. It runs 10 times a second, starts at
burnout, and shares the target as an output:

.. jupyter-execute::

    from rocketpy import Controller

    def guidance_function(context):
        context.outputs.guidance.roll_rate_target = 0.0 if context.time < 12 else 1.0

    guidance = Controller(
        guidance_function,
        sampling_rate=10,
        name="Guidance",
        outputs=("roll_rate_target",),
        enabled=False,
        enable_on="burnout",
        disable_on="apogee",
    )
    calisto.add_controller(guidance)

**The roll controller.** A second controller runs 50 times a second and
drives the wheel to reach the target. It:

- waits while the target is ``None``, which is the case until the guidance
  first runs;
- reads the gyroscope and smooths the reading, keeping the smoothed value in
  its memory;
- works out the command from the error and from the integral of the error,
  also kept in its memory;
- adds up the angular momentum the wheel has stored, and when the wheel is at
  its top speed it stops the torque and disables itself;
- records the reading, the error and the stored momentum.

.. jupyter-execute::

    KP = 2.0  # command per rad/s of error
    KI = 1.0  # command per rad of accumulated error
    SMOOTHING = 0.3  # share of each new reading in the smoothed roll rate
    MAX_MOMENTUM = 1.0  # N m s, what the wheel stores at its top speed

    def roll_function(context):
        target = context.outputs.guidance.roll_rate_target
        if target is None:  # the guidance has not started yet
            return

        memory = context.memory
        measured = context.sensors.gyroscope.wz
        memory["smoothed"] += SMOOTHING * (measured - memory["smoothed"])

        error = target - memory["smoothed"]
        command = KP * error + KI * memory["integral"]
        if abs(command) < 1:  # stop adding up while at full torque
            memory["integral"] += error * context.dt
        command = max(-1, min(1, command))

        # the wheel gains what it gives the rocket, with the opposite sign
        memory["momentum"] -= MAX_TORQUE * command * context.dt
        if abs(memory["momentum"]) > MAX_MOMENTUM:
            context.controls.wheel.command = 0
            context.commands.disable()
            return

        context.controls.wheel.command = command
        context.record(measured=measured, error=error, momentum=memory["momentum"])

    roll = Controller(
        roll_function,
        sampling_rate=50,
        name="Roll Controller",
        memory={"smoothed": 0.0, "integral": 0.0, "momentum": 0.0},
        disable_on="apogee",
    )
    calisto.add_controller(roll)

    wheel.info()
    guidance.info()
    roll.info()

**The flight.** Nothing changes in the :class:`rocketpy.Flight`: it finds the
sensor, the wheel and the controllers on the rocket.

.. jupyter-execute::

    flight = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    flight_without_wheel = Flight(
        rocket=calisto_without_wheel,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

**The results.** The wheel stops the roll, then brings it to 1 rad/s, until
it reaches its top speed and the canted fins take over again:

.. jupyter-execute::

    flight.prints.controllers()
    flight.prints.controls()
    flight.plots.controls()
    flight.plots.records()

    command = flight.controls.wheel.command
    target = flight.event_outputs.guidance.roll_rate_target
    momentum = flight.records.roll_controller.momentum
    print(f"Command at 10 s: {command(10.0):.2f}")
    print(f"First target: {target[0][1]} at {target[0][0]:.1f} s")
    print(f"Wheel at its top speed at {momentum[-1][0]:.2f} s")
    for time in (5, 10, 16, 20):
        print(
            f"Roll rate at {time} s: {flight.w3(time):.2f} rad/s with the wheel, "
            f"{flight_without_wheel.w3(time):.2f} rad/s without"
        )

See also
--------

- :ref:`eventusage` for everything a controller function can read and do.
- :ref:`airbrakes` and :ref:`controlfins` for complete examples.
- :ref:`sensorsusage` for feeding a controller with sensor readings.
