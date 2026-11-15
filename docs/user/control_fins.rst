.. _controlfins:

Control Fins
============

Control fins are fins a controller turns during the flight: canards near the
nose, or movable tail fins. Turned together they roll the rocket, turned in
pairs they pitch or yaw it. This example keeps a rocket from rolling: Calisto's
tail fins are canted, so left alone it spins up, and four small canards driven
by a controller hold the roll rate at zero.

We use the same rocket as in the :ref:`First Simulation <firstsimulation>`
example and add the canards to it. :ref:`controllers` describes everything a
controller can do; here we go through one from start to finish.

Setting Up The Simulation
-------------------------

First, everything up to the rocket. Note the ``cant_angle`` of the tail fins:
it is what makes the rocket roll.

.. jupyter-execute::

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
        cant_angle=0.5,
        airfoil=("../data/airfoils/NACA0012-radians.txt", "radians"),
    )

    calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

We keep a copy of the rocket as it is now, without canards, to build the
variants at the end of this page from it.

.. jupyter-execute::

    from copy import deepcopy

    calisto_without_canards = deepcopy(calisto)

Adding the Canards
------------------

A :class:`rocketpy.ControlFins` is built from one fin, which is copied ``n``
times around the rocket, or from a list of fins given one by one, each at its
own angular position and with its own shape (``ControlFins([fin_a, fin_b])``). Each copy has one control, its deflection in
radians, named ``fin_1`` to ``fin_n``. ``fin_1`` sits at the
``angular_position`` of the fin given, and the numbering follows the way the
angular position grows: with four fins and ``angular_position=0``, ``fin_1``
is at 0°, ``fin_2`` at 90°, ``fin_3`` at 180° and ``fin_4`` at 270°.

A deflection turns a fin exactly as a cant angle does, about its hinge line,
and a set held at one deflection gives the same forces as the same fins added
with that cant angle. So the sign convention is the cant angle's: **a positive
deflection rolls the rocket toward negative** ``w3``, the angular velocity
about the rocket's axis. To oppose a roll rate, deflect the fins the same way
as the roll rate's sign. Deflecting one fin alone pushes the rocket sideways
at that fin. :ref:`controlfins_which_fin` below gives where each fin sits and
which way it turns, so you can match them to the fins of your rocket.

``hinge_position`` is where the hinge line crosses the root chord, as a
fraction of its length from the leading edge; the default, 0.5, is where a
cant angle turns a fixed fin. The fin model has no stall, so keep the
deflections small, a few tenths of a radian at most.

``deflection_limits`` is the most a fin can turn, in radians. A controller
that asks for more gets the limit.

.. important::

    The forces of the fins are computed from their shape, not measured. The
    model is the one RocketPy uses for fixed fins, described in
    :ref:`individual_fins`:

    - Each fin makes a force normal to itself, equal to its lift slope times
      the angle between the fin and the air it meets: the angle of attack at
      that fin, plus its deflection, plus what the rocket's rotation adds.
      The force grows in proportion to that angle, so the fin never stalls.
    - The lift slope comes from the fin's planform (Diederich's correlation,
      as in the Barrowman method), corrected for the Mach number
      (Prandtl-Glauert below Mach 0.8) and for the presence of the body. It
      is that of a flat plate unless the fin is given an ``airfoil``.
    - The force acts at one fixed point of the fin, its center of pressure.
    - A fin has no drag of its own. A deflected fin adds drag only because
      its normal force then leans back.
    - Each fin is computed alone. The gap between a turning fin and the
      body, the effect of one fin on its neighbors and the effect of the
      canards on the tail fins are not modelled.

    The roll moment, the sideways force and the hinge torque on this page all
    come from this model, so read them as estimates, best at low speed and
    small angles. If you have wind tunnel or CFD data for your fins, use it
    instead: a :class:`rocketpy.GenericSurface` takes coefficients given as
    functions of the deflections (see :ref:`surface_controls`).

.. jupyter-execute::

    from rocketpy import ControlFins, TrapezoidalFin

    canard = TrapezoidalFin(
        root_chord=0.05,
        tip_chord=0.025,
        span=0.04,
        rocket_radius=calisto.radius,
        angular_position=0,
    )

    canards = ControlFins(
        canard,
        n=4,
        name="Canards",
        deflection_limits=(-0.2, 0.2),
    )

    calisto.add_surfaces(canards, positions=0.3)

    canards.info()

The name of the set names its control inputs: ``"Canards"`` gives
``canards.fin_1`` and so on, which is how a controller writes them.

.. jupyter-execute::

    calisto.control_names

.. _controlfins_which_fin:

Which Fin Is Which
~~~~~~~~~~~~~~~~~~

To fly the controller you design here on a real rocket, each ``fin_i`` must be
the right fin of the rocket, and a positive deflection must turn it the right
way. Both follow from the rocket's body axes: ``z`` runs along the rocket
toward the nose, and ``x`` and ``y`` are fixed to the airframe (see
:ref:`rocket_axes`).

- **Where a fin sits** is its angular position: the angle from the ``+y`` axis
  to the fin, growing counterclockwise when you look at the rocket from the
  nose toward the tail (see :ref:`Angular Position Inputs <angular_position>`).
- **Which way it turns**: seen from the nose, a positive deflection moves the
  leading edge of every fin (the edge nearer the nose) clockwise around the
  rocket. The air then pushes the rocket, at that fin, the way the leading
  edge moved.

For the four canards of this page, with the sides and directions given in the
body axes:

.. list-table::
   :header-rows: 1
   :widths: 14 18 18 25 25

   * - Control
     - Angular position
     - Side of the rocket
     - A positive deflection moves the leading edge toward
     - and pushes the rocket there toward
   * - ``fin_1``
     - 0°
     - ``+y``
     - ``+x``
     - ``+x``
   * - ``fin_2``
     - 90°
     - ``-x``
     - ``+y``
     - ``+y``
   * - ``fin_3``
     - 180°
     - ``-y``
     - ``-x``
     - ``-x``
   * - ``fin_4``
     - 270°
     - ``+x``
     - ``-y``
     - ``-y``

To match this to your rocket:

1. Mark the line along the airframe that is your ``+y`` side. Every angular
   position is measured from it: those of the fins and of the rail buttons.
   The rail buttons' ``angular_position`` and the launch ``heading`` set
   where that line points on the pad.
2. Give the fin handed to ``ControlFins`` the ``angular_position`` of the fin
   your first servo turns. The other fins are numbered counterclockwise from
   it, seen from the nose. If your servos are numbered in another order, or
   the fins are not evenly spaced, give the fins one by one in your own
   order: ``ControlFins([fin_a, fin_b, fin_c, fin_d])``, each built with its
   own ``angular_position``.
3. On the bench, command a small positive angle to one servo and look at the
   rocket from the nose. The leading edge of that fin must move clockwise. If
   it moves the other way, change the sign of that servo in the flight
   computer. Deflections are in radians.

A controller rarely thinks in single fins. ``canards.deflections`` turns a
roll command and two sideways commands into the deflection of every fin, from
their angular positions: ``roll`` is the same deflection on every fin, ``x``
pushes the set toward ``+x`` and ``y`` toward ``+y``.

.. jupyter-execute::

    print("roll:", canards.deflections(roll=0.02))
    print("x:   ", canards.deflections(x=0.05))
    print("y:   ", canards.deflections(y=0.05))
    print("all: ", canards.deflections(roll=0.02, x=0.05))

A sideways push on canards, which sit ahead of the center of mass, turns the
nose the way of the push. The same push on movable tail fins turns the nose
the other way.

How much each command does is ``canards.effectiveness``: the roll moment
coefficient per radian of ``roll``, and the force coefficients per radian of
``x`` and ``y``, as functions of Mach. Multiplied by the dynamic pressure and
the reference area (and length, for the moment) they are the gains a control
design starts from.

.. jupyter-execute::

    effectiveness = canards.effectiveness()
    print(f"Roll moment coefficient per rad at Mach 0.3: {effectiveness['roll'](0.3):.3f}")
    canards.plots.effectiveness()

The torque the air puts on each fin about its hinge line, the load the
mechanism turning the fin has to hold, is ``canards.hinge_moments``: one
function of Mach per fin, in N m, at a dynamic pressure and a set of
deflections you give. It is the fin's normal force times the distance from
the hinge line to the fin's center of pressure, so it depends on
``hinge_position``. The fin model gives one fixed center of pressure, while
the real one moves with speed and angle of attack, so with the hinge close to
it trust the size of the torque, not its sign.

.. jupyter-execute::

    torque = canards.hinge_moments(dynamic_pressure=30e3, deflections=0.1)
    print(f"Torque on fin 1 at Mach 0.3: {torque['fin_1'](0.3):.3f} N m")

Canards ahead of the center of mass move the center of pressure forward, so
check the stability margin is still what you want:

.. jupyter-execute::

    calisto.plots.stability_margin()

Writing the Controller
----------------------

The controller is a function of one argument, the context, called at the
sampling rate. It reads the roll rate, ``context.state.w3`` in rad/s, and
writes the four deflections. This one is a PI controller: the proportional
term damps the roll rate, the integral term finds the steady deflection that
cancels the canted tail fins. The integral is what the next call needs, so it
lives in ``context.memory``; what we want to look at afterwards goes to
``context.record``.

.. jupyter-execute::

    from rocketpy import Controller

    KP = 0.02  # rad of deflection per rad/s of roll rate
    KI = 0.2  # rad of deflection per rad of accumulated roll
    TARGET = 0.0  # rad/s

    def roll_controller(context):
        error = context.state.w3 - TARGET
        context.memory["integral"] += error * context.dt
        # The same sign as the roll rate: see the convention above
        deflection = KP * error + KI * context.memory["integral"]
        context.controls.canards.update(canards.deflections(roll=deflection))
        context.record(error=error, deflection=deflection)

    roll_control = Controller(
        roll_controller,
        sampling_rate=100,
        memory={"integral": 0.0},
        name="Roll Control",
    )

    calisto.add_controller(roll_control)

    roll_control.info()

A few things worth knowing:

- ``context.dt`` is the time since the controller last ran, ``1 / sampling_rate``
  on the first call.
- ``context.memory`` starts again from ``{"integral": 0.0}`` at every flight, so
  the rocket can be flown many times, in a Monte Carlo study for instance,
  with no leftover from the previous flight.
- A deflection outside ``(-0.2, 0.2)`` is clamped when the function returns.
- The controller only writes the deflection; the fins keep it until the next
  sample. To read the latest sensor values instead of the true state, use
  ``context.sensors``; see :ref:`sensorsusage`.

Simulating a Flight
-------------------

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

The deflections the fins flew with are stored with the flight.
``flight.solution.controls`` holds each one as an array of time and value and
``flight.controls`` as a function of time, by the set's name as given or in
snake_case, and ``flight.plots.controls()`` draws them all, held between
samples as they were in flight.

.. jupyter-execute::

    test_flight.plots.controls()

    fin_1 = test_flight.controls.canards.fin_1
    print(f"Deflection at 2 s: {fin_1(2.0):.4f} rad")
    print(f"Deflection at 10 s: {fin_1(10.0):.4f} rad")

The torque the air put on each fin about its hinge line during the flight,
from the air the set met and the deflections it flew with:

.. jupyter-execute::

    canards.plots.hinge_moments(test_flight)

The roll rate the controller was holding:

.. jupyter-execute::

    test_flight.w3.plot(0, test_flight.apogee_time)

What the function recorded is in ``flight.records``, named after the
controller:

.. jupyter-execute::

    test_flight.plots.records()
    test_flight.prints.controls()

Compare with the same rocket and no canards, which the canted tail fins spin
up:

.. jupyter-execute::

    uncontrolled_flight = Flight(
        rocket=calisto_without_canards,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    uncontrolled_flight.w3.plot(0, uncontrolled_flight.apogee_time)

The rocket itself was not changed by flying: ``canards`` is still at its
default deflection, and the next flight starts from it again.

Steering With the Fins
----------------------

The roll controller gives every fin the same deflection. A sideways command
does not: here the same controller also pushes the canards toward ``+x`` for
one second of the coast, from 6 s to 7 s.

.. jupyter-execute::

    def steering_controller(context):
        error = context.state.w3 - TARGET
        context.memory["integral"] += error * context.dt
        roll = KP * error + KI * context.memory["integral"]
        x = 0.05 if 6 <= context.time < 7 else 0.0
        context.controls.canards.update(canards.deflections(roll=roll, x=x))

    steering_canards = ControlFins(
        canard, n=4, name="Canards", deflection_limits=(-0.2, 0.2)
    )
    calisto_steered = deepcopy(calisto_without_canards)
    calisto_steered.add_surfaces(steering_canards, positions=0.3)
    calisto_steered.add_controller(
        Controller(
            steering_controller,
            sampling_rate=100,
            memory={"integral": 0.0},
            name="Steering",
        )
    )

    steered_flight = Flight(
        rocket=calisto_steered,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

During that second ``fin_1`` (on ``+y``) and ``fin_3`` (on ``-y``) move apart
by the ``x`` command, on top of the deflection that holds the roll, while
``fin_2`` and ``fin_4`` keep it:

.. jupyter-execute::

    for name in steering_canards.controls:
        deflection = steered_flight.controls.canards[name]
        print(f"{name}: {deflection(5.5):+.4f} rad at 5.5 s, {deflection(6.5):+.4f} rad at 6.5 s")

    steered_flight.plots.controls()

Each fin now carries its own load:

.. jupyter-execute::

    steering_canards.plots.hinge_moments(steered_flight)

The push toward ``+x`` ahead of the center of mass turns the nose toward
``+x``, a positive rotation about the body ``y`` axis, ``w2``. The rocket then
swings back and forth about its new heading, as a stable rocket does:

.. jupyter-execute::

    steered_flight.w2.plot(5.5, 9)

Fins That Lag
-------------

So far a deflection takes effect the instant the controller writes it. A real
servo takes time to get there. To model that, give the fins states of their
own: the angle each fin has actually reached. The flight integrates these
states with the rocket's, and the fins act with the reached angle, not the
commanded one. Here each fin follows its command with a first-order lag of
50 ms:

.. jupyter-execute::

    class ServoCanards(ControlFins):
        states = ("a1", "a2", "a3", "a4")  # the angle each fin has reached

        def state_derivative(self, states, commands, conditions):
            return [(c - s) / 0.05 for c, s in zip(commands, states)]

        def actual_controls(self, states, commands, conditions):
            return list(states)

    calisto_lagged = deepcopy(calisto_without_canards)
    calisto_lagged.add_surfaces(
        ServoCanards(canard, n=4, name="Canards", deflection_limits=(-0.2, 0.2)),
        positions=0.3,
    )
    calisto_lagged.add_controller(deepcopy(roll_control))

    lagged_flight = Flight(
        rocket=calisto_lagged,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

The reached angles are states of the flight, read by the set's name in
``flight.state_history``, each a function of time like the commands in
``flight.controls``:

.. jupyter-execute::

    commanded = lagged_flight.controls.canards.fin_1
    reached = lagged_flight.state_history.canards.a1
    print(f"Commanded at 2 s: {commanded(2.0):.4f} rad")
    print(f"Reached at 2 s:   {reached(2.0):.4f} rad")
    lagged_flight.w3.plot(0, lagged_flight.apogee_time)

A controller tuned on instant fins may oscillate on lagging ones. Add the lag
before tuning the gains.

A Continuous Controller
-----------------------

The controller above runs 100 times a second and holds its output in between,
as a flight computer does. To study the law without the sampling, as control
theory writes it, use a :class:`rocketpy.ContinuousController`: its law runs
inside the equations of motion, at every instant. It gets the flight's
conditions, its own states and the dictionary of control inputs, which it
writes.
The integral term is a state of the controller, with its derivative:

.. jupyter-execute::

    from rocketpy import ContinuousController

    def integral_derivative(states, controls, conditions):
        return [conditions.state.w3 - TARGET]

    def continuous_roll_law(states, controls, conditions):
        deflection = KP * (conditions.state.w3 - TARGET) + KI * states[0]
        controls.canards.update(canards.deflections(roll=deflection))

    continuous_roll_control = ContinuousController(
        continuous_roll_law,
        states=("integral",),
        state_derivative=integral_derivative,
        name="Roll Control",
    )

    calisto_continuous = deepcopy(calisto_without_canards)
    calisto_continuous.add_surfaces(
        ControlFins(canard, n=4, name="Canards", deflection_limits=(-0.2, 0.2)),
        positions=0.3,
    )
    calisto_continuous.add_controller(continuous_roll_control)

    continuous_flight = Flight(
        rocket=calisto_continuous,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    continuous_flight.w3.plot(0, continuous_flight.apogee_time)

A continuous controller has no memory, records or commands. The deflections
it wrote are kept with every stored row of the flight, in
``flight.controls.canards.fin_1``, and its integral is a state of the flight,
``flight.state_history.roll_control.integral``.

If the sampled controller behaves like the continuous one, the sampling rate
is high enough. If it oscillates where the continuous one does not, the rate
is too low for these gains.

Not Covered
-----------

- The interference between the canards and the tail fins. Each surface is
  computed on its own, so the swirl a deflected canard sends to the tail, which
  reduces the roll the canards give and can reverse it, is not modelled. The
  effectiveness above is that of the canards alone.
- There is no stochastic version of ``ControlFins`` yet: a Monte Carlo study
  flies the set unchanged in every sampled rocket.

.. seealso::

    :ref:`controllers` for everything a controller can read and do, the
    other things it can move, and how controllers share values.
    :class:`rocketpy.ControlFins` for the class itself.
