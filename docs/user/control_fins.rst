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

.. note::

    There are two ways to model control fins:

    - **From their shape**, with :class:`rocketpy.ControlFins`, as on this
      page. RocketPy computes the forces of each fin from its geometry with
      the Barrowman method, the model it uses for fixed fins. That method has
      limits: small angles, subsonic flight, and each fin computed alone. See
      :ref:`the limits of the model <controlfins_model>` and
      :ref:`controlfins_not_covered` before relying on the numbers.
    - **From coefficients** measured in a wind tunnel or computed elsewhere,
      with a :class:`rocketpy.GenericSurface` whose coefficients depend on
      the deflections (see :ref:`surface_controls`). The sign of each moment
      coefficient is given in :ref:`genericsurfaces`.

    The controller is written the same way for both.

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

A :class:`rocketpy.ControlFins` is built from one fin, copied ``n`` times
around the rocket, or from a list of fins given one by one
(``ControlFins([fin_a, fin_b])``). Each fin has one control, its deflection in
radians, named ``fin_1`` to ``fin_n``. ``fin_1`` sits at the
``angular_position`` of the fin given and the others follow as the angular
position grows: four fins starting at 0° sit at 0°, 90°, 180° and 270°.

A deflection turns a fin about its hinge line, exactly as a cant angle does.
So **a positive deflection rolls the rocket toward negative** ``w3``, the
angular velocity about the rocket's axis: to oppose a roll rate, deflect the
fins with the sign of that rate. One fin deflected alone pushes the rocket
sideways at that fin. :ref:`controlfins_which_fin` gives where each fin sits
and which way it turns, and :ref:`controlfins_commands` how to command the
set by roll, pitch and yaw without minding these signs.

``hinge_position`` is where the hinge line crosses the root chord, as a
fraction of its length from the leading edge; the default, 0.5, is where a
cant angle turns a fixed fin. The fin model has no stall, so keep the
deflections small, a few tenths of a radian at most.

``deflection_limits`` is the most a fin can turn, in radians. A controller
that asks for more gets the limit.

.. _controlfins_model:

.. important::

    The forces of the fins are computed from their shape, not measured. They
    come from the Barrowman method, the model RocketPy uses for fixed fins
    (see :ref:`individual_fins`), with each fin turned by its deflection. The
    limits of that method apply here too:

    - **Small angles.** The force of a fin grows in proportion to its
      deflection plus the angle of attack it meets, so a fin never stalls.
    - **Subsonic flight.** The method was derived for subsonic flow. It is
      corrected for the Mach number up to Mach 0.8 and is not valid near and
      above Mach 1.
    - **Each fin alone.** The effect of one fin on its neighbors, the effect
      of canards on the tail fins and the gap between a turning fin and the
      body are left out.
    - **A fixed center of pressure.** The force of a fin always acts at the
      same point of the fin.
    - **No drag of the fin itself.** A deflected fin adds drag only because
      its force then leans back.

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
3. With the rocket on the ground, command a small positive angle to one servo
   and look at the rocket from the nose. The leading edge of that fin must
   move clockwise. If it moves the other way, change the sign of that servo
   in the flight computer. Deflections are in radians.
4. Match the gyroscope of your flight computer to the angular velocities a
   controller reads here. ``w1``, ``w2`` and ``w3`` are the angular velocities
   about the body ``x``, ``y`` and ``z`` axes, each positive by the right-hand
   rule:

   .. list-table::
      :header-rows: 1
      :widths: 20 20 60

      * - Angular velocity
        - About
        - When it is positive
      * - ``w1``
        - ``x``
        - the nose moves toward ``-y``
      * - ``w2``
        - ``y``
        - the nose moves toward ``+x``
      * - ``w3``
        - ``z``
        - seen from the nose, the rocket turns counterclockwise

   A gyroscope whose axes are not the body axes reads other components. Add
   it to the rocket as a sensor with its ``orientation`` and read it with
   ``context.sensors`` (see :ref:`sensorsusage`).

.. _controlfins_commands:

The Commands and What They Do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A controller rarely thinks in single fins. ``canards.deflections`` turns a
roll, a pitch and a yaw command into the deflection of every fin. A positive
command turns the rocket toward the positive angular velocity of that name:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Command
     - Angular velocity it raises
     - What you see
   * - ``roll``
     - ``w3``
     - seen from the nose, the rocket turns counterclockwise
   * - ``pitch``
     - ``w1``
     - the nose moves toward ``-y``
   * - ``yaw``
     - ``w2``
     - the nose moves toward ``+x``

This holds wherever the set sits. Canards and movable tail fins turn opposite
ways for the same pitch or yaw command, and the set works that out from where
it was added to the rocket. A controller that opposes the three angular
velocities therefore uses ``roll=-K * w3``, ``pitch=-K * w1`` and
``yaw=-K * w2``, on canards and on tail fins alike.

.. jupyter-execute::

    print("roll: ", canards.deflections(roll=0.02))
    print("pitch:", canards.deflections(pitch=0.05))
    print("yaw:  ", canards.deflections(yaw=0.05))
    print("all:  ", canards.deflections(roll=0.02, yaw=0.05))

.. note::

    The pitch and yaw commands need the set to be on a rocket, since they
    depend on which side of the center of dry mass it is. Build the set, add
    it with ``add_surfaces``, then call ``deflections``.

How much each command does is ``canards.effectiveness``: the moment
coefficient per radian of each command, as functions of Mach. Multiplied by
the dynamic pressure, the reference area and the reference length they are
the moments in N m per radian, the gains a control design starts from. Here
for a command of 0.05 rad at Mach 0.3:

.. jupyter-execute::

    effectiveness = canards.effectiveness()
    canards.plots.effectiveness()

    mach = 0.3
    dynamic_pressure = 5500  # Pa, near Mach 0.3 at this altitude
    command = 0.05  # rad
    to_moment = dynamic_pressure * canards.reference_area * canards.reference_length
    for name in ("roll", "pitch", "yaw"):
        moment = effectiveness[name](mach) * to_moment * command
        print(f"{name}={command}: a moment of {moment:+.3f} N m")

The pitch and yaw moments are taken about the rocket's center of dry mass,
where the center of mass is once the propellant has burned. While the motor
burns the center of mass is further aft, so these canards turn the rocket a
little more than the numbers say. Divide a moment by the rocket's moment of
inertia about that axis to get the angular acceleration it causes.

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

    KP = 0.02  # rad of roll command per rad/s of roll rate
    KI = 0.2  # rad of roll command per rad of accumulated roll
    TARGET = 0.0  # rad/s

    def roll_controller(context):
        error = context.state.w3 - TARGET
        context.memory["integral"] += error * context.dt
        # Against the error: a positive roll command raises w3
        roll = -(KP * error + KI * context.memory["integral"])
        context.controls.canards.update(canards.deflections(roll=roll))
        context.record(error=error, roll=roll)

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

If your flight computer already computes one angle per servo, write each fin
yourself, with your own mixing, in place of ``canards.deflections``. This
one holds the roll and damps the pitch and yaw rates:

.. code-block:: python

    K_ROLL = 0.02  # rad of deflection per rad/s of roll rate
    K_RATE = 0.05  # rad of deflection per rad/s of pitch or yaw rate

    def flight_computer(context):
        state = context.state
        roll = K_ROLL * state.w3
        fins = context.controls.canards
        fins.fin_1 = roll - K_RATE * state.w2  # on +y, pushes toward +x
        fins.fin_2 = roll + K_RATE * state.w1  # on -x, pushes toward +y
        fins.fin_3 = roll + K_RATE * state.w2  # on -y, pushes toward -x
        fins.fin_4 = roll - K_RATE * state.w1  # on +x, pushes toward -y

    calisto.add_controller(Controller(flight_computer, sampling_rate=100))

RocketPy computes the forces and moments from the four deflections, so
simulating a controller you have already designed needs no gain from this
page: only the right fin, the right sign and the right axes, as in
:ref:`controlfins_which_fin`. A fin in the wrong place or a wrong sign makes
the flight diverge, which looks the same as a badly tuned controller. To tell
them apart, fly a few seconds with one fixed small deflection and no control
law, and check that ``w1``, ``w2`` and ``w3`` move the way your rocket would.

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

The roll controller gives every fin the same deflection. A pitch or yaw
command does not: here the same controller also gives a yaw command for one
second of the coast, from 6 s to 7 s.

.. jupyter-execute::

    def steering_controller(context):
        error = context.state.w3 - TARGET
        context.memory["integral"] += error * context.dt
        roll = -(KP * error + KI * context.memory["integral"])
        yaw = 0.05 if 6 <= context.time < 7 else 0.0
        context.controls.canards.update(canards.deflections(roll=roll, yaw=yaw))

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
by the yaw command, on top of the deflection that holds the roll, while
``fin_2`` and ``fin_4`` keep it:

.. jupyter-execute::

    for name in steering_canards.controls:
        deflection = steered_flight.controls.canards[name]
        print(f"{name}: {deflection(5.5):+.4f} rad at 5.5 s, {deflection(6.5):+.4f} rad at 6.5 s")

    steered_flight.plots.controls()

Each fin now carries its own load:

.. jupyter-execute::

    steering_canards.plots.hinge_moments(steered_flight)

The positive yaw command raises ``w2``, the angular velocity about the body
``y`` axis: the nose turns toward the rocket's ``+x`` side by about one
degree, and the flight path follows it. Compared with ``test_flight``, which
only holds the roll, the rocket moves away sideways from the moment of the
command:

.. jupyter-execute::

    def distance_apart(time):
        east = steered_flight.x(time) - test_flight.x(time)
        north = steered_flight.y(time) - test_flight.y(time)
        return (east**2 + north**2) ** 0.5

    apogee_time = min(steered_flight.apogee_time, test_flight.apogee_time)
    for time in (6, 7, 10, 15, apogee_time):
        print(f"{time:5.1f} s: {distance_apart(time):5.1f} m apart")

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
        roll = -(KP * (conditions.state.w3 - TARGET) + KI * states[0])
        controls.canards.update(canards.deflections(roll=roll))

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

.. _controlfins_not_covered:

Not Covered
-----------

- The interference between the canards and the tail fins. Each surface is
  computed on its own, so the swirl a deflected canard sends to the tail, which
  reduces the roll the canards give and can reverse it, is not modelled. The
  effectiveness above is that of the canards alone.
- Stall. The force of a fin grows in proportion to its deflection plus the
  angle of attack it meets, without limit. Past a few tenths of a radian a
  real fin loses force, so a simulated controller that asks for large
  deflections has more authority than the real one.
- The servos. A fin takes its deflection the instant the controller writes
  it. The time a servo needs to get there and the speed it can turn at are
  not modelled yet.
- There is no stochastic version of ``ControlFins`` yet: a Monte Carlo study
  flies the set unchanged in every sampled rocket.

.. seealso::

    :ref:`controllers` for everything a controller can read and do, the
    other things it can move, and how controllers share values.
    :class:`rocketpy.ControlFins` for the class itself.
