.. _rocketusage:

Rocket Class Usage
==================

Defining a Rocket in RocketPy is simple and requires a few steps:

1. Define the rocket itself by passing in the rocket's dry mass, inertia,
   drag coefficient and radius;
2. Add a motor;
3. Add, if desired, aerodynamic surfaces;
4. Add, if desired, parachutes;
5. Set, if desired, rail guides;
6. See results.
7. Inertia Tensors.

Lets go through each of these steps in detail.

1. Defining the Rocket
----------------------

The first step is to define the rocket itself. This is done by creating a
Rocket object and passing in the rocket's dry mass, inertia, drag coefficient
and radius:

.. jupyter-execute::

    from rocketpy import Rocket

    calisto = Rocket(
        radius=127 / 2000,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

.. caution::
    Pay special attention to the following:

    - ``mass`` is the rocket's mass, **without the motor**, in kg.
    - All ``inertia`` values are given in relation to the rocket's center of
      mass without motor.
    - ``inertia`` is defined as a tuple of the form ``(I11, I22, I33)``.
      Where ``I11`` and ``I22`` are the inertia of the mass around the
      perpendicular axes to the rocket, and ``I33`` is the inertia around the
      rocket center axis.
    - Alternatively, ``inertia`` can be defined as a tuple of the form
      ``(I11, I22, I33, I12, I13, I23)``. Where ``I12``, ``I13`` and ``I23``
      are the component of the inertia tensor in the directions ``12``, ``13``
      and ``23`` respectively.
    - ``center_of_mass_without_motor`` and
      ``coordinate_system_orientation`` are :ref:`position <positions>`
      parameters. They must be treated with care. See the
      :doc:`Positions and Coordinate Systems </user/positions>` section for more
      information.

.. seealso::
    For more information on the :class:`rocketpy.Rocket` class initialization, see
    :class:`rocketpy.Rocket.__init__` section.

Drag Curves
~~~~~~~~~~~

The ``Rocket`` class requires two drag curves, one for when the motor is off
and one for when the motor is on. When the motor is on, due to the exhaust
gases, the drag coefficient is lower than when the motor is off.

.. note::
    If you do not have a drag curve for when the motor is on, you can use the
    same drag curve for both cases.

These curves give the rocket's drag coefficient when it flies straight into
the air, with zero angle of attack. When the rocket flies at an angle to the
air, the simulation applies this drag along the rocket's centerline and scales
it by the cosine of the angle of attack. The force therefore fades to zero when
the rocket is sideways to the air and brakes the rocket when it moves tail
first. The extra drag at an angle comes from the forces on the nose cone, fins
and tail, which push the rocket sideways and partly against the air. In a 3-DOF
simulation, which does not model the rocket's attitude, the drag acts against
the velocity instead.

.. note::
    The same scaling applies to a drag coefficient that changes with the angle
    of attack: the value you give at each angle is multiplied by the cosine of
    that angle and applied along the centerline.

A drag curve can be given as:

1. a number, for a constant drag coefficient;
2. the path to a CSV file as a string;
3. a list of points, ``[[mach, cd], ...]``;
4. a function that returns the drag coefficient given the Mach number, such
   as ``lambda mach: ...``;
5. a :class:`rocketpy.Function`.

CSV files and lists of points must have the Mach number in the first column
and the drag coefficient in the second. Here is an example of a drag curve
file:

.. code-block::

    0.0, 0.0
    0.1, 0.4018816
    0.2, 0.38821269
    0.3, 0.38150576
    0.4, 0.37946785
    0.5, 0.38118499
    0.6, 0.38947261
    0.7, 0.40604949
    0.8, 0.40110651
    0.9, 0.45696342
    1.0, 0.62744566

.. note::
    A drag curve may also depend on more than the Mach number. A function can
    take any of ``alpha``, ``beta``, ``mach``, ``reynolds``, ``pitch_rate``,
    ``yaw_rate`` and ``roll_rate`` as arguments (for example
    ``lambda alpha, mach: ...``), and a CSV file can have a header naming its
    columns after them, with the drag coefficient in the last column. The angles
    are in radians, the Reynolds number is based on the rocket's diameter, and
    the three rates are non-dimensional: the rotation rate in rad/s times the
    rocket's diameter, divided by twice the airspeed. These are the same
    variables used by :ref:`generic surfaces <genericsurfaces>`.

    For a drag that grows with the angle between the rocket and the air, whatever
    side the wind comes from, use the total angle of attack ``alpha_total`` (or
    ``alpha_total_deg``), for example ``lambda alpha_total, mach: ...``. ``alpha``
    alone is the angle in one plane only and misses a sideslip.

    If what you have are the coefficients of the whole rocket (such as lift,
    drag and pitch moment against angle of attack), use
    :meth:`rocketpy.Rocket.add_full_body_aerodynamics` instead.

.. tip::
    Getting a drag curve can be a challenging task. To get really accurate
    drag curves, you can use CFD software or wind tunnel data.

    However, if you do not have access to these, you can always use
    `RASAero II <https://www.rasaero.com/>`_ software. In there you need
    only define the geometry of the rocket and access *AeroPlots*.

2. Adding a Motor
-----------------

The second step is to add a motor to the rocket. This is done by creating a
Motor object.

.. seealso::
    For more information on defining motors, see:

    .. grid:: auto

        .. grid-item::

            .. button-ref:: /user/motors/solidmotor
                :ref-type: doc
                :color: primary

                Solid Motors

        .. grid-item::

            .. button-ref:: /user/motors/hybridmotor
                :ref-type: doc
                :color: secondary

                Hybrid Motors

        .. grid-item::

            .. button-ref:: /user/motors/liquidmotor
                :ref-type: doc
                :color: success

                Liquid Motors

With the motor defined, you can add it to the rocket:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from rocketpy import SolidMotor
    example_motor =  SolidMotor(
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

    calisto.add_motor(example_motor, position=-1.255)

.. caution::

    Again, pay special attention to the ``position`` parameter. See
    the :doc:`Positions and Coordinate Systems </user/positions>` section for
    more information.

3. Adding Aerodynamic Surfaces
------------------------------

The third step is to add aerodynamic surfaces to the rocket. These surfaces are
used to calculate the rocket's aerodynamic forces and moments. They can be the
rocket's parts, described by their geometry (nose cone, fins and tail, whose
coefficients RocketPy computes), or surfaces described directly by their
aerodynamic coefficients: a :class:`rocketpy.GenericSurface` (coefficient
tables or functions) or a :class:`rocketpy.LinearGenericSurface` (coefficient
slopes). Coefficients for the whole rocket, for example from a wind tunnel or
another program, go in through :meth:`rocketpy.Rocket.add_full_body_aerodynamics`;
see :ref:`genericsurfaces` for the details.

Differently from the motor, the aerodynamic surfaces do not need to be
defined before being added to the rocket. They can be defined and added
to the rocket in one step:

.. jupyter-execute::

    nose_cone = calisto.add_nose(
        length=0.55829, kind="von karman", position=1.278
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

.. caution::

    Once again, pay special attention to the ``position`` parameter. Check \
    the :meth:`rocketpy.Rocket.add_surfaces` method for more information.

.. seealso::

    For more information on adding aerodynamic surfaces, see:

    - :meth:`rocketpy.Rocket.add_nose`
    - :meth:`rocketpy.Rocket.add_trapezoidal_fins`
    - :meth:`rocketpy.Rocket.add_elliptical_fins`
    - :meth:`rocketpy.Rocket.add_free_form_fins`
    - :meth:`rocketpy.Rocket.add_tail`
    - :meth:`rocketpy.Rocket.add_surfaces` (any surface, including generic ones)
    - :meth:`rocketpy.Rocket.add_full_body_aerodynamics`

Now we can see a representation of the rocket, this will guarantee that the
rocket has been constructed correctly:

.. jupyter-execute::

    calisto.draw()


Adding Airfoil Profile to Fins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Rocket.add_trapezoidal_fins`` and ``Rocket.add_elliptical_fins`` methods
have an optional parameter called ``airfoil``. This parameter allows you to
specify an airfoil profile for the fins.

The ``airfoil`` parameter can be ``None``, in which case fins will be treated as
flat plates. Otherwise, it can be a tuple of the form ``(path, units)``.

The ``path`` is the path to the airfoil CSV file in which the first column is
the angle of attack and the second column is the lift coefficient.

The ``units`` is the unit of the first column of the CSV file.
It can be either ``"radians"`` or ``"degrees"``.

An example of a valid CSV file for a *NACA0012* airfoil is:

.. code-block::

    0.0,          0.0
    0.017453293,  0.11
    0.034906585,  0.22
    0.052359878,  0.33
    0.06981317,   0.44
    0.087266463,  0.55
    0.104719755,  0.66
    0.122173048,  0.746
    0.13962634,   0.8274
    0.157079633,  0.8527
    0.174532925,  0.1325
    0.191986218,  0.1095
    0.20943951,   0.1533

.. note::

    This CSV file has the angle of attack in radians. It is important that the
    CSV file has angle of attack points until the stall point.

.. tip::

    You can find airfoil CSV files in
    `Airfoil Tools <http://airfoiltools.com/>`_

4. Adding Parachutes
--------------------

The fourth step is to add parachutes to the rocket. For that, we need:

- The parachute drag coefficient times reference area for parachute ``cd_s``
- The parachute trigger ``trigger``. More details on
  :ref:`Trigger Details <triggerdetails>`.
- The parachute trigger system sampling rate ``sampling_rate``.

Optionally, we can also define:

- The parachute trigger system lag ``lag``.
- The parachute trigger system noise ``noise``.

Lets add two parachutes to the rocket, one that will be deployed at
apogee and another that will be deployed at 800 meters above ground level:

.. jupyter-execute::

    main = calisto.add_parachute(
        name="Main",
        cd_s=10.0,
        trigger=800,
        sampling_rate=105,
        lag=1.5,
        radius=1.5,
        height=1.5,
        porosity=0.0432,
    )

    drogue = calisto.add_parachute(
        name="Drogue",
        cd_s=1.0,
        trigger="apogee",
        sampling_rate=105,
        lag=1.5,
        radius=1.5,
        height=1.5,
        porosity=0.0432,
    )

.. seealso::

    For more information on adding parachutes, see
    :class:`rocketpy.Rocket.add_parachute`


.. _triggerdetails:

Parachute Trigger Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The parachute trigger is a very important parameter. It is used to determine
when the parachute will be deployed. It can be either a number, a string
``"apogee"``, or a callable.

If it is a number, it is the altitude (height above ground level, in meters) at
which the parachute will be deployed during descent.

If it is a string ``"apogee"``, the parachute will be deployed at apogee.

If it is a callable, it must return ``True`` when the parachute should be
deployed and ``False`` otherwise. Internally, the parachute is wrapped in an
:class:`rocketpy.Event`, so the trigger receives the same information available
to any event trigger.

The trigger must be defined as a function taking one argument, ``context``,
from which you read the values you need, such as ``context.height_agl``. The
most used ones are listed below. For the complete list, with every value
``context`` holds, see :ref:`eventcontext`.

**Simulation time and state:**

- ``time`` (float): current simulation time in seconds.
- ``state``: the rocket's states, read by name. ``context.state.x``, ``.y``
  and ``.z`` are the position in meters (``z`` is the altitude above sea
  level), ``.vx``, ``.vy`` and ``.vz`` are the velocity in m/s, ``.e0`` to
  ``.e3`` are the quaternion orientation components, and ``.w1``, ``.w2`` and
  ``.w3`` are the angular velocity in rad/s.
- ``state_dot``: the derivative of each state with respect to time.
  ``context.state_dot.ax``, ``.ay`` and ``.az`` are the acceleration components
  in m/s², so ``context.state_dot.az`` is the vertical acceleration.
- ``pressure`` (float): current atmospheric pressure in Pa at the rocket's
  altitude.
- ``height_agl`` (float): height above ground level in meters.
- ``step_size`` (float): how long the simulation has been inside the solver
  step being evaluated, in seconds.

**Simulation objects:**

- ``flight`` (:class:`rocketpy.Flight`): the Flight instance orchestrating the
  simulation.
- ``rocket`` (:class:`rocketpy.Rocket`): the Rocket object being simulated.
- ``environment`` (:class:`rocketpy.Environment`): the Environment conditions
  for the flight.

**Sensor and Event data:**

- ``sensors`` (list): the sensors attached to the rocket, each exposing its
  most recent ``measurement``.
- ``sensors_by_name`` (dict): the same sensors keyed by name (or class name).
  If several sensors share a name, the value is a list.
- ``sampling_rate`` (float or None): the sampling rate of the parachute trigger
  in Hz (or ``None`` for a continuous trigger).
- ``event`` (:class:`rocketpy.Event`): a reference to the wrapping Event object,
  giving access to ``context``, ``commands``, and other event state. See the
  :ref:`eventusage` section for more information.

This function is called throughout the simulation. Therefore, you can
use it to deploy the parachute at any time.

The following example shows how to define trigger functions that will
deploy the parachute when the vertical velocity is negative
(post-apogee) and the height above ground level is less than 800 meters:

Because ``context`` exposes the full event context, you can combine any of the
available values. For example, you can use the acceleration components from
``state_dot`` (and the simulation time) to gate deployment:

.. jupyter-input::

    def main_trigger(context):
        vz = context.state.vz  # vertical velocity
        az = context.state_dot.az  # vertical acceleration
        h = context.height_agl
        time = context.time

        # activate main when descending (vz < 0) and decelerating (az > 0),
        # below 800 m, and at least 5 s into the flight
        return vz < 0 and az > 0 and h < 800 and time > 5.0

.. note::
    You can import ``c`` or ``cpp`` code into Python and use it as a callable
    trigger function. This allows you to simulate the parachute trigger system
    that will be used in the real rocket.

Legacy positional trigger signature
""""""""""""""""""""""""""""""""""""

Older trigger functions declared the following positional arguments instead of
reading them from ``context``:

- ``p`` (float): pressure in Pa **considering the parachute noise signal**.
- ``h`` (float): height above ground level in meters, **considering the
  parachute noise signal**.
- ``y`` (list of float): the state vector (the same thirteen values as
  ``context.state``, in the same order).
- ``sensors`` (list, optional fourth argument): the same list as
  ``context.sensors``.

.. note::
    The legacy positional ``p`` and ``h`` carry the parachute noise signal,
    whereas ``context.pressure`` and ``context.height_agl`` are the
    clean, noise-free values. For pressure or height signals with noise, use
    Sensor objects instead.

A legacy trigger therefore looked like this:

.. jupyter-input::

    def main_trigger(p, h, y):

        # activate main when vz < 0 m/s and h < 800 m
        return y[5] < 0 and h < 800

5. Setting Rail Guides
----------------------

In RocketPy, any rail guides are simulated as *rail buttons*. The rail buttons
are defined by their positions.

.. note::

    Rail buttons are optional for the simulation, but are very important to
    have realistic out of rail speeds and behavior.

Here is an example of how to set rail buttons:

.. jupyter-execute::

    rail_buttons = calisto.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.618,
        angular_position=45,
    )

.. caution::

    Again, pay special attention to both ``positions`` parameter. See
    the :ref:`Setting Rail Guides <setrail>` section for more information.

.. seealso::

    For more information on setting rail buttons, see
    :class:`rocketpy.Rocket.set_rail_buttons`

6. See Results
--------------

Now that we have defined the rocket, we can plot and see a bit of information
about our rocket, and double check if everything is correct.

First, lets guarantee that the rocket is stable, by plotting the static margin:

.. jupyter-execute::

    calisto.plots.static_margin()

.. danger::

    Always check the static margin of your rocket.

    If it is **negative**, your rocket is **unstable** and the simulation
    will most likely **fail**.

    If it is unreasonably **high**, your rocket is **super stable** and the
    simulation will most likely **fail**.

The stability margin at a given Mach number and time is read from
``calisto.stability_margin(mach, time)``. It is the margin with the rocket
flying straight into the air, at zero angle of attack.

The lets check all the information available about the rocket:

.. jupyter-execute::

    calisto.all_info()

7. Inertia Tensors
------------------

The inertia tensor relative to the center of dry mass of the rocket at a
given time can be obtained using the ``get_inertia_tensor_at_time`` method.
This method evaluates each component of the inertia tensor at the specified
time and returns a :class:`rocketpy.mathutils.Matrix` object.

The inertia tensor is a matrix that looks like this:

.. math::
    :label: inertia_tensor

    \mathbf{I} = \begin{bmatrix}
    I_{11} & I_{12} & I_{13} \\
    I_{21} & I_{22} & I_{23} \\
    I_{31} & I_{32} & I_{33}
    \end{bmatrix}

For example, to get the inertia tensor of the rocket at time 0.5 seconds, you
can use the following code:

.. jupyter-execute::

    calisto.get_inertia_tensor_at_time(0.5)

Derivative of the Inertia Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also get the derivative of the inertia tensor at a given time using the
``get_inertia_tensor_derivative_at_time`` method. Here's an example:

.. jupyter-execute::

    calisto.get_inertia_tensor_derivative_at_time(0.5)

Implications from these results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inertia tensor reveals important information about the rocket's symmetry
and ease of rotation:

1. **Axis Symmetry**: If I\ :sub:`11` and I\ :sub:`22` are equal, the rocket is symmetric around the axes perpendicular to the rocket's center axis. In our defined rocket, I\ :sub:`11` and I\ :sub:`22` are indeed equal, indicating that our rocket is axisymmetric.

2. **Zero Products of Inertia**: The off-diagonal elements of the inertia tensor are zero, which means the products of inertia are zero. This indicates that the rocket is symmetric around its center axis.

3. **Ease of Rotation**: The I\ :sub:`33` value is significantly lower than the other two. This suggests that the rocket is easier to rotate around its center axis than around the axes perpendicular to the rocket. This is an important factor when considering the rocket's stability and control.

However, these conclusions are based on the assumption that the inertia tensor is calculated with respect to the rocket's center of mass and aligned with the principal axes of the rocket. If the inertia tensor is calculated with respect to a different point or not aligned with the principal axes, the conclusions may not hold.

