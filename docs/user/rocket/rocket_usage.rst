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

These curves are used to calculate the drag coefficient of the rocket at any
given time.

The drag curves can be defined in two ways:

1. Passing in the path to the drag curve CSV file as a string;
2. Passing in a function that returns the drag coefficient given the Mach
   number.

Curves defined in CSV files must have the first column as the Mach number
and the second column as the drag coefficient.
Here is an example of a drag curve file:

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

The third step is to add aerodynamic surfaces (i.e. nose cone, fins and tail)
to the rocket. These surfaces are used to calculate the rocket's aerodynamic
forces and moments.

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

    - :class:`rocketpy.Rocket.add_nose`
    - :class:`rocketpy.Rocket.add_trapezoidal_fins`
    - :class:`rocketpy.Rocket.add_elliptical_fins`
    - :class:`rocketpy.Rocket.add_tail`

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
        noise=(0, 8.3, 0.5),
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
        noise=(0, 8.3, 0.5),
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

If it is a number, it is the altitude at which the parachute will be deployed.

If it is a string ``"apogee"``, the parachute will be deployed at apogee.

If it is a callable, it must be a function that takes three parameters:

- ``p``: pressure considering parachute noise signal.
- ``h``: height above ground level considering parachute noise signal.
- ``y``: state vector in the from ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.

The function must return ``True`` if the parachute should be deployed and
``False`` otherwise.

The ``p`` and ``h`` parameters are useful if you want to deploy the parachute
based on the pressure or height above ground level. The ``y`` parameter is
useful if you want to deploy the parachute based on the state vector (velocity,
attitude angle, etc).

This function is called throughout the simulation. Therefore, you can
use it to deploy the parachute at any time.

The following example shows how to define a callable trigger function that will
deploy the drogue parachute when the vertical velocity is negative (apogee)
and will deploy the main parachute when the vertical velocity is negative
(post-apogee) and the height above ground level is less than 800 meters:

.. jupyter-input::

    def drogue_trigger(p, h, y):

        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False


    def main_trigger(p, h, y):

        # activate main when vz < 0 m/s and z < 800 m
        return True if y[5] < 0 and h < 800 else False

.. note::
    You can import ``c`` or ``cpp`` code into Python and use it as a callable
    trigger function. This allows you to simulate the parachute trigger system
    that will be used in the real rocket.

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

