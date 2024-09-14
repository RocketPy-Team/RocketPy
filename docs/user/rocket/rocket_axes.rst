.. _rocket_axes:

Rocket Class Axes Definitions
=============================

The Rocket class has two different coordinate systems:

1. **User Defined Coordinate System**: Used for geometrical inputs of the \
   aerodynamic surfaces and motor.
2. **Body Axes Coordinate System**: Used during the flight simulation to assess \
   the governing equations of motion.

All inputs are automatically converted from the user's coordinate system to the
rocket body axes coordinate system for use during the simulation.

Let's dive into the definitions of these coordinate systems:


1. User Defined Coordinate System
---------------------------------

Two things are set by the user in the user input coordinate system:

1. **Coordinate System Origin**: The origin of the coordinate system is set at \
   any point along the rocket's center line. This point can be arbitrarily chosen \
   and is not explicitly defined. All inputs must be given relative to this \
   point.
2. **Direction of Center Axis**: Specified by the ``coordinate_system_orientation`` \
   argument when initializing the Rocket (:class:`rocketpy.Rocket.__init__`). This \
   argument defines the direction of the axis that follows the rocket's center \
   line. It can be either ``"nose_to_tail"`` or ``"tail_to_nose"``.

.. tip::

   If you are using some CAD software to design your rocket, you can imagine the \
   coordinate system as the one used in the CAD software. The origin of the \
   coordinate system is the origin of the rocket in the CAD software, and the \
   direction of the center axis is the direction of the rocket's centerline. \
   You don't need to worry about the exact position of the origin, as long as \
   all inputs are given relative to this point.

.. seealso:: 
     
   See `Positions and Coordinate Systems <positions.rst>`_ for more \
   information on the definitions of the rocket's aerodynamic surfaces and motor.

The ``x`` and ``y`` axes are defined at the plane perpendicular to the center axis,
while the ``z`` axis is defined along the center axis. Depending on the choice of
``coordinate_system_orientation``, the ``x`` axis and ``y`` axis can be inverted.

The following figure shows the two possibilities for the user input coordinate system:

.. figure:: ../../static/rocket/3dcsys.png
  :align: center
  :alt: Rocket axes

.. note::

   When ``coordinate_system_orientation`` is set to ``"tail_to_nose"``, the direction \
   of the ``x``, ``y``, and ``z`` axes of the **User Defined Coordinate System** is \
   the same as the **Body Axes Coordinate System**. The origin of the coordinate \
   system may still be different.

Angular Position Inputs
~~~~~~~~~~~~~~~~~~~~~~~

Angular position inputs (``angular_position``) refer to the roll angle position
of that surface along the rocket's tube. The roll angle is defined as the angle 
from the ``y`` axis to the surface.
Currently, only the :class:`rocketpy.RailButtons` class uses this kind of input.  

The following figure shows the roll angle
definition for both ``coordinate_system_orientation`` options:

.. figure:: ../../static/rocket/angularpos.png
  :align: center
  :alt: Angular position


.. note::

   The positive direction of the roll angle is defined as the direction that \
   rotates the surface in the positive direction of the ``z`` axis.

.. _rocket_axes_body_axes:

2. Body Axes Coordinate System
------------------------------

The body axes coordinate system is used inside the simulation to assess the
governing equations of motion. The body axes coordinate system is defined as follows:

- The origin is at the rocket's center of dry mass (``center_of_dry_mass_position``).
- The ``z`` axis is defined along the rocket's centerline, pointing from the center of dry mass towards the nose.
- The ``x`` and ``y`` axes are perpendicular.

3. Relation to Flight Coordinates
---------------------------------

The ``Flight`` class uses a coordinate system defined as follows:

- The origin is at the launch rail.
- The ``Z`` axis is positive upwards.
- The ``X`` axis is position eastwards.
- The ``Y`` axis is positive northwards.

The following figure shows the rotational relationship between the
**Body Axes Coordinate System** and the **Flight Coordinate System**:

.. figure:: ../../static/rocket/flightcsys.png
  :align: center
  :alt: Flight coordinate system

In the figure above, :math:`\bf{i}` is the ``inclination`` and :math:`\bf{h}`
is the ``heading`` of the launch rail.

The heading and inclination can be described in terms of Euler angles.
The relation is given by:

.. math::
    \begin{aligned}
        &\text{Precession:} \quad &\psi &= -\bf{h} \\
        &\text{Nutation:} \quad &\theta &= \bf{i} - 90° \\
   \end{aligned}

A last rotation is defined by the ``angular_position`` of the rocket's rail buttons.
This is a rotation around the rocket's centerline, and describes the last 
Euler angle:

.. math::
    \begin{aligned}
        &\text{Spin:} \quad &φ & \\
    \end{aligned}

If no rail buttons pair ir present, the spin angle is set to **0°**.

.. note::
   
   With spin angle set to **0°**, if the launch rail ``heading`` is set to \
   **0°** and rail ``inclination`` to **90°**, the **Body Axes Coordinate \
   System** is aligned with the **Flight Coordinate System**.

Rocket's initial orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial orientation of the rocket is expressed in Euler parameters (quaternions).
The Euler parameters are defined using the 3-1-3 rotation sequence:

.. math::

   \begin{aligned}
      e_{0} &= \cos\left(\frac{φ}{2}\right) \cos\left(\frac{θ}{2}\right) \cos\left(\frac{ψ}{2}\right) - \sin\left(\frac{φ}{2}\right) \cos\left(\frac{θ}{2}\right) \sin\left(\frac{ψ}{2}\right) \\
      e_{1} &= \cos\left(\frac{φ}{2}\right) \cos\left(\frac{ψ}{2}\right) \sin\left(\frac{θ}{2}\right) + \sin\left(\frac{φ}{2}\right) \sin\left(\frac{θ}{2}\right) \sin\left(\frac{ψ}{2}\right) \\
      e_{2} &= \cos\left(\frac{φ}{2}\right) \sin\left(\frac{θ}{2}\right) \sin\left(\frac{ψ}{2}\right) - \sin\left(\frac{φ}{2}\right) \cos\left(\frac{ψ}{2}\right) \sin\left(\frac{θ}{2}\right) \\
      e_{3} &= \cos\left(\frac{φ}{2}\right) \cos\left(\frac{θ}{2}\right) \sin\left(\frac{ψ}{2}\right) + \cos\left(\frac{θ}{2}\right) \cos\left(\frac{ψ}{2}\right) \sin\left(\frac{φ}{2}\right) \\
   \end{aligned}


