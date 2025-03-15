.. _solidmotor:

SolidMotor Class Usage
======================

Here we explore different features of the SolidMotor class.

Key Assumptions
---------------

A few comments on the grain configuration:

- Only BATES grain configuration is currently supported;
- Exhaust velocity is assumed to be constant throughout the burn, meaning the
  specific impulse is constant.

Creating a Solid Motor
----------------------

To define a solid motor, we will need a few information about our motor:

- The thrust source file, which is a file containing the thrust curve of the motor.
  This file can be a .eng file, a .rse file, or a .csv file. See more details in
  :doc:`Thrust Source Details </user/motors/thrust>`
- Physical parameters, such as the dry mass, inertia and nozzle radius
- Propellant parameters, such as the number of grains, their geometry,
  and their density
- Position related parameters, such as the position of the center of mass of the
  dry mass, the position of the center of mass of the grains, and the position
  of the nozzle. See more details in
  :ref:`Motor Coordinate Systems <motorcsys>`

Let's instantiate a ``SolidMotor`` object:

.. jupyter-execute::

  from rocketpy import SolidMotor

  example_solid = SolidMotor(
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

.. caution::
    Pay special attention to:

    - ``dry_inertia`` is defined as a tuple of the form ``(I11, I22, I33)``.
      Where ``I11`` and ``I22`` are the inertia of the dry mass around the
      perpendicular axes to the motor, and ``I33`` is the inertia around the
      motor center axis.
    - ``dry_inertia`` is defined in relation to the **center of dry mass**, and
      not in relation to the coordinate system origin.
    - ``grains_center_of_mass_position``, ``center_of_dry_mass_position`` and
      ``nozzle_position`` are defined in relation to the
      :ref:`coordinate system origin <motorcsys>`, which is the nozzle outlet in
      this case.

.. seealso::

    You can find details on each of these parameters in
    :class:`rocketpy.SolidMotor.__init__`

We can now access all the information about our motor:

.. jupyter-execute::

  example_solid.all_info()
