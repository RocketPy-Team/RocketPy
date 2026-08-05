.. _ringclustermotor:

RingClusterMotor Class Usage
============================

Here we explore different features of the ``RingClusterMotor`` class.

``RingClusterMotor`` models a ring (annular) configuration of ``N`` identical
motors arranged symmetrically around a circular perimeter of a given radius,
with no central motor along the rocket's longitudinal axis. It is a thin
wrapper around a single, already-defined motor: all thrust-, mass- and
inertia-related quantities are derived automatically from that base motor,
its ``number`` of copies and their ``radius`` from the rocket's central axis.

Key Assumptions
---------------

- ``number`` must be an integer ``>= 2``;
- ``radius`` must be non-negative;
- The base motor is currently expected to be a ``SolidMotor``, since
  ``RingClusterMotor`` reuses its grain properties directly;
- Motors are assumed identical and evenly spaced around the ring
  (``360 / number`` degrees apart);
- The transverse inertia (``I11``/``I22``) contribution of each motor is
  computed explicitly via the parallel axis (Steiner) theorem for every
  angular position, which keeps the result accurate even for the
  asymmetric ``number=2`` case.

Creating a Ring Cluster Motor
------------------------------

To define a ``RingClusterMotor``, we first need a fully defined base motor
(typically a ``SolidMotor``), then wrap it with the number of motors in the
cluster and their radial distance from the rocket's central axis:

.. jupyter-execute::

    from rocketpy import SolidMotor, RingClusterMotor

    base_motor = SolidMotor(
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

    cluster_motor = RingClusterMotor(
        motor=base_motor,
        number=4,
        radius=0.1,
    )

    # Print the cluster (and underlying single-motor) information
    cluster_motor.info()

.. note::

    ``RingClusterMotor`` does not read a thrust curve file directly. Instead,
    it scales the ``thrust``, ``dry_mass``, ``propellant_mass`` and inertia
    properties of the ``motor`` passed in by ``number``, so any changes to
    ``base_motor`` must be made before it is wrapped.

Since a ``RingClusterMotor`` is a ``Motor`` like any other, it can be attached
to a ``Rocket`` with :meth:`Rocket.add_motor` exactly like a ``SolidMotor``,
``HybridMotor`` or ``LiquidMotor`` would be.

Visualizing the Cluster Layout
-------------------------------

The ``draw_cluster_layout`` method plots the position of each motor around
the ring, which is useful to double-check the ``number`` and ``radius``
parameters before running a simulation:

.. jupyter-execute::

    cluster_motor.draw_cluster_layout(rocket_radius=0.15)

.. tip::

    Passing ``rocket_radius`` draws the rocket's outer tube as a dashed
    circle, which helps confirm that the motors fit inside the airframe.
