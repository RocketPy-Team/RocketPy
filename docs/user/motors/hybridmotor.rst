HybridMotor Class Usage
=======================

Here we explore different features of the HybridMotor class.

A hybrid motor is motor composed by a solid fuel grain and a liquid oxidizer.
The solid fuel grain is usually made of a rubber-like material, and the liquid
oxidizer is usually nitrous oxide. The solid fuel grain is ignited by a spark
plug, and the liquid oxidizer is injected into the combustion chamber through
an injector. The combustion of the solid fuel grain and the liquid oxidizer
generates the thrust.

When simulating this kind of motor in RocketPy, we internally use two motors:
one for the solid fuel grain and another for the liquid oxidizer.

The solid fuel behavior is simulated by a solid motor, and the liquid oxidizer
behavior is simulated by a liquid motor. The combination of these two create
a valid hybrid motor. Everything is then defined at the same time using the
HybridMotor class.

Creating a Hybrid Motor
-----------------------

To define a hybrid motor, we will need a few information about our motor:

- The thrust source file, which is a file containing the thrust curve of the 
  motor. This file can be a .eng file, a .rse file, or a .csv file. See more 
  details in :doc:`Thrust Source Details </user/motors/thrust>`
- A Tank object, which defines the liquid oxidizer tank. See more details in 
  :doc:`Tank Usage</user/motors/tanks>`
- Solid fuel parameters, such as the number of grains, their geometry, 
  and their density
- Position related parameters, such as the position of the center of mass of the
  dry mass, the position of the center of mass of the grains, and the position 
  of the nozzle. See more details in 
  :ref:`Motor Coordinate Systems <motorcsys>`

So, lets first import the necessary classes:

.. jupyter-execute::

  from rocketpy import Fluid, CylindricalTank, MassFlowRateBasedTank, HybridMotor 

Then we must first define the oxidizer tank:

.. seealso::
  :doc:`Tank Usage </user/motors/tanks>`


.. jupyter-execute::

  # Define the fluids
  oxidizer_liq = Fluid(name="N2O_l", density=1220)
  oxidizer_gas = Fluid(name="N2O_g", density=1.9277)

  # Define tank geometry
  tank_shape = CylindricalTank(115 / 2000, 0.705)

  # Define tank
  oxidizer_tank = MassFlowRateBasedTank(
      name="oxidizer tank",
      geometry=tank_shape,
      flux_time=5.2,
      initial_liquid_mass=4.11,
      initial_gas_mass=0,
      liquid_mass_flow_rate_in=0,
      liquid_mass_flow_rate_out=(4.11 - 0.5) / 5.2,
      gas_mass_flow_rate_in=0,
      gas_mass_flow_rate_out=0,
      liquid=oxidizer_liq,
      gas=oxidizer_gas,
  )

.. note::
  This tank is a :ref:`MassFlowRateBasedTank <mass_flow_rate_based_tank>`,
  which means that the mass flow rates of the liquid and gas are defined by the
  user. In this case, the tank has only a liquid mass flow rate out, which is
  constant.

Now we can define our hybrid motor. We are using a lambda function as the thrust
curve, but keep in mind that you can use 
:doc:`different formats </user/motors/thrust>` here.

.. jupyter-execute::

  example_hybrid = HybridMotor(
      thrust_source=lambda t: 2000 - (2000 - 1400) / 5.2 * t,
      dry_mass=2,
      dry_inertia=(0.125, 0.125, 0.002),
      nozzle_radius=63.36 / 2000,
      grain_number=4,
      grain_separation=0,
      grain_outer_radius=0.0575,
      grain_initial_inner_radius=0.025,
      grain_initial_height=0.1375,
      grain_density=900,
      grains_center_of_mass_position=0.384,
      center_of_dry_mass_position=0.284,
      nozzle_position=0,
      burn_time=5.2,
      throat_radius=26 / 2000,
  )

.. caution::
    Pay special attention to:

    - ``dry_inertia`` is defined as a tuple of the form ``(I11, I22, I33)``.
      Where ``I11`` and ``I22`` are the inertia of the dry mass around the
      perpendicular axes to the motor, and ``I33`` is the inertia around the
      motor center axis. 
    - ``dry inertia`` is defined in relation to the **center of dry mass**, and 
      not in relation to the coordinate system origin.
    - ``grains_center_of_mass_position``, ``center_of_dry_mass_position`` and 
      ``nozzle_position`` are defined in relation to the 
      :ref:`coordinate system origin <motorcsys>`, which is the nozzle outlet in
      this case.
    - Both ``dry_mass`` **and** ``center_of_dry_mass_position`` must consider
      the mass of the tanks.

.. seealso:: 
    
    You can find details on each of these parameters in 
    :class:`rocketpy.HybridMotor.__init__`

Finally we can add the oxidizer tank to the hybrid motor. This is done using the
:ref:`add_tank <Adding Tanks>` method.

.. jupyter-execute::

  example_hybrid.add_tank(
    tank = oxidizer_tank, position = 1.0615
  )

And we can see all the results with:

.. jupyter-execute::

  example_hybrid.all_info()
