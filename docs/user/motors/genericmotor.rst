.. _genericmotor:

GenericMotor Class Usage
========================

Here we explore different features of the GenericMotor class.

Class that represents a simple motor defined mainly by its thrust curve.
There is no distinction between the propellant types (e.g. Solid, Liquid).
This class is meant for rough estimations of the motor performance,
therefore for more accurate results, use the ``SolidMotor``, ``HybridMotor``
or ``LiquidMotor`` classes.

Creating a Generic Motor
------------------------

To define a generic motor, we will need a few information about our motor:

- The thrust source file, which is a file containing the thrust curve of the motor. \
  This file can be a .eng file, a .rse file, or a .csv file. See more details in \
  :doc:`Thrust Source Details </user/motors/thrust>`
- A few physical parameters, which the most important are:
    - The burn time of the motor.
    - The combustion chamber radius;
    - The combustion chamber height;
    - The combustion chamber position;
    - The propellant initial mass;
    - The nozzle radius;
    - The motor dry mass.

The usage of the GenericMotor class is very similar to the other motor classes.
See more details in the
:doc:`SolidMotor Class Usage </user/motors/solidmotor>`,
:doc:`LiquidMotor Class Usage </user/motors/liquidmotor>`, and
:doc:`HybridMotor Class Usage </user/motors/hybridmotor>` pages.


.. jupyter-execute::

    from rocketpy.motors import GenericMotor

    # Define the motor parameters
    motor = GenericMotor(
      thrust_source = "../data/motors/cesaroni/Cesaroni_M1670.eng",
      burn_time = 3.9,
      chamber_radius = 33 / 100,
      chamber_height = 600 / 1000,
      chamber_position = 0,
      propellant_initial_mass = 2.5,
      nozzle_radius = 33 / 1000,
      dry_mass = 1.815,
      center_of_dry_mass_position = 0,
      dry_inertia = (0.125, 0.125, 0.002),
      nozzle_position = 0,
      reshape_thrust_curve = False,
      interpolation_method = "linear",
      coordinate_system_orientation = "nozzle_to_combustion_chamber",
    )

    # Print the motor information
    motor.info()

.. note::

    The GenericMotor is a simplified model of a rocket motor. If you need more \
    accurate results, use the ``SolidMotor``, ``HybridMotor`` or ``LiquidMotor`` classes.


.. _load_from_eng_file:

The ``load_from_eng_file`` method
---------------------------------

The ``GenericMotor`` class has a method called ``load_from_eng_file`` that allows
the user to build a GenericMotor object by providing just the path to an .eng file.
The parameters available in the method are the same as the ones used in the
constructor of the GenericMotor class. But the method will automatically read
the .eng file and extract the required information if the user does not
provide it. In this case, the following assumptions about the most
relevant parameters are made:

- The ``chamber_radius`` is assumed to be the same as the motor diameter in the .eng file;
- The ``chamber_height`` is assumed to be the same as the motor length in the .eng file;
- The ``chamber_position`` is assumed to be 0;
- The ``propellant_initial_mass`` is assumed to be the same as the propellant mass in the .eng file;
- The ``nozzle_radius`` is assumed to be 85% of the ``chamber_radius``;
- The ``dry_mass`` is assumed to be the total mass minus the propellant mass in the .eng file;

As an example, we can demonstrate:

.. jupyter-execute::

    from rocketpy.motors import GenericMotor


    # Load the motor from an .eng file
    motor = GenericMotor.load_from_eng_file("../data/motors/cesaroni/Cesaroni_M1670.eng")

    # Print the motor information
    motor.info()

Although the ``load_from_eng_file`` method is very useful, it is important to
note that the user can still provide the parameters manually if needed.

.. tip::

  The ``load_from_eng_file`` method is a very useful tool for simulating motors \
  when the user does not have all the information required to build a ``SolidMotor`` yet.


The ``list_motors_dataset`` function
------------------------------------

RocketPy includes a set of pre-registered solid rocket motors in the ``.eng`` format,
which are stored in the ``rocketpy/datasets/motors/`` directory. These motors can be used
directly to create ``GenericMotor`` objects, making it easier to get started with simulations
without needing to search for external motor files.

The ``list_motors_dataset`` function allows users to quickly inspect which pre-registered
motors are currently available. It returns a list of motor names that can be used with
the ``load_motor_from_dataset`` function.

.. jupyter-execute::

    from rocketpy.utilities import list_motors_dataset

    # List all available motors in the dataset
    motors = list_motors_dataset()
    print(motors)


The ``load_motor_from_dataset`` function
----------------------------------------

The ``load_motor_from_dataset`` function loads a pre-registered motor by name,
returning a ``GenericMotor`` object. 

Internally, it uses the ``load_from_eng_file``
method from the ``GenericMotor`` class to parse the corresponding ``.eng`` file.
Therefore, it applies the same assumptions described previously in the
:ref:`load_from_eng_file <load_from_eng_file>` section. This includes default values for parameters such as
``chamber_radius``, ``nozzle_radius``, and ``dry_mass``.

.. jupyter-execute::

    from rocketpy.utilities import load_motor_from_dataset

    # Load a motor using its dataset name
    motor = load_motor_from_dataset("Cesaroni_M1670")

    # Print motor info
    motor.info()


The ``show_motors_dataset`` function
-----------------------------------------

The ``show_motors_dataset`` function is a utility that prints the list of
available pre-registered motors directly to the terminal or notebook output,
including how many motors are available.
It is helpful for quick visual inspection when an explicit return value is not needed.

.. jupyter-execute::

    from rocketpy.utilities import show_motors_dataset

    # Show the list of available motors (prints to output)
    show_motors_dataset()

