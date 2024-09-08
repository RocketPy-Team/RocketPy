.. _genericmotor:

GenericMotor Class Usage
======================

Here we explore different features of the GenericMotor class.

Class that represents a simple motor defined mainly by its thrust curve.
There is no distinction between the propellant types (e.g. Solid, Liquid).
This class is meant for rough estimations of the motor performance,
therefore for more accurate results, use the ``SolidMotor``, ``HybridMotor``
or ``LiquidMotor`` classes.

Creating a Generic Motor
------------------------

To define a generic motor, we will need a few information about our motor:

- The thrust source file, which is a file containing the thrust curve of the motor.
  This file can be a .eng file, a .rse file, or a .csv file. See more details in
  :doc:`Thrust Source Details </user/motors/thrust>`
- A few physical parameters, which the most important are:
    - The burn time of the motor.
    - The chamber_radius;
    - The chamber_height;
    - The chamber_position;
    - The propellant initial mass;
    - The radius of the nozzle;
    - The dry mass of the motor;

The usage of the GenericMotor class is very similar to the other motor classes. See
more details in
:doc:`SolidMotor Class Usage </user/motors/solidmotor>`,
:doc:`LiquidMotor Class Usage </user/motors/liquidmotor>`,
:doc:`HybridMotor Class Usage </user/motors/hybridmotor>`.

The ``load_from_eng_file method``
-----------------------------

The GenericMotor class has a method called ``load_from_eng_file`` that allows
the user to build a GenericMotor object only by providing the path to the .eng file.

The parameters available in the method are the same as the ones used in the
constructor of the GenericMotor class. But the method will automatically read 
the .eng file and extract the required information if the user does not 
provide it. In this case, the following assumptions about the most 
relevant parameters are made:

- The chamber_radius is assumed to be the same as the motor diameter in the .eng file;
- The chamber_height is assumed to be the same as the motor length in the .eng file;
- The chamber_position is assumed to be 0;
- The propellant initial mass is assumed to be the same as the propellant mass in the .eng file;
- The nozzle_radius is assumed to be 85% of the chamber_radius;
- The dry mass is assumed to be the total mass minus the propellant mass in the .eng file;

