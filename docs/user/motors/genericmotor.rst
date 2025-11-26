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

The ``load_from_thrustcurve_api`` method
----------------------------------------

The ``GenericMotor`` class provides a convenience loader that downloads a temporary
`.eng` file from the ThrustCurve.org public API and builds a ``GenericMotor``
instance from it. This is useful when you know a motor designation (for example
``"M1670"``) but do not want to manually download and
save the `.eng` file.

.. note::

    This method performs network requests to the ThrustCurve API. Use it only
    when you have network access. For automated testing or reproducible runs,
    prefer using local `.eng` files.

Signature
----------

``GenericMotor.load_from_thrustcurve_api(name: str, **kwargs) -> GenericMotor``

Parameters
----------
name : str
    Motor name to search on ThrustCurve (example:
    ``"M1670"``).Only shorthand names are accepted (e.g. ``"M1670"``, not
    ``"Cesaroni M1670"``).
    when multiple matches occur the first result returned by the API is used.
**kwargs :
    Same optional arguments accepted by the :class:`GenericMotor` constructor
    (e.g. ``dry_mass``, ``nozzle_radius``, ``interpolation_method``). Any
    parameters provided here override values parsed from the downloaded file.

Returns
----------
GenericMotor
    A new ``GenericMotor`` instance created from the .eng data downloaded from
    ThrustCurve.

Raises
----------
ValueError
    If the API search returns no motor, or if the download endpoint returns no
    .eng file or empty/invalid data.
requests.exceptions.RequestException

Behavior notes
---------------
- The method first performs a search on ThrustCurve using the provided name.
  If no results are returned a :class:`ValueError` is raised.
- If a motor is found, the method requests the .eng file in RASP format, decodes
  it and temporarily writes it to disk. A ``GenericMotor`` is then constructed
  using the existing .eng file loader. The temporary file is removed even if an
  error occurs.
- The function emits a non-fatal informational warning when a motor is found
  (``warnings.warn(...)``). This follows the repository convention for
  non-critical messages; callers can filter or suppress warnings as needed.

Example
---------------

.. jupyter-execute::

    from rocketpy.motors import GenericMotor

    # Build a motor by name (requires network access)
    motor = GenericMotor.load_from_thrustcurve_api("M1670")

    # Use the motor as usual
    motor.info()

Testing advice
---------------
- ``pytest``'s ``caplog`` or ``capfd`` to assert on log/warning output.

Security & reliability
----------------
- The method makes outgoing HTTP requests and decodes base64-encoded content;
  validate inputs in upstream code if you accept motor names from untrusted
  sources.
- Network failures, API rate limits, or changes to the ThrustCurve API may
  break loading; consider caching downloaded `.eng` files for production use.