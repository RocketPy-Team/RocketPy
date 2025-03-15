Thrust Source
=============

The thrust source is of most importance when analyzing several trajectory
attributes, including the out of rail velocity, thrust to weight ratio, apogee
and many others. Let's create a new motor and take a closer look at this
functionality.

Constant Thrust
---------------

When passing an ``int`` or ``float``, the thrust will be considered constant in time.

.. jupyter-execute::

    from rocketpy import SolidMotor

    solid_constant = SolidMotor(
        thrust_source=1500,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        burn_time=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

Let's call the ``info()`` method to see what kind of information we get.
Alternatively, we could use the ``all_info()`` method to get a more complete
output.

.. jupyter-execute::

    solid_constant.info()

Thrust From Static Firings (CSV Files)
--------------------------------------

Usually one has much more precise information about the motor and wishes to
specify a directory containing a .csv file.

.. note::

        The first column of the .csv file must be time in seconds and the second
        column must be thrust in Newtons. The file can contain a single line header.

That can be done as follows:

.. jupyter-execute::

    solid_csv = SolidMotor(
        thrust_source="../data/motors/projeto-jupiter/keron_thrust_curve.csv",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        burn_time=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    solid_csv.info()

.. attention::

    Beware of noisy data. RocketPy will not filter the data for you.

    The thrust curve will be integrated to obtain the total impulse. This can
    lead to a significant error in the total impulse if the data is too
    noisy.

Also one is able to specify a certain interpolation method. That can be done by
simply changing the ``interpolation_method`` parameter to ``spline`` , ``akima``
or ``linear``. Default is set to ``linear``.

Eng Files (RASP)
----------------

Most rocket motors providers share the thrust curve from their motors using
the RASP file format (``.eng`` files). RocketPy can import such files as the
thrust source.

.. note::

    If you have a thrust curve in a ``.csv`` file, RocketPy can also read your
    data and exported as a ``.eng`` file. This can be done by using the
    :class:`rocketpy.Motor.export_eng()` Motor method.

.. jupyter-execute::

    solid_eng = SolidMotor(
        thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        burn_time=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

This time we want to try the all_info() to capture more details about the motor.

.. jupyter-execute::

    solid_eng.all_info()

Lambda Functions
----------------

There is also a fourth option where one specifies the thrust source parameter by
passing a callable function like below.

Lambda functions are particularly useful in Python, and therefore the SolidMotor
class also supports them. Let's see how to use it.

.. jupyter-execute::

    solid_lambda = SolidMotor(
        thrust_source=lambda x: 1 / (x + 1),
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        burn_time=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    solid_lambda.info()

.. _reshaping_thrust_curve:

Reshaping and interpolating the thrust curve
--------------------------------------------

RocketPy can rescale a given curve to match new specifications when impulse
and burn out time are expected to vary only slightly. That can be done by
passing the ``reshape_thrust_curve`` parameter as a list of two elements. The
first element is the new burn out time in seconds and the second element is the
new total impulse in Ns.

Here we will reshape the thrust curve by setting the new burn out time in
seconds to 10 and the new total impulse to be 6000 Ns.

.. jupyter-execute::
    :emphasize-lines: 5

    solid_reshaped = SolidMotor(
        thrust_source="../data/motors/projeto-jupiter/keron_thrust_curve.csv",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        reshape_thrust_curve=[10, 6000],
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        burn_time=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

Pay close attention to the newly generated curve and be aware of the changes the
rescale has produced regarding the physical quantities.

.. jupyter-execute::

    solid_reshaped.all_info()