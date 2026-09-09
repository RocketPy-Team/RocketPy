.. _stochastic_usage:

Working with Stochastic objects
===============================

For each RocketPy object (e.g. Environment, SolidMotor, Rocket, etc.), we can
create a ``Stochastic`` counterpart that extends the initial model, allowing us
to define the uncertainties of each input parameter.

The idea of the ``Stochastic`` classes is to take the deterministic classes and
assign uncertainties to their input parameters. This will reflect the inherent
uncertainties in the real-world data and provide a more realistic simulation.

.. seealso::
    For ``Stochastic`` class API details, see :ref:`stochastic_reference`.

.. note::
    In this tutorial, classes without the ``Stochastic`` prefix are considered \
    deterministic. For instance, ``SolidMotor`` is a deterministic class, while \
    its stochastic counterpart is named ``StochasticSolidMotor``.

Initialization Parameters
-------------------------

In terms of initialization parameters, the ``Stochastic`` classes are very
similar to the deterministic classes.
We will separate the parameters into three categories: the "deterministic object",
the "optional parameters", and the "additional parameters".
Let's take a look at their nuances:

.. note::
    In Python, we use the terms "argument", "parameter", and "initialization parameter"
    interchangeably to refer to the values passed to a function or class during initialization.

    For the sake of clarity, we will use the term "argument" from now on.

Arguments
^^^^^^^^^

1. **Deterministic Object**: \
    All ``Stochastic`` classes **must** receive a \
    deterministic object as an argument. This is the only mandatory argument.

2. **Optional Arguments**: \
    The remaining parameters are the same as in the \
    deterministic classes, and they are optional. They only need to be passed \
    if you want to define the uncertainty for that argument. If you don't pass an \
    argument, it will not be varied during the simulation, and the nominal value \
    will be taken from the deterministic object.

3. **Additional Arguments**: \
    Some ``Stochastic`` classes may present additional \
    arguments that are not present in the deterministic classes. These are used \
    for specific purposes, such as a multiplication factor for the drag curves.


Specifying Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^

Furthermore, the optional arguments - which define the uncertainties - can be
passed in a few different ways:

1. **As a single value**: \
    This will be the standard deviation for that parameter. \
    The default distribution used will be a normal distribution, and the nominal \
    value will be the value of that same argument from the deterministic object.

2. **As a tuple of two numbers**: \
    The first number will be the nominal value of \
    the distribution, and the second number will be the standard deviation. The \
    default distribution used will be a normal distribution.

3. **As a tuple of two numbers and a string**: \
    The first number will be the \
    nominal value of the distribution, the second number will be the standard \
    deviation, and the string will be the distribution type. The distribution \
    type can be one of the following: *"normal"*, *"binomial"*, *"chisquare"*, \
    *"exponential"*, *"gamma"*, *"gumbel"*, *"laplace"*, *"logistic"*, \
    *"poisson"*, *"uniform"*, and *"wald"*.

3. **As a tuple of a number and a string**: \
    The number will be the standard \
    deviation, and the string will be the distribution type. The nominal value \
    will be taken from the standard object.

4. **As a list of values**: \
    The values will be randomly chosen from this list and \
    used as the parameter value during the simulation. You cannot assign standard \
    deviations when using lists, nor can you assign different distribution types.

5. **A CustomSampler object**: \
    An object from a class that inherits from ``CustomSampler``. This object \
    gives you the full control of how the samples are generated. See 
    :ref:`custom_sampler` for more details.

.. note::
    The formats above assume each argument holds a single number. The
    ``shape_points`` of :class:`rocketpy.stochastic.StochasticFreeFormFins` is
    the exception: a fin outline is only meaningful as a complete set of points,
    so the deviation given applies to the outline as a block, with every
    coordinate of every point perturbed by its own draw. The fin root is held on
    the body line, so a point nominally at ``y = 0`` keeps that value and no
    point ends up inside the airframe.

    Because the deviation has to centre on the nominal coordinate, only the
    distributions that read their first argument as that centre can be used here:
    *"normal"*, *"gumbel"*, *"laplace"* and *"logistic"*. The others take bounds
    (*"uniform"*) or shape parameters (*"wald"*, *"gamma"*, ...), which a set of
    coordinates cannot be, and are rejected when the object is created.

    A list means either one fixed outline, used as given, or a list of candidate
    outlines to choose between, which do not have to have the same number of
    points::

        # One millimetre of deviation on every coordinate
        StochasticFreeFormFins(free_form_fins=fins, shape_points=0.001)

        # One fixed outline, not randomized
        StochasticFreeFormFins(
            free_form_fins=fins,
            shape_points=[(0, 0), (0.08, 0.1), (0.12, 0)],
        )

        # Choose between two outlines
        StochasticFreeFormFins(
            free_form_fins=fins,
            shape_points=[[(0, 0), (0.08, 0.1), (0.12, 0)], [(0, 0), (0.06, 0.12), (0.12, 0)]],
        )

    A ``CustomSampler`` given for this argument has to yield a whole outline per
    sample, since what it returns replaces the outline instead of perturbing it.

.. note::
    In statistics, the terms "Normal" and "Gaussian" refer to the same type of \
    distribution. This distribution is commonly used and is the default for the \
    ``Stochastic`` classes in RocketPy.

    In this context, a "distribution" refers to a function that describes the \
    probability of a parameter assuming a certain value. The type of distribution \
    determines the shape of this function. We use the term "distribution" to \
    simplify the explanation of the stochastic classes.

Examples
--------

Here is a better explanation of the arguments with examples:

Example 1: Stochastic Solid Motor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the ``StochasticSolidMotor`` object:

.. jupyter-execute::

    from rocketpy import SolidMotor, StochasticSolidMotor

    motor = SolidMotor(
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

    stochastic_motor = StochasticSolidMotor(
        solid_motor=motor,
        burn_start_time=(0, 0.1, "binomial"),
        grains_center_of_mass_position=0.001,
        grain_density=10,
        grain_separation=1 / 1000,
        grain_initial_height=1 / 1000,
        grain_initial_inner_radius=0.375 / 1000,
        grain_outer_radius=0.375 / 1000,
        total_impulse=(6500, 100),
        throat_radius=0.5 / 1000,
        nozzle_radius=0.5 / 1000,
        nozzle_position=0.001,
    )

    stochastic_motor.visualize_attributes()

Interpreting the Output
"""""""""""""""""""""""

To illustrate the example above, you can notice that:

- The ``burn_start_time`` argument was specified as a tuple of 3 items (0, 0.1, "binomial"), meaning the nominal value is 0, the standard deviation is 0.1, and the distribution type is binomial. You can check that it was correctly set being reading the ``burn_start_time: 0.00000 ± 0.10000 (numpy.random.binomial)`` line in the output.
- ``total_impulse`` was given as a tuple of 2 numbers (6500, 100), indicating a nominal value of 6500 and a standard deviation of 1000, with the default distribution being normal, which is the default distribution type.

.. note::
    Always remember to run ``stochastic_object.visualize_attributes()`` to check \
    if the uncertainties were correctly set.

Sampling a Stochastic Object
""""""""""""""""""""""""""""

Continuing with the example, you can use the ``stochastic_motor`` object to generate
a random `SolidMotor` object considering the uncertainties defined in the initialization.

.. jupyter-execute::

    sampled_motor = stochastic_motor.create_object()
    print(sampled_motor)

This will create a new ``SolidMotor`` object in memory and assign it to the
variable ``sampled_motor``. This behaves exactly like a ``SolidMotor`` object, but
considering that each parameter was randomly sampled from the defined distributions.
We can compare the nominal values of the ``motor`` object with the sampled values
of the ``sampled_motor`` object:

.. jupyter-execute::

    print("Deterministic Motor with nominal values:\n")
    motor.prints.all()
    print("\n\nSampled Motor considering uncertainties:\n")
    sampled_motor.prints.all()

As you can notice, the values from the ``sampled_motor`` object are slightly different
from the nominal values of the ``motor`` object.

.. important::
    If you run the ``create_object()`` method multiple times, you will get different
    results each time, as the values are always randomly sampled from the defined
    distributions.


Determining Uncertainties
-------------------------

Determining the uncertainties for each parameter is crucial for accurate simulations.
Here are some practical methods:

1. **Empirical Measurements**: \
    For geometric properties and other parameters that \
    can be measured, you can take multiple measurements and calculate the standard \
    deviation. This method provides a direct and reliable estimate of uncertainty. \
    Some examples include: rocket mass, dimensions or positions and material density.

2. **Historical Data**: \
    Use historical data from previous experiments or similar \
    projects to base your standard deviations. For example, if you are designing a \
    rocket with similar characteristics to a previous project, you can use the \
    uncertainties from that project as a starting point.

3. **Literature Review**: \
    Review literature and technical documents to find \
    estimation values for uncertainties. For example, for aerodynamic coefficients, \
    you can find typical values in textbooks or research papers, these usually \
    come from wind tunnel tests. A good resource to base your uncertainties is the \
    `RocketPy article <https://doi.org/10.1061/(ASCE)AS.1943-5525.0001331>`_.

5. **Rule of Thumb**: \
    In the absence of specific data, you can use general rules \
    of thumb. For example, assigning a standard deviation of 10% of the nominal \
    value is a common practice.

As your rocket project evolves, you will likely gather more data and refine your
models. Consequently, the uncertainties should decrease, resulting in stochastic
models with less variance. This iterative process will enhance the accuracy and
reliability of your simulations over time.

.. Determining Which Arguments to Vary
.. -----------------------------------

.. Choosing which arguments to vary is crucial for effective Monte Carlo simulations.
.. RocketPy offers a ``Sensitivity Analysis toolkit``, which can help you to identify
.. which parameters most significantly impact your simulation results.


Seeding a rocket's components
-----------------------------

A ``StochasticRocket`` holds nested stochastic objects: the motors, the
aerodynamic surfaces, the rail buttons, the parachutes and the air brakes. Each
time the rocket is reset, every component attached to it at that moment is given
a stream of its own, spawned from the seed the reset was given. A main and a
drogue parachute built from the same ``cd_s`` and ``lag`` are then no longer
made to consume the same draws as each other. Independent streams can still land
on equal values, and a specification with no spread always will.

The reset is what builds the tree, not ``add_parachute`` or ``add_nose``. A
rocket resets itself once while being constructed, when it has no components
yet, so anything attached afterwards keeps the generator it was built with until
something resets it again. A serial ``MonteCarlo`` run does not reset it at
all. A parallel one tries to, once per worker, but hands the model a
``SeedSequence`` where an integer is wanted, so that path does not get as far
as building the tree either. Resetting per simulation, from an integer seed the
caller chooses, is what the Monte Carlo seeding work adds.

Each collection is spawned separately, so adding an aerodynamic surface does not
move what the parachutes draw. Within a collection the stream follows insertion
order, so adding a component ahead of another does change what the later one
draws under a fixed seed. The rocket's own inputs, such as ``mass`` and
``radius``, use the seed exactly as given.

.. note::
    A component's *position* is a property of the rocket rather than of the
    component, so it is drawn from the rocket's own stream.

.. note::
    A stream belongs to one stochastic wrapper. Storing the same wrapper twice,
    or sharing one between two rockets, is not supported: the second reset
    replaces the first, and the two entries end up drawing from one generator.

    A shared ``CustomSampler.seed_group`` keeps its own rule: a group belongs to
    one model. Sharing one between two components leaves each of them seeding it
    from their own child, and the last one to be reset decides what both draw.

Conclusion
----------

The ``Stochastic`` classes in RocketPy provide a powerful way to introduce and
manage uncertainties in your simulations. By defining distributions for each
input parameter, you can perform more realistic and robust Monte Carlo simulations,
better reflecting the inherent uncertainties in rocketry.

.. note::
    See the ``MonteCarlo`` class documentation for more information on how to run \
    Monte Carlo simulations with stochastic objects.
