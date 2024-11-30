.. _testing_guidelines:

Testing Guidelines
==================

This page describes the testing philosophy used throughout RocketPy's
development with pytest. That includes the definition
and some technical discussion regarding:

* Testing philosophy and style
* Testing naming conventions
* Directory structure
* Unit tests definition
* Integration tests definition
* Acceptance tests definition

However, some other topics such as naming conventions are going to be treated.

Testing philosophy and style
----------------------------

First of all, it is worth noting the role of tests within the framework of RocketPy. Developers must be aware that:

* Unit tests are the minimum requirement for a feature to be accepted.

That is, for each feature must correspond a testing unit which properly documents and tests the newly implemented feature.
In even more practical terms that means the Pull Request containing the feature should include an unit test together with it.

Testing Naming Conventions
--------------------------

Consider the following test naming:

.. code-block:: python

    def test_add_motor_coordinates(
        calisto_motorless,
        cdm_position,
        grain_cm_position,
        nozzle_position,
        coord_direction,
        motor_position,
        expected_motor_cdm,
        expected_motor_cpp,
    ):

RocketPy Team agreed upon following the testing convention where it's name exactly follows one of those:

* ``test_methodname``
* ``test_methodname_stateundertest``
* ``test_methodname_expectedbehaviour``

However, in any case, it is of utmost importance that the expected behaviour or state to be tested
**should be included within the docstring of the test**, just as illustrated below by the docstring
of the same test:

    Test the method add_motor and related position properties in a Rocket
    instance.

    This test checks the correctness of the `add_motor` method and the computed
    `motor_center_of_dry_mass_position` and `center_of_propellant_position`
    properties in the `Rocket` class using various parameters related to the
    motor's position, nozzle's position, and other related coordinates.
    Different scenarios are tested using parameterization, checking scenarios
    moving from the nozzle to the combustion chamber and vice versa, and with
    various specific physical and geometrical characteristics of the motor.


Do not get caught by the size of that docstring. The only requirements it has to satisfy is
that the docstring contains precise information on the expected behaviour and/or behaviour
to be tested.

Testing Structure
-----------------

In order to keep the tests easily readable and maintainable, RocketPy encourages
the use of the AAA pattern (Arrange, Act, Assert) for structuring the tests.

* **Arrange:** Set up the necessary preconditions and inputs (often done by *Fixtures* as it will be described below);
* **Act:** Execute the code under test;
* **Assert:** Verify that the code under test behaves as expected.

The following example illustrates the AAA pattern:

.. code-block:: python

    @pytest.mark.parametrize(
        "latitude, longitude", [(-21.960641, -47.482122), (0, 0), (21.960641, 47.482122)]
    ) # Arrange: Done by the fixtures and the parameters of the test
    def test_location_set_location_saves_location(latitude, longitude, example_plain_env):
        """Tests location is saved correctly in the environment obj.

        Parameters
        ----------
        example_plain_env : rocketpy.Environment
        latitude: float
            The latitude in decimal degrees.
        longitude: float
            The longitude in decimal degrees.
        """
        # Act: Set the location
        example_plain_env.set_location(latitude, longitude)
        # Assert: Check if the location was saved correctly
        assert example_plain_env.latitude == latitude
        assert example_plain_env.longitude == longitude

This pattern is a general guideline, of course specific tests cases might
modify it to better fit the specific needs of the test.

.. note::

    Parameterization is a powerful feature of ``pytest.mark.parametrize`` that allows
    you to run the same test with different inputs. This is highly recommended when
    there multiple testing scenarios for the same method.

Directory Structure
-------------------

RocketPy organizes its tests as follows:

::

    tests/
    ├── acceptance/
    │   ├── acceptance_file_1.py
    │   └── acceptance_file_2.py
    ├── fixtures/
    │   ├── fixtures_file_1.py
    │   └── fixtures_file_2.py
    ├── integration/
    │   ├── integration_file_1.py
    │   └── integration_file_2.py
    └── unit/
        ├── unit_file_1.py
        ├── unit_file_2.py
        └── stochastic/
            ├── stochastic_file_1.py
            └── stochastic_file_2.py

As one might guess, each kind of test should be included within it's correspondent kind of test. For instance, if one is writing
an unit testing module called ``test_flight.py``, it should be included within the ``unit`` folder. The same holds for other tests.
For a more detailed treatment of the directory containing the fixtures, read the next section.

Fixtures
--------

Fixtures play a significant role within testing. In RocketPy it is no different. In fact, so many features are needed
to properly test the code that the RocketPy Team decided to organize them a little different then one might find in
small projects. The directory is structured as follows:

::

    tests/
    ├── fixtures/
    │   ├── acceptance/
    │   ├── airfoils/
    │   ├── environment/
    │   ├── flight/
    │   ├── function/
    │   ├── hybrid/
    │   ├── monte_carlo/
    │   ├── motor/
    │   ├── parachutes/
    │   ├── rockets/
    │   ├── surfaces/
    │   ├── units/
    │   └── utilities/

Rocketpy Team opted for this kind of structure since it allowed for a more convenient way of organizing
fixtures. Additionally, it serves the purpose of putting the tests in a position where only strictly needed
fixtures are imported.

**Important:** If a new module containing fixtures is to be created, do not forget to look for the
``conftest.py`` file within the tests folder to include your newly created module.

To finish, let's take a quick look inside the tests directory structure. Consider the **motor**
folder containing its fixtures:

.. code-block:: rst

    motor/
    ├── __init__.py
    ├── Cesaroni_M1670_shifted.eng
    ├── Cesaroni_M1670.eng
    ├── generic_motor_fixtures.py
    ├── hybrid_fixtures.py
    ├── liquid_fixtures.py
    ├── solid_motor_fixtures.py
    └── tanks_fixtures.py

Observe the naming convention (**RocketPy prefers Hungarian Notation**) for the fixtures within the modules and also how the fixtures were
structured, such that each kind of motor contains a module loaded with its needed fixtures.

Unit tests definition
---------------------

Within a complex code such as RocketPy, some definitions or agreements need to be reviewed or sophisticated
to make sense within a project. In RocketPy, unit tests are/can be **sociable**, which **still** means that:

* (Speed) They have to be **fast**.
* (Isolated behavior) They focus on a **small part** of the system. Here we define unit in the method-level.

*However*, as already said, they are/can be sociable:

* (Sociable) The tested unit relies on other units to fulfill its behavior.

The classification depends on whether the test isolates the unit under test from its dependencies or allows them
to interact naturally. In practical terms, consider the test:

.. code-block:: python

    def test_evaluate_total_mass(calisto_motorless):
        """Tests the evaluate_total_mass method of the Rocket class.
        Both with respect to return instances and expected behaviour.

        Parameters
        ----------
        calisto_motorless : Rocket instance
            A predefined instance of a Rocket without a motor, used as a base for testing.
        """
        assert isinstance(calisto_motorless.evaluate_total_mass(), Function)

This test is **sociable** because it relies on the actual Rocket instance and tests its real behavior without
isolating the Rocket class from its potential interactions with other classes or methods within its implementation.
It checks the real implementation of ``evaluate_total_mass`` rather than a mocked or stubbed version, ensuring that
the functionality being tested is part of the integrated system.

Please note that writing an unit test which is solitary is allowed, however: make sure to back it up with proper contract
tests when applicable.

The classification regarding solitary and sociable tests was clarified due to the specific needs developers
naturally encountered within the software, while also hoping that since the developers had the need to further
identify them, external contributors would probably fall into the same problem.

Integration tests definition
----------------------------

Integration tests verify that individual modules or components of a software system work together as expected.
Unlike unit tests that isolate specific units of code, integration tests contain an interesting feature:

* (Non-isolated behavior) Focus on interactions between different parts of the system, such as modules, services, databases, or external APIs.

Consider the following integration test:

.. code-block:: python

    @patch("matplotlib.pyplot.show")
    def test_wyoming_sounding_atmosphere(mock_show, example_plain_env):
        """Tests the Wyoming sounding model in the environment object.

        Parameters
        ----------
        mock_show : mock
            Mock object to replace matplotlib.pyplot.show() method.
        example_plain_env : rocketpy.Environment
            Example environment object to be tested.
        """
        # TODO:: this should be added to the set_atmospheric_model() method as a
        #        "file" option, instead of receiving the URL as a string.
        URL = "http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=2019&MONTH=02&FROM=0500&TO=0512&STNM=83779"
        # give it at least 5 times to try to download the file
        example_plain_env.set_atmospheric_model(type="wyoming_sounding", file=URL)

        assert example_plain_env.all_info() is None
        assert abs(example_plain_env.pressure(0) - 93600.0) < 1e-8
        assert (
            abs(example_plain_env.barometric_height(example_plain_env.pressure(0)) - 722.0)
            < 1e-8
        )
        assert abs(example_plain_env.wind_velocity_x(0) - -2.9005178894925043) < 1e-8
        assert abs(example_plain_env.temperature(100) - 291.75) < 1e-8

This test contains two fundamental traits which defines it as an integration test:

* (I/O Access) Communication with external dependencies that may not be stable or quick to access. Emphasis on I/O and functionality of public interfaces.
* Contains the ``all_info()`` method, which is an integration test by convention for RocketPy.

**Observation:** The ``all_info()`` method present in the code is considered to be an integration test.
The motivation behind lies in the fact that it interacts and calls too many methods, being too broad
to be considered an unit test.

Please be aware that Integration tests are not solely classified when interacting with external dependencies,
but also encompass verifying the interaction between classes or too many methods at once, such as ``all_info()``.

Further clarification: Even if the test contains traits of unit tests and use dependencies which are stable, such as
.csv or .eng files contained within the project or any other external dependencies which are easy to access
and do not make the test slow, **then your test is still an integration test, since those are strongly I/O related.**

Acceptance tests definition
---------------------------

Acceptance tests configure the final phase of the testing lifecycle within RocketPy. These tests are designed to
account for user-centered scenarios where usually real flights and configurations are setup and launched.

This phase of testing presents the task of letting the developers know if the system still satisfies well enough the
requirements of normal use of the software, including for instance:

* Error free use of the software within the setup of a real launch.
* Assertions regarding the accuracy of simulations. Thresholds are put and should be checked. RocketPy Paper results are a good reference.
* Usually include prior knowledge of real flight data.

In practical terms, acceptance tests come through the form of a notebook where a certain flight is tested.
It is an important feature and also defining feature of the acceptance tests that thresholds are compared
to real flight data allowing for true comparison.

Docstrings
----------

Some tests are also defined within the docstring of some methods. That has been done so far for example and
documenting purposes, such as below:

.. code-block:: python

    def to_frequency_domain(self, lower, upper, sampling_frequency, remove_dc=True):
        """Performs the conversion of the Function to the Frequency Domain and
        returns the result. This is done by taking the Fourier transform of the
        Function. The resulting frequency domain is symmetric, i.e., the
        negative frequencies are included as well.

        Parameters
        ----------
        lower : float
            Lower bound of the time range.
        upper : float
            Upper bound of the time range.
        sampling_frequency : float
            Sampling frequency at which to perform the Fourier transform.
        remove_dc : bool, optional
            If True, the DC component is removed from the Fourier transform.

        Returns
        -------
        Function
            The Function in the frequency domain.

        Examples
        --------
        >>> from rocketpy import Function
        >>> import numpy as np
        >>> main_frequency = 10 # Hz
        >>> time = np.linspace(0, 10, 1000)
        >>> signal = np.sin(2 * np.pi * main_frequency * time)
        >>> time_domain = Function(np.array([time, signal]).T)
        >>> frequency_domain = time_domain.to_frequency_domain(
        ...     lower=0, upper=10, sampling_frequency=100
        ... )
        >>> peak_frequencies_index = np.where(frequency_domain[:, 1] > 0.001)
        >>> peak_frequencies = frequency_domain[peak_frequencies_index, 0]
        >>> print(peak_frequencies)
        [[-10.  10.]]
        """

This is not common practice, but it is optional and can be done, specially to provide
an usage example for the function under testing. RocketPy however encourages the use
of other means to test its software, as described.
