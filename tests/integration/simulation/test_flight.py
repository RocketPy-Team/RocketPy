from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Flight

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.mark.parametrize(
    "flight_fixture", ["flight_calisto_robust", "flight_calisto_robust_solid_eom"]
)
@patch("matplotlib.pyplot.show")
# pylint: disable=unused-argument
def test_all_info(mock_show, request, flight_fixture):
    """Test that the flight class is working as intended. This basically calls
    the all_info() method and checks if it returns None. It is not testing if
    the values are correct, but whether the method is working without errors.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    request : _pytest.fixtures.FixtureRequest
        Request object to access the fixture dynamically.
    flight_fixture : str
        Name of the flight fixture to be tested.
    """
    flight = request.getfixturevalue(flight_fixture)
    assert flight.all_info() is None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("solver_method", ["RK45", "DOP853", "Radau", "BDF"])
# RK23 is unstable and requires a very low tolerance to work
# pylint: disable=unused-argument
def test_all_info_different_solvers(
    mock_show, calisto_robust, example_spaceport_env, solver_method
):
    """Test that the flight class is working as intended with different solver
    methods. This basically calls the all_info() method and checks if it returns
    None. It is not testing if the values are correct, but whether the method is
    working without errors.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    calisto_robust : rocketpy.Rocket
        Rocket to be simulated. See the conftest.py file for more info.
    example_spaceport_env : rocketpy.Environment
        Environment to be simulated. See the conftest.py file for more info.
    solver_method : str
        The solver method to be used in the simulation.
    """
    test_flight = Flight(
        environment=example_spaceport_env,
        rocket=calisto_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
        ode_solver=solver_method,
    )
    assert test_flight.all_info() is None


@patch("matplotlib.pyplot.show")
def test_hybrid_motor_flight(mock_show, flight_calisto_hybrid_modded):  # pylint: disable=unused-argument
    """Test the flight of a rocket with a hybrid motor. This test only validates
    that a flight simulation can be performed with a hybrid motor; it does not
    validate the results.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    flight_calisto_hybrid_modded : rocketpy.Flight
        Sample Flight to be tested. See the conftest.py file for more info.
    """
    assert flight_calisto_hybrid_modded.all_info() is None


@patch("matplotlib.pyplot.show")
def test_liquid_motor_flight(mock_show, flight_calisto_liquid_modded):  # pylint: disable=unused-argument
    """Test the flight of a rocket with a liquid motor. This test only validates
    that a flight simulation can be performed with a liquid motor; it does not
    validate the results.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    flight_calisto_liquid_modded : rocketpy.Flight
        Sample Flight to be tested. See the conftest.py file for more info.
    """
    assert flight_calisto_liquid_modded.all_info() is None


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_time_overshoot_false(mock_show, calisto_robust, example_spaceport_env):  # pylint: disable=unused-argument
    """Test the time_overshoot parameter of the Flight class. This basically
    calls the all_info() method for a simulation without time_overshoot and
    checks if it returns None. It is not testing if the values are correct,
    just if the flight simulation is not breaking.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        The rocket to be simulated. In this case, the fixture rocket is used.
        See the conftest.py file for more information.
    example_spaceport_env : rocketpy.Environment
        The environment to be simulated. In this case, the fixture environment
        is used. See the conftest.py file for more information.
    """

    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
    )

    assert test_flight.all_info() is None


@patch("matplotlib.pyplot.show")
def test_simpler_parachute_triggers(mock_show, example_plain_env, calisto_robust):  # pylint: disable=unused-argument
    """Tests different types of parachute triggers. This is important to ensure
    the code is working as intended, since the parachute triggers can have very
    different format definitions. It will add 3 parachutes using different
    triggers format and check if the parachute events are being at the correct
    altitude

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    example_plain_env : rocketpy.Environment
        Environment to be simulated. See the conftest.py file for more info.
    calisto_robust : rocketpy.Rocket
        Rocket to be simulated. See the conftest.py file for more info.
    """
    calisto_robust.parachutes = []

    _ = calisto_robust.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=400,
        sampling_rate=105,
        lag=0,
    )

    _ = calisto_robust.add_parachute(
        "Drogue2",
        cd_s=5.5,
        trigger=lambda pressure, height, state: height < 800 and state[5] < 0,
        sampling_rate=105,
        lag=0,
    )

    _ = calisto_robust.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger="apogee",
        sampling_rate=105,
        lag=0,
    )

    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5,
        inclination=85,
        heading=0,
    )

    assert (
        abs(test_flight.z(test_flight.parachute_events[0][0]) - test_flight.apogee) <= 1
    )
    assert (
        abs(
            test_flight.z(test_flight.parachute_events[1][0])
            - (800 + example_plain_env.elevation)
        )
        <= 1
    )
    assert (
        abs(
            test_flight.z(test_flight.parachute_events[2][0])
            - (400 + example_plain_env.elevation)
        )
        <= 1
    )
    assert calisto_robust.all_info() is None
    assert test_flight.all_info() is None


@patch("matplotlib.pyplot.show")
def test_rolling_flight(  # pylint: disable=unused-argument
    mock_show,
    example_plain_env,
    cesaroni_m1670,
    calisto,
    calisto_nose_cone,
    calisto_tail,
    calisto_main_chute,
    calisto_drogue_chute,
):
    test_rocket = calisto

    test_rocket.set_rail_buttons(0.082, -0.618)
    test_rocket.add_motor(cesaroni_m1670, position=-1.373)
    test_rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956,
        cant_angle=0.5,
    )
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)

    test_flight = Flight(
        rocket=test_rocket,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert test_flight.all_info() is None


@patch("matplotlib.pyplot.show")
def test_eccentricity_on_flight(  # pylint: disable=unused-argument
    mock_show,
    example_plain_env,
    cesaroni_m1670,
    calisto,
    calisto_nose_cone,
    calisto_trapezoidal_fins,
    calisto_tail,
):
    test_rocket = calisto

    test_rocket.set_rail_buttons(0.082, -0.618)
    test_rocket.add_motor(cesaroni_m1670, position=-1.373)
    calisto.add_surfaces(calisto_trapezoidal_fins, -1.04956)
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)
    calisto.add_cm_eccentricity(x=-0.01, y=-0.01)

    test_flight = Flight(
        rocket=test_rocket,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    assert test_flight.all_info() is None


@patch("matplotlib.pyplot.show")
def test_air_brakes_flight(mock_show, flight_calisto_air_brakes):  # pylint: disable=unused-argument
    """Test the flight of a rocket with air brakes. This test only validates
    that a flight simulation can be performed with air brakes; it does not
    validate the results.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    flight_calisto_air_brakes_clamp_on : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test_flight = flight_calisto_air_brakes
    air_brakes = test_flight.rocket.air_brakes[0]

    assert air_brakes.plots.all() is None
    assert air_brakes.prints.all() is None


@patch("matplotlib.pyplot.show")
def test_air_brakes_flight_with_overshoot(
    mock_show, flight_calisto_air_brakes_time_overshoot
):  # pylint: disable=unused-argument
    """
    Same as test_air_brakes_flight but with time_overshoot=True.
    """
    test_flight = flight_calisto_air_brakes_time_overshoot
    air_brakes = test_flight.rocket.air_brakes[0]
    assert air_brakes.plots.all() is None
    assert air_brakes.prints.all() is None


@patch("matplotlib.pyplot.show")
def test_initial_solution(mock_show, example_plain_env, calisto_robust):  # pylint: disable=unused-argument
    """Tests the initial_solution option of the Flight class. This test simply
    simulates the flight using the initial_solution option and checks if the
    all_info method returns None.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    example_plain_env : rocketpy.Environment
        Environment to be simulated. See the conftest.py file for more info.
    calisto_robust : rocketpy.Rocket
        Rocket to be simulated. See the conftest.py file for more info.
    """
    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5,
        inclination=85,
        heading=0,
        rtol=1e-8,
        atol=1e-6,
        verbose=True,
        initial_solution=[
            0.0,
            0.0,
            0.0,
            1.5e3,
            10,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    assert test_flight.all_info() is None


@patch("matplotlib.pyplot.show")
def test_empty_motor_flight(mock_show, example_plain_env, calisto_motorless):  # pylint: disable=unused-argument
    flight = Flight(
        rocket=calisto_motorless,
        environment=example_plain_env,
        rail_length=5,
        initial_solution=[  # a random flight starting at apogee
            22.945995194368354,
            277.80976806186936,
            353.29457980509113,
            3856.1112773441596,
            12.737953434495966,
            15.524649322067267,
            -0.00011874766384947776,
            -0.06086838708814366,
            0.019695167217632138,
            -0.35099420532705555,
            -0.9338841410225396,
            0.25418555574446716,
            0.03632002739509155,
            2.0747266017020563,
        ],
    )
    assert flight.all_info() is None


def test_freestream_speed_at_apogee(example_plain_env, calisto):
    """
    Asserts that a rocket at apogee has a free stream speed of near 0.0 m/s
    in all directions given that the environment doesn't have any wind. Any
    speed values comes from coriolis effect.
    """
    # NOTE: this rocket doesn't move in x or z direction. There's no wind.
    hard_atol = 1e-12
    soft_atol = 1e-6
    test_flight = Flight(
        environment=example_plain_env,
        rocket=calisto,
        rail_length=5.2,
        inclination=90,
        heading=0,
        terminate_on_apogee=False,
        atol=13 * [hard_atol],
    )

    assert np.isclose(
        test_flight.stream_velocity_x(test_flight.apogee_time),
        0.4641492104717301,
        atol=hard_atol,
    )
    assert np.isclose(
        test_flight.stream_velocity_y(test_flight.apogee_time), 0.0, atol=hard_atol
    )
    # NOTE: stream_velocity_z has a higher error due to apogee detection estimation
    assert np.isclose(
        test_flight.stream_velocity_z(test_flight.apogee_time), 0.0, atol=soft_atol
    )
    assert np.isclose(
        test_flight.free_stream_speed(test_flight.apogee_time),
        0.4641492104717798,
        atol=hard_atol,
    )
    assert np.isclose(
        test_flight.apogee_freestream_speed, 0.4641492104717798, atol=hard_atol
    )


def test_rocket_csys_equivalence(
    flight_calisto_robust, flight_calisto_nose_to_tail_robust
):
    """Test the equivalence of the rocket coordinate systems between two
    different flight simulations.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info.
    flight_calisto_nose_to_tail_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info.
    """
    assert np.isclose(
        flight_calisto_robust.apogee, flight_calisto_nose_to_tail_robust.apogee
    )
    assert np.isclose(
        flight_calisto_robust.apogee_time,
        flight_calisto_nose_to_tail_robust.apogee_time,
    )
    assert np.isclose(
        flight_calisto_robust.x_impact,
        flight_calisto_nose_to_tail_robust.x_impact,
        atol=1e-3,
    )
    assert np.isclose(
        flight_calisto_robust.y_impact,
        flight_calisto_nose_to_tail_robust.y_impact,
    )
    assert np.allclose(
        flight_calisto_robust.initial_solution,
        flight_calisto_nose_to_tail_robust.initial_solution,
    )


def test_air_brakes_with_environment_parameter(
    calisto_robust, controller_function_with_environment, example_plain_env
):
    """Test that air brakes controller can access environment parameter during flight.

    This test verifies that:
    - The 8-parameter controller signature works correctly
    - Environment data is accessible within the controller
    - The flight simulation completes successfully
    - Controller observed variables are properly stored

    This addresses issue #853 where environment had to be accessed via global variables.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    controller_function_with_environment : function
        Controller function using the new 8-parameter signature
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """
    # Add air brakes with 8-parameter controller
    calisto_robust.parachutes = []  # Remove parachutes for cleaner test
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function_with_environment,
        sampling_rate=10,
        clamp=True,
    )

    # Run flight simulation
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    # Verify flight completed successfully
    assert flight.t_final > 0
    assert flight.apogee > 0

    # Verify controller was called and observed variables were stored
    # Controller is attached to the rocket, not the air brakes object
    controllers = [c for c in calisto_robust._controllers if "AirBrakes" in c.name]
    assert len(controllers) > 0
    controller = controllers[0]
    assert len(controller.observed_variables) > 0

    # Verify observed variables contain expected data (time, deployment_level, mach_number)
    for observed in controller.observed_variables:
        if observed is not None:
            assert len(observed) == 3
            time, deployment_level, mach_number = observed
            assert time >= 0
            assert 0 <= deployment_level <= 1  # Should be clamped
            assert mach_number >= 0


def test_air_brakes_serialization_with_environment(
    calisto_robust, controller_function_with_environment, example_plain_env
):
    """Test that rockets with air brakes using environment parameter can be serialized.

    This test specifically addresses issue #853 - serialization of rockets with
    air brakes that use controllers should work without relying on global variables.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    controller_function_with_environment : function
        Controller function using the new 8-parameter signature
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """
    # Add air brakes with 8-parameter controller
    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function_with_environment,
        sampling_rate=10,
        clamp=True,
    )

    # Serialize the rocket
    rocket_dict = calisto_robust.to_dict()

    # Verify serialization succeeded and contains air brakes data
    assert "air_brakes" in rocket_dict
    assert len(rocket_dict["air_brakes"]) > 0

    # Run a flight with the original rocket
    flight_original = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    # Verify flight completed
    assert flight_original.t_final > 0
    assert flight_original.apogee > 0


def test_backward_compatibility_6_parameter_controller(
    calisto_robust, controller_function, example_plain_env
):
    """Test that old 6-parameter controllers still work (backward compatibility).

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    controller_function : function
        Controller function using the old 6-parameter signature
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """
    # Add air brakes with old-style 6-parameter controller
    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        clamp=True,
    )

    # Run flight simulation
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    # Verify flight completed successfully
    assert flight.t_final > 0
    assert flight.apogee > 0

    # Verify controller exists
    controllers = [c for c in calisto_robust._controllers if "AirBrakes" in c.name]
    assert len(controllers) > 0


def test_7_parameter_controller_with_sensors(calisto_robust, example_plain_env):
    """Test that 7-parameter controllers (with sensors, no environment) work correctly.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """

    # Define a 7-parameter controller
    def controller_7_params(  # pylint: disable=unused-argument
        time,
        sampling_rate,
        state,
        state_history,
        observed_variables,
        air_brakes,
        sensors,
    ):
        """Controller with 7 parameters (includes sensors, but not environment)."""
        altitude = state[2]
        vz = state[5]

        if time < 3.9:
            return None

        if altitude < 1500:
            air_brakes.deployment_level = 0
        else:
            # Simple proportional control
            air_brakes.deployment_level = min(0.5, max(0, vz / 100))

        return (time, air_brakes.deployment_level)

    # Add air brakes with 7-parameter controller
    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_7_params,
        sampling_rate=10,
        clamp=True,
    )

    # Run flight simulation
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    # Verify flight completed successfully
    assert flight.t_final > 0
    assert flight.apogee > 0


def test_invalid_controller_parameter_count(calisto_robust):
    """Test that controllers with invalid parameter counts raise ValueError.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    """

    # Define controller with wrong number of parameters (5)
    def invalid_controller_5_params(  # pylint: disable=unused-argument
        time, sampling_rate, state, state_history, observed_variables
    ):
        """Invalid controller with only 5 parameters."""
        return None

    # Define controller with wrong number of parameters (9)
    def invalid_controller_9_params(  # pylint: disable=unused-argument
        time,
        sampling_rate,
        state,
        state_history,
        observed_variables,
        air_brakes,
        sensors,
        environment,
        extra_param,
    ):
        """Invalid controller with 9 parameters."""
        return None

    calisto_robust.parachutes = []

    # Test that 5-parameter controller raises ValueError
    with pytest.raises(ValueError, match="must have 6, 7, or 8 arguments"):
        calisto_robust.add_air_brakes(
            drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
            controller_function=invalid_controller_5_params,
            sampling_rate=10,
            clamp=True,
        )

    # Test that 9-parameter controller raises ValueError
    with pytest.raises(ValueError, match="must have 6, 7, or 8 arguments"):
        calisto_robust.add_air_brakes(
            drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
            controller_function=invalid_controller_9_params,
            sampling_rate=10,
            clamp=True,
        )


def make_controller_test_environment_access(methods_called):
    def _call_env_methods(environment, altitude_asl):
        _ = environment.elevation
        methods_called["elevation"] = True
        _ = environment.wind_velocity_x(altitude_asl)
        methods_called["wind_velocity_x"] = True
        _ = environment.wind_velocity_y(altitude_asl)
        methods_called["wind_velocity_y"] = True
        _ = environment.speed_of_sound(altitude_asl)
        methods_called["speed_of_sound"] = True
        _ = environment.pressure(altitude_asl)
        methods_called["pressure"] = True
        _ = environment.temperature(altitude_asl)
        methods_called["temperature"] = True

    def controller(  # pylint: disable=unused-argument
        time,
        sampling_rate,
        state,
        state_history,
        observed_variables,
        air_brakes,
        sensors,
        environment,
    ):
        """Controller that tests access to various environment methods."""
        altitude_asl = state[2]

        if time < 3.9:
            return None

        try:
            _call_env_methods(environment, altitude_asl)
            air_brakes.deployment_level = 0.3
        except AttributeError as e:
            raise AssertionError(f"Environment method not accessible: {e}") from e

        return (time, air_brakes.deployment_level)

    return controller


def test_environment_methods_accessible_in_controller(
    calisto_robust, example_plain_env
):
    """Test that all environment methods are accessible within the controller.

    This test verifies that the environment object passed to the controller
    provides access to all necessary atmospheric and environmental data.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """
    # Track which environment methods were successfully called
    methods_called = {
        "elevation": False,
        "wind_velocity_x": False,
        "wind_velocity_y": False,
        "speed_of_sound": False,
        "pressure": False,
        "temperature": False,
    }

    controller = make_controller_test_environment_access(methods_called)

    # Add air brakes with environment-testing controller
    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller,
        sampling_rate=10,
        clamp=True,
    )

    # Run flight simulation
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    # Verify flight completed
    assert flight.t_final > 0
    assert flight.all_info() is None

    # Verify all environment methods were successfully called
    assert all(methods_called.values()), f"Not all methods called: {methods_called}"
