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
def test_time_overshoot(mock_show, calisto_robust, example_spaceport_env):  # pylint: disable=unused-argument
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
