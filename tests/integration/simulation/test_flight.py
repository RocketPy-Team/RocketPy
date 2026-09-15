from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest

from rocketpy import Flight
from rocketpy.simulation.helpers.dynamics import FULL_POST_PROCESS_VARS

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


def test_a_parachute_keeps_falling_freely_during_its_lag(
    example_plain_env, calisto_robust
):
    """The lag between a parachute firing and opening is flown, not skipped.

    The rocket must keep falling under the current equations of motion until
    the parachute phase begins: rows are stored during the lag, and the
    vertical speed when the parachute opens is what free fall gives.
    """
    lag = 1.5
    calisto_robust.parachutes.clear()
    calisto_robust.add_parachute(
        "Drogue", cd_s=1.0, trigger="apogee", sampling_rate=105, lag=lag
    )
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=60,
    )

    fired_at = flight.parachute_events[0][0]
    descent = next(p for p in flight.solution.phases if "Drogue" in p.name)
    assert descent.t_start == pytest.approx(fired_at + lag)
    # the lag is flown: rows are stored strictly inside it
    times = flight.time
    assert np.sum((times > fired_at) & (times < fired_at + lag)) > 3
    # and the rocket falls freely through it: vz drops by about g * lag
    g = example_plain_env.gravity(flight.apogee)
    vz_fired = flight.vz(fired_at)
    vz_opened = flight.vz(fired_at + lag)
    assert vz_opened == pytest.approx(vz_fired - g * lag, abs=0.5)


def test_solid_propulsion_equations_damp_rotation(example_plain_env, calisto_robust):
    """The solid-propulsion equations must see the angular rates.

    They once zeroed the rates before the aerodynamics, which removed all
    aerodynamic damping: a pitch rate produced no restoring moment and a
    canted fin spun the rocket up without bound. At a coasting state, a pitch
    rate must produce an angular acceleration that opposes it, and the roll
    rate of a canted-fin flight must match the generalized equations.
    """
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        equations_of_motion="solid_propulsion",
        terminate_on_apogee=True,
    )
    assert flight.u_dot_generalized is flight.u_dot

    # a coasting state a few seconds after burnout, with a pitch rate added
    t = flight.rocket.motor.burn_out_time + 3.0
    still = list(flight.solution.at(t).values())[:13]
    pitching = [*still[:10], 2.0, 0.0, 0.0]
    alpha_still = flight.u_dot(t, still)[10]
    alpha_pitching = flight.u_dot(t, pitching)[10]
    assert alpha_pitching != alpha_still
    assert alpha_pitching < alpha_still  # the moment opposes the rate

    # a rolling flight settles at the same roll rate with either set of equations
    calisto_robust.aerodynamic_surfaces.clear()  # rebuild with canted fins
    calisto_robust.add_nose(length=0.55829, kind="vonkarman", position=1.160)
    calisto_robust.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.313
    )
    calisto_robust.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
        cant_angle=1.5,
    )
    roll = {}
    for equations in ("solid_propulsion", "standard"):
        rolling = Flight(
            rocket=calisto_robust,
            environment=example_plain_env,
            rail_length=5.2,
            inclination=85,
            heading=0,
            equations_of_motion=equations,
            terminate_on_apogee=True,
        )
        roll[equations] = np.max(np.abs(rolling.w3[:, 1]))
    assert roll["solid_propulsion"] == pytest.approx(roll["standard"], rel=0.02)
    assert roll["standard"] < 100  # rad/s; unbounded spin-up would give thousands


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

    # The canted fins spin the rocket up and their damping balances it, so the
    # roll rate tracks the airspeed during the coast. Without damping it grows
    # past a thousand rad/s.
    times = test_flight.w3[:, 0]
    coast = (times > test_flight.rocket.motor.burn_out_time + 1) & (
        times < test_flight.apogee_time - 3
    )
    roll_rate = np.abs(test_flight.w3[coast, 1])
    assert 1.0 < roll_rate.max() < 100.0
    roll_per_speed = roll_rate / test_flight.speed(times[coast])
    assert roll_per_speed.std() < 0.15 * roll_per_speed.mean()


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


def test_the_rail_phase_reports_forces_but_no_angular_acceleration(
    example_plain_env, calisto_robust
):
    """On the rail the rocket cannot rotate, but it still feels thrust and drag.

    The rail phase works its post-process variables out from the free-flight
    equations and replaces only the accelerations, so this pins both halves of
    that: the angular accelerations are exactly zero, and the forces and thrust
    are the real ones.
    """
    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    rail = flight.solution.post.phase_values(0)
    column = FULL_POST_PROCESS_VARS.index

    assert flight.solution.phases[0].dynamics.name == "rail"
    start, stop = flight.solution.phase_span(0)
    assert len(rail) == stop - start > 1
    # the rail holds the rocket, so it cannot turn
    for name in ("alpha1", "alpha2", "alpha3"):
        assert np.all(rail[:, column(name)] == 0.0), name
    # but it burns, it accelerates upwards, and it feels drag
    assert np.max(rail[:, column("net_thrust")]) > 0
    assert np.max(rail[:, column("az")]) > 0
    assert np.min(rail[:, column("R3")]) < 0


def test_initial_solution_from_a_previous_flight(example_plain_env, calisto_robust):
    """A flight can start where another one ended.

    Passing a Flight rather than a list takes the previous flight's last state,
    canonicalized in case it ended in a phase following fewer than the thirteen
    states of the full flight state.
    """
    first = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    continued = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        # the rail is not used, since the state given is already off it, but
        # it still has to be a real length
        rail_length=5,
        initial_solution=first,
    )

    # the second flight picks up exactly where the first stopped
    assert continued.t_initial == pytest.approx(
        first.solution.raw_row(-1)[0], abs=1e-12
    )
    assert continued.solution[0] == pytest.approx(first.solution[-1], rel=1e-12)
    # and then goes on to descend under its parachutes and land
    assert continued.t_final > continued.t_initial
    assert continued.parachute_events
    assert continued.z(continued.t_final) < continued.z(continued.t_initial)
    # post-process variables are reported for the continued flight too
    assert np.isfinite(continued.az(continued.t_initial))


def test_initial_solution_still_on_the_rail(example_plain_env, calisto_robust):
    """A state inside the rail starts in the rail phase, not in free flight.

    Both other ways of giving an initial state begin out of the rail, so this
    is the one that exercises the rail dynamics being chosen.
    """
    reference = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    on_rail = next(
        row for row in reference.solution if 0 < row[0] < reference.out_of_rail_time
    )

    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5,
        initial_solution=list(on_rail),
        terminate_on_apogee=True,
    )

    assert flight.initial_dynamics is flight.udot_rail1
    assert flight.out_of_rail_time > flight.t_initial
    # starting partway up the same rail must reach the same apogee
    assert flight.apogee == pytest.approx(reference.apogee, rel=1e-2)


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
    assert len(controller.log) > 0

    # Verify observed variables contain expected data (time, deployment_level, mach_number)
    for observed in controller.log:
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


def test_backward_compatibility_5_parameter_controller(
    calisto_robust, example_plain_env
):
    """Test that old 5-parameter controllers still work (backward compatibility).

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """

    def controller_5_params(  # pylint: disable=unused-argument
        time, sampling_rate, state, observed_variables, air_brakes
    ):
        """Controller with the shortest supported positional signature."""
        altitude = state[2]
        vz = state[5]

        if time < 3.9:
            return None

        if altitude < 1500:
            air_brakes.deployment_level = 0
        else:
            air_brakes.deployment_level = min(0.5, max(0, vz / 100))
        return None

    # Add air brakes with old-style 5-parameter controller
    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_5_params,
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


def test_6_parameter_controller_with_sensors(calisto_robust, example_plain_env):
    """Test that 6-parameter controllers (with sensors, no environment) work correctly.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """

    # Define a 6-parameter controller
    def controller_6_params(  # pylint: disable=unused-argument
        time,
        sampling_rate,
        state,
        observed_variables,
        air_brakes,
        sensors,
    ):
        """Controller with 6 parameters (includes sensors, but not environment)."""
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

    # Add air brakes with 6-parameter controller
    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_6_params,
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


@pytest.mark.parametrize("param_count", [4, 8])
def test_invalid_controller_parameter_count(calisto_robust, param_count):
    """Test that positional controllers with an invalid parameter count raise
    ValueError. Only 6, 7 or 8 positional arguments are supported.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    param_count : int
        Number of positional parameters of the invalid controller function.
    """
    arg_names = [
        "time",
        "sampling_rate",
        "state",
        "observed_variables",
        "air_brakes",
        "sensors",
        "environment",
        "extra_param",
    ][:param_count]
    namespace = {}
    exec(  # pylint: disable=exec-used
        f"def invalid_controller({', '.join(arg_names)}): return None",
        namespace,
    )
    invalid_controller = namespace["invalid_controller"]

    calisto_robust.parachutes = []

    with pytest.raises(ValueError, match="must have 5, 6, or 7"):
        calisto_robust.add_air_brakes(
            drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
            controller_function=invalid_controller,
            sampling_rate=10,
            clamp=True,
        )


@pytest.mark.parametrize("param_count", [5, 6, 7])
def test_deprecated_positional_controller(calisto_robust, param_count):
    """Test that positional controllers with a valid parameter count (6, 7 or
    8) are accepted but emit a DeprecationWarning.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    param_count : int
        Number of positional parameters of the controller function.
    """
    arg_names = [
        "time",
        "sampling_rate",
        "state",
        "observed_variables",
        "air_brakes",
        "sensors",
        "environment",
    ][:param_count]
    namespace = {}
    exec(  # pylint: disable=exec-used
        f"def positional_controller({', '.join(arg_names)}): return None",
        namespace,
    )
    positional_controller = namespace["positional_controller"]

    calisto_robust.parachutes = []

    with pytest.warns(DeprecationWarning, match="positional arguments is"):
        calisto_robust.add_air_brakes(
            drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
            controller_function=positional_controller,
            sampling_rate=10,
            clamp=True,
        )


def test_context_controller_no_warning(calisto_robust, recwarn):
    """Test that a controller using the recommended ``controller(context)`` signature is
    accepted without raising or emitting a deprecation warning.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    recwarn : pytest.WarningsRecorder
        Fixture that records warnings raised during the test.
    """

    def context_controller(context):  # pylint: disable=unused-argument
        return None

    calisto_robust.parachutes = []

    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=context_controller,
        sampling_rate=10,
        clamp=True,
    )

    assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)


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


def test_air_brakes_context_controller(calisto_robust, example_plain_env):
    """Test that an air brakes controller using the recommended controller(context) API works.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        Calisto rocket without air brakes
    example_plain_env : rocketpy.Environment
        Environment object for the simulation
    """

    def controller_function(context):
        time = context["time"]
        sampling_rate = context["sampling_rate"]
        state = context["state"]
        previous_state = context["previous_state"]
        air_brakes = context["air_brakes"]
        environment = context["environment"]

        altitude_agl = context["height_agl"]
        altitude_asl = state[2]
        vx, vy, vz = state[3], state[4], state[5]

        wind_x = environment.wind_velocity_x(altitude_asl)
        wind_y = environment.wind_velocity_y(altitude_asl)
        free_stream_speed = ((wind_x - vx) ** 2 + (wind_y - vy) ** 2 + vz**2) ** 0.5
        mach_number = free_stream_speed / environment.speed_of_sound(altitude_asl)

        if time < 3.9:
            return None

        previous_vz = previous_state[5] if previous_state is not None else vz
        if altitude_agl < 1500:
            air_brakes.deployment_level = 0
        else:
            new_level = air_brakes.deployment_level + 0.1 * vz + 0.01 * previous_vz**2
            max_change = 0.2 / sampling_rate
            new_level = max(
                air_brakes.deployment_level - max_change,
                min(air_brakes.deployment_level + max_change, new_level),
            )
            air_brakes.deployment_level = new_level

        return (time, air_brakes.deployment_level, mach_number)

    calisto_robust.parachutes = []
    calisto_robust.add_air_brakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        clamp=True,
    )

    flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    assert flight.t_final > 0
    assert flight.apogee > 0
