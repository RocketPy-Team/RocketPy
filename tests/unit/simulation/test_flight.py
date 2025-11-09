import json
import os
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest
from scipy import optimize

from rocketpy import Components, Flight, Function, Rocket

plt.rcParams.update({"figure.max_open_warning": 0})

# Helper functions


def setup_rocket_with_given_static_margin(rocket, static_margin):
    """Takes any rocket, removes its aerodynamic surfaces and adds a set of
    nose, fins and tail specially designed to have a given static margin.
    The rocket is modified in place.

    Parameters
    ----------
    rocket : Rocket
        Rocket to be modified
    static_margin : float
        Static margin that the given rocket shall have

    Returns
    -------
    rocket : Rocket
        Rocket with the given static margin.
    """

    def compute_static_margin_error_given_distance(position, static_margin, rocket):
        """Computes the error between the static margin of a rocket and a given
        static margin. This function is used by the scipy.optimize.root_scalar
        function to find the position of the aerodynamic surfaces that will
        result in the given static margin.

        Parameters
        ----------
        position : float
            Position of the trapezoidal fins
        static_margin : float
            Static margin that the given rocket shall have
        rocket : rocketpy.Rocket
            Rocket to be modified. Only the trapezoidal fins will be modified.

        Returns
        -------
        error : float
            Error between the static margin of the rocket and the given static
            margin.
        """
        rocket.aerodynamic_surfaces = Components()
        rocket.add_nose(length=0.5, kind="vonKarman", position=1.0 + 0.5)
        rocket.add_trapezoidal_fins(
            4,
            span=0.100,
            root_chord=0.100,
            tip_chord=0.100,
            position=position,
        )
        rocket.add_tail(
            top_radius=0.0635,
            bottom_radius=0.0435,
            length=0.060,
            position=-1.194656,
        )
        return rocket.static_margin(0) - static_margin

    _ = optimize.root_scalar(
        compute_static_margin_error_given_distance,
        bracket=[-2.0, 2.0],
        method="brentq",
        args=(static_margin, rocket),
    )

    return rocket


# Tests


def test_get_solution_at_time(flight_calisto):
    """Test the get_solution_at_time method of the Flight class. This test
    simply calls the method at the initial and final time and checks if the
    returned values are correct. Also, checking for valid return instance.

    Parameters
    ----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    assert isinstance(flight_calisto.get_solution_at_time(0), np.ndarray)
    assert isinstance(
        flight_calisto.get_solution_at_time(flight_calisto.t_final), np.ndarray
    )

    assert np.allclose(
        flight_calisto.get_solution_at_time(0),
        np.array([0, 0, 0, 0, 0, 0, 0, 0.99904822, -0.04361939, 0, 0, 0, 0, 0]),
        rtol=1e-05,
        atol=1e-08,
    )
    assert np.allclose(
        flight_calisto.get_solution_at_time(flight_calisto.t_final),
        np.array(
            [
                48.43719482805657,
                -14.836008075478597,
                985.9858934483618,
                -3.4415459237894554e-05,
                0.0007572309307800201,
                11.21695000766671,
                -341.1460775169661,
                0.9990482215818578,
                -0.043619387365336,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        rtol=1e-02,
        atol=5e-03,
    )


def test_get_controller_observed_variables(flight_calisto_air_brakes):
    """Tests whether the method Flight.get_controller_observed_variables is
    working as intended."""
    obs_vars = flight_calisto_air_brakes.get_controller_observed_variables()
    assert isinstance(obs_vars, list)
    assert len(obs_vars) == 0


def test_initial_stability_margin(flight_calisto_custom_wind):
    """Test the initial_stability_margin method of the Flight class.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
    """
    res = flight_calisto_custom_wind.initial_stability_margin
    assert isinstance(res, float)
    assert res == flight_calisto_custom_wind.stability_margin(0)
    assert np.isclose(res, 2.05, atol=0.1)


def test_out_of_rail_stability_margin(flight_calisto_custom_wind):
    """Test the out_of_rail_stability_margin method of the Flight class.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
    """
    res = flight_calisto_custom_wind.out_of_rail_stability_margin
    assert isinstance(res, float)
    assert res == flight_calisto_custom_wind.stability_margin(
        flight_calisto_custom_wind.out_of_rail_time
    )
    assert np.isclose(res, 2.14, atol=0.1)


def test_export_sensor_data(flight_calisto_with_sensors):
    """Test the export of sensor data.

    Parameters
    ----------
    flight_calisto_with_sensors : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal accelerometer and a gyroscope.
    """
    flight_calisto_with_sensors.export_sensor_data("test_sensor_data.json")
    # read the json and parse as dict
    filename = "test_sensor_data.json"
    with open(filename, "r") as f:
        data = f.read()
        sensor_data = json.loads(data)
    # convert list of tuples into list of lists to compare with the json
    flight_calisto_with_sensors.sensors[0].measured_data[0] = [
        list(measurement)
        for measurement in flight_calisto_with_sensors.sensors[0].measured_data[0]
    ]
    flight_calisto_with_sensors.sensors[1].measured_data[1] = [
        list(measurement)
        for measurement in flight_calisto_with_sensors.sensors[1].measured_data[1]
    ]
    flight_calisto_with_sensors.sensors[2].measured_data = [
        list(measurement)
        for measurement in flight_calisto_with_sensors.sensors[2].measured_data
    ]
    assert (
        sensor_data["Accelerometer"][0]
        == flight_calisto_with_sensors.sensors[0].measured_data[0]
    )
    assert (
        sensor_data["Accelerometer"][1]
        == flight_calisto_with_sensors.sensors[1].measured_data[1]
    )
    assert (
        sensor_data["Gyroscope"] == flight_calisto_with_sensors.sensors[2].measured_data
    )
    os.remove(filename)


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (0.25886, -0.649623, 0)),
        ("out_of_rail_time", (0.792028, -1.987634, 0)),
        ("apogee_time", (-0.509420, -0.732933, -2.089120e-14)),
        ("t_final", (0, 0, 0)),
    ],
)
def test_aerodynamic_moments(flight_calisto_custom_wind, flight_time, expected_values):
    """Tests if the aerodynamic moments in some particular points of the
    trajectory is correct. The expected values were NOT calculated by hand, it
    was just copied from the test results. The results are not expected to
    change, unless the code is changed for bug fixes or accuracy improvements.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info.
    flight_time : str
        The name of the attribute of the flight object that contains the time
        of the point to be tested.
    expected_values : tuple
        The expected values of the aerodynamic moments vector at the point to
        be tested.
    """
    expected_attr, expected_moment = flight_time, expected_values

    test = flight_calisto_custom_wind
    t = getattr(test, expected_attr)
    atol = 5e-3

    assert pytest.approx(expected_moment, abs=atol) == (
        test.M1(t),
        test.M2(t),
        test.M3(t),
    ), f"Assertion error for moment vector at {expected_attr}."


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (1.654150, 0.659142, -0.067103)),
        ("out_of_rail_time", (5.052628, 2.013361, -1.75370)),
        ("apogee_time", (2.321838, -1.613641, -0.962108)),
        ("t_final", (-0.019802, 0.012030, 159.051604)),
    ],
)
def test_aerodynamic_forces(flight_calisto_custom_wind, flight_time, expected_values):
    """Tests if the aerodynamic forces in some particular points of the
    trajectory is correct. The expected values were NOT calculated by hand, it
    was just copied from the test results. The results are not expected to
    change, unless the code is changed for bug fixes or accuracy improvements.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info.
    flight_time : str
        The name of the attribute of the flight object that contains the time
        of the point to be tested.
    expected_values : tuple
        The expected values of the aerodynamic forces vector at the point to be
        tested.
    """
    expected_attr, expected_forces = flight_time, expected_values

    test = flight_calisto_custom_wind
    t = getattr(test, expected_attr)
    atol = 5e-3

    assert pytest.approx(expected_forces, abs=atol) == (
        test.R1(t),
        test.R2(t),
        test.R3(t),
    ), f"Assertion error for aerodynamic forces vector at {expected_attr}."


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (0, 0, 0)),
        ("out_of_rail_time", (0, 2.248540, 25.700928)),
        (
            "apogee_time",
            (-14.826350, 15.670022, -0.000264),
        ),
        ("t_final", (5, 2, -5.660155)),
    ],
)
def test_velocities(flight_calisto_custom_wind, flight_time, expected_values):
    """Tests if the velocity in some particular points of the trajectory is
    correct. The expected values were NOT calculated by hand, it was just
    copied from the test results. The results are not expected to change,
    unless the code is changed for bug fixes or accuracy improvements.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info.
    flight_time : str
        The name of the attribute of the flight object that contains the time
        of the point to be tested.
    expected_values : tuple
        The expected values of the velocity vector at the point to be tested.
    """
    expected_attr, expected_vel = flight_time, expected_values

    test = flight_calisto_custom_wind
    t = getattr(test, expected_attr)
    atol = 5e-3

    assert pytest.approx(expected_vel, abs=atol) == (
        test.vx(t),
        test.vy(t),
        test.vz(t),
    ), f"Assertion error for velocity vector at {expected_attr}."


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (0, 0, 0)),
        ("out_of_rail_time", (0, 7.8067, 89.2315)),
        ("apogee_time", (0.07649, -0.053530, -9.620037)),
        ("t_final", (0, 0, 0.0019548)),
    ],
)
def test_accelerations(flight_calisto_custom_wind, flight_time, expected_values):
    """Tests if the acceleration in some particular points of the trajectory is
    correct. The expected values were NOT calculated by hand, it was just
    copied from the test results. The results are not expected to change,
    unless the code is changed for bug fixes or accuracy improvements.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info.
    flight_time : str
        The name of the attribute of the flight object that contains the time
        of the point to be tested.
    expected_values : tuple
        The expected values of the acceleration vector at the point to be
        tested.
    """
    expected_attr, expected_acc = flight_time, expected_values

    test = flight_calisto_custom_wind
    t = getattr(test, expected_attr)
    atol = 5e-3

    assert pytest.approx(expected_acc, abs=atol) == (
        test.ax(t),
        test.ay(t),
        test.az(t),
    ), f"Assertion error for acceleration vector at {expected_attr}."


def test_rail_buttons_forces(flight_calisto_custom_wind):
    """Test the rail buttons forces. This tests if the rail buttons forces are
    close to the expected values. However, the expected values were NOT
    calculated by hand, it was just copied from the test results. The results
    are not expected to change, unless the code is changed for bug fixes or
    accuracy improvements.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test = flight_calisto_custom_wind
    atol = 5e-3
    assert pytest.approx(1.795539, abs=atol) == test.max_rail_button1_normal_force
    assert pytest.approx(0.715483, abs=atol) == test.max_rail_button1_shear_force
    assert pytest.approx(3.257089, abs=atol) == test.max_rail_button2_normal_force
    assert pytest.approx(1.297878, abs=atol) == test.max_rail_button2_shear_force


def test_max_values(flight_calisto_robust):
    """Test the max values of the flight. This tests if the max values are
    close to the expected values. However, the expected values were NOT
    calculated by hand, it was just copied from the test results. This is
    because the expected values are not easy to calculate by hand, and the
    results are not expected to change. If the results change, the test will
    fail, and the expected values must be updated. If the values are updated,
    always double check if the results are really correct. Acceptable reasons
    for changes in the results are: 1) changes in the code that improve the
    accuracy of the simulation, 2) a bug was found and fixed. Keep in mind that
    other tests may be more accurate than this one, for example, the acceptance
    tests, which are based on the results of real flights.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test = flight_calisto_robust
    rtol = 5e-3
    assert pytest.approx(105.1599, rel=rtol) == test.max_acceleration_power_on
    assert pytest.approx(105.1599, rel=rtol) == test.max_acceleration
    assert pytest.approx(0.85999, rel=rtol) == test.max_mach_number
    assert pytest.approx(285.94948, rel=rtol) == test.max_speed


def test_effective_rail_length(flight_calisto_robust, flight_calisto_nose_to_tail):
    """Tests the effective rail length of the flight simulation. The expected
    values are calculated by hand, and should be valid as long as the rail
    length and the position of the buttons and nozzle do not change in the
    fixtures. If the fixtures change, this test must be updated. It is important
    to keep

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    flight_calisto_nose_to_tail : rocketpy.Flight
        Flight object to be tested. The difference here is that the rocket is
        defined with the "nose_to_tail" orientation instead of the
        "tail_to_nose" orientation. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test1 = flight_calisto_robust
    test2 = flight_calisto_nose_to_tail
    atol = 1e-8

    rail_length = 5.2
    upper_button_position = 0.082
    lower_button_position = -0.618
    nozzle_position = -1.373

    effective_1rl = rail_length - abs(upper_button_position - nozzle_position)
    effective_2rl = rail_length - abs(lower_button_position - nozzle_position)

    # test 1: Rocket with "tail_to_nose" orientation
    assert pytest.approx(test1.effective_1rl, abs=atol) == effective_1rl
    assert pytest.approx(test1.effective_2rl, abs=atol) == effective_2rl
    # test 2: Rocket with "nose_to_tail" orientation
    assert pytest.approx(test2.effective_1rl, abs=atol) == effective_1rl
    assert pytest.approx(test2.effective_2rl, abs=atol) == effective_2rl


def test_surface_wind(flight_calisto_custom_wind):
    """Tests the surface wind of the flight simulation. The expected values
    are provided by the definition of the 'light_calisto_custom_wind' fixture.
    If the fixture changes, this test must be updated.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test = flight_calisto_custom_wind
    atol = 1e-8
    assert pytest.approx(2.0, abs=atol) == test.frontal_surface_wind
    assert pytest.approx(-5.0, abs=atol) == test.lateral_surface_wind


@pytest.mark.skip(reason="legacy tests")
@pytest.mark.parametrize(
    "rail_length, out_of_rail_time",
    [
        (0.52, 0.5180212542878443),
        (5.2, 5.180378138072207),
        (50.2, 50.00897551720473),
        (100000, 100003.35594050681),
    ],
)
def test_rail_length(calisto_robust, example_plain_env, rail_length, out_of_rail_time):
    """Tests the rail length parameter of the Flight class. This test simply
    simulate the flight using different rail lengths and checks if the expected
    out of rail altitude is achieved. Four different rail lengths are
    tested: 0.001, 1, 10, and 100000 meters. This provides a good test range.
    Currently, if a rail length of 0 is used, the simulation will fail in a
    ZeroDivisionError, which is not being tested here.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        The rocket to be simulated. In this case, the fixture rocket is used.
        See the conftest.py file for more information.
    example_plain_env : rocketpy.Environment
        The environment to be simulated. In this case, the fixture environment
        is used. See the conftest.py file for more information.
    rail_length : float, int
        The length of the rail in meters. It must be a positive number. See the
        Flight class documentation for more information.
    out_of_rail_time : float, int
        The expected time at which the rocket leaves the rail in seconds.
    """
    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=rail_length,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    assert abs(test_flight.z(test_flight.out_of_rail_time) - out_of_rail_time) < 1e-6


@patch("matplotlib.pyplot.show")
def test_lat_lon_conversion_robust(mock_show, example_spaceport_env, calisto_robust):  # pylint: disable=unused-argument
    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        inclination=85,
        heading=45,
    )

    # Check for initial and final lat/lon coordinates based on launch pad coordinates
    assert abs(test_flight.latitude(0)) - abs(test_flight.env.latitude) < 1e-6
    assert abs(test_flight.longitude(0)) - abs(test_flight.env.longitude) < 1e-6
    assert test_flight.latitude(test_flight.t_final) > test_flight.env.latitude
    assert test_flight.longitude(test_flight.t_final) > test_flight.env.longitude


@patch("matplotlib.pyplot.show")
def test_lat_lon_conversion_from_origin(mock_show, example_plain_env, calisto_robust):  # pylint: disable=unused-argument
    "additional tests to capture incorrect behaviors during lat/lon conversions"

    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert abs(test_flight.longitude(test_flight.t_final)) < 1e-4
    assert test_flight.latitude(test_flight.t_final) > 0


@pytest.mark.parametrize("wind_u, wind_v", [(0, 10), (0, -10), (10, 0), (-10, 0)])
@pytest.mark.parametrize(
    "static_margin, max_time",
    [(-0.1, 2), (-0.01, 5), (0, 5), (0.01, 20), (0.1, 20), (1.0, 20)],
)
def test_stability_static_margins(
    wind_u, wind_v, static_margin, max_time, example_plain_env, dummy_empty_motor
):
    """Test stability margins for a constant velocity flight, 100 m/s, wind a
    lateral wind speed of 10 m/s. Rocket has infinite mass to prevent side motion.
    Check if a restoring moment exists depending on static margins.

    Parameters
    ----------
    wind_u : float
        Wind speed in the x direction
    wind_v : float
        Wind speed in the y direction
    static_margin : float
        Static margin to be tested
    max_time : float
        Maximum time to be simulated
    example_plain_env : rocketpy.Environment
        This is a fixture.
    dummy_empty_motor : rocketpy.SolidMotor
        This is a fixture.
    """

    # Create an environment with ZERO gravity to keep the rocket's speed constant
    example_plain_env.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=wind_u,
        wind_v=wind_v,
        pressure=101325,
        temperature=300,
    )
    # Make sure that the free_stream_mach will always be 0, so that the rocket
    # behaves as the STATIC (free_stream_mach=0) margin predicts
    example_plain_env.speed_of_sound = Function(1e16)

    # create a rocket with zero drag and huge mass to keep the rocket's speed constant
    dummy_rocket = Rocket(
        radius=0.0635,
        mass=1e16,
        inertia=(1, 1, 0.034),
        power_off_drag=0,
        power_on_drag=0,
        center_of_mass_without_motor=0,
    )
    dummy_rocket.set_rail_buttons(0.082, -0.618)
    dummy_rocket.add_motor(dummy_empty_motor, position=-1.373)

    setup_rocket_with_given_static_margin(dummy_rocket, static_margin)

    # Simulate
    init_pos = [0, 0, 100]  # Start at 100 m of altitude
    init_vel = [0, 0, 100]  # Start at 100 m/s
    init_att = [1, 0, 0, 0]  # Inclination of 90 deg and heading of 0 deg
    init_angvel = [0, 0, 0]
    initial_solution = [0] + init_pos + init_vel + init_att + init_angvel
    test_flight = Flight(
        rocket=dummy_rocket,
        rail_length=1,
        environment=example_plain_env,
        initial_solution=initial_solution,
        max_time=max_time,
        max_time_step=1e-2,
        verbose=False,
    )

    # Check stability according to static margin
    if wind_u == 0:
        moments = test_flight.M1.get_source()[:, 1]
        wind_sign = np.sign(wind_v)
    else:  # wind_v == 0
        moments = test_flight.M2.get_source()[:, 1]
        wind_sign = -np.sign(wind_u)

    if static_margin > 0:
        assert np.max(moments) * np.min(moments) < 0
    elif static_margin < 0:
        assert np.all(moments / wind_sign <= 0)
    else:  # static_margin == 0
        assert np.all(np.abs(moments) <= 1e-10)
