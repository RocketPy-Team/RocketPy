import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest
from scipy import optimize

from rocketpy import Components, Flight, Function, LinearGenericSurface, Rocket
from rocketpy.rocket._helpers import aerodynamic_damping
from rocketpy.simulation.helpers.flight_derivatives import u_dot, u_dot_generalized

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


def test_solution_time_is_monotonically_non_decreasing(flight_calisto_robust):
    """Test that solution timestamps never go backwards across all phase transitions.

    This covers flights with exact-time events and new flight phases (drogue and
    main parachute deployments), which are the scenarios most likely to produce
    out-of-order timestamps at phase boundaries.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Full flight with both drogue and main parachutes enabled.
    """
    times = flight_calisto_robust.solution.time
    backward_steps = np.where(np.diff(times) < 0)[0]
    assert backward_steps.size == 0, (
        f"Solution time goes backward at {backward_steps.size} step(s). "
        f"First occurrence: index {backward_steps[0]}, "
        f"t={times[backward_steps[0]]:.6f} -> t={times[backward_steps[0] + 1]:.6f}"
    )
    # Confirm both parachutes actually fired so the phase transitions were exercised
    assert len(flight_calisto_robust.parachute_events) >= 2, (
        "Expected at least 2 parachute events (drogue + main) but got "
        f"{len(flight_calisto_robust.parachute_events)}"
    )


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
    # This rocket has no aerodynamic surfaces, so nothing turns it: it keeps its
    # launch attitude and falls tail first, with the drag braking the fall.
    assert np.allclose(
        flight_calisto.get_solution_at_time(flight_calisto.t_final),
        np.array(
            [
                52.61962020640597,
                -15.5982732,
                1165.22136,
                -2.68709202e-06,
                0.000755621118,
                27.5202515,
                -195.691415,
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
    # The basic air brakes controller mutates the deployment level but does not
    # return observed variables, so every logged entry is None.
    assert len(obs_vars) > 0
    assert all(var is None for var in obs_vars)


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
        ("apogee_time", (-0.652631, -0.734179, 2.701482e-16)),
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
        ("t_initial", (1.654150, 0.659142, 0.002172)),
        ("out_of_rail_time", (5.052628, 2.013361, -1.716290)),
        ("apogee_time", (2.266519, -2.014795, -0.792216)),
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
            (-11.635005, 16.697127, -0.000000),
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
        ("apogee_time", (0.072063, -0.061586, -9.613866)),
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


@pytest.mark.parametrize(
    "flight_time_attr",
    ["t_initial", "out_of_rail_time", "apogee_time", "t_final"],
)
def test_axial_acceleration(flight_calisto_custom_wind, flight_time_attr):
    """Tests the axial_acceleration property by manually calculating the
    dot product of the acceleration vector and the attitude vector at
    specific time steps.

    Parameters
    ----------
    flight_calisto_custom_wind : rocketpy.Flight
        Flight object to be tested.
    flight_time_attr : str
        The name of the attribute of the flight object that contains the time
        of the point to be tested.
    """
    flight = flight_calisto_custom_wind
    t = getattr(flight, flight_time_attr)

    calculated_axial_acc = flight.axial_acceleration(t)

    expected_axial_acc = (
        flight.ax(t) * flight.attitude_vector_x(t)
        + flight.ay(t) * flight.attitude_vector_y(t)
        + flight.az(t) * flight.attitude_vector_z(t)
    )

    assert pytest.approx(expected_axial_acc, abs=1e-9) == calculated_axial_acc


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


def test_linear_generic_surface_flight_is_stable(
    calisto_linear_generic, example_plain_env
):
    """A Calisto whose fin set is a body-frame LinearGenericSurface flies stably.

    The linear surface builds its forces and moments directly in the body frame
    from the coefficient derivatives (no wind-to-body rotation). With a positive
    normal-force slope placed aft it must give a positive static margin and the
    rocket must reach a finite apogee while staying aligned with the flow (a
    small angle of attack, i.e. no tumbling).
    """
    rocket = calisto_linear_generic
    assert any(
        isinstance(surface, LinearGenericSurface)
        for surface, _ in rocket.aerodynamic_surfaces
    )
    assert rocket.static_margin(0) > 0

    test_flight = Flight(
        environment=example_plain_env,
        rocket=rocket,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

    assert test_flight.apogee_time > test_flight.out_of_rail_time
    assert np.isfinite(test_flight.apogee)
    assert test_flight.apogee > example_plain_env.elevation
    # A stable rocket keeps a small angle of attack throughout the ascent. Only
    # the ascent off the rail is checked: while the rocket is still on the rail
    # its speed is ~0, so the angle of attack is reported as a degenerate 90
    # degrees (arccos of 0) for every launcher, stable or not.
    aoa_source = test_flight.angle_of_attack.get_source()
    ascent = aoa_source[:, 0] > test_flight.out_of_rail_time
    angle_of_attack = aoa_source[ascent, 1]
    assert np.nanmax(np.abs(angle_of_attack)) < 45


@pytest.mark.parametrize(
    "generic_rocket_name, apogee_rel_tol",
    [
        ("calisto_linear_generic", 5e-3),
        ("calisto_generic", 5e-3),
        ("calisto_full_aerodynamics", 1e-2),
    ],
)
def test_generic_surface_calisto_flight_matches_barrowman(
    request, calisto_robust, example_plain_env, generic_rocket_name, apogee_rel_tol
):
    """A Calisto rebuilt from generic surfaces flies the same as the Barrowman
    Calisto.

    The reference ``calisto_robust`` models each surface with the classic
    Barrowman method. The three generic-surface Calistos carry the same
    aerodynamics through different code paths: per-surface
    ``LinearGenericSurface`` (coefficient slopes), per-surface
    ``GenericSurface`` (total coefficients), and a single full-body
    ``LinearGenericSurface`` added with ``add_full_body_aerodynamics`` (the lumped
    stability-derivative set, including the rate damping the distributed
    surfaces produce through their lever arms). Flown from the same launcher in
    still air, each must reach essentially the same apogee and leave the rail at
    the same time and speed. Only the ascent is compared (``terminate_on_apogee``);
    the descent under identical parachutes adds nothing to the comparison.

    The per-surface models reproduce the Barrowman forces almost exactly; the
    lumped full-body model is a point approximation of the distributed
    surfaces, so it is held to a slightly looser apogee tolerance.
    """
    generic_rocket = request.getfixturevalue(generic_rocket_name)

    launch = dict(
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    reference_flight = Flight(rocket=calisto_robust, **launch)
    generic_flight = Flight(rocket=generic_rocket, **launch)

    # Leaving the rail is driven by thrust and the shared drag curve, so every
    # model must agree tightly here.
    assert generic_flight.out_of_rail_time == pytest.approx(
        reference_flight.out_of_rail_time, rel=1e-3
    )
    assert generic_flight.out_of_rail_velocity == pytest.approx(
        reference_flight.out_of_rail_velocity, rel=1e-3
    )
    # Apogee reflects the whole aerodynamic ascent.
    assert generic_flight.apogee == pytest.approx(
        reference_flight.apogee, rel=apogee_rel_tol
    )
    assert generic_flight.apogee_time == pytest.approx(
        reference_flight.apogee_time, rel=apogee_rel_tol
    )


def test_max_acceleration_power_off_time_with_controllers(
    flight_calisto_air_brakes,
):
    """Test that max_acceleration_power_off_time returns a valid time when
    controllers are present (e.g., air brakes). This is a regression test for
    a bug where the time was always returned as 0.0.

    Parameters
    ----------
    flight_calisto_air_brakes : rocketpy.Flight
        Flight object with air brakes. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test = flight_calisto_air_brakes
    burn_out_time = test.rocket.motor.burn_out_time

    # The max_acceleration_power_off_time should be at or after motor burn out
    # It should NOT be 0.0, which was the bug behavior
    assert test.max_acceleration_power_off_time > 0, (
        "max_acceleration_power_off_time should not be zero"
    )
    assert test.max_acceleration_power_off_time >= burn_out_time - 0.01, (
        f"max_acceleration_power_off_time ({test.max_acceleration_power_off_time}) "
        f"should be at or after burn_out_time ({burn_out_time})"
    )

    # Also verify max_acceleration_power_off is positive
    assert test.max_acceleration_power_off > 0, (
        "max_acceleration_power_off should be greater than zero"
    )


# ---------------------------------------------------------------------------
# The equations of motion themselves, evaluated on a fixed state
# ---------------------------------------------------------------------------


def _coasting_state(flight, seconds_after_burnout=3.0):
    """Return ``(t, state)`` a few seconds into the coast of a flown flight."""
    t = flight.rocket.motor.burn_out_time + seconds_after_burnout
    return t, list(flight.solution.at(t).values())[:13]


def _with_rates(state, w1=0.0, w2=0.0, w3=0.0):
    return [*state[:10], w1, w2, w3]


@pytest.mark.parametrize("derivative", [u_dot, u_dot_generalized])
def test_an_angular_rate_is_aerodynamically_damped(flight_calisto_robust, derivative):
    """A rate about any body axis must produce an angular acceleration against it.

    The solid propulsion equations once zeroed the rates before the
    aerodynamics, so no rate was ever damped: a canted fin spun the rocket up
    without bound. This checks each axis on a coasting state, for both sets of
    equations.
    """
    flight = flight_calisto_robust
    t, state = _coasting_state(flight)
    at_rest = derivative(flight, t, state)
    for axis, rate in ((0, 2.0), (1, 2.0), (2, 20.0)):
        slot = 10 + axis
        positive = derivative(flight, t, _with_rates(state, **{f"w{axis + 1}": rate}))
        negative = derivative(flight, t, _with_rates(state, **{f"w{axis + 1}": -rate}))
        assert positive[slot] < at_rest[slot] < negative[slot], f"axis {axis + 1}"
        # damping is close to linear in the rate at these magnitudes
        assert positive[slot] - at_rest[slot] == pytest.approx(
            -(negative[slot] - at_rest[slot]), rel=0.2
        )


def test_solid_propulsion_and_generalized_equations_agree_when_rotating(
    flight_calisto_robust,
):
    """With a pitch rate applied, both models give the same damping moment.

    They differ in how variable mass is handled, not in the aerodynamics, so
    the angular acceleration a rate produces must agree closely.
    """
    flight = flight_calisto_robust
    t, state = _coasting_state(flight)
    pitching = _with_rates(state, w1=2.0)
    solid = u_dot(flight, t, pitching)
    generalized = u_dot_generalized(flight, t, pitching)
    assert solid[10] == pytest.approx(generalized[10], rel=0.1)
    assert solid[3:6] == pytest.approx(generalized[3:6], rel=0.05)


def test_static_margin_yaw_is_the_one_of_the_rocket():
    """The flight gives both planes of the static margin, as it does for the
    stability margin. No simulation is needed to check it."""
    rocket = SimpleNamespace(static_margin="pitch", static_margin_yaw="yaw")
    flight = SimpleNamespace(rocket=rocket)
    assert Flight.static_margin.func(flight) == "pitch"
    assert Flight.static_margin_yaw.func(flight) == "yaw"


def test_yaw_stability_margin_values_follow_the_yaw_curve():
    """The single yaw-plane margins (initial, at rail exit, largest, smallest)
    are read from ``stability_margin_yaw``, as the pitch ones are read from
    ``stability_margin``. No simulation is needed to check it."""
    curve = Function([[0.0, 2.0], [1.0, 3.5], [2.0, 1.2], [3.0, 2.4]], "t", "margin")
    flight = SimpleNamespace(
        stability_margin_yaw=curve, time=[0.0, 1.0, 2.0, 3.0], out_of_rail_time=1.0
    )
    assert Flight.initial_stability_margin_yaw.fget(flight) == pytest.approx(2.0)
    assert Flight.out_of_rail_stability_margin_yaw.fget(flight) == pytest.approx(3.5)

    flight.max_stability_margin_yaw_time = Flight.max_stability_margin_yaw_time.func(
        flight
    )
    flight.min_stability_margin_yaw_time = Flight.min_stability_margin_yaw_time.func(
        flight
    )
    assert flight.max_stability_margin_yaw_time == pytest.approx(1.0)
    assert flight.min_stability_margin_yaw_time == pytest.approx(2.0)
    assert Flight.max_stability_margin_yaw.func(flight) == pytest.approx(3.5)
    assert Flight.min_stability_margin_yaw.func(flight) == pytest.approx(1.2)


# Moments about the center of mass while the motor burns


def _rigid_burning_calisto(
    calisto_motorless, calisto_nose_cone, calisto_tail, calisto_trapezoidal_fins
):
    """Calisto with a full grain whose motor gives a milli-newton of thrust and
    a negligible mass flow: a rigid body whose center of mass sits well behind
    the center of dry mass."""
    from rocketpy import SolidMotor  # pylint: disable=import-outside-toplevel

    motor = SolidMotor(
        thrust_source=1e-3,
        burn_time=1000.0,
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
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    rocket = calisto_motorless
    rocket.add_motor(motor, position=-1.255)
    rocket.add_surfaces(calisto_nose_cone, 1.160)
    rocket.add_surfaces(calisto_trapezoidal_fins, -1.168)
    rocket.add_surfaces(calisto_tail, -1.313)
    return rocket


def _center_of_mass_inertia(rocket, t):
    """Position of the center of mass relative to the center of dry mass in
    the body frame, and the inertia tensor about it."""
    from rocketpy.mathutils.vector_matrix import Vector  # pylint: disable=import-outside-toplevel

    r_cm = Vector([0, 0, -rocket.com_to_cdm_function.get_value_opt(t)])
    mass = rocket.total_mass.get_value_opt(t)
    inertia = (
        rocket.get_inertia_tensor_at_time(t)
        - (r_cm.cross_matrix @ -r_cm.cross_matrix) * mass
    )
    return r_cm, inertia


@pytest.mark.parametrize(
    "derivative, tolerance", [(u_dot_generalized, 1e-6), (u_dot, 0.02)]
)
def test_dynamics_take_moments_about_the_center_of_mass(
    calisto_motorless,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    example_plain_env,
    derivative,
    tolerance,
):
    """With the center of mass behind the center of dry mass, a lateral force
    turns the rocket about the center of mass: ``I_cm w_dot`` must equal the
    aerodynamic moment about the center of dry mass transferred there,
    ``M + R x r_cm``, not the untransferred moment (legacy) nor the transfer
    the other way (generalized, before the fix).
    """
    from rocketpy.mathutils.vector_matrix import Vector  # pylint: disable=import-outside-toplevel

    rocket = _rigid_burning_calisto(
        calisto_motorless, calisto_nose_cone, calisto_tail, calisto_trapezoidal_fins
    )
    flight = Flight(
        rocket=rocket,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=90,
        heading=0,
        max_time=0.05,
    )
    t = 1.0
    r_cm, inertia_cm = _center_of_mass_inertia(rocket, t)
    assert r_cm[2] < -0.1  # the full grain pulls the center of mass aft

    # Vertical, 100 m/s up with 10 m/s sideways: sideslip, no rotation
    state = [0, 0, 1500, 10.0, 0, 100.0, 1, 0, 0, 0, 0, 0, 0]
    out = derivative(flight, t, state, post_processing=True)
    w_dot, forces, moments = Vector(out[3:6]), Vector(out[6:9]), Vector(out[9:12])
    assert abs(moments[1]) > 10  # the fins push back
    about_center_of_mass = moments + (forces ^ r_cm)
    assert list(inertia_cm @ w_dot) == pytest.approx(
        list(about_center_of_mass), rel=tolerance, abs=1e-9
    )
    assert abs(about_center_of_mass[1]) < 0.8 * abs(moments[1])


def test_generalized_dynamics_match_a_rigid_body_when_rotating(
    calisto_motorless,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    example_plain_env,
):
    """Rotating about all three axes with the center of mass off the center of
    dry mass, the angular acceleration is that of a rigid body about its
    center of mass, ``I_cm w_dot = M_cm - w x (I_cm w)``, and the acceleration
    of the center of dry mass follows from the center of mass' one,
    ``a_O = F / m - w_dot x r_cm - w x (w x r_cm)``."""
    from rocketpy.mathutils.vector_matrix import Vector  # pylint: disable=import-outside-toplevel

    rocket = _rigid_burning_calisto(
        calisto_motorless, calisto_nose_cone, calisto_tail, calisto_trapezoidal_fins
    )
    flight = Flight(
        rocket=rocket,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=90,
        heading=0,
        max_time=0.05,
    )
    t = 1.0
    r_cm, inertia_cm = _center_of_mass_inertia(rocket, t)
    mass = rocket.total_mass.get_value_opt(t)
    w = Vector([0.4, -0.3, 2.0])
    state = [0, 0, 1500, 10.0, -5.0, 100.0, 1, 0, 0, 0, *w]
    out = u_dot_generalized(flight, t, state, post_processing=True)
    a_cdm, w_dot = Vector(out[:3]), Vector(out[3:6])
    forces, moments, thrust = Vector(out[6:9]), Vector(out[9:12]), out[12]

    gravity = Vector([0, 0, -mass * example_plain_env.gravity.get_value_opt(1500)])
    total_force = forces + gravity + Vector([0, 0, thrust])
    moment_cm = moments + (forces ^ r_cm)  # gravity and thrust give none
    expected_w_dot = inertia_cm.inverse @ (moment_cm - (w ^ (inertia_cm @ w)))
    # The milli-newton motor still has a whisper of mass flow, hence the slack
    assert list(w_dot) == pytest.approx(list(expected_w_dot), rel=2e-3, abs=1e-4)

    # The attitude is the identity, so body and inertial components coincide;
    # the integrator also adds the Earth-rotation Coriolis acceleration
    earth = Vector(example_plain_env.earth_rotation_vector)
    expected_a_cdm = (
        total_force / mass
        - (expected_w_dot ^ r_cm)
        - (w ^ (w ^ r_cm))
        - 2 * (earth ^ Vector(state[3:6]))
    )
    assert list(a_cdm) == pytest.approx(list(expected_a_cdm), rel=1e-3, abs=1e-4)


def test_integrator_damping_matches_the_oscillator(flight_calisto_robust):
    """During the burn, the moment the equations of motion set against a pitch
    rate equals the oscillator's ``C2`` (aerodynamic plus jet damping in
    Thomson's form), once the rate is taken about the same point: the state
    rotates about the center of dry mass, ``a`` ahead of the center of mass,
    which adds ``C1 a / V`` of angle-of-attack coupling."""
    from rocketpy.mathutils.vector_matrix import Vector  # pylint: disable=import-outside-toplevel

    flight = flight_calisto_robust
    rocket = flight.rocket
    # A solution time about one second into the burn
    index = int(np.argmin(np.abs(flight.time - 1.0)))
    t = flight.time[index]
    r_cm, inertia_cm = _center_of_mass_inertia(rocket, t)
    speed = 100.0
    state = [0, 0, 1500, 0, 0, speed, 1, 0, 0, 0, 0, 0, 0]

    def pitch_acceleration(rate):
        out = u_dot_generalized(flight, t, [*state[:10], rate, 0, 0])
        return (inertia_cm @ Vector(out[10:13]))[0]

    step = 1e-3
    integrator = -(pitch_acceleration(step) - pitch_acceleration(-step)) / (2 * step)

    density = flight.env.density.get_value_opt(1500)
    mach = speed / flight.env.speed_of_sound.get_value_opt(1500)
    aero = density * speed * aerodynamic_damping(rocket, 0.0, 0.0, mach, t, "pitch")
    inertia, inertia_rate = flight._lateral_inertia(rocket.I_11)
    jet = (
        abs(rocket.total_mass_flow_rate.get_value_opt(t))
        * (rocket.nozzle_position - rocket.center_of_mass.get_value_opt(t)) ** 2
        + inertia_rate[index]
    )
    assert inertia[index] == pytest.approx(inertia_cm[0][0], rel=1e-6)
    assert (
        0
        < jet
        < 0.7
        * abs(rocket.total_mass_flow_rate.get_value_opt(t))
        * (rocket.nozzle_position - rocket.center_of_mass.get_value_opt(t)) ** 2
    )
    # Rotating about the center of dry mass gives every surface an extra
    # angle of attack a w / V, felt through the restoring moment C1
    slope = rocket.total_lift_coeff_der.get_value_opt(mach)
    margin = rocket.stability_margin.get_value_opt(mach, t)
    dynamic_pressure = 0.5 * density * speed**2
    corrective = dynamic_pressure * rocket.area * slope * margin * 2 * rocket.radius
    coupling = corrective * abs(r_cm[2]) / speed
    assert integrator == pytest.approx(aero + jet + coupling, rel=0.01)


def test_disturbance_response_matches_the_rocket_at_the_flight_condition(
    flight_calisto_robust,
):
    """The response at an instant of the flight is the rocket's own response at
    that instant's airspeed, density and Mach number; on the rail there is
    none."""
    flight = flight_calisto_robust
    t = 4.0  # just after burnout
    response = flight.disturbance_response(t)
    speed = flight.free_stream_speed.get_value_opt(t)
    expected = flight.rocket.disturbance_response(
        speed,
        time=t,
        density=flight.density.get_value_opt(t),
        speed_of_sound=speed / flight.mach_number.get_value_opt(t),
        duration=response.x_array[-1],
    )
    assert response.y_array[0] == pytest.approx(5.0)
    # The flight reads the coefficients at its own angle of attack and between
    # its time steps, so the two agree closely, not exactly
    assert response.y_array == pytest.approx(expected.y_array, abs=0.05)
    with pytest.raises(ValueError, match="rail"):
        flight.disturbance_response(0.0)
