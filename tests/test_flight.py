import os
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest
from scipy import optimize

from rocketpy import Components, Environment, Flight, Function, Rocket, SolidMotor

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


@patch("matplotlib.pyplot.show")
def test_all_info(mock_show, flight_calisto_robust):
    """Test that the flight class is working as intended. This basically calls
    the all_info() method and checks if it returns None. It is not testing if
    the values are correct, but whether the method is working without errors.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    assert flight_calisto_robust.all_info() == None


@patch("matplotlib.pyplot.show")
def test_initial_solution(mock_show, example_env, calisto_robust):
    """Tests the initial_solution option of the Flight class. This test simply
    simulates the flight using the initial_solution option and checks if the
    all_info method returns None.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    example_env : rocketpy.Environment
        Environment to be simulated. See the conftest.py file for more info.
    calisto_robust : rocketpy.Rocket
        Rocket to be simulated. See the conftest.py file for more info.
    """
    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_env,
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

    assert test_flight.all_info() == None


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
                48.4313533,
                0.0,
                985.7665845,
                -0.00000229951048,
                0.0,
                11.2223284,
                -341.028803,
                0.999048222,
                -0.0436193874,
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


@patch("matplotlib.pyplot.show")
def test_empty_motor_flight(mock_show, example_env, calisto_motorless):
    flight = Flight(
        rocket=calisto_motorless,
        environment=example_env,
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
    assert flight.all_info() == None


@pytest.mark.parametrize("wind_u, wind_v", [(0, 10), (0, -10), (10, 0), (-10, 0)])
@pytest.mark.parametrize(
    "static_margin, max_time",
    [(-0.1, 2), (-0.01, 5), (0, 5), (0.01, 20), (0.1, 20), (1.0, 20)],
)
def test_stability_static_margins(wind_u, wind_v, static_margin, max_time):
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
    """

    # Create an environment with ZERO gravity to keep the rocket's speed constant
    env = Environment(gravity=0, latitude=0, longitude=0, elevation=0)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=wind_u,
        wind_v=wind_v,
        pressure=101325,
        temperature=300,
    )
    # Make sure that the free_stream_mach will always be 0, so that the rocket
    # behaves as the STATIC (free_stream_mach=0) margin predicts
    env.speed_of_sound = Function(1e16)

    # Create a motor with ZERO thrust and ZERO mass to keep the rocket's speed constant
    # TODO: why don t we use these same values to create EmptyMotor class?
    dummy_motor = SolidMotor(
        thrust_source=1e-300,
        burn_time=1e-10,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1e-300,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        nozzle_position=0,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

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
    dummy_rocket.add_motor(dummy_motor, position=-1.373)

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
        environment=env,
        initial_solution=initial_solution,
        max_time=max_time,
        max_time_step=1e-2,
        verbose=False,
    )
    test_flight.post_process(interpolation="linear")

    # Check stability according to static margin
    if wind_u == 0:
        moments = test_flight.M1.get_source()[:, 1]
        wind_sign = np.sign(wind_v)
    else:  # wind_v == 0
        moments = test_flight.M2.get_source()[:, 1]
        wind_sign = -np.sign(wind_u)

    assert (
        (static_margin > 0 and np.max(moments) * np.min(moments) < 0)
        or (static_margin < 0 and np.all(moments / wind_sign <= 0))
        or (static_margin == 0 and np.all(np.abs(moments) <= 1e-10))
    )


@patch("matplotlib.pyplot.show")
def test_rolling_flight(
    mock_show,
    example_env,
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
    fin_set = test_rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956,
        cant_angle=0.5,
    )
    calisto.aerodynamic_surfaces.add(calisto_nose_cone, 1.160)
    calisto.aerodynamic_surfaces.add(calisto_tail, -1.313)
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)

    test_flight = Flight(
        rocket=test_rocket,
        environment=example_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert test_flight.all_info() == None


@patch("matplotlib.pyplot.show")
def test_simpler_parachute_triggers(mock_show, example_env, calisto_robust):
    """Tests different types of parachute triggers. This is important to ensure
    the code is working as intended, since the parachute triggers can have very
    different format definitions. It will add 3 parachutes using different
    triggers format and check if the parachute events are being at the correct
    altitude

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    example_env : rocketpy.Environment
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
        environment=example_env,
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
            - (800 + example_env.elevation)
        )
        <= 1
    )
    assert (
        abs(
            test_flight.z(test_flight.parachute_events[2][0])
            - (400 + example_env.elevation)
        )
        <= 1
    )
    assert calisto_robust.all_info() == None
    assert test_flight.all_info() == None


def test_export_data(flight_calisto):
    """Tests wether the method Flight.export_data is working as intended

    Parameters:
    -----------
    flight_calisto : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test_flight = flight_calisto

    # Basic export
    test_flight.export_data("test_export_data_1.csv")

    # Custom export
    test_flight.export_data(
        "test_export_data_2.csv",
        "z",
        "vz",
        "e1",
        "w3",
        "angle_of_attack",
        time_step=0.1,
    )

    # Load exported files and fixtures and compare them
    test_1 = np.loadtxt("test_export_data_1.csv", delimiter=",")
    test_2 = np.loadtxt("test_export_data_2.csv", delimiter=",")

    # Delete files
    os.remove("test_export_data_1.csv")
    os.remove("test_export_data_2.csv")

    # Check if basic exported content matches data
    assert np.allclose(test_flight.x[:, 0], test_1[:, 0], atol=1e-5) == True
    assert np.allclose(test_flight.x[:, 1], test_1[:, 1], atol=1e-5) == True
    assert np.allclose(test_flight.y[:, 1], test_1[:, 2], atol=1e-5) == True
    assert np.allclose(test_flight.z[:, 1], test_1[:, 3], atol=1e-5) == True
    assert np.allclose(test_flight.vx[:, 1], test_1[:, 4], atol=1e-5) == True
    assert np.allclose(test_flight.vy[:, 1], test_1[:, 5], atol=1e-5) == True
    assert np.allclose(test_flight.vz[:, 1], test_1[:, 6], atol=1e-5) == True
    assert np.allclose(test_flight.e0[:, 1], test_1[:, 7], atol=1e-5) == True
    assert np.allclose(test_flight.e1[:, 1], test_1[:, 8], atol=1e-5) == True
    assert np.allclose(test_flight.e2[:, 1], test_1[:, 9], atol=1e-5) == True
    assert np.allclose(test_flight.e3[:, 1], test_1[:, 10], atol=1e-5) == True
    assert np.allclose(test_flight.w1[:, 1], test_1[:, 11], atol=1e-5) == True
    assert np.allclose(test_flight.w2[:, 1], test_1[:, 12], atol=1e-5) == True
    assert np.allclose(test_flight.w3[:, 1], test_1[:, 13], atol=1e-5) == True

    # Check if custom exported content matches data
    timePoints = np.arange(test_flight.t_initial, test_flight.t_final, 0.1)
    assert np.allclose(timePoints, test_2[:, 0], atol=1e-5) == True
    assert np.allclose(test_flight.z(timePoints), test_2[:, 1], atol=1e-5) == True
    assert np.allclose(test_flight.vz(timePoints), test_2[:, 2], atol=1e-5) == True
    assert np.allclose(test_flight.e1(timePoints), test_2[:, 3], atol=1e-5) == True
    assert np.allclose(test_flight.w3(timePoints), test_2[:, 4], atol=1e-5) == True
    assert (
        np.allclose(test_flight.angle_of_attack(timePoints), test_2[:, 5], atol=1e-5)
        == True
    )


def test_export_kml(flight_calisto_robust):
    """Tests weather the method Flight.export_kml is working as intended.

    Parameters:
    -----------
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """

    test_flight = flight_calisto_robust

    # Basic export
    test_flight.export_kml(
        "test_export_data_1.kml", time_step=None, extrude=True, altitude_mode="absolute"
    )

    # Load exported files and fixtures and compare them
    test_1 = open("test_export_data_1.kml", "r")

    for row in test_1:
        if row[:29] == "                <coordinates>":
            r = row[29:-15]
            r = r.split(",")
            for i, j in enumerate(r):
                r[i] = j.split(" ")
    lon, lat, z, coords = [], [], [], []
    for i in r:
        for j in i:
            coords.append(j)
    for i in range(0, len(coords), 3):
        lon.append(float(coords[i]))
        lat.append(float(coords[i + 1]))
        z.append(float(coords[i + 2]))

    # Delete temporary test file
    test_1.close()
    os.remove("test_export_data_1.kml")

    assert np.allclose(test_flight.latitude[:, 1], lat, atol=1e-3) == True
    assert np.allclose(test_flight.longitude[:, 1], lon, atol=1e-3) == True
    assert np.allclose(test_flight.z[:, 1], z, atol=1e-3) == True


@patch("matplotlib.pyplot.show")
def test_lat_lon_conversion_robust(mock_show, example_env_robust, calisto_robust):
    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_env_robust,
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
def test_lat_lon_conversion_from_origin(mock_show, example_env, calisto_robust):
    "additional tests to capture incorrect behaviors during lat/lon conversions"

    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert abs(test_flight.longitude(test_flight.t_final) - 0) < 1e-12
    assert test_flight.latitude(test_flight.t_final) > 0


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
def test_rail_length(calisto_robust, example_env, rail_length, out_of_rail_time):
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
    example_env : rocketpy.Environment
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
        environment=example_env,
        rail_length=rail_length,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    assert abs(test_flight.z(test_flight.out_of_rail_time) - out_of_rail_time) < 1e-6


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_time_overshoot(mock_show, calisto_robust, example_env_robust):
    """Test the time_overshoot parameter of the Flight class. This basically
    calls the all_info() method for a simulation without time_overshoot and
    checks if it returns None. It is not testing if the values are correct,
    just if the flight simulation is not breaking.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        The rocket to be simulated. In this case, the fixture rocket is used.
        See the conftest.py file for more information.
    example_env_robust : rocketpy.Environment
        The environment to be simulated. In this case, the fixture environment
        is used. See the conftest.py file for more information.
    """

    test_flight = Flight(
        rocket=calisto_robust,
        environment=example_env_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
    )

    assert test_flight.all_info() == None


@patch("matplotlib.pyplot.show")
def test_liquid_motor_flight(mock_show, calisto_liquid_modded):
    """Test the flight of a rocket with a liquid motor. This test only validates
    that a flight simulation can be performed with a liquid motor; it does not
    validate the results.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    calisto_liquid_modded : rocketpy.Rocket
        Sample Rocket to be simulated. See the conftest.py file for more info.
    """
    test_flight = Flight(
        rocket=calisto_liquid_modded,
        environment=Environment(),
        rail_length=5,
        inclination=85,
        heading=0,
        max_time_step=0.25,
    )

    assert test_flight.all_info() == None


@patch("matplotlib.pyplot.show")
def test_hybrid_motor_flight(mock_show, calisto_hybrid_modded):
    """Test the flight of a rocket with a hybrid motor. This test only validates
    that a flight simulation can be performed with a hybrid motor; it does not
    validate the results.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mock object to replace matplotlib.pyplot.show
    calisto_hybrid_modded : rocketpy.Rocket
        Sample rocket to be simulated. See the conftest.py file for more info.
    """
    test_flight = Flight(
        rocket=calisto_hybrid_modded,
        environment=Environment(),
        rail_length=5,
        inclination=85,
        heading=0,
        max_time_step=0.25,
    )

    assert test_flight.all_info() == None


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


def test_max_values(flight_calisto_robust):
    """Test the max values of the flight. This tests if the max values are
    close to the expected values. However, the expected values were NOT
    calculated by hand, it was just copied from the test results. This is
    because the expected values are not easy to calculate by hand, and the
    results are not expected to change. If the results change, the test will
    fail, and the expected values must be updated. If if want to update the
    values, always double check if the results are really correct. Acceptable
    reasons for changes in the results are: 1) changes in the code that
    improve the accuracy of the simulation, 2) a bug was found and fixed. Keep
    in mind that other tests may be more accurate than this one, for example,
    the acceptance tests, which are based on the results of real flights.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight object to be tested. See the conftest.py file for more info
        regarding this pytest fixture.
    """
    test = flight_calisto_robust
    atol = 1e-2
    assert pytest.approx(105.2774, abs=atol) == test.max_acceleration_power_on
    assert pytest.approx(105.2774, abs=atol) == test.max_acceleration
    assert pytest.approx(0.85999, abs=atol) == test.max_mach_number
    assert pytest.approx(285.90240, abs=atol) == test.max_speed


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
    assert pytest.approx(3.80358, abs=atol) == test.max_rail_button1_normal_force
    assert pytest.approx(1.63602, abs=atol) == test.max_rail_button1_shear_force
    assert pytest.approx(1.19353, abs=atol) == test.max_rail_button2_normal_force
    assert pytest.approx(0.51337, abs=atol) == test.max_rail_button2_shear_force


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (0, 0, 0)),
        ("out_of_rail_time", (0, 7.8068, 89.2325)),
        ("apogee_time", (0.07534, -0.058127, -9.614386)),
        ("t_final", (0, 0, 0.0017346294117130806)),
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


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (0, 0, 0)),
        ("out_of_rail_time", (0, 2.248727, 25.703072)),
        ("apogee_time", (-13.209436, 16.05115, -0.000257)),
        ("t_final", (5, 2, -5.334289)),
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
        ("t_initial", (1.6542528, 0.65918, -0.067107)),
        ("out_of_rail_time", (5.05334, 2.01364, -1.7541)),
        ("apogee_time", (2.35291, -1.8275, -0.87851)),
        ("t_final", (0, 0, 141.42421)),
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
    expected_attr, expected_R = flight_time, expected_values

    test = flight_calisto_custom_wind
    t = getattr(test, expected_attr)
    atol = 5e-3

    assert pytest.approx(expected_R, abs=atol) == (
        test.R1(t),
        test.R2(t),
        test.R3(t),
    ), f"Assertion error for aerodynamic forces vector at {expected_attr}."


@pytest.mark.parametrize(
    "flight_time, expected_values",
    [
        ("t_initial", (0.17179073815516033, -0.431117, 0)),
        ("out_of_rail_time", (0.547026, -1.3727895, 0)),
        ("apogee_time", (-0.5874848151271623, -0.7563596, 0)),
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
    expected_attr, expected_M = flight_time, expected_values

    test = flight_calisto_custom_wind
    t = getattr(test, expected_attr)
    atol = 5e-3

    assert pytest.approx(expected_M, abs=atol) == (
        test.M1(t),
        test.M2(t),
        test.M3(t),
    ), f"Assertion error for moment vector at {expected_attr}."
