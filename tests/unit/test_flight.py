import json
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
