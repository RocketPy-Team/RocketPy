import json
import os

import numpy as np
import pytest

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.components import Components
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gyroscope import Gyroscope


def test_sensor_on_rocket(calisto_sensors):
    """Test the sensor on the rocket.

    Parameters
    ----------
    calisto_accel_gyro : Rocket
        Pytest fixture for the calisto rocket with a set of ideal sensors.
    """
    sensors = calisto_sensors.sensors
    assert isinstance(sensors, Components)
    assert isinstance(sensors[0].component, Accelerometer)
    assert isinstance(sensors[1].position, Vector)
    assert isinstance(sensors[2].component, Gyroscope)
    assert isinstance(sensors[2].position, Vector)


@pytest.mark.parametrize(
    "sensor_index, measured_data_key, sim_method, tolerance",
    [
        (0, "measured_data[0]", lambda flight, time: flight.acceleration(time), 1e-12),
        (
            2,
            "measured_data",
            lambda flight, time: np.sqrt(
                flight.w1(time) ** 2 + flight.w2(time) ** 2 + flight.w3(time) ** 2
            ),
            1e-12,
        ),
        (3, "measured_data", lambda flight, time: flight.pressure(time), 1e-12),
    ],
)
def test_ideal_sensors(
    flight_calisto_sensors, sensor_index, measured_data_key, sim_method, tolerance
):
    """Test the ideal sensors. All types of sensors are here to reduce
    testing time.

    Parameters
    ----------
    flight_calisto_sensors : Flight
        Pytest fixture for the flight of the calisto rocket with a set of ideal
        sensors.
    sensor_index : int
        Index of the sensor in the rocket's sensor list.
    measured_data_key : str
        Key to access the measured data from the sensor component.
    sim_method : function
        Function to compute the simulated data.
    tolerance : float
        Tolerance level for the comparison between measured and simulated data.
    """
    sensor = flight_calisto_sensors.rocket.sensors[sensor_index].component
    measured_data = eval(f"sensor.{measured_data_key}")

    if sensor_index == 0:  # Accelerometer
        time, ax, ay, az = zip(*measured_data)
        ax = np.array(ax)
        ay = np.array(ay)
        az = np.array(az)
        a = np.sqrt(ax**2 + ay**2 + az**2)
        sim_data = sim_method(flight_calisto_sensors, time)
        assert np.allclose(a, sim_data, atol=tolerance)

        # Check if both added accelerometer instances saved the same data
        assert (
            flight_calisto_sensors.sensors[0].measured_data[0]
            == flight_calisto_sensors.sensors[0].measured_data[1]
        )

    elif sensor_index == 2:  # Gyroscope
        time, wx, wy, wz = zip(*measured_data)
        wx = np.array(wx)
        wy = np.array(wy)
        wz = np.array(wz)
        w = np.sqrt(wx**2 + wy**2 + wz**2)
        sim_data = sim_method(flight_calisto_sensors, time)
        assert np.allclose(w, sim_data, atol=tolerance)

    elif sensor_index == 3:  # Barometer
        time, pressure = zip(*measured_data)
        pressure = np.array(pressure)
        sim_data = sim_method(flight_calisto_sensors, time)
        assert np.allclose(pressure, sim_data, atol=tolerance)


def test_export_sensor_data(flight_calisto_sensors):
    """Test the export of sensor data.

    Parameters
    ----------
    flight_calisto_sensors : Flight
        Pytest fixture for the flight of the calisto rocket with a set of ideal
        sensors.
    """
    flight_calisto_sensors.export_sensor_data("test_sensor_data.json")
    # read the json and parse as dict
    filename = "test_sensor_data.json"
    with open(filename, "r") as f:
        data = f.read()
        sensor_data = json.loads(data)
    # convert list of tuples into list of lists to compare with the json
    flight_calisto_sensors.sensors[0].measured_data[0] = [
        list(measurement)
        for measurement in flight_calisto_sensors.sensors[0].measured_data[0]
    ]
    flight_calisto_sensors.sensors[1].measured_data[1] = [
        list(measurement)
        for measurement in flight_calisto_sensors.sensors[1].measured_data[1]
    ]
    flight_calisto_sensors.sensors[2].measured_data = [
        list(measurement)
        for measurement in flight_calisto_sensors.sensors[2].measured_data
    ]
    flight_calisto_sensors.sensors[3].measured_data = [
        list(measurement)
        for measurement in flight_calisto_sensors.sensors[3].measured_data
    ]
    assert (
        sensor_data["Accelerometer"]["1"]
        == flight_calisto_sensors.sensors[0].measured_data[0]
    )
    assert (
        sensor_data["Accelerometer"]["2"]
        == flight_calisto_sensors.sensors[1].measured_data[1]
    )
    assert sensor_data["Gyroscope"] == flight_calisto_sensors.sensors[2].measured_data
    assert sensor_data["Barometer"] == flight_calisto_sensors.sensors[3].measured_data
    os.remove(filename)
