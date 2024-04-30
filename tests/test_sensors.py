import json
import os
import numpy as np

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.components import Components
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gyroscope import Gyroscope


def test_sensor_on_rocket(calisto_accel_gyro):
    """Test the sensor on the rocket.

    Parameters
    ----------
    calisto_accel_gyro : Rocket
        Pytest fixture for the calisto rocket with an accelerometer and a gyroscope.
    """
    sensors = calisto_accel_gyro.sensors
    assert isinstance(sensors, Components)
    assert isinstance(sensors[0].component, Accelerometer)
    assert isinstance(sensors[1].position, Vector)
    assert isinstance(sensors[2].component, Gyroscope)
    assert isinstance(sensors[2].position, Vector)


def test_ideal_sensors(flight_calisto_accel_gyro):
    """Test the ideal sensors. All types of sensors are here to reduvce
    testing time.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal accelerometer and a gyroscope.
    """
    accelerometer = flight_calisto_accel_gyro.rocket.sensors[0].component
    time, ax, ay, az = zip(*accelerometer.measured_data[0])
    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    a = np.sqrt(ax**2 + ay**2 + az**2)
    sim_accel = flight_calisto_accel_gyro.acceleration(time)

    # tolerance is bounded to numerical errors in the transformation matrixes
    assert np.allclose(a, sim_accel, atol=1e-12)
    # check if both added accelerometer instances saved the same data
    assert (
        flight_calisto_accel_gyro.sensors[0].measured_data[0]
        == flight_calisto_accel_gyro.sensors[0].measured_data[1]
    )

    gyroscope = flight_calisto_accel_gyro.rocket.sensors[2].component
    time, wx, wy, wz = zip(*gyroscope.measured_data)
    wx = np.array(wx)
    wy = np.array(wy)
    wz = np.array(wz)
    w = np.sqrt(wx**2 + wy**2 + wz**2)
    flight_wx = np.array(flight_calisto_accel_gyro.w1(time))
    flight_wy = np.array(flight_calisto_accel_gyro.w2(time))
    flight_wz = np.array(flight_calisto_accel_gyro.w3(time))
    sim_w = np.sqrt(flight_wx**2 + flight_wy**2 + flight_wz**2)
    assert np.allclose(w, sim_w, atol=1e-12)


def test_export_sensor_data(flight_calisto_accel_gyro):
    """Test the export of sensor data.

    Parameters
    ----------
    flight_calisto_accel_gyro : Flight
        Pytest fixture for the flight of the calisto rocket with an ideal accelerometer and a gyroscope.
    """
    flight_calisto_accel_gyro.export_sensor_data("test_sensor_data.json")
    # read the json and parse as dict
    filename = "test_sensor_data.json"
    with open(filename, "r") as f:
        data = f.read()
        sensor_data = json.loads(data)
    # convert list of tuples into list of lists to compare with the json
    flight_calisto_accel_gyro.sensors[0].measured_data[0] = [
        list(measurement)
        for measurement in flight_calisto_accel_gyro.sensors[0].measured_data[0]
    ]
    flight_calisto_accel_gyro.sensors[1].measured_data[1] = [
        list(measurement)
        for measurement in flight_calisto_accel_gyro.sensors[1].measured_data[1]
    ]
    flight_calisto_accel_gyro.sensors[2].measured_data = [
        list(measurement)
        for measurement in flight_calisto_accel_gyro.sensors[2].measured_data
    ]
    assert (
        sensor_data["Accelerometer"]["1"]
        == flight_calisto_accel_gyro.sensors[0].measured_data[0]
    )
    assert (
        sensor_data["Accelerometer"]["2"]
        == flight_calisto_accel_gyro.sensors[1].measured_data[1]
    )
    assert (
        sensor_data["Gyroscope"] == flight_calisto_accel_gyro.sensors[2].measured_data
    )
    os.remove(filename)
