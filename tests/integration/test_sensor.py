import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.components import Components
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.barometer import Barometer
from rocketpy.sensors.gnss_receiver import GnssReceiver
from rocketpy.sensors.gyroscope import Gyroscope


def test_sensor_on_rocket(calisto_with_sensors):
    """Test the sensor on the rocket.

    Parameters
    ----------
    calisto_with_sensors : Rocket
        Pytest fixture for the calisto rocket with a set of ideal sensors.
    """
    sensors = calisto_with_sensors.sensors
    assert isinstance(sensors, Components)
    assert isinstance(sensors[0].component, Accelerometer)
    assert isinstance(sensors[1].position, Vector)
    assert isinstance(sensors[2].component, Gyroscope)
    assert isinstance(sensors[2].position, Vector)
    assert isinstance(sensors[3].component, Barometer)
    assert isinstance(sensors[3].position, Vector)
    assert isinstance(sensors[4].component, GnssReceiver)
    assert isinstance(sensors[4].position, Vector)


class TestIdealSensors:
    """Test a flight with ideal sensors on the rocket."""

    @pytest.fixture(autouse=True)
    def setup(self, flight_calisto_with_sensors):
        """Setup an flight fixture for all tests."""
        self.flight = flight_calisto_with_sensors

    def test_accelerometer(self):
        """Test an ideal accelerometer."""
        accelerometer = self.flight.rocket.sensors[0].component
        time, ax, ay, az = zip(*accelerometer.measured_data[0])
        sim_ax = self.flight.ax_body_frame(time)
        sim_ay = self.flight.ay_body_frame(time)
        sim_az = self.flight.az_body_frame(time)

        assert np.allclose(ax, sim_ax, atol=1e-12)
        assert np.allclose(ay, sim_ay, atol=1e-12)
        assert np.allclose(az, sim_az, atol=1e-12)

        # check if both added accelerometer instances saved the same data
        assert (
            self.flight.sensors[0].measured_data[0]
            == self.flight.sensors[0].measured_data[1]
        )

    def test_gyroscope(self):
        """Test an ideal gyroscope."""
        gyroscope = self.flight.rocket.sensors[2].component
        time, wx, wy, wz = zip(*gyroscope.measured_data)
        sim_wx = self.flight.w1(time)
        sim_wy = self.flight.w2(time)
        sim_wz = self.flight.w3(time)

        assert np.allclose(wx, sim_wx, atol=1e-12)
        assert np.allclose(wy, sim_wy, atol=1e-12)
        assert np.allclose(wz, sim_wz, atol=1e-12)

    def test_barometer(self):
        """Test an ideal barometer."""
        barometer = self.flight.rocket.sensors[3].component
        time, pressure = zip(*barometer.measured_data)
        pressure = np.array(pressure)
        sim_data = self.flight.pressure(time)
        assert np.allclose(pressure, sim_data, atol=1e-12)

    def test_gnss_receiver(self):
        """Test an ideal GnssReceiver."""
        gnss = self.flight.rocket.sensors[4].component
        time, latitude, longitude, altitude = zip(*gnss.measured_data)
        sim_latitude = self.flight.latitude(time)
        sim_longitude = self.flight.longitude(time)
        sim_altitude = self.flight.altitude(time)
        assert np.allclose(np.array(latitude), sim_latitude, atol=1e-12)
        assert np.allclose(np.array(longitude), sim_longitude, atol=1e-12)
        assert np.allclose(np.array(altitude), sim_altitude, atol=1e-12)


@pytest.mark.parametrize("plane", ["xz", "yz"])
@patch("matplotlib.pyplot.show")
def test_draw(mock_show, calisto_with_sensors, plane):  # pylint: disable=unused-argument
    """Test the drawing of the sensors."""
    calisto_with_sensors.draw(plane=plane)


def test_export_all_sensors_data(flight_calisto_with_sensors):
    """Test the export of sensor data.

    Parameters
    ----------
    flight_calisto_with_sensors : Flight
        Pytest fixture for the flight of the calisto rocket with a set of ideal
        sensors.
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
    flight_calisto_with_sensors.sensors[3].measured_data = [
        list(measurement)
        for measurement in flight_calisto_with_sensors.sensors[3].measured_data
    ]
    flight_calisto_with_sensors.sensors[4].measured_data = [
        list(measurement)
        for measurement in flight_calisto_with_sensors.sensors[4].measured_data
    ]
    assert (
        sensor_data["Accelerometer"]
        == flight_calisto_with_sensors.sensors[0].measured_data
    )
    assert (
        sensor_data["Gyroscope"] == flight_calisto_with_sensors.sensors[2].measured_data
    )
    assert (
        sensor_data["Barometer"] == flight_calisto_with_sensors.sensors[3].measured_data
    )
    assert (
        sensor_data["GnssReceiver"]
        == flight_calisto_with_sensors.sensors[4].measured_data
    )
    os.remove(filename)


def test_export_single_sensor_data(flight_calisto_with_sensors):
    """Test the export of a single sensor data.

    Parameters
    ----------
    flight_calisto_with_sensors : Flight
        Pytest fixture for the flight of the calisto rocket with a set of ideal
        sensors.
    """
    flight_calisto_with_sensors.export_sensor_data("test_sensor_data.json", "Gyroscope")
    # read the json and parse as dict
    filename = "test_sensor_data.json"
    with open(filename, "r") as f:
        data = f.read()
        sensor_data = json.loads(data)
    # convert list of tuples into list of lists to compare with the json
    flight_calisto_with_sensors.sensors[2].measured_data = [
        list(measurement)
        for measurement in flight_calisto_with_sensors.sensors[2].measured_data
    ]
    assert (
        sensor_data["Gyroscope"] == flight_calisto_with_sensors.sensors[2].measured_data
    )
    os.remove(filename)
