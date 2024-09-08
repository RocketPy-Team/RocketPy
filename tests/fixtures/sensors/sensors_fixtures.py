import pytest

from rocketpy import Accelerometer, Gyroscope
from rocketpy.sensors.barometer import Barometer
from rocketpy.sensors.gnss_receiver import GnssReceiver


@pytest.fixture
def noisy_rotated_accelerometer():
    """Returns an accelerometer with all parameters set to non-default values,
    i.e. with noise and rotation."""
    # mpu6050 approx values, variances are made up
    return Accelerometer(
        sampling_rate=100,
        orientation=(60, 60, 60),
        noise_density=[0, 0.03, 0.05],
        noise_variance=1.01,
        random_walk_density=[0, 0.01, 0.02],
        random_walk_variance=[1, 1, 1.05],
        constant_bias=[0, 0.3, 0.5],
        operating_temperature=25 + 273.15,
        temperature_bias=[0, 0.01, 0.02],
        temperature_scale_factor=[0, 0.01, 0.02],
        cross_axis_sensitivity=0.5,
        consider_gravity=True,
        name="Accelerometer",
    )


@pytest.fixture
def noisy_rotated_gyroscope():
    """Returns a gyroscope with all parameters set to non-default values,
    i.e. with noise and rotation."""
    # mpu6050 approx values, variances are made up
    return Gyroscope(
        sampling_rate=100,
        orientation=(-60, -60, -60),
        noise_density=[0, 0.03, 0.05],
        noise_variance=1.01,
        random_walk_density=[0, 0.01, 0.02],
        random_walk_variance=[1, 1, 1.05],
        constant_bias=[0, 0.3, 0.5],
        operating_temperature=25 + 273.15,
        temperature_bias=[0, 0.01, 0.02],
        temperature_scale_factor=[0, 0.01, 0.02],
        cross_axis_sensitivity=0.5,
        acceleration_sensitivity=[0, 0.0008, 0.0017],
        name="Gyroscope",
    )


@pytest.fixture
def noisy_barometer():
    """Returns a barometer with all parameters set to non-default values,
    i.e. with noise and temperature drift."""
    return Barometer(
        sampling_rate=50,
        noise_density=19,
        noise_variance=19,
        random_walk_density=0.01,
        constant_bias=1000,
        operating_temperature=25 + 273.15,
        temperature_bias=0.02,
        temperature_scale_factor=0.02,
    )


@pytest.fixture
def noisy_gnss():
    return GnssReceiver(
        sampling_rate=1,
        position_accuracy=1,
        altitude_accuracy=1,
    )


@pytest.fixture
def quantized_accelerometer():
    """Returns an accelerometer with all parameters set to non-default values,
    i.e. with noise and rotation."""
    return Accelerometer(
        sampling_rate=100,
        measurement_range=2,
        resolution=0.4882,
    )


@pytest.fixture
def quantized_gyroscope():
    """Returns a gyroscope with all parameters set to non-default values,
    i.e. with noise and rotation."""
    return Gyroscope(
        sampling_rate=100,
        measurement_range=15,
        resolution=0.4882,
    )


@pytest.fixture
def quantized_barometer():
    """Returns a barometer with all parameters set to non-default values,
    i.e. with noise and temperature drift."""
    return Barometer(
        sampling_rate=50,
        measurement_range=7e4,
        resolution=0.16,
    )


@pytest.fixture
def ideal_accelerometer():
    return Accelerometer(
        sampling_rate=10,
    )


@pytest.fixture
def ideal_gyroscope():
    return Gyroscope(
        sampling_rate=10,
    )


@pytest.fixture
def ideal_barometer():
    return Barometer(
        sampling_rate=10,
    )


@pytest.fixture
def ideal_gnss():
    return GnssReceiver(
        sampling_rate=1,
    )
