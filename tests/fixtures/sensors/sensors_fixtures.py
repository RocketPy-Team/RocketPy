import pytest
from rocketpy import Accelerometer, Gyroscope


@pytest.fixture
def noisy_rotated_accelerometer():
    """Returns an accelerometer with all parameters set to non-default values,
    i.e. with noise and rotation."""
    # mpu6050 approx values
    return Accelerometer(
        sampling_rate=100,
        orientation=(60, 60, 60),
        noise_density=[0, 0.03, 0.05],
        random_walk=[0, 0.01, 0.02],
        constant_bias=[0, 0.3, 0.5],
        operating_temperature=25,
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
    # mpu6050 approx values
    return Gyroscope(
        sampling_rate=100,
        orientation=(-60, -60, -60),
        noise_density=[0, 0.03, 0.05],
        random_walk=[0, 0.01, 0.02],
        constant_bias=[0, 0.3, 0.5],
        operating_temperature=25,
        temperature_bias=[0, 0.01, 0.02],
        temperature_scale_factor=[0, 0.01, 0.02],
        cross_axis_sensitivity=0.5,
        acceleration_sensitivity=[0, 0.0008, 0.0017],
        name="Gyroscope",
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
def ideal_accelerometer():
    return Accelerometer(
        sampling_rate=100,
    )


@pytest.fixture
def ideal_gyroscope():
    return Gyroscope(
        sampling_rate=100,
    )
