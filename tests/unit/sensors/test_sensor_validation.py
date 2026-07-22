"""Validation and dunder coverage for the sensor base classes.

These exercise the argument-validation error paths and the small ``__repr__`` /
``__call__`` helpers on ``Sensor`` / ``InertialSensor`` that the noise-focused
tests never reach, so the base class is fully covered.
"""

import pytest

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.barometer import Barometer


def test_measurement_range_wrong_length_raises():
    with pytest.raises(ValueError, match="measurement range"):
        Accelerometer(sampling_rate=1, measurement_range=(1, 2, 3))


def test_measurement_range_wrong_type_raises():
    with pytest.raises(ValueError, match="measurement range"):
        Accelerometer(sampling_rate=1, measurement_range="not-a-range")


def test_orientation_matrix_is_accepted():
    accel = Accelerometer(
        sampling_rate=1, orientation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert accel.rotation_sensor_to_body is not None


def test_orientation_wrong_length_raises():
    with pytest.raises(ValueError, match="orientation"):
        Accelerometer(sampling_rate=1, orientation=(1, 2))


def test_vectorize_input_wrong_type_raises():
    with pytest.raises(ValueError, match="noise_density"):
        Accelerometer(sampling_rate=1, noise_density="not-a-vector")


def test_repr_returns_name():
    assert repr(Barometer(sampling_rate=1, name="baro")) == "baro"


def test_export_measured_data_rejects_bad_format(tmp_path):
    with pytest.raises(ValueError, match="file_format"):
        Barometer(sampling_rate=1).export_measured_data(
            str(tmp_path / "out"), file_format="xml"
        )


def test_call_dispatches_to_measure(example_plain_env):
    """Calling a sensor forwards to ``measure`` and records one sample."""
    barometer = Barometer(sampling_rate=1)
    barometer(
        3.3,
        u=[0, 0, 1000, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        relative_position=Vector([0, 0, 0]),
        environment=example_plain_env,
    )
    assert len(barometer.measured_data) == 1
