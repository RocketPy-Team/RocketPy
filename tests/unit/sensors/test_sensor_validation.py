"""Validation and dunder coverage for the sensor base classes.

These exercise the argument-validation error paths and the small ``__repr__`` /
``__call__`` helpers on ``Sensor`` / ``InertialSensor`` that the noise-focused
tests never reach, so the base class is fully covered.
"""

import numpy as np
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


@pytest.mark.parametrize(
    "seed",
    [np.random.default_rng(5), np.random.PCG64(5)],
    ids=["generator", "bit_generator"],
)
def test_live_rng_objects_are_rejected(seed):
    """A generator's state advances as noise is drawn, so it cannot describe
    the stream the way an int or a ``SeedSequence`` does."""
    with pytest.raises(TypeError, match="seed"):
        Barometer(sampling_rate=1, seed=seed)


@pytest.mark.parametrize(
    "seed",
    [None, 0, 5, np.int64(5), 2**128 - 1],
    ids=["none", "zero", "int", "numpy_int", "wide_int"],
)
def test_int_and_none_seeds_are_accepted(seed):
    """The check must not catch seeds that already work.

    numpy integers serialize through ``RocketPyEncoder``, and #1054 hands each
    model a plain 128-bit int, so both have to pass.
    """
    assert Barometer(sampling_rate=1, seed=seed).to_dict()["seed"] == seed


def test_seed_sequence_is_accepted():
    """#1124 made ``SeedSequence`` serializable, so this check must let it by."""
    seed = np.random.SeedSequence(5)
    assert Barometer(sampling_rate=1, seed=seed).to_dict()["seed"] is seed


def test_sequence_of_ints_is_accepted():
    """``default_rng`` takes a sequence of ints and json writes it out as a
    list, so the signature names it and the check has to let it by."""
    assert Barometer(sampling_rate=1, seed=[1, 2]).to_dict()["seed"] == [1, 2]


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
